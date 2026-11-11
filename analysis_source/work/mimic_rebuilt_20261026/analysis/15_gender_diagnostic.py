"""Post-result numeric-only diagnostic; retain registered 18-input main results."""
from pathlib import Path
import importlib
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples,adjusted_rand_score,adjusted_mutual_info_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
c=importlib.import_module('14_cluster_training')

def numeric_gower(a,b,features,limits):
    def encode(frame):
        result=[]
        for col in features:
            lo,hi=limits[col]
            result.append(np.zeros(len(frame)) if hi==lo else (np.clip(frame[col].to_numpy(dtype=float),lo,hi)-lo)/(hi-lo))
        return np.column_stack(result)
    x,y=encode(a),encode(b)
    output=np.empty((len(a),len(b)),dtype=np.float32)
    for start in range(0,len(a),256):output[start:start+256]=cdist(x[start:start+256],y,'cityblock')/len(features)
    assert np.isfinite(output).all() and output.min()>=0 and output.max()<=1.000001
    return output

def main():
    started=time.time()
    loc=json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))
    main_loc=json.loads((ROOT/'logs/clustering_latest.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/'config/clustering_v4.json').read_text(encoding='utf-8'))
    extra=json.loads((ROOT/'config/clustering_gender_diagnostic.json').read_text(encoding='utf-8'))
    features=[f for f in json.loads((ROOT/'config/analysis_v2.json').read_text(encoding='utf-8'))['primary_features'] if f!='gender']
    parent=Path(loc['data_path'])
    reports=Path(main_loc['report_path'])/'gender_diagnostic';reports.mkdir(exist_ok=True)
    tables=Path(main_loc['tables_path'])/'gender_diagnostic';tables.mkdir(exist_ok=True)
    data=Path(main_loc['data_path'])/'gender_diagnostic';data.mkdir(exist_ok=True)
    cols=['subject_id','stay_id','hadm_id','split','phenotype_eligible','gender']+features
    def read(name):
        df=pd.read_parquet(parent/name,columns=cols)
        return df[df.split.eq('train')&df.phenotype_eligible].sort_values('stay_id').reset_index(drop=True)
    train=read('reviewed_unimputed.parquet');strict=read('strict_screen_sensitivity_unimputed.parquet')
    assert train.stay_id.equals(strict.stay_id)
    filled,params,limits=c.fit_transform(train,features)
    strict_filled,strict_params,strict_limits=c.fit_transform(strict,features)
    manifest={'status':'running','scope':extra,'primary_analysis_replaced':False,'outcome_columns_loaded':False,'script_sha256':c.sha(__file__),'config_sha256':c.sha(ROOT/'config/clustering_gender_diagnostic.json'),'input_sha256':{n:c.sha(parent/n) for n in ['reviewed_unimputed.parquet','strict_screen_sensitivity_unimputed.parquet']}}
    for file in [Path(__file__),ROOT/'config/clustering_gender_diagnostic.json']:(reports/file.name).write_bytes(file.read_bytes())
    def record():(reports/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    record()
    try:
        D=numeric_gower(filled,filled,features,limits);np.fill_diagonal(D,0)
        fits={};metrics=[];profiles=[];sizes_rows=[]
        for k in config['candidate_k']:
            print(f'Gender-omitted diagnostic full-training PAM k={k}',flush=True)
            fit,_=c.fit_pam(D,k,extra['restarts'],config['seed']+k,config['max_iter']);fits[k]=fit
            silhouette=silhouette_samples(D,fit['labels'],metric='precomputed')
            size=np.bincount(fit['labels'],minlength=k)
            metrics.append({'k':k,'silhouette_mean':float(silhouette.mean()),'min_cluster_fraction':float(size.min()/len(train)),'gender_ami':float(adjusted_mutual_info_score(train.gender,fit['labels'])),'loss':fit['loss']})
            labels=train[['subject_id','hadm_id','stay_id']].copy();labels['cluster']=fit['labels']+1
            labels.to_parquet(data/f'numeric17_k{k}_training_labels.parquet',index=False)
            filled.iloc[fit['medoids']].assign(cluster=np.arange(1,k+1)).to_parquet(data/f'numeric17_k{k}_medoids.parquet',index=False)
            for cluster in range(k):
                mask=fit['labels']==cluster
                sizes_rows.append({'k':k,'cluster':cluster+1,'n':int(mask.sum()),'female_fraction':float(train.loc[mask,'gender'].eq('F').mean()),'mean_missing_fraction':float(train.loc[mask,features].isna().mean(axis=1).mean())})
                for feature in features:
                    observed=train.loc[mask,feature]
                    profiles.append({'k':k,'cluster':cluster+1,'feature':feature,'observed_median':float(observed.median()) if observed.notna().any() else None,'observed_q1':float(observed.quantile(.25)) if observed.notna().any() else None,'observed_q3':float(observed.quantile(.75)) if observed.notna().any() else None,'missing_fraction':float(observed.isna().mean())})
            pd.DataFrame(metrics).to_csv(tables/'numeric17_k_metrics_partial.csv',index=False)
        stability=[];rng=np.random.default_rng(config['seed']+10000)
        for rep in range(extra['stability_repetitions']):
            ids=np.sort(rng.choice(len(train),int(extra['stability_fraction']*len(train)),replace=False))
            sample=train.iloc[ids].reset_index(drop=True);sub,imputer,scales=c.fit_transform(sample,features)
            Ds=numeric_gower(sub,sub,features,scales);np.fill_diagonal(Ds,0)
            entire=c.apply_transform(train,imputer)
            for k in config['candidate_k']:
                fit,_=c.fit_pam(Ds,k,extra['stability_restarts'],config['seed']+20000+100*rep+k,config['max_iter'])
                labels=np.argmin(numeric_gower(entire,sub.iloc[fit['medoids']],features,scales),axis=1)
                agreement,_,_=c.aligned(fits[k]['labels'],labels,k)
                stability.append({'rep':rep+1,'k':k,'ari':float(adjusted_rand_score(fits[k]['labels'],labels)),'aligned_agreement':agreement})
            del Ds
            pd.DataFrame(stability).to_csv(tables/'numeric17_training_stability.csv',index=False)
            print(f'Gender diagnostic stability {rep+1}/{extra["stability_repetitions"]}',flush=True)
        summary=pd.DataFrame(stability).groupby('k').agg(stability_ari_mean=('ari','mean'),stability_ari_min=('ari','min')).reset_index()
        metrics=pd.DataFrame(metrics).merge(summary,on='k')
        metrics['passes_size_stability_screen']=(metrics.min_cluster_fraction>=config['selection_min_cluster_fraction'])&(metrics.stability_ari_mean>=config['selection_min_mean_stability_ari'])
        candidates=metrics[metrics.passes_size_stability_screen]
        tentative=None
        if not candidates.empty:tentative=int(candidates.loc[candidates.silhouette_mean.ge(candidates.silhouette_mean.max()-config['silhouette_tie_tolerance']),'k'].min())
        sensitivity=[]
        for name,scales in [('strict_main_scales',limits),('strict_refit_scales',strict_limits)]:
            S=numeric_gower(strict_filled,strict_filled,features,scales);np.fill_diagonal(S,0)
            for k in config['candidate_k']:
                fit,_=c.fit_pam(S,k,extra['restarts'],config['seed']+k,config['max_iter'])
                agreement,mapped,_=c.aligned(fits[k]['labels'],fit['labels'],k)
                ari=float(adjusted_rand_score(fits[k]['labels'],fit['labels']))
                sensitivity.append({'variant':name,'k':k,'ari_vs_numeric17':ari,'aligned_agreement':agreement,'changed_fraction':1-agreement,'triggers_influence_review':ari<config['influence_review_ari_below'] or 1-agreement>config['influence_review_aligned_disagreement_above']})
                labels=train[['subject_id','hadm_id','stay_id']].copy();labels['primary_numeric17_cluster']=fits[k]['labels']+1;labels['strict_cluster_aligned']=mapped+1
                labels.to_parquet(data/f'{name}_numeric17_k{k}_labels.parquet',index=False)
            del S
            pd.DataFrame(sensitivity).to_csv(tables/'numeric17_outlier_sensitivity.csv',index=False)
            print('Gender diagnostic outlier comparison '+name+' complete',flush=True)
        metrics.to_csv(tables/'numeric17_k_metrics.csv',index=False)
        pd.DataFrame(profiles).to_csv(tables/'numeric17_profiles.csv',index=False)
        pd.DataFrame(sizes_rows).to_csv(tables/'numeric17_cluster_sizes_missingness.csv',index=False)
        decision={'diagnostic_candidate_k':tentative,'replaces_registered_primary':False,'clinical_k_frozen':False,'outlier_influence_at_candidate':None if tentative is None else bool(pd.DataFrame(sensitivity).query('k==@tentative').triggers_influence_review.any())}
        manifest.update(status='gender_diagnostic_complete',elapsed_seconds=time.time()-started,decision=decision,data_sha256={file.name:c.sha(file) for file in data.glob('*.parquet')})
        record();print(json.dumps(decision),flush=True)
    except Exception as exc:
        manifest.update(status='failed',failure={'type':type(exc).__name__,'reason':str(exc)});record();raise

if __name__=='__main__':main()
