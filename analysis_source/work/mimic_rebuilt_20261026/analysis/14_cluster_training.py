"""Outcome-blinded full-training Gower/PAM and outlier sensitivity analysis."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_samples, adjusted_rand_score, adjusted_mutual_info_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime/clustering'))
import kmedoids

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fit_transform(frame,features):
    filled=frame[features].copy();params={};limits={}
    for col in features:
        value=filled[col].mode().iloc[0] if col=='gender' else float(filled[col].median())
        if pd.isna(value):raise ValueError('Entire fitting feature missing: '+col)
        params[col]=str(value) if col=='gender' else value
        filled[col]=filled[col].fillna(value)
        if col!='gender':limits[col]=[float(filled[col].min()),float(filled[col].max())]
    return filled,params,limits

def apply_transform(frame,params):
    return frame[list(params)].fillna(params)

def encode(frame,features,limits):
    numeric=[]
    for col in features:
        if col=='gender':continue
        lo,hi=limits[col]
        values=frame[col].to_numpy(dtype=float)
        numeric.append(np.zeros(len(frame)) if hi==lo else (np.clip(values,lo,hi)-lo)/(hi-lo))
    num=np.column_stack(numeric)
    cat=frame.gender.astype(str).to_numpy()
    if not np.isfinite(num).all():raise ValueError('Non-finite Gower input')
    return num,cat

def gower(left,right,features,limits):
    a,ac=encode(left,features,limits); b,bc=encode(right,features,limits)
    result=np.empty((len(left),len(right)),dtype=np.float32)
    for start in range(0,len(left),256):
        stop=min(start+256,len(left))
        result[start:stop]=(cdist(a[start:stop],b,metric='cityblock')+(ac[start:stop,None]!=bc[None,:]))/len(features)
    if not np.isfinite(result).all() or result.min()<0 or result.max()>1.000001:raise ValueError('Invalid Gower distance')
    return result

def fit_pam(distance,k,restarts,seed,max_iter):
    rng=np.random.default_rng(seed);best=None;rows=[]
    for restart in range(restarts):
        init=rng.choice(len(distance),size=k,replace=False).astype(np.uintp)
        result=kmedoids.fastpam1(distance,init,max_iter=max_iter)
        medoids=np.sort(np.asarray(result.medoids,dtype=int))
        labels=np.argmin(distance[:,medoids],axis=1)
        loss=float(distance[np.arange(len(distance)),medoids[labels]].sum(dtype=np.float64))
        if len(np.unique(labels))!=k:raise ValueError('PAM has empty cluster')
        if result.n_iter>=max_iter:raise ValueError('PAM reached iteration limit')
        rows.append({'restart':restart,'loss':loss,'iterations':int(result.n_iter),'swaps':int(result.n_swap)})
        if best is None or loss<best['loss']:best={'labels':labels,'medoids':medoids,'loss':loss}
    return best,rows

def aligned(left,right,k):
    table=np.zeros((k,k),dtype=int)
    np.add.at(table,(left,right),1)
    rows,cols=linear_sum_assignment(-table)
    mapping={int(c):int(r) for r,c in zip(rows,cols)}
    mapped=np.array([mapping[int(x)] for x in right],dtype=int)
    return float(np.mean(left==mapped)),mapped,table

def main():
    start=time.time()
    config=json.loads((ROOT/'config/clustering_v4.json').read_text(encoding='utf-8'))
    base_config=json.loads((ROOT/'config/analysis_v2.json').read_text(encoding='utf-8'))
    loc=json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))
    features=base_config['primary_features'];parent=Path(loc['data_path'])
    input_manifest=json.loads((Path(loc['report_path'])/'run_manifest.json').read_text(encoding='utf-8'))
    run=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    data=parent/'clustering_v4'/run;data.mkdir(parents=True)
    reports=ROOT/'outputs/reports/clustering_v4'/run;reports.mkdir(parents=True)
    tables=ROOT/'outputs/tables/clustering_v4'/run;tables.mkdir(parents=True)
    cols=['subject_id','hadm_id','stay_id','split','phenotype_eligible']+features
    frames={};paths={}
    for variant,name in [('primary','reviewed_unimputed.parquet'),('strict','strict_screen_sensitivity_unimputed.parquet')]:
        path=parent/name
        if sha(path)!=input_manifest['output_sha256'][name]:raise ValueError('Preprocessing input hash changed')
        # Project only permitted predictors, identifiers and partition fields; no outcome fields loaded.
        frame=pd.read_parquet(path,columns=cols)
        frame=frame[frame.split.eq('train')&frame.phenotype_eligible].sort_values('stay_id').reset_index(drop=True)
        frames[variant]=frame;paths[variant]=path
    train=frames['primary'];strict=frames['strict']
    if not train[['subject_id','stay_id']].equals(strict[['subject_id','stay_id']]):raise ValueError('Sensitivity population mismatch; compare common IDs explicitly')
    filled,params,limits=fit_transform(train,features)
    strict_filled,strict_params,strict_limits=fit_transform(strict,features)
    manifest={'status':'running','date':run,'config':config,'input_sha256':{v:sha(p) for v,p in paths.items()},'script_sha256':sha(__file__),'outcome_columns_loaded':False,'partition_used':'eligible_train_only','n_train':len(train),'failure':None,'sources':{'kmedoids':'https://python-kmedoids.readthedocs.io/en/latest/index.html','silhouette':'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html'}}
    snapshot=reports/'source_snapshot';snapshot.mkdir()
    for rel in ['analysis/14_cluster_training.py','config/clustering_v4.json','config/analysis_v2.json','config/preprocessing_review_v3.json']:
        file=ROOT/rel;(snapshot/file.name).write_bytes(file.read_bytes())
    manifest['software_versions']={name:metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn']}
    manifest['software_versions']['kmedoids']=metadata.distribution('kmedoids').version
    def record():
        (reports/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        (ROOT/'logs/clustering_latest.json').write_text(json.dumps({'data_path':str(data),'report_path':str(reports),'tables_path':str(tables),'status':manifest['status']},indent=2,ensure_ascii=False),encoding='utf-8')
    record()
    try:
        print(f'Building full training Gower distance: n={len(train)},p={len(features)}',flush=True)
        D=gower(filled,filled,features,limits);np.fill_diagonal(D,0)
        if not np.allclose(D,D.T,rtol=0,atol=1e-7):raise ValueError('Asymmetric distance')
        np.save(data/'primary_training_gower.npy',D)
        train[['subject_id','hadm_id','stay_id']].to_parquet(data/'training_order.parquet',index=False)
        (data/'primary_transform.json').write_text(json.dumps({'features':features,'imputer':params,'numeric_minmax':limits},indent=2),encoding='utf-8')
        results={};metrics=[];restarts=[];profiles=[];missing_rows=[]
        missing=train[features].isna()
        for k in config['candidate_k']:
            print(f'Primary full-training PAM k={k}',flush=True)
            fit,runs=fit_pam(D,k,config['main_restarts'],config['seed']+k,config['max_iter'])
            results[k]=fit
            silhouette=silhouette_samples(D,fit['labels'],metric='precomputed')
            sizes=np.bincount(fit['labels'],minlength=k)
            genders=pd.crosstab(fit['labels'],filled.gender).reindex(range(k),fill_value=0)
            purity=float(genders.max(axis=1).sum()/len(train))
            fractions=missing.mean(axis=1).groupby(fit['labels']).mean()
            metrics.append({'k':k,'n':len(train),'silhouette_mean':float(silhouette.mean()),'silhouette_negative_fraction':float((silhouette<0).mean()),'min_cluster_n':int(sizes.min()),'min_cluster_fraction':float(sizes.min()/len(train)),'loss':fit['loss'],'gender_purity':purity,'gender_ami':float(adjusted_mutual_info_score(filled.gender,fit['labels'])),'cluster_missing_fraction_spread':float(fractions.max()-fractions.min())})
            assign=train[['subject_id','hadm_id','stay_id']].copy();assign['cluster']=fit['labels']+1;assign['silhouette']=silhouette
            assign.to_parquet(data/f'primary_k{k}_training_labels.parquet',index=False)
            filled.iloc[fit['medoids']].assign(cluster=np.arange(1,k+1)).to_parquet(data/f'primary_k{k}_medoid_features.parquet',index=False)
            train.iloc[fit['medoids']][['subject_id','hadm_id','stay_id']].assign(cluster=np.arange(1,k+1)).to_parquet(data/f'primary_k{k}_medoid_ids.parquet',index=False)
            for row in runs:restarts.append({'variant':'primary','k':k,**row})
            for cluster in range(k):
                mask=fit['labels']==cluster
                for feature in features:
                    observed=train.loc[mask,feature]
                    row={'k':k,'cluster':cluster+1,'feature':feature,'n_cluster':int(mask.sum()),'missing_fraction':float(observed.isna().mean())}
                    if feature=='gender':row['female_fraction']=float(observed.eq('F').mean())
                    else:row.update(observed_median=float(observed.median()) if observed.notna().any() else None,observed_q1=float(observed.quantile(.25)) if observed.notna().any() else None,observed_q3=float(observed.quantile(.75)) if observed.notna().any() else None,imputed_median=float(filled.loc[mask,feature].median()))
                    profiles.append(row)
                missing_rows.append({'k':k,'cluster':cluster+1,'n':int(mask.sum()),'mean_missing_fraction':float(missing.loc[mask].mean(axis=1).mean()),'median_missing_features':float(missing.loc[mask].sum(axis=1).median()),'gender_female_fraction':float(filled.loc[mask,'gender'].eq('F').mean())})
            pd.DataFrame(metrics).to_csv(tables/'primary_k_metrics_partial.csv',index=False)
        stability=[]
        rng=np.random.default_rng(config['seed']+10000)
        for rep in range(config['stability_repetitions']):
            sample=np.sort(rng.choice(len(train),int(config['stability_fraction']*len(train)),replace=False))
            sampled=train.iloc[sample].reset_index(drop=True)
            sub,subparams,sublimits=fit_transform(sampled,features)
            Ds=gower(sub,sub,features,sublimits);np.fill_diagonal(Ds,0)
            sampled[['subject_id','hadm_id','stay_id']].to_parquet(data/f'stability_subsample_{rep+1:02d}.parquet',index=False)
            entire=apply_transform(train,subparams)
            for k in config['candidate_k']:
                fit,_=fit_pam(Ds,k,config['stability_restarts'],config['seed']+20000+rep*100+k,config['max_iter'])
                predicted=np.argmin(gower(entire,sub.iloc[fit['medoids']],features,sublimits),axis=1)
                baseline=results[k]['labels'];agreement,_,_=aligned(baseline,predicted,k)
                stability.append({'rep':rep+1,'k':k,'ari':float(adjusted_rand_score(baseline,predicted)),'aligned_agreement':agreement,'subsample_n':len(sample)})
            del Ds
            pd.DataFrame(stability).to_csv(tables/'training_subsample_stability.csv',index=False)
            print(f'Training subsample stability {rep+1}/{config["stability_repetitions"]} complete',flush=True)
        summary=pd.DataFrame(stability).groupby('k').agg(stability_ari_mean=('ari','mean'),stability_ari_min=('ari','min'),stability_ari_q10=('ari',lambda x:x.quantile(.1)),stability_agreement_mean=('aligned_agreement','mean')).reset_index()
        metrics=pd.DataFrame(metrics).merge(summary,on='k',validate='one_to_one')
        metrics['passes_size_stability_screen']=(metrics.min_cluster_fraction>=config['selection_min_cluster_fraction'])&(metrics.stability_ari_mean>=config['selection_min_mean_stability_ari'])
        candidates=metrics[metrics.passes_size_stability_screen]
        recommended=None
        if not candidates.empty:
            best=float(candidates.silhouette_mean.max())
            recommended=int(candidates.loc[candidates.silhouette_mean.ge(best-config['silhouette_tie_tolerance']),'k'].min())
        sensitivity=[];changes=[];switch_counts=np.zeros(len(train),dtype=int)
        for variant,scales in [('strict_main_scales',limits),('strict_refit_scales',strict_limits)]:
            print('Outlier sensitivity: '+variant,flush=True)
            S=gower(strict_filled,strict_filled,features,scales);np.fill_diagonal(S,0)
            for k in config['candidate_k']:
                fit,runs=fit_pam(S,k,config['sensitivity_restarts'],config['seed']+k,config['max_iter'])
                baseline=results[k]['labels'];agree,mapped,cross=aligned(baseline,fit['labels'],k)
                ari=float(adjusted_rand_score(baseline,fit['labels']))
                sizes=np.bincount(fit['labels'],minlength=k)
                row={'variant':variant,'k':k,'ari_vs_primary':ari,'aligned_agreement':agree,'changed_assignment_fraction':1-agree,'silhouette_mean':float(silhouette_samples(S,fit['labels'],metric='precomputed').mean()),'min_cluster_fraction':float(sizes.min()/len(train))}
                row['triggers_influence_review']=ari<config['influence_review_ari_below'] or 1-agree>config['influence_review_aligned_disagreement_above']
                sensitivity.append(row)
                for a in range(k):
                    for b in range(k):changes.append({'variant':variant,'k':k,'primary_cluster':a+1,'sensitivity_cluster_unaligned':b+1,'n':int(cross[a,b])})
                assign=train[['subject_id','hadm_id','stay_id']].copy();assign['primary_cluster']=baseline+1;assign['sensitivity_cluster']=fit['labels']+1;assign['sensitivity_cluster_aligned']=mapped+1
                assign.to_parquet(data/f'{variant}_k{k}_labels.parquet',index=False)
                for r in runs:restarts.append({'variant':variant,'k':k,**r})
                if k==recommended:switch_counts+=(mapped!=baseline)
            del S
            pd.DataFrame(sensitivity).to_csv(tables/'outlier_sensitivity.csv',index=False)
        metrics.to_csv(tables/'primary_k_metrics.csv',index=False)
        pd.DataFrame(restarts).to_csv(tables/'pam_restarts.csv',index=False)
        pd.DataFrame(profiles).to_csv(tables/'training_cluster_profiles.csv',index=False)
        pd.DataFrame(missing_rows).to_csv(tables/'training_cluster_missingness.csv',index=False)
        pd.DataFrame(changes).to_csv(tables/'sensitivity_cluster_crosstab.csv',index=False)
        decisions={'provisional_k':recommended,'final_clinical_k_frozen':False,'selection_rule':config['selection_rule'],'outlier_influence_review_needed_at_provisional_k':None,'gender_dominance_flag':None,'missingness_spread_flag':None,'models_started':False}
        if recommended is not None:
            row=metrics.set_index('k').loc[recommended]
            decisions['gender_dominance_flag']=bool(row.gender_purity>=config['diagnostic_gender_purity_above'])
            decisions['missingness_spread_flag']=bool(row.cluster_missing_fraction_spread>config['diagnostic_missing_fraction_spread_above'])
            decisions['outlier_influence_review_needed_at_provisional_k']=bool(pd.DataFrame(sensitivity).query('k==@recommended').triggers_influence_review.any())
            if decisions['outlier_influence_review_needed_at_provisional_k']:
                changed=(train[features].isna()!=strict[features].isna()).any(axis=1)
                influence=train.loc[changed,['subject_id','hadm_id','stay_id']].copy();influence['assignment_changed_conditions']=switch_counts[changed]
                influence.to_parquet(data/'flagged_patients_for_focused_review.parquet',index=False)
                pd.DataFrame({'feature':features,'training_values_newly_screened':[(train[f].notna()&strict[f].isna()).sum() for f in features],'main_scale_range':[None if f=='gender' else limits[f][1]-limits[f][0] for f in features],'strict_scale_range':[None if f=='gender' else strict_limits[f][1]-strict_limits[f][0] for f in features]}).to_csv(tables/'outlier_scale_influence.csv',index=False)
        (reports/'selection_review.json').write_text(json.dumps(decisions,indent=2,ensure_ascii=False),encoding='utf-8')
        manifest.update(status='training_clustering_and_sensitivity_complete',elapsed_seconds=time.time()-start,selection_review=decisions,output_data_sha256={file.name:sha(file) for file in data.iterdir() if file.is_file()})
        record()
        print('Clustering completed; provisional selection: '+str(recommended),flush=True)
        print(json.dumps(decisions,ensure_ascii=False),flush=True)
    except Exception as exc:
        manifest.update(status='failed',failure={'type':type(exc).__name__,'reason':str(exc)},elapsed_seconds=time.time()-start)
        record();raise

if __name__=='__main__':main()
