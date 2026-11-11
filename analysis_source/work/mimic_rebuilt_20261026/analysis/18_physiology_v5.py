"""Outcome-blinded development: coverage, numerical Gower/PAM, RF-PMM comparisons.

Patient-level artifacts remain in data/. Custom RF-PMM is an exploratory
stochastic imputation comparison, not a certified proper Bayesian MI procedure.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import importlib
import importlib.metadata as metadata
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_samples, adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
c=importlib.import_module('14_cluster_training')
g=importlib.import_module('15_gender_diagnostic')
VITALS=['heart_rate','sbp','map','resp_rate','temp_c','spo2']
LABS=['lactate','wbc','platelet','hemoglobin','creatinine','bun','sodium','potassium','bicarbonate','glucose']
FEATURES=VITALS+LABS
IDS=['subject_id','stay_id','hadm_id']

def initialize_pam_worker(path):
    global WORKER_DISTANCE
    WORKER_DISTANCE=np.load(path,mmap_mode='r')

def pam_restart_worker(task):
    restart,initial,max_iter=task
    result=c.kmedoids.fastpam1(WORKER_DISTANCE,np.asarray(initial,dtype=np.uintp),max_iter=max_iter)
    medoids=np.sort(np.asarray(result.medoids,dtype=int))
    labels=np.argmin(WORKER_DISTANCE[:,medoids],axis=1)
    loss=float(WORKER_DISTANCE[np.arange(len(labels)),medoids[labels]].sum(dtype=np.float64))
    if len(np.unique(labels))!=len(medoids) or result.n_iter>=max_iter:raise ValueError('Invalid PAM convergence')
    return {'labels':labels,'medoids':medoids,'loss':loss},dict(restart=restart,loss=loss,iterations=int(result.n_iter),swaps=int(result.n_swap))

def parallel_pam(pool,n,k,restarts,seed,max_iter):
    rng=np.random.default_rng(seed)
    tasks=[(r,rng.choice(n,size=k,replace=False).tolist(),max_iter) for r in range(restarts)]
    outputs=list(pool.map(pam_restart_worker,tasks))
    return min((x[0] for x in outputs),key=lambda x:x['loss']),[x[1] for x in outputs]

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def eligible(frame,threshold):
    return (frame[FEATURES].notna().sum(axis=1).ge(threshold)
            &frame[VITALS].notna().any(axis=1)&frame[LABS].notna().any(axis=1))

def donor_draw(predicted,observed_prediction,observed_y,rng,n_donors):
    """Draw from the nearest ORIGINAL observed outcome donor values."""
    order=np.argsort(observed_prediction,kind='stable')
    scores=observed_prediction[order];values=observed_y[order]
    n=min(n_donors,len(scores));pos=np.searchsorted(scores,predicted)
    offsets=np.arange(-n,n+1)
    # Out-of-bounds candidates are invalid, not repeated edge donors.
    indices=pos[:,None]+offsets[None,:]
    valid=(indices>=0)&(indices<len(scores))
    clipped=np.clip(indices,0,len(scores)-1)
    distances=np.abs(scores[clipped]-predicted[:,None])
    distances[~valid]=np.inf
    nearest=np.argsort(distances,axis=1,kind='stable')[:,:n]
    # Include every strictly closer donor; randomize nth-absolute-distance
    # boundary ties across BOTH sides, including whole equal-score groups.
    radius=distances[np.arange(len(predicted)),nearest[:,-1]]
    tolerance=4*np.finfo(float).eps*np.maximum.reduce([np.ones(len(predicted)),np.abs(predicted),radius])
    lo=np.searchsorted(scores,predicted-radius-tolerance,side='left')
    inner_lo=np.searchsorted(scores,predicted-radius+tolerance,side='right')
    inner_hi=np.searchsorted(scores,predicted+radius-tolerance,side='left')
    hi=np.searchsorted(scores,predicted+radius+tolerance,side='right')
    inner_count=np.maximum(inner_hi-inner_lo,0)
    overlap=inner_hi<inner_lo
    left_count=np.where(overlap,hi-lo,inner_lo-lo)
    right_count=np.where(overlap,0,hi-inner_hi)
    inner_choice=inner_lo+rng.integers(np.maximum(inner_count,1))
    boundary_slot=rng.integers(np.maximum(left_count+right_count,1))
    boundary_choice=np.where(boundary_slot<left_count,lo+boundary_slot,inner_hi+boundary_slot-left_count)
    selected=np.where(rng.integers(n,size=len(predicted))<inner_count,inner_choice,boundary_choice)
    assert ((selected>=0)&(selected<len(scores))).all()
    return values[selected]

def rf_pmm(frame,config,seed,model_path=None,progress=None):
    original=frame[FEATURES].to_numpy(dtype=float)
    missing=np.isnan(original);x=original.copy();rng=np.random.default_rng(seed)
    medians=np.nanmedian(x,axis=0)
    if not np.isfinite(medians).all():raise ValueError('All-missing imputation fitting feature')
    x[missing]=np.broadcast_to(medians,x.shape)[missing]
    trace=[];models={}
    for cycle in range(config['cycles']):
        for j in range(len(FEATURES)):
            obs=~missing[:,j];mis=missing[:,j]
            if not mis.any():continue
            others=[i for i in range(len(FEATURES)) if i!=j]
            model=RandomForestRegressor(n_estimators=config['trees'],max_depth=config['max_depth'],
                min_samples_leaf=config['min_samples_leaf'],n_jobs=config['n_jobs'],
                random_state=int(rng.integers(2**31-1)),bootstrap=True)
            model.fit(x[obs][:,others],original[obs,j])
            observed_prediction=model.predict(x[obs][:,others])
            x[mis,j]=donor_draw(model.predict(x[mis][:,others]),observed_prediction,
                original[obs,j],rng,config['donors'])
            models[j]={'model':model,'donor_scores':observed_prediction,
                'donor_values':original[obs,j],'other_columns':others}
            trace.append({'cycle':cycle+1,'feature':FEATURES[j],
                'imputed_mean':float(x[mis,j].mean()),'imputed_sd':float(x[mis,j].std()),
                'n_imputed':int(mis.sum())})
        if progress:progress(cycle+1)
    assert np.array_equal(x[~missing],original[~missing])
    assert np.isfinite(x).all()
    if model_path:
        joblib.dump({'models':models,'medians':medians,'features':FEATURES,'seed':seed,
            'cycles':config['cycles'],'donors':config['donors'],
            'heldout_transform_validated':False},model_path,compress=3)
    return pd.DataFrame(x,columns=FEATURES),trace

def load(loc):
    cfg=json.loads((ROOT/'config/clustering_v5.json').read_text(encoding='utf-8'))
    parent=Path(loc['source_data_path']);meta=json.loads(Path(loc['source_manifest']).read_text(encoding='utf-8'))
    frames={}
    for key,name in [('main','reviewed_unimputed.parquet'),('strict','strict_screen_sensitivity_unimputed.parquet')]:
        if c.sha(parent/name)!=meta['output_sha256'][name]:raise ValueError('Input hash changed')
        frame=pd.read_parquet(parent/name,columns=IDS+['split','age','gender']+FEATURES)
        frames[key]=frame[frame.split.eq('train')].sort_values('stay_id').reset_index(drop=True)
        if frames[key][IDS].isna().any().any() or frames[key].subject_id.duplicated().any() or frames[key].stay_id.duplicated().any():
            raise ValueError('Invalid training identity keys')
        if np.isinf(frames[key][FEATURES].to_numpy(dtype=float)).any():
            raise ValueError('Infinite source feature')
    assert frames['main'][IDS].equals(frames['strict'][IDS])
    return cfg,frames

def fit_grid(filled,raw,limits,tag,data,tables,cfg,distance=None):
    D=g.numeric_gower(filled,filled,FEATURES,limits) if distance is None else distance
    np.fill_diagonal(D,0);fits={};metrics=[];profiles=[];centres=[]
    work_path=data/f'{tag}_temporary_distance.npy'
    np.save(work_path,D)
    pool=ProcessPoolExecutor(max_workers=4,initializer=initialize_pam_worker,initargs=(str(work_path),))
    for k in cfg['candidate_k']:
        print(f'{tag}: full-training PAM k={k}, n={len(raw)}',flush=True)
        fit,restarts=parallel_pam(pool,len(D),k,cfg['pam_restarts'],cfg['seed']+k,cfg['max_iter']);fits[k]=fit
        sizes=np.bincount(fit['labels'],minlength=k)
        sil=silhouette_samples(D,fit['labels'],metric='precomputed')
        metrics.append({'analysis':tag,'k':k,'n':len(raw),'loss':fit['loss'],
            'silhouette_mean':float(sil.mean()),'min_cluster_fraction':float(sizes.min()/len(raw))})
        labels=raw[IDS].copy();labels['cluster']=fit['labels']+1
        labels.to_parquet(data/f'{tag}_k{k}_labels.parquet',index=False)
        raw.iloc[fit['medoids']].assign(cluster=np.arange(1,k+1)).to_parquet(data/f'{tag}_k{k}_observed_medoids.parquet',index=False)
        pd.DataFrame(restarts).to_csv(tables/f'{tag}_k{k}_restarts.csv',index=False)
        for cluster in range(k):
            mask=fit['labels']==cluster;centre=raw.iloc[fit['medoids'][cluster]]
            centres.append({'analysis':tag,'k':k,'cluster':cluster+1,'n':int(mask.sum()),
                'medoid_observed':int(centre[FEATURES].notna().sum()),
                'median_observed':float(raw.loc[mask,FEATURES].notna().sum(axis=1).median()),
                'age_median':float(raw.loc[mask,'age'].median()),
                'female_fraction':float(raw.loc[mask,'gender'].eq('F').mean())})
            for feature in FEATURES:
                s=raw.loc[mask,feature]
                profiles.append({'analysis':tag,'k':k,'cluster':cluster+1,'feature':feature,
                    'n_observed':int(s.notna().sum()),'missing_fraction':float(s.isna().mean()),
                    'observed_median':float(s.median()),'observed_q1':float(s.quantile(.25)),
                    'observed_q3':float(s.quantile(.75))})
        pd.DataFrame(metrics).to_csv(tables/f'{tag}_metrics.csv',index=False)
    pd.DataFrame(profiles).to_csv(tables/f'{tag}_profiles.csv',index=False)
    pd.DataFrame(centres).to_csv(tables/f'{tag}_medoid_observability.csv',index=False)
    pool.shutdown(wait=True)
    work_path.unlink()
    return fits,D

def baseline(loc):
    started=time.time();cfg,frames=load(loc);all_train=frames['main']
    data=Path(loc['data_path'])/'baseline';data.mkdir(exist_ok=True)
    tables=Path(loc['tables_path'])/'baseline';tables.mkdir(exist_ok=True)
    coverage=[];descriptions=[];fits_by={};raw_by={}
    for threshold in [6,8,10]:
        mask=eligible(all_train,threshold)
        coverage.append({'minimum_observed':threshold,'n_train_total':len(all_train),
            'n_eligible':int(mask.sum()),'n_excluded':int((~mask).sum()),'fraction_retained':float(mask.mean())})
        for retained in [True,False]:
            group=all_train[mask.eq(retained)]
            for f in ['age']+FEATURES:
                s=group[f]
                descriptions.append({'threshold':threshold,'group':'retained' if retained else 'excluded',
                    'feature':f,'n':len(group),'n_observed':int(s.notna().sum()),
                    'median':float(s.median()),'q1':float(s.quantile(.25)),'q3':float(s.quantile(.75))})
            descriptions.append({'threshold':threshold,'group':'retained' if retained else 'excluded',
                'feature':'female_fraction','n':len(group),'median':float(group.gender.eq('F').mean())})
        raw=all_train[mask].reset_index(drop=True);raw_by[threshold]=raw
        filled,params,limits=c.fit_transform(raw,FEATURES)
        dump(data/f'coverage{threshold}_transform.json',{'imputer':params,'limits':limits})
        filled.to_parquet(data/f'coverage{threshold}_imputed.parquet',index=False)
        raw.to_parquet(data/f'coverage{threshold}_unimputed.parquet',index=False)
        fits,D=fit_grid(filled,raw,limits,f'coverage{threshold}',data,tables,cfg);fits_by[threshold]=fits
        if threshold==8:
            strict=frames['strict'].loc[mask].reset_index(drop=True)
            strict_filled,_,strict_limits=c.fit_transform(strict,FEATURES)
            sensitivity=[]
            for scale_name,scale in [('main_limits',limits),('refitted_strict_limits',strict_limits)]:
                Ds=g.numeric_gower(strict_filled,strict_filled,FEATURES,scale);np.fill_diagonal(Ds,0)
                for k in cfg['candidate_k']:
                    fit,_=c.fit_pam(Ds,k,cfg['pam_restarts'],cfg['seed']+k,cfg['max_iter'])
                    agreement,_,_=c.aligned(fits[k]['labels'],fit['labels'],k)
                    sensitivity.append({'k':k,'scale':scale_name,
                        'ari':float(adjusted_rand_score(fits[k]['labels'],fit['labels'])),
                        'aligned_agreement':agreement})
                del Ds
            pd.DataFrame(sensitivity).to_csv(tables/'outlier_sensitivity.csv',index=False)
            rng=np.random.default_rng(cfg['seed']+3000);stability=[]
            for rep in range(cfg['stability']['repetitions']):
                indices=np.sort(rng.choice(len(raw),int(len(raw)*cfg['stability']['fraction']),replace=False))
                sample=raw.iloc[indices].reset_index(drop=True)
                sub,imputer,scale=c.fit_transform(sample,FEATURES)
                Ds=g.numeric_gower(sub,sub,FEATURES,scale);np.fill_diagonal(Ds,0)
                for k in cfg['candidate_k']:
                    fit,_=c.fit_pam(Ds,k,cfg['pam_restarts'],cfg['seed']+10000+100*rep+k,cfg['max_iter'])
                    labels=np.argmin(g.numeric_gower(c.apply_transform(raw,imputer),sub.iloc[fit['medoids']],FEATURES,scale),axis=1)
                    agreement,_,_=c.aligned(fits[k]['labels'],labels,k)
                    stability.append({'rep':rep+1,'k':k,'ari':float(adjusted_rand_score(fits[k]['labels'],labels)),
                        'aligned_agreement':agreement})
                del Ds
                pd.DataFrame(stability).to_csv(tables/'coverage8_stability.csv',index=False)
                print(f'Coverage8 median stability {rep+1}/{cfg["stability"]["repetitions"]}',flush=True)
        del D
    comparisons=[]
    for threshold in [6,10]:
        a=raw_by[8];b=raw_by[threshold]
        for k in cfg['candidate_k']:
            joined=pd.DataFrame({'stay_id':a.stay_id,'a':fits_by[8][k]['labels']}).merge(
                pd.DataFrame({'stay_id':b.stay_id,'b':fits_by[threshold][k]['labels']}),on='stay_id',validate='one_to_one')
            agreement,_,_=c.aligned(joined.a.to_numpy(),joined.b.to_numpy(),k)
            comparisons.append({'reference_threshold':8,'compared_threshold':threshold,'k':k,
                'n_common':len(joined),'ari':float(adjusted_rand_score(joined.a,joined.b)),
                'aligned_agreement':agreement})
    pd.DataFrame(coverage).to_csv(tables/'coverage_counts.csv',index=False)
    pd.DataFrame(descriptions).to_csv(tables/'retained_excluded_descriptions.csv',index=False)
    pd.DataFrame(comparisons).to_csv(tables/'coverage_common_patient_comparison.csv',index=False)
    dump(Path(loc['report_path'])/'baseline_status.json',{'status':'complete','elapsed_seconds':time.time()-started})

def multiple(loc):
    started=time.time();cfg,frames=load(loc)
    raw=frames['main'][eligible(frames['main'],8)].reset_index(drop=True)
    data=Path(loc['data_path'])/'multiple_imputation';data.mkdir(exist_ok=True)
    tables=Path(loc['tables_path'])/'multiple_imputation';tables.mkdir(exist_ok=True)
    raw.to_parquet(data/'coverage8_unimputed.parquet',index=False)
    _,_,limits=c.fit_transform(raw,FEATURES);dump(data/'limits.json',limits)
    pooled=None;individual={};trace=[]
    for number in range(cfg['multiple_imputation']['datasets']):
        tag=f'imputation{number+1:02d}'
        filled,t=rf_pmm(raw,cfg['multiple_imputation'],cfg['seed']+50000+number,
            data/f'{tag}_models.joblib',lambda cycle:print(f'{tag}: RF-PMM cycle {cycle}',flush=True))
        trace.extend(dict(imputation=number+1,**row) for row in t)
        pd.DataFrame(trace).to_csv(tables/'imputation_trajectory.csv',index=False)
        filled.to_parquet(data/f'{tag}_filled.parquet',index=False)
        fits,D=fit_grid(filled,raw,limits,tag,data,tables,cfg)
        individual[number]=fits
        if pooled is None:pooled=D/cfg['multiple_imputation']['datasets']
        else:pooled+=D/cfg['multiple_imputation']['datasets']
        del D
    np.fill_diagonal(pooled,0)
    np.save(data/'mean_gower_distance.npy',pooled)
    # Pooled medoid rows correspond to patients, not an invented average patient.
    pooled_fits,_=fit_grid(raw[FEATURES],raw,limits,'pooled10',data,tables,cfg,distance=pooled)
    uncertainty=[];agreement_rows=[]
    for k in cfg['candidate_k']:
        support=np.zeros((len(raw),k),dtype=int)
        for number,fits in individual.items():
            agreement,mapped,_=c.aligned(pooled_fits[k]['labels'],fits[k]['labels'],k)
            support[np.arange(len(raw)),mapped]+=1
            agreement_rows.append({'k':k,'imputation':number+1,
                'ari':float(adjusted_rand_score(pooled_fits[k]['labels'],fits[k]['labels'])),
                'aligned_agreement':agreement})
        p=support/cfg['multiple_imputation']['datasets']
        entropy=-(p*np.log(np.maximum(p,1e-30))).sum(axis=1)/np.log(k)
        assigned_support=p[np.arange(len(raw)),pooled_fits[k]['labels']]
        private=raw[IDS].copy();private['cluster']=pooled_fits[k]['labels']+1
        private['assigned_label_support']=assigned_support;private['entropy']=entropy
        private.to_parquet(data/f'pooled10_k{k}_assignment_uncertainty.parquet',index=False)
        uncertainty.append({'k':k,'mean_assigned_support':float(assigned_support.mean()),
            'fraction_support_below_0_8':float((assigned_support<.8).mean()),
            'mean_normalized_entropy':float(entropy.mean())})
    pd.DataFrame(agreement_rows).to_csv(tables/'individual_vs_pooled_agreement.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(tables/'assignment_uncertainty.csv',index=False)
    del pooled
    # Hide known training measurements BEFORE fitting; evaluate stochastic draws
    # individually and their mean. This tests artificial masking, not MNAR truth.
    truth=raw[FEATURES].to_numpy(dtype=float);rng=np.random.default_rng(cfg['seed']+90000)
    hidden=(rng.random(truth.shape)<.10)&np.isfinite(truth)
    np.save(data/'masked_observation_boolean_mask.npy',hidden)
    masked=raw.copy();masked[FEATURES]=raw[FEATURES].mask(hidden)
    median,_,_=c.fit_transform(masked,FEATURES);predictions={'median':median.to_numpy()}
    draws=[]
    for rep in range(cfg['masked_check_repetitions']):
        result,_=rf_pmm(masked,cfg['multiple_imputation'],cfg['seed']+91000+rep,
            progress=lambda cycle:print(f'Masked RF-PMM {rep+1}: cycle {cycle}',flush=True))
        draws.append(result.to_numpy());predictions[f'rf_pmm_draw{rep+1}']=draws[-1]
    predictions['rf_pmm_mean']=np.mean(draws,axis=0)
    scores=[];observed_count=raw[FEATURES].notna().sum(axis=1).to_numpy()
    for tag,pred in predictions.items():
        for j,feature in enumerate(FEATURES):
            iqr=float(np.nanquantile(truth[:,j],.75)-np.nanquantile(truth[:,j],.25))
            for stratum,rows in [('all',np.ones(len(raw),dtype=bool)),('observed8_9',observed_count<10),('observed10_plus',observed_count>=10)]:
                select=hidden[:,j]&rows;err=pred[select,j]-truth[select,j]
                if not len(err):continue
                scores.append({'method':tag,'feature':feature,'stratum':stratum,'n_hidden':len(err),
                    'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.mean(err**2))),
                    'mae_over_iqr':float(np.abs(err).mean()/iqr) if iqr>0 else None,
                    'rmse_over_iqr':float(np.sqrt(np.mean(err**2))/iqr) if iqr>0 else None})
    pd.DataFrame(scores).to_csv(tables/'masked_observation_errors.csv',index=False)
    dump(Path(loc['report_path'])/'multiple_imputation_status.json',{'status':'complete','elapsed_seconds':time.time()-started,
        'models_saved_for_reproduction':True,'heldout_transform_validated':False,
        'proper_bayesian_mi_claim':False,'convergence_claim':False,
        'masked_observation_mask_sha256':c.sha(data/'masked_observation_boolean_mask.npy')})

def initialize():
    previous=json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))
    cfg_path=ROOT/'config/clustering_v5.json'
    run=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    loc={'run_id':run,'data_path':str(Path(previous['data_path'])/'clustering_v5'/run),
        'tables_path':str(ROOT/'outputs/tables/clustering_v5'/run),
        'report_path':str(ROOT/'outputs/reports/clustering_v5'/run),
        'source_data_path':previous['data_path'],
        'source_manifest':str(Path(previous['report_path'])/'run_manifest.json'),
        'status':'running','outcomes_loaded':False,'validation_test_used':False,'clinical_k_frozen':False}
    for key in ['data_path','tables_path','report_path']:Path(loc[key]).mkdir(parents=True)
    for file in [Path(__file__),cfg_path]: (Path(loc['report_path'])/file.name).write_bytes(file.read_bytes())
    dump(Path(loc['report_path'])/'run_manifest.json',dict(loc,config_sha256=c.sha(cfg_path),script_sha256=c.sha(__file__),
        dependencies={x:metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','joblib']}))
    dump(ROOT/'logs/clustering_v5_latest.json',loc)
    print('Initialized v5 run '+run,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','baseline','multiple'])
    args=parser.parse_args()
    if args.stage=='init':initialize()
    else:
        loc=json.loads((ROOT/'logs/clustering_v5_latest.json').read_text(encoding='utf-8'))
        try:
            (baseline if args.stage=='baseline' else multiple)(loc)
        except Exception as exc:
            dump(Path(loc['report_path'])/(args.stage+'_failure.json'),{'type':type(exc).__name__,'message':str(exc)})
            raise
