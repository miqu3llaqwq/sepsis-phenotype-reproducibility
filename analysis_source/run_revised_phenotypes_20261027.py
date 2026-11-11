"""Resumable, training-only rerun; preserve MICE primary and external diagnostic roles."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse, hashlib, importlib, importlib.metadata, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, adjusted_rand_score
from statsmodels.imputation.mice import MICEData

ROOT=Path(__file__).resolve().parent
B=ROOT/'work/mimic_rebuilt_20261026'
OUT=B/'data/revised_analysis_20261027'
TABLES=B/'tables/revised_analysis_20261027'
REPORT=B/'reports/revised_analysis_20261027'
sys.path.insert(0,str(B/'analysis'))
v=importlib.import_module('26_domain_balanced_v7')
c=importlib.import_module('14_cluster_training')
measure=importlib.import_module('39_missingness_measurement_sensitivity')
CFG=json.loads((B/'config/phenotype_primary_v1.json').read_text())
FEATS=CFG['features_primary']; KS=CFG['clustering']['candidate_k']; SEED=CFG['clustering']['seed']
IDS=['subject_id','stay_id','hadm_id']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)

def limits(raw): return v.domain_limits(raw,{f:[f] for f in FEATS})

def eq_distance(left,right,lim,block=256):
    """Same per-feature float32 accumulation as original single-feature-domain Gower."""
    x=left[FEATS].to_numpy(float);y=right[FEATS].to_numpy(float)
    px=np.isfinite(x);py=np.isfinite(y)
    for j,f in enumerate(FEATS):
        lo,hi=lim[f];assert np.isfinite([lo,hi]).all() and hi>lo
        x[:,j]=(np.clip(np.nan_to_num(x[:,j],nan=0.),lo,hi)-lo)/(hi-lo)
        y[:,j]=(np.clip(np.nan_to_num(y[:,j],nan=0.),lo,hi)-lo)/(hi-lo)
    out=np.empty((len(x),len(y)),np.float32)
    for a in range(0,len(x),block):
        z=min(a+block,len(x));numer=np.zeros((z-a,len(y)),np.float32);denom=np.zeros_like(numer)
        for j in range(len(FEATS)):
            common=px[a:z,j,None]&py[None,:,j]
            part=np.where(common,np.abs(x[a:z,j,None]-y[None,:,j]),0.).astype(np.float32)
            numer+=part;denom+=common
        out[a:z]=np.divide(numer,denom,out=np.full_like(numer,np.nan),where=denom>0)
    if not np.isfinite(out).all(): raise ValueError('Undefined common-observation distance')
    return out

def check_distance(raw,lim):
    sub=raw.sample(n=min(160,len(raw)),random_state=SEED)
    expected=v.missing_domain_gower_rect(sub,sub,FEATS,{f:[f] for f in FEATS},lim)
    actual=eq_distance(sub,sub,lim)
    assert np.array_equal(actual,expected),np.max(np.abs(actual-expected))
    assert np.array_equal(actual,actual.T) and np.all(np.diag(actual)==0)
    return {'bitwise_original_distance_match':True,'sample_n':len(sub)}

def fit_worker(args):
    matrix,k,seed=args
    d=np.load(matrix,mmap_mode='r')
    fit,restarts=c.fit_pam(d,k,CFG['clustering']['pam_restarts'],seed,200)
    sil=silhouette_samples(d,fit['labels'],metric='precomputed')
    sizes=np.bincount(fit['labels'],minlength=k)
    return k,fit,restarts,{'k':k,'n':len(d),'silhouette_mean':float(sil.mean()),'negative_silhouette_fraction':float((sil<0).mean()),'min_cluster_fraction':float(sizes.min()/len(d)),'loss':fit['loss']}

def fit_grid(raw,lim,tag,seedbase,kind='equal',pool=None):
    dest=OUT/tag;dest.mkdir(parents=True,exist_ok=True)
    done=dest/'grid_manifest.json'
    if done.exists():
        manifest=json.loads(done.read_text());assert manifest['status']=='complete'
        assert manifest['n']==len(raw) and manifest['seedbase']==seedbase and manifest['kind']==kind and manifest['limits']==lim
        assert manifest['input_hash']==hashlib.sha256(pd.util.hash_pandas_object(raw,index=True).values.tobytes()).hexdigest()
        for file,h in manifest['sha256'].items(): assert sha(dest/file)==h
        print('reuse completed',tag,flush=True)
        return {k:pd.read_parquet(dest/f'k{k}_labels.parquet').cluster.to_numpy()-1 for k in KS}
    start=time.time();print('distance',tag,'n=',len(raw),flush=True)
    if kind=='equal': D=eq_distance(raw,raw,lim)
    elif kind=='domain': D=v.missing_domain_gower(raw,FEATS,CFG['domains'],lim)
    elif kind=='indicator': D=measure.indicator_balanced_distance(raw,FEATS,lim)
    elif kind=='availability': D=measure.availability_distance(raw,FEATS)
    else: raise ValueError(kind)
    np.fill_diagonal(D,0.)
    assert np.isfinite(D).all() and D.min()>=0 and D.max()<=1.000001
    assert np.array_equal(D,D.T)
    matrix=dest/'working_distance.npy';np.save(matrix,D);del D
    print('fit grid',tag,'distance seconds',round(time.time()-start,1),flush=True)
    tasks=[(str(matrix),k,seedbase+k) for k in KS]
    results={};metrics=[]
    for future in as_completed([pool.submit(fit_worker,t) for t in tasks]):
        k,fit,restarts,row=future.result();results[k]=fit['labels'];metrics.append(row)
        labels=raw[IDS].copy();labels['cluster']=fit['labels']+1
        labels.to_parquet(dest/f'k{k}_labels.parquet',index=False)
        raw.iloc[fit['medoids']][IDS+FEATS].assign(cluster=np.arange(1,k+1)).to_parquet(dest/f'k{k}_medoids.parquet',index=False)
        pd.DataFrame(restarts).to_csv(dest/f'k{k}_restarts.csv',index=False)
        print(tag,'k',k,'silhouette',round(row['silhouette_mean'],4),'seconds',round(time.time()-start,1),flush=True)
    pd.DataFrame(metrics).sort_values('k').to_csv(dest/'metrics.csv',index=False)
    # Exact file only; disposable distance is outside the frozen bundle.
    matrix.unlink()
    files={p.name:sha(p) for p in dest.iterdir() if p.suffix in ['.parquet','.csv']}
    dump(done,{'status':'complete','n':len(raw),'seedbase':seedbase,'kind':kind,'limits':lim,'sha256':files,'elapsed_seconds':time.time()-start,
               'input_hash':hashlib.sha256(pd.util.hash_pandas_object(raw,index=True).values.tobytes()).hexdigest()})
    return results

def impute(raw,mi,tag):
    dest=OUT/tag;dest.mkdir(parents=True,exist_ok=True);path=dest/'imputed.parquet';manifest=dest/'imputation.json'
    digest=hashlib.sha256(pd.util.hash_pandas_object(raw[FEATS],index=True).values.tobytes()).hexdigest()
    if manifest.exists():
        prior=json.loads(manifest.read_text());assert prior['source_hash']==digest and prior['sha256']==sha(path)
        completed=pd.read_parquet(path)
    else:
        print('MICE-PMM',tag,'n',len(raw),'20 cycles, donors 10',flush=True)
        source=raw[FEATS].astype(float)
        chain=MICEData(source.copy(),perturbation_method='boot',k_pmm=10,rng=np.random.default_rng(SEED+50000+mi))
        for cycle in range(20):
            chain.update_all(1)
            if (cycle+1)%5==0:print(tag,'MICE cycles',cycle+1,flush=True)
        completed=chain.data[FEATS].astype(float)
        original=source.to_numpy();imputed=completed.to_numpy();observed=np.isfinite(original)
        assert np.array_equal(original[observed],imputed[observed]) and np.isfinite(imputed).all()
        for f in FEATS: assert completed.loc[source[f].isna(),f].isin(source[f].dropna()).all()
        completed.to_parquet(path,index=False)
        dump(manifest,{'status':'complete','source_hash':digest,'sha256':sha(path),'seed':SEED+50000+mi,'n':len(raw),'method':'statsmodels MICEData bootstrap PMM','burn_in':10,'between_draw':10,'donors':10,'observed_values_unchanged':True,'all_imputations_are_observed_training_donors':True,'fit_scope':'eligible training rows supplied to this chain only','outcomes_used':False})
    result=raw[IDS].copy().reset_index(drop=True)
    result[FEATS]=completed[FEATS].to_numpy()
    return result

def compare(base,labels,tag):
    rows=[]
    for k in KS:
        agreement,_,_=c.aligned(base[k],labels[k],k)
        rows.append({'analysis':tag,'k':k,'ari':float(adjusted_rand_score(base[k],labels[k])),'aligned_agreement':agreement})
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    for path in [OUT,TABLES,REPORT]:path.mkdir(parents=True,exist_ok=True)
    acceptance=json.loads((B/'reports/source_acceptance_20261027/acceptance.json').read_text())
    assert acceptance['status']=='accepted_for_revised_analysis'
    source=B/'data/core14_preparation/eligible_train_unimputed.parquet'
    raw=pd.read_parquet(source,columns=IDS+['split']+FEATS).sort_values('stay_id').reset_index(drop=True)
    assert raw.split.eq('train').all() and raw.subject_id.is_unique and len(raw)==17810
    raw=raw[IDS+FEATS];lim=limits(raw);distance_checks=check_distance(raw,lim)
    contract={'date':'2026-10-27','source_sha256':sha(source),'cohort_sha256':acceptance['dataset_sha256'],'config_sha256':sha(B/'config/phenotype_primary_v1.json'),'features':FEATS,'n':len(raw),'k':KS,'restarts':5,'max_iter':200,
      'primary':'10 training-only MICE-PMM datasets; equal-feature Gower; no single imputation selected',
      'external_diagnostic':'observed-value missing-aware equal-feature Gower, fixed k2-6 medoids; v9 external protocol',
      'stability':'10 independent 80% subsamples with refit limits for missing-aware; 10 independently reimputed 80% subsamples for MICE, compare on included subjects to matched full chain',
      'MICE_stability_interpretation':'joint subsampling and re-imputation perturbation, not full-data out-of-sample imputation or pure sampling variance',
      'sensitivity':['domain-balanced core14','availability-only','indicator-balanced','complete14'],
      'clinical_k_frozen':False,'outcomes_used':False,'validation_test_used':False,'distance_checks':distance_checks,
      'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','statsmodels','kmedoids']}}
    contract_path=REPORT/'execution_contract.json'
    if contract_path.exists(): assert json.loads(contract_path.read_text())==contract
    else:dump(contract_path,contract)
    dump(REPORT/'source_code_hashes.json',{str(p.relative_to(B)):sha(p) for p in (B/'analysis').glob('*.py')})
    print('preflight passed',json.dumps(distance_checks),'n',len(raw),flush=True)
    if args.check_only:return
    started=time.time();dump(REPORT/'run_status.json',{'status':'running','stage':'observed','started_epoch':started})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        base=fit_grid(raw,lim,'observed_equal',SEED+70000,pool=pool)
        st=[];ms=[];mi_labels={k:[] for k in KS};mi_rows=[]
        rng=np.random.default_rng(SEED)
        for rep in range(10):
            ids=np.sort(rng.choice(len(raw),int(len(raw)*.8),replace=False));sub=raw.iloc[ids].reset_index(drop=True);sub_lim=limits(sub)
            tag=f'resample_{rep+1:02d}';dump(REPORT/'run_status.json',{'status':'running','stage':tag,'started_epoch':started})
            labels=fit_grid(sub,sub_lim,tag,SEED+60000+rep*100,pool=pool)
            sub[IDS].to_parquet(OUT/tag/'subsample_ids.parquet',index=False)
            for k in KS:
                med=pd.read_parquet(OUT/tag/f'k{k}_medoids.parquet')
                predicted=np.argmin(eq_distance(raw,med,sub_lim),axis=1)
                agreement,_,_=c.aligned(base[k],predicted,k)
                st.append({'rep':rep+1,'k':k,'n_subsample':len(sub),'ari':float(adjusted_rand_score(base[k],predicted)),'aligned_agreement':agreement})
            pd.DataFrame(st).to_csv(TABLES/'observed_resampling_detail.csv',index=False)
            # Each full independent chain is retained; no best-draw selection.
            full_tag=f'mice_{rep+1:02d}';completed=impute(raw,rep,full_tag)
            ml=fit_grid(completed,lim,full_tag,SEED+50000+(rep+1)*100,pool=pool)
            rows=compare(base,ml,full_tag)
            for row in rows:row['imputation']=rep+1
            mi_rows.extend(rows)
            for k in KS:mi_labels[k].append(ml[k])
            pd.DataFrame(mi_rows).to_csv(TABLES/'mice_vs_observed_detail.csv',index=False)
            # Refit the imputer solely on this subsample; no held-out subjects enter fitting.
            sub_tag=f'mice_resample_{rep+1:02d}';cs=impute(sub,rep,sub_tag)
            sl=fit_grid(cs,sub_lim,sub_tag,SEED+60000+rep*100,pool=pool)
            rows=compare({k:ml[k][ids] for k in KS},sl,sub_tag)
            for row in rows:row.update(rep=rep+1,n_subsample=len(sub),evaluation='overlapping subjects; paired full-chain comparator')
            ms.extend(rows);pd.DataFrame(ms).to_csv(TABLES/'mice_joint_resampling_detail.csv',index=False)
        summaries=[]
        for k in KS:
            pair=[adjusted_rand_score(mi_labels[k][i],mi_labels[k][j]) for i in range(10) for j in range(i+1,10)]
            summaries.append({'k':k,'imputations':10,'pair_count':len(pair),'between_imputation_ari_mean':float(np.mean(pair)),'between_imputation_ari_min':float(np.min(pair))})
        pd.DataFrame(summaries).to_csv(TABLES/'mice_between_imputation_summary.csv',index=False)
        for file in ['observed_resampling','mice_joint_resampling']:
            d=pd.read_csv(TABLES/(file+'_detail.csv'))
            d.groupby('k').agg(repetitions=('ari','size'),ari_mean=('ari','mean'),ari_min=('ari','min'),agreement_mean=('aligned_agreement','mean')).reset_index().to_csv(TABLES/(file+'_summary.csv'),index=False)
        comparisons=[]
        for kind,offset in [('domain',0),('indicator',61000),('availability',62000)]:
            labels=fit_grid(raw,lim,'sensitivity_'+kind,SEED+offset,kind=kind,pool=pool)
            comparisons.extend(compare(base,labels,kind))
        complete=raw[raw[FEATS].notna().all(axis=1)].reset_index(drop=True)
        fit_grid(complete,limits(complete),'complete14',SEED+70000,pool=pool)
        pd.DataFrame(comparisons).to_csv(TABLES/'sensitivity_agreement.csv',index=False)
    # Descriptive feature profiles and missingness keep outcomes excluded.
    profile=[]
    for k in KS:
        for cluster in range(k):
            part=raw.loc[base[k]==cluster]
            for f in FEATS:profile.append({'k':k,'cluster':cluster+1,'feature':f,'n':len(part),'missing_fraction':float(part[f].isna().mean()),'median':float(part[f].median()),'q1':float(part[f].quantile(.25)),'q3':float(part[f].quantile(.75))})
    pd.DataFrame(profile).to_csv(TABLES/'observed_cluster_profiles.csv',index=False)
    pd.DataFrame({'feature':FEATS,'missing_fraction':raw[FEATS].isna().mean().to_numpy()}).to_csv(TABLES/'training_missingness.csv',index=False)
    allmetrics=[]
    for p in sorted(OUT.glob('*/metrics.csv')):
        d=pd.read_csv(p);d.insert(0,'analysis',p.parent.name);allmetrics.append(d)
    pd.concat(allmetrics,ignore_index=True).to_csv(TABLES/'all_clustering_metrics.csv',index=False)
    dump(REPORT/'run_status.json',{'status':'computations_complete_freeze_pending','elapsed_seconds_this_invocation':time.time()-started,'primary_MICE_draws':10,'MICE_subsample_refits':10,'observed_resamples':10,'k':KS,'clinical_k_frozen':False,'external_validation_complete':False,'script_sha256':sha(Path(__file__))})
    print('ALL TRAINING COMPUTATIONS COMPLETE; freeze QA next',flush=True)

if __name__=='__main__':main()
