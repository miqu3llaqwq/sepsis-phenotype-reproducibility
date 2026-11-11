"""TRAIN-only domain-balanced, missing-aware Gower diagnostic."""
from pathlib import Path
import importlib
import json
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
m=importlib.import_module('18_physiology_v5')

def domain_limits(raw,domains):
    feats=[f for fs in domains.values() for f in fs]
    return {f:[float(raw[f].quantile(.01)),float(raw[f].quantile(.99))] for f in feats}

def eligibility(frame,cfg):
    domains=cfg['domains'];
    domain_n=sum(frame[fs].notna().any(axis=1) for fs in domains.values())
    feature_n=frame[cfg['features']].notna().sum(axis=1)
    required=np.logical_and.reduce([frame[domains[d]].notna().any(axis=1).to_numpy() for d in cfg['eligibility']['required_domains']])
    return domain_n.ge(cfg['eligibility']['minimum_domains_observed']) & feature_n.ge(cfg['eligibility']['minimum_features_observed']) & required

def missing_domain_gower(frame,features,domains,limits,block=256):
    return missing_domain_gower_rect(frame,frame,features,domains,limits,block=block,square=True)

def missing_domain_gower_rect(left,right,features,domains,limits,block=256,square=False):
    x=left[features].to_numpy(dtype=float);y=right[features].to_numpy(dtype=float)
    n=len(left);mright=len(right);out=np.empty((n,mright),dtype=np.float32)
    scaled=np.empty_like(x);present=np.isfinite(x)
    scaled_right=np.empty_like(y);present_right=np.isfinite(y)
    for j,f in enumerate(features):
        lo,hi=limits[f]
        if not np.isfinite(lo) or not np.isfinite(hi): raise ValueError(f'Invalid training limit for {f}')
        if hi==lo:
            scaled[:,j]=0.;scaled_right[:,j]=0.
        else:
            scaled[:,j]=np.clip(np.nan_to_num(x[:,j],nan=0.),lo,hi)-lo;scaled[:,j]/=(hi-lo)
            scaled_right[:,j]=np.clip(np.nan_to_num(y[:,j],nan=0.),lo,hi)-lo;scaled_right[:,j]/=(hi-lo)
    positions={f:j for j,f in enumerate(features)}
    domain_arrays=[]
    for fs in domains.values():
        js=[positions[f] for f in fs];domain_arrays.append((js,scaled[:,js],present[:,js]))
    domain_arrays_right=[([positions[f] for f in fs],scaled_right[:,[positions[f] for f in fs]],present_right[:,[positions[f] for f in fs]]) for fs in domains.values()]
    for start in range(0,n,block):
        stop=min(start+block,n);numer=np.zeros((stop-start,mright),dtype=np.float32);denom=np.zeros_like(numer)
        for (_,values,valid),(_,values_right,valid_right) in zip(domain_arrays,domain_arrays_right):
            common=valid[start:stop,None,:]&valid_right[None,:,:]
            dif=np.abs(values[start:stop,None,:]-values_right[None,:,:])
            count=common.sum(axis=2);part=np.divide((dif*common).sum(axis=2),count,out=np.zeros_like(numer),where=count>0)
            numer+=part*(count>0);denom+=(count>0)
        out[start:stop]=np.divide(numer,denom,out=np.full_like(numer,np.nan),where=denom>0)
    if square:
        np.fill_diagonal(out,0.)
    offdiag=~np.eye(n,dtype=bool) if square else np.ones(out.shape,dtype=bool)
    if np.isnan(out[offdiag]).any():
        raise ValueError('At least one pair has no common observed domain; define a compatible-pair rule before PAM')
    return out

def incompatible_positions(frame,features,domains,block=256):
    """Return rows that have at least one pair with no shared variable.

    This is an outcome-blinded compatibility screen required by the complete
    PAM dissimilarity matrix. It never substitutes a distance for an
    unobserved comparison.
    """
    x=frame[features].to_numpy(dtype=float);valid=np.isfinite(x);n=len(frame)
    positions={f:j for j,f in enumerate(features)};bad=set()
    for start in range(0,n,block):
        stop=min(start+block,n);shared=np.zeros((stop-start,n),dtype=bool)
        for fs in domains.values():
            js=[positions[f] for f in fs]
            shared |= (valid[start:stop,None,js]&valid[None,:,js]).any(axis=2)
        rows=np.argwhere(~shared)
        bad.update(start+int(i) for i,j in rows);bad.update(int(j) for i,j in rows)
    return sorted(bad)

def fit_parallel(D,k,cfg,seed):
    # Keep the same FastPAM objective and seeded starts as prior runs. The
    # matrix is already memory-heavy; serial starts avoid Windows worker
    # pickling and duplicate-memory risk while preserving reproducibility.
    return m.c.fit_pam(D,k,cfg['pam_restarts'],seed,cfg['max_iter'])

def run_grid(raw,cfg,limits,tag,data,tables):
    D=missing_domain_gower(raw,cfg['features'],cfg['domains'],limits)
    fits={};metrics=[];profiles=[]
    for k in cfg['candidate_k']:
        print(f'{tag}: PAM k={k}, n={len(raw)}',flush=True)
        fit,restarts=fit_parallel(D,k,cfg,cfg['seed']+k);fits[k]=fit
        sizes=np.bincount(fit['labels'],minlength=k);sil=silhouette_samples(D,fit['labels'],metric='precomputed')
        metrics.append({'analysis':tag,'k':k,'n':len(raw),'loss':fit['loss'],'silhouette_mean':float(sil.mean()),'min_cluster_fraction':float(sizes.min()/len(raw))})
        labels=raw[m.IDS].copy();labels['cluster']=fit['labels']+1;labels.to_parquet(data/f'{tag}_k{k}_labels.parquet',index=False)
        raw.iloc[fit['medoids']].assign(cluster=np.arange(1,k+1)).to_parquet(data/f'{tag}_k{k}_medoids.parquet',index=False)
        pd.DataFrame(restarts).to_csv(tables/f'{tag}_k{k}_restarts.csv',index=False)
        for cidx in range(k):
            mask=fit['labels']==cidx;centre=raw.iloc[fit['medoids'][cidx]]
            profiles.append({'analysis':tag,'k':k,'cluster':cidx+1,'n':int(mask.sum()),'medoid_observed':int(centre[cfg['features']].notna().sum()),'median_domains':float(sum(raw.loc[mask,fs].notna().any(axis=1) for fs in cfg['domains'].values()).median()),'median_features':float(raw.loc[mask,cfg['features']].notna().sum(axis=1).median())})
        pd.DataFrame(metrics).to_csv(tables/f'{tag}_metrics.csv',index=False)
    pd.DataFrame(profiles).to_csv(tables/f'{tag}_profiles.csv',index=False);return fits,D

def main():
    started=time.time();cfg=json.loads((ROOT/'config/clustering_v7.json').read_text(encoding='utf-8'));v6=json.loads((ROOT/'logs/clustering_v6_latest.json').read_text(encoding='utf-8'));v3=json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))
    source=Path(v3['data_path'])/'reviewed_unimputed.parquet';cols=m.IDS+['split','age','gender']+cfg['features'];frame=pd.read_parquet(source,columns=cols);train=frame[frame.split.eq('train')].sort_values('stay_id').reset_index(drop=True);mask=eligibility(train,cfg);raw_pre=train[mask].reset_index(drop=True)
    incompatible=incompatible_positions(raw_pre,cfg['features'],cfg['domains'])
    raw=raw_pre.drop(index=incompatible).reset_index(drop=True)
    run=__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S');data=Path(v6['data_path']).parent/'clustering_v7'/run;tables=ROOT/'outputs/tables/clustering_v7'/run;reports=ROOT/'outputs/reports/clustering_v7'/run
    for p in [data,tables,reports]:p.mkdir(parents=True)
    limits=domain_limits(raw,cfg['domains']);m.dump(data/'run_inputs.json',{'n_train_all':len(train),'n_eligible_before_pairwise_compatibility':len(raw_pre),'n_pairwise_incompatible_excluded':len(incompatible),'n_eligible':len(raw),'eligible_domain_rule':cfg['eligibility'],'pairwise_compatibility_rule':'exclude any eligible row involved in a pair with no shared observed variable across any domain; no distance substitution','limits':limits});raw.to_parquet(data/'eligible_train_unimputed.parquet',index=False)
    domain_names=list(cfg['domains'])
    domain_present=np.column_stack([raw[fs].notna().any(axis=1).to_numpy() for fs in cfg['domains'].values()])
    common_counts=domain_present.astype(np.uint8)@domain_present.astype(np.uint8).T
    upper=common_counts[np.triu_indices(len(raw),1)]
    coverage_audit={'n_train_all':int(len(train)),'n_eligible_before_pairwise_compatibility':int(len(raw_pre)),'n_pairwise_incompatible_excluded':int(len(incompatible)),'n_eligible':int(len(raw)),'domain_count_distribution':{str(k):int(v) for k,v in zip(*np.unique(domain_present.sum(axis=1),return_counts=True))},'pair_count':int(len(upper)),'pairs_with_zero_common_domains':int((upper==0).sum()),'pairs_with_one_common_domain':int((upper==1).sum()),'pairs_with_two_or_more_common_domains':int((upper>=2).sum()),'minimum_common_domains_non_diagonal':int(upper.min()),'median_common_domains':float(np.median(upper)),'p01_common_domains':float(np.quantile(upper,.01)),'p05_common_domains':float(np.quantile(upper,.05)),'domain_names':domain_names}
    m.dump(reports/'domain_coverage_audit.json',coverage_audit)
    metrics=[];fits_main,D=run_grid(raw,cfg,limits,'primary_domain_balanced',data,tables);metrics.extend(pd.read_csv(tables/'primary_domain_balanced_metrics.csv').to_dict('records'))
    # Complete 16-feature subgroup diagnostic, using frozen eligible-domain limits.
    complete=raw[raw[cfg['features']].notna().all(axis=1)].reset_index(drop=True);complete.to_parquet(data/'complete16_train_unimputed.parquet',index=False);run_grid(complete,cfg,limits,'complete16_domain_balanced',data,tables)
    # Fixed-parameter 80% stability, no outcomes, no label selection.
    rng=np.random.default_rng(cfg['seed']);stability=[]
    for rep in range(cfg['stability']['repetitions']):
        ids=np.sort(rng.choice(len(raw),int(len(raw)*cfg['stability']['fraction']),replace=False));sub=raw.iloc[ids].reset_index(drop=True);sub_limits=domain_limits(sub,cfg['domains']);Ds=missing_domain_gower(sub,cfg['features'],cfg['domains'],sub_limits)
        for k in cfg['candidate_k']:
            fit,_=fit_parallel(Ds,k,cfg,cfg['seed']+10000+rep*100+k)
            global_medoids=ids[fit['medoids']]
            Dall=missing_domain_gower_rect(raw,sub.iloc[fit['medoids']],cfg['features'],cfg['domains'],sub_limits)
            assigned=np.argmin(Dall,axis=1)
            _,mapped,_=m.c.aligned(fits_main[k]['labels'],assigned,k)
            stability.append({'rep':rep+1,'k':k,'ari':float(adjusted_rand_score(fits_main[k]['labels'],assigned)),'aligned_agreement':float(np.mean(fits_main[k]['labels']==mapped)),'n_subsample':len(sub),'medoid_global_positions':','.join(map(str,global_medoids))})
        print(f'v7 stability {rep+1}/{cfg["stability"]["repetitions"]}',flush=True)
    pd.DataFrame(stability).to_csv(tables/'primary_stability.csv',index=False)
    manifest={'status':'computation_complete','run_id':run,'n_train_all':len(train),'n_eligible_before_pairwise_compatibility':len(raw_pre),'n_pairwise_incompatible_excluded':len(incompatible),'n_eligible':len(raw),'n_complete16':len(complete),'pairwise_no_common_domain_pairs_for_primary':0,'no_common_domain_pairs_set_to_one':False,'outcomes_loaded':False,'validation_test_used':False,'clinical_k_frozen':False,'script_sha256':m.c.sha(__file__),'config_sha256':m.c.sha(ROOT/'config/clustering_v7.json'),'elapsed_seconds':time.time()-started}
    m.dump(reports/'run_manifest.json',manifest);m.dump(ROOT/'logs/clustering_v7_latest.json',{'run_id':run,'data_path':str(data),'tables_path':str(tables),'report_path':str(reports),**manifest});print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
