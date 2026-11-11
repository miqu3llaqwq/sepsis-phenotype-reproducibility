"""Post-hoc, outcome-blind controlled perturbations and patient instability audit."""
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='1'
from pathlib import Path
import argparse,hashlib,importlib,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from statsmodels.imputation.mice import MICEData
ROOT=Path(__file__).resolve().parent;B=ROOT/'work/mimic_rebuilt_20261026'
F=B/'frozen/20261027';OLD=B/'data/revised_analysis_20261027'
O=ROOT/'outputs/missingness_depth_20261028';W=ROOT/'work/missingness_depth_20261028'
sys.path.insert(0,str(B/'analysis'));C=importlib.import_module('14_cluster_training')
CFG=json.loads((F/'core14_primary_transform.json').read_text());FEATS=CFG['features'];KS=list(range(2,7))
IDS=['subject_id','stay_id','hadm_id'];SEED=202610280;LO=np.array([CFG['limits'][f][0] for f in FEATS]);HI=np.array([CFG['limits'][f][1] for f in FEATS])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def scaled(x):return (np.clip(np.asarray(x,float),LO,HI)-LO)/(HI-LO)
def distance(x,y):
    x=scaled(x);y=scaled(y)
    if np.isfinite(x).all() and np.isfinite(y).all():return (cdist(x,y,'cityblock')/14).astype('float32')
    out=np.empty((len(x),len(y)),np.float32)
    for a in range(0,len(x),256):
        b=min(a+256,len(x));num=np.zeros((b-a,len(y)),np.float64);den=np.zeros_like(num)
        for j in range(14):
            ok=np.isfinite(x[a:b,j,None])&np.isfinite(y[None,:,j]);num+=np.where(ok,np.abs(x[a:b,j,None]-y[None,:,j]),0.);den+=ok
        out[a:b]=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    return out
def align(a,b,k):
    t=np.zeros((k,k),int);np.add.at(t,(a,b),1);r,c=linear_sum_assignment(-t);m=np.zeros(k,int);m[c]=r
    return m[b],m,t
def metric(a,b,k):
    mapped,_,_=align(a,b,k)
    return {'ari':float(adjusted_rand_score(a,b)),'aligned_disagreement':float(np.mean(a!=mapped))}
def domains(obs):
    d=np.column_stack([obs[:,[FEATS.index(f) for f in fs]].any(axis=1) for fs in CFG['domains'].values()])
    names=list(CFG['domains']);ok=(obs.sum(axis=1)>=6)&(d.sum(axis=1)>=4)&d[:,names.index('circulation')]&d[:,names.index('respiratory')]
    return d.sum(axis=1),ok
def compatible(obs):
    bits=(obs.astype(np.uint16)*(1<<np.arange(14,dtype=np.uint16))).sum(axis=1)
    unique,back=np.unique(bits,return_inverse=True)
    bad=np.array([np.any((b & unique)==0) for b in unique])
    return ~bad[back]
def load_source():
    p=B/'data/core14_preparation/eligible_train_unimputed.parquet'
    raw=pd.read_parquet(p,columns=IDS+['split']+FEATS).sort_values('stay_id').reset_index(drop=True)
    assert len(raw)==17810 and raw.subject_id.is_unique and raw.split.eq('train').all()
    return raw
def preflight(raw):
    O.mkdir(exist_ok=True,parents=True);W.mkdir(exist_ok=True,parents=True)
    bm=read(F/'bundle_manifest.json');assert all(sha(F/p)==h for p,h in bm['sha256'].items())
    sources={str(B/'data/core14_preparation/eligible_train_unimputed.parquet'):sha(B/'data/core14_preparation/eligible_train_unimputed.parquet'),str(F/'bundle_manifest.json'):sha(F/'bundle_manifest.json')}
    for mi in range(1,11):
        p=OLD/f'mice_{mi:02d}';m=read(p/'imputation.json');assert sha(p/'imputed.parquet')==m['sha256']
        for n,h in read(p/'grid_manifest.json')['sha256'].items():assert sha(p/n)==h
        sources[str(p/'imputation.json')]=sha(p/'imputation.json');sources[str(p/'grid_manifest.json')]=sha(p/'grid_manifest.json')
    for n,h in read(OLD/'observed_equal/grid_manifest.json')['sha256'].items():assert sha(OLD/'observed_equal'/n)==h
    sources[str(OLD/'observed_equal/grid_manifest.json')]=sha(OLD/'observed_equal/grid_manifest.json')
    protocol={'date':'2026-10-28','design':'post-hoc diagnostic study; not preregistration or clinical validation','seed':SEED,
      'n_natural':len(raw),'n_complete':int(raw[FEATS].notna().all(axis=1).sum()),'features':FEATS,'k':KS,
      'scales':'unchanged full eligible-training P1-P99','outcomes_used':False,'validation_test_used':False,
      'natural':'all 10 saved imputations; observed anchors versus completed same-anchor identities versus refitted MICE anchors; order-dependent contrasts, not additive causal effects',
      'patient_uncertainty':'mean pairwise Hungarian-aligned disagreement over 45 imputation pairs; cross-path disagreement over 10 draws; observed medoid gap is not calibrated probability',
      'mask_scenarios':['mcar20','mcar40','empirical_incomplete_pattern'], 'mask_replicates':5,'imputations_per_mask':3,
      'empirical_pattern':'sample entire missingness masks from incomplete eligible training rows independently of complete-case values; preserves pattern co-occurrence, does not identify MAR or MNAR',
      'eligibility':'apply frozen >=6 features >=4 domains including circulation and respiratory, then remove every row involved in any undefined common-observation pair; same retained rows in all contrasts of one replicate',
      'mask_imputation':'training-only statsmodels MICEData bootstrap PMM, 20 updates, 10 donors; held-out hidden truth never enters imputer',
      'mask_baseline':'fit complete values of retained rows at fixed training scales; unmasked labels are a computational comparator, not clinical ground truth',
      'mask_contrasts':['complete_baseline','observed_query_fixed_complete_anchors','observed_both_fixed_identities','observed_refit','pmm_query_fixed_complete_anchors','pmm_both_fixed_identities','pmm_refit'],
      'oracle_anchor_note':'fixed complete anchors intentionally retain hidden reference values to isolate query perturbation; diagnostic only, not a deployable missing-data estimator',
      'fitting':'all k2-6, same five initialization seeds and max_iter 200 per paired replicate; separate alternate seeds on unmasked data measure optimizer variation',
      'replication_limits':'five mask seeds and three PMM seeds quantify sensitivity; no claim of precise Monte Carlo uncertainty; shared patients/repetitions not independent clinical samples',
      'reporting':'all scenarios/k; no select-best-k, outcome association, p-values or causal attribution; complete-case selection limits generalization','input_sha256':sources}
    p=O/'protocol.json'
    if p.exists():assert read(p)==protocol,'Protocol change requires new version'
    else:dump(p,protocol)
    dump(O/'run_status.json',{'status':'running','stage':'preflight','script_sha256':sha(__file__),'frozen_files_verified':len(bm['sha256'])})
    return protocol
def natural(raw):
    dest=W/'natural';dest.mkdir(exist_ok=True)
    x=raw[FEATS].to_numpy(float);obs=np.isfinite(x);nd,elig=domains(obs);assert elig.all()
    missing=14-obs.sum(axis=1);base={};ref={};gaps={};fixed={k:[] for k in KS};ml={k:[] for k in KS};rows=[];identities={}
    for k in KS:
        b=pd.read_parquet(OLD/f'observed_equal/k{k}_labels.parquet');assert np.array_equal(b.stay_id,raw.stay_id)
        base[k]=b.cluster.to_numpy()-1
        med=pd.read_parquet(F/f'equal_feature_k{k}_medoids.parquet').sort_values('cluster')
        ids=raw.stay_id.to_numpy();index={int(t):i for i,t in enumerate(ids)};ind=np.array([index[int(t)] for t in med.stay_id]);identities[k]=ind
        dd=distance(x,x[ind]);assert np.isfinite(dd).all();pred=dd.argmin(axis=1)
        # Equivalent float64 accumulation may differ at genuine machine-precision ties: fail rather than silently remap.
        assert np.array_equal(pred,base[k]),'Frozen replay mismatch'
        ds=np.sort(dd,axis=1);gaps[k]=ds[:,1]-ds[:,0]
    for mi in range(1,11):
        z=pd.read_parquet(OLD/f'mice_{mi:02d}/imputed.parquet')[FEATS].to_numpy(float)
        assert np.array_equal(x[obs],z[obs]) and np.isfinite(z).all()
        for k in KS:
            m=pd.read_parquet(OLD/f'mice_{mi:02d}/k{k}_labels.parquet');assert np.array_equal(m.stay_id,raw.stay_id)
            lab=m.cluster.to_numpy()-1;ml[k].append(lab)
            same_anchor=distance(z,z[identities[k]]).argmin(axis=1);fixed[k].append(same_anchor)
            for name,a,b in [('observed_to_completed_fixed_identities',base[k],same_anchor),('completed_fixed_to_refitted',same_anchor,lab),('observed_to_refitted_mice',base[k],lab)]:
                rows.append({'imputation':mi,'k':k,'contrast':name,'n':len(raw),**metric(a,b,k)})
        print('natural bridge draw',mi,flush=True)
    pd.DataFrame(rows).to_csv(O/'natural_bridge_detail.csv',index=False)
    summaries=[];strata=[];allpatients=[]
    for k in KS:
        pair=np.zeros(len(raw));fixedpair=np.zeros(len(raw));cross=np.zeros(len(raw));crossfixed=np.zeros(len(raw))
        for i in range(10):
            cross+=(align(base[k],ml[k][i],k)[0]!=base[k]);crossfixed+=(fixed[k][i]!=base[k])
            for j in range(i):
                pair+=(align(ml[k][i],ml[k][j],k)[0]!=ml[k][i]);fixedpair+=(fixed[k][i]!=fixed[k][j])
        patient=raw[IDS].copy();patient['k']=k;patient['missing_features']=missing;patient['observed_domains']=nd
        patient['observed_anchor_gap']=gaps[k];patient['mi_pair_disagreement']=pair/45;patient['fixed_identity_mi_pair_disagreement']=fixedpair/45
        patient['cross_path_disagreement']=cross/10;patient['fixed_identity_cross_path_disagreement']=crossfixed/10
        allpatients.append(patient)
        groups={'missing_count':np.select([missing==0,missing<=2,missing<=4],['0','1-2','3-4'],default='5-8'),
          'anchor_gap_quartile':pd.qcut(pd.Series(gaps[k]),4,labels=False,duplicates='drop').astype(str).to_numpy()}
        for f in FEATS:groups[f+'_missing']=np.where(obs[:,FEATS.index(f)],'observed','missing')
        groups['observed_domains']=nd.astype(str)
        for name,group in groups.items():
            for level in np.unique(group):
                p=patient.loc[group==level]
                strata.append({'k':k,'stratifier':name,'level':str(level),'n':len(p),'mi_pair_disagreement_mean':float(p.mi_pair_disagreement.mean()),'cross_path_disagreement_mean':float(p.cross_path_disagreement.mean()),'fixed_identity_pair_disagreement_mean':float(p.fixed_identity_mi_pair_disagreement.mean()),'gap_median':float(p.observed_anchor_gap.median())})
        summaries.append({'k':k,'n':len(raw),'mi_pair_disagreement_mean':float(patient.mi_pair_disagreement.mean()),'fixed_identity_mi_pair_disagreement_mean':float(patient.fixed_identity_mi_pair_disagreement.mean()),'cross_path_disagreement_mean':float(patient.cross_path_disagreement.mean()),'no_pair_change_n':int((pair==0).sum()),'pair_disagreement_ge_half_n':int((pair/45>=.5).sum())})
    pd.concat(allpatients,ignore_index=True).to_parquet(dest/'patient_instability_local.parquet',index=False)
    pd.DataFrame(strata).to_csv(O/'patient_instability_strata.csv',index=False);pd.DataFrame(summaries).to_csv(O/'patient_instability_summary.csv',index=False)
    pd.DataFrame(rows).groupby(['k','contrast']).agg(ari_mean=('ari','mean'),ari_min=('ari','min'),ari_max=('ari','max'),disagreement_mean=('aligned_disagreement','mean')).reset_index().to_csv(O/'natural_bridge_summary.csv',index=False)
    dump(O/'natural_manifest.json',{'status':'complete','n':17810,'imputations':10,'pair_count':45,'label_alignment':'Hungarian maximum overlap per pair on full same cohort; labels are arbitrary; not calibrated uncertainty','outcomes_used':False})
def fit_worker(task):
    path,k,seed=task;d=np.load(path,mmap_mode='r');fit,restarts=C.fit_pam(d,k,5,seed+k,200)
    return k,fit,restarts
def fit_grid(x,path,seed,pool):
    path.mkdir(exist_ok=True,parents=True);digest=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest();done=path/'manifest.json'
    if done.exists():
        m=read(done);assert m['input_hash']==digest and m['seed']==seed
        assert all(sha(path/p)==h for p,h in m['sha256'].items())
        return {k:np.load(path/f'k{k}.npz') for k in KS}
    t=time.time();d=distance(x,x);assert np.isfinite(d).all() and np.array_equal(d,d.T);np.fill_diagonal(d,0)
    matrix=path/'distance.npy';np.save(matrix,d);del d
    result={}
    for fu in as_completed([pool.submit(fit_worker,(str(matrix),k,seed)) for k in KS]):
        k,fit,restarts=fu.result();np.savez_compressed(path/f'k{k}.npz',labels=fit['labels'],medoids=fit['medoids'],loss=fit['loss'])
        pd.DataFrame(restarts).to_csv(path/f'k{k}_starts.csv',index=False);result[k]=np.load(path/f'k{k}.npz')
    matrix.unlink()
    dump(done,{'status':'complete','input_hash':digest,'seed':seed,'n':len(x),'seconds':time.time()-t,'sha256':{p.name:sha(p) for p in path.iterdir() if p.suffix in ['.npz','.csv']}})
    print('grid',path.name,'n',len(x),'seconds',round(time.time()-t,1),flush=True)
    return result
def impute(x,seed,path):
    digest=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest();mf=path.with_suffix('.json')
    if mf.exists():
        m=read(mf);assert m['seed']==seed and m['input_hash']==digest and m['sha256']==sha(path);return pd.read_parquet(path)[FEATS].to_numpy(float)
    source=pd.DataFrame(x,columns=FEATS);chain=MICEData(source.copy(),perturbation_method='boot',k_pmm=10,rng=np.random.default_rng(seed))
    chain.update_all(20);z=chain.data[FEATS].to_numpy(float);ok=np.isfinite(x)
    assert np.isfinite(z).all() and np.array_equal(x[ok],z[ok])
    for j in range(14):assert np.isin(z[~ok[:,j],j],x[ok[:,j],j]).all()
    pd.DataFrame(z,columns=FEATS).to_parquet(path,index=False)
    dump(mf,{'seed':seed,'input_hash':digest,'sha256':sha(path),'observed_unchanged':True,'donors_only_observed_after_mask':True,'cycles':20,'donors':10})
    return z
def controlled(raw,protocol):
    full=raw.loc[raw[FEATS].notna().all(axis=1)].reset_index(drop=True);x0=full[FEATS].to_numpy(float);assert len(full)==8338
    empirical=raw[FEATS].isna().to_numpy();empirical=empirical[empirical.any(axis=1)]
    rows=[];flow=[];errors=[];baseline_optimizer=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for si,scenario in enumerate(protocol['mask_scenarios']):
            for rep in range(1,6):
                tag=f'{scenario}_r{rep:02d}';p=W/tag;p.mkdir(exist_ok=True)
                seed=SEED+si*10000+rep*100;rng=np.random.default_rng(seed)
                hide=empirical[rng.integers(0,len(empirical),len(x0))] if scenario=='empirical_incomplete_pattern' else rng.random(x0.shape)<({'mcar20':.2,'mcar40':.4}[scenario])
                _,eligible=domains(~hide);ids=np.flatnonzero(eligible);pair_ok=compatible(~hide[ids]);ids=ids[pair_ok]
                truth=x0[ids].copy();mask=hide[ids];masked=truth.copy();masked[mask]=np.nan
                assert len(ids)>1000 and np.all(compatible(~mask))
                np.savez_compressed(p/'mask_and_order_local.npz',complete_case_row=ids,hidden=mask)
                full.iloc[ids][IDS].to_parquet(p/'patient_order_local.parquet',index=False)
                pd.DataFrame(masked,columns=FEATS).to_parquet(p/'masked_local.parquet',index=False)
                flow.append({'scenario':scenario,'replicate':rep,'complete_n':len(x0),'feature_eligible_n':int(eligible.sum()),'pair_incompatible_removed':int((~pair_ok).sum()),'analyzed_n':len(ids),'retained_hidden_fraction':float(mask.mean()),'all_hidden_fraction':float(hide.mean())})
                dump(O/'run_status.json',{'status':'running','stage':tag,'script_sha256':sha(__file__)})
                base=fit_grid(truth,p/'complete_baseline',seed,pool);observed=fit_grid(masked,p/'observed_refit',seed,pool)
                # Optimizer-only control: identical unmasked matrix and a separate five-start seed set, every replicate.
                alternate=fit_grid(truth,p/'complete_alternate_seeds',seed+5000000,pool)
                for k in KS:
                    baseline_optimizer.append({'scenario':scenario,'replicate':rep,'k':k,**metric(base[k]['labels'],alternate[k]['labels'],k)})
                    anchors=base[k]['medoids'];ref=base[k]['labels'];q=distance(masked,truth[anchors]).argmin(axis=1);both=distance(masked,masked[anchors]).argmin(axis=1)
                    for name,lab in [('observed_query_fixed_complete_anchors',q),('observed_both_fixed_identities',both),('observed_refit',observed[k]['labels'])]:
                        rows.append({'scenario':scenario,'replicate':rep,'imputation':0,'k':k,'contrast':name,'n':len(ids),**metric(ref,lab,k)})
                    np.savez_compressed(p/f'observed_k{k}_fixed.npz',query=q,both=both)
                for mi in range(1,4):
                    z=impute(masked,seed+1000000+mi,p/f'imputed_{mi:02d}.parquet');fit=fit_grid(z,p/f'pmm_refit_{mi:02d}',seed,pool)
                    for k in KS:
                        anchors=base[k]['medoids'];ref=base[k]['labels'];q=distance(z,truth[anchors]).argmin(axis=1);both=distance(z,z[anchors]).argmin(axis=1)
                        labs={'pmm_query_fixed_complete_anchors':q,'pmm_both_fixed_identities':both,'pmm_refit':fit[k]['labels']}
                        for name,lab in labs.items():rows.append({'scenario':scenario,'replicate':rep,'imputation':mi,'k':k,'contrast':name,'n':len(ids),**metric(ref,lab,k)})
                        rows.append({'scenario':scenario,'replicate':rep,'imputation':mi,'k':k,'contrast':'pmm_fixed_to_refitted','n':len(ids),**metric(both,fit[k]['labels'],k)})
                        np.savez_compressed(p/f'pmm_{mi:02d}_k{k}_fixed.npz',query=q,both=both)
                    err=np.abs(scaled(z)-scaled(truth));errors.extend({'scenario':scenario,'replicate':rep,'imputation':mi,'feature':f,'hidden_n':int(mask[:,j].sum()),'scaled_hidden_mae':float(err[mask[:,j],j].mean()) if mask[:,j].any() else None} for j,f in enumerate(FEATS))
                    print('finished',tag,'imputation',mi,flush=True)
                pd.DataFrame(rows).to_csv(O/'controlled_detail.csv',index=False);pd.DataFrame(flow).to_csv(O/'controlled_sample_flow.csv',index=False)
                pd.DataFrame(errors).to_csv(O/'controlled_imputation_errors.csv',index=False);pd.DataFrame(baseline_optimizer).to_csv(O/'optimizer_control.csv',index=False)
    # Within-mask imputation averages, then summarize five masks equally; no pseudo-independent confidence intervals.
    df=pd.DataFrame(rows);rr=df.groupby(['scenario','replicate','k','contrast']).agg(ari=('ari','mean'),disagreement=('aligned_disagreement','mean'),n=('n','first')).reset_index()
    rr.to_csv(O/'controlled_mask_level_summary.csv',index=False)
    rr.groupby(['scenario','k','contrast']).agg(mask_replicates=('replicate','size'),ari_mean=('ari','mean'),ari_min=('ari','min'),ari_max=('ari','max'),disagreement_mean=('disagreement','mean'),n_min=('n','min'),n_max=('n','max')).reset_index().to_csv(O/'controlled_summary.csv',index=False)
    dump(O/'controlled_manifest.json',{'status':'complete','mask_replicates':15,'imputed_datasets':45,'k':KS,'source_complete_n':8338,'outcomes_used':False,'validation_test_used':False})
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','natural','controlled'],default='all');args=parser.parse_args()
    raw=load_source();protocol=preflight(raw);t=time.time()
    if args.phase in ['all','natural']:natural(raw)
    if args.phase in ['all','controlled']:controlled(raw,protocol)
    dump(O/'run_status.json',{'status':'computations_complete_qa_pending','phase':args.phase,'elapsed_seconds':time.time()-t,'script_sha256':sha(__file__)})
if __name__=='__main__':main()
