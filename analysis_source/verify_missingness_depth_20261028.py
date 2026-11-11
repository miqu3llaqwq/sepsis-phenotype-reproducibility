"""Independent replay from saved tables; does not import the producing analysis."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parent;O=ROOT/'outputs/missingness_depth_20261028';W=ROOT/'work/missingness_depth_20261028';B=ROOT/'work/mimic_rebuilt_20261026';F=B/'frozen/20261027';OLD=B/'data/revised_analysis_20261027'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(F/'core14_primary_transform.json');fs=cfg['features'];lo=np.array([cfg['limits'][f][0] for f in fs]);hi=np.array([cfg['limits'][f][1] for f in fs])
def dis(x,y):
    a=(np.clip(x,lo,hi)-lo)/(hi-lo);b=(np.clip(y,lo,hi)-lo)/(hi-lo)
    ans=[]
    for row in b:
        common=np.isfinite(a)&np.isfinite(row);den=common.sum(axis=1)
        assert (den>0).all()
        ans.append(np.nansum(np.abs(a-row),axis=1)/den)
    return np.column_stack(ans).astype(np.float32)
def aligned(a,b,k):
    tab=pd.crosstab(pd.Series(a),pd.Series(b)).reindex(index=range(k),columns=range(k),fill_value=0).to_numpy();r,c=linear_sum_assignment(-tab);mapping=dict(zip(c,r))
    return np.array([mapping[v] for v in b])
def close(a,b):assert np.allclose(a,b,rtol=1e-7,atol=1e-7),(a,b)
def verify_metric(row,a,b,k):
    close(row.ari,adjusted_rand_score(a,b));close(row.aligned_disagreement,np.mean(a!=aligned(a,b,k)))
def main():
    assert read(O/'controlled_manifest.json')['status']=='complete'
    protocol=read(O/'protocol.json');assert all(sha(p)==h for p,h in protocol['input_sha256'].items())
    bm=read(F/'bundle_manifest.json');assert all(sha(F/p)==h for p,h in bm['sha256'].items())
    raw=pd.read_parquet(B/'data/core14_preparation/eligible_train_unimputed.parquet',columns=['subject_id','stay_id','hadm_id','split']+fs).sort_values('stay_id').reset_index(drop=True)
    assert raw.split.eq('train').all() and len(raw)==17810
    x=raw[fs].to_numpy();obs=np.isfinite(x);complete=obs.all(axis=1);truth_all=x[complete]
    source=pd.read_parquet(W/'natural/patient_instability_local.parquet');factor=pd.read_parquet(W/'natural/query_anchor_patient_sensitivity_local.parquet')
    natural=pd.read_csv(O/'natural_bridge_detail.csv');draws=[pd.read_parquet(OLD/f'mice_{i:02d}/imputed.parquet')[fs].to_numpy() for i in range(1,11)]
    for z in draws:assert np.array_equal(x[obs],z[obs])
    ids={int(s):i for i,s in enumerate(raw.stay_id)};natural_checks=0;factor_distance_cells=0
    sample=np.random.default_rng(81723).choice(len(raw),size=250,replace=False)
    for k in range(2,7):
        base=pd.read_parquet(OLD/f'observed_equal/k{k}_labels.parquet').cluster.to_numpy()-1
        med=pd.read_parquet(F/f'equal_feature_k{k}_medoids.parquet').sort_values('cluster');ind=np.array([ids[int(s)] for s in med.stay_id])
        dd=dis(x,x[ind]);assert np.array_equal(base,dd.argmin(axis=1));gaps=np.sort(dd,axis=1)[:,1]-np.sort(dd,axis=1)[:,0]
        p=source[source.k.eq(k)].reset_index(drop=True);assert np.array_equal(p.stay_id,raw.stay_id);close(p.observed_anchor_gap,gaps)
        labs=[pd.read_parquet(OLD/f'mice_{i:02d}/k{k}_labels.parquet').cluster.to_numpy()-1 for i in range(1,11)]
        disagreement=np.zeros(len(raw));cross=np.zeros(len(raw))
        for i,a in enumerate(labs):
            cross+=base!=aligned(base,a,k)
            fixed=dis(draws[i],draws[i][ind]).argmin(axis=1)
            for name,left,right in [('observed_to_completed_fixed_identities',base,fixed),('completed_fixed_to_refitted',fixed,a),('observed_to_refitted_mice',base,a)]:
                row=natural[natural.k.eq(k)&natural.imputation.eq(i+1)&natural.contrast.eq(name)].iloc[0];verify_metric(row,left,right,k);natural_checks+=1
            for b in labs[:i]:disagreement+=a!=aligned(a,b,k)
        close(p.mi_pair_disagreement,disagreement/45);close(p.cross_path_disagreement,cross/10)
        f=np.load(W/f'natural/factorial_k{k}_labels_local.npz')['labels'];assert f.shape==(10,10,len(raw))
        for i in range(10):
            for j in range(10):assert np.array_equal(f[i,j,sample],dis(draws[i][sample],draws[j][ind]).argmin(axis=1));factor_distance_cells+=len(sample)*k
        q=np.zeros(len(raw));a=np.zeros(len(raw));joint=np.zeros(len(raw));qa=np.zeros(len(raw));aa=np.zeros(len(raw))
        for i,j in itertools.combinations(range(10),2):
            joint+=f[i,i]!=f[j,j]
            q+=(f[i,:,:]!=f[j,:,:]).sum(axis=0);a+=(f[:,i,:]!=f[:,j,:]).sum(axis=0)
            for held in range(10):
                for left,right,total in [(f[i,held],f[j,held],qa),(f[held,i],f[held,j],aa)]:
                    tab=np.bincount(left.astype(int)*k+right,minlength=k*k).reshape(k,k);rr,cc=linear_sum_assignment(-tab);mapping=np.empty(k,int);mapping[cc]=rr
                    total+=left!=mapping[right]
        pp=factor[factor.k.eq(k)].reset_index(drop=True)
        close(pp.query_draw_disagreement,q/450);close(pp.anchor_draw_disagreement,a/450);close(pp.joint_draw_disagreement,joint/45)
        close(pp.query_aligned_disagreement,qa/450);close(pp.anchor_aligned_disagreement,aa/450)
        assert (pp.query_draw_disagreement[complete]==0).all()
    flow=pd.read_csv(O/'controlled_sample_flow.csv');detail=pd.read_csv(O/'controlled_detail.csv');opt=pd.read_csv(O/'optimizer_control.csv');errs=pd.read_csv(O/'controlled_imputation_errors.csv')
    assert len(flow)==15 and len(detail)==1125 and len(opt)==75 and len(errs)==630
    assert flow.groupby('scenario').size().eq(5).all()
    controlled_checks=0;imputation_checks=0;fit_checks=0
    empirical=np.isnan(x);empirical=empirical[empirical.any(axis=1)]
    for row in flow.itertuples():
        p=W/f'{row.scenario}_r{row.replicate:02d}';m=np.load(p/'mask_and_order_local.npz');idx=m['complete_case_row'];hide=m['hidden'];truth=truth_all[idx]
        si=protocol['mask_scenarios'].index(row.scenario);rng=np.random.default_rng(protocol['seed']+si*10000+row.replicate*100)
        fullhide=empirical[rng.integers(0,len(empirical),len(truth_all))] if row.scenario=='empirical_incomplete_pattern' else rng.random(truth_all.shape)<({'mcar20':.2,'mcar40':.4}[row.scenario])
        fullobs=~fullhide;full_domains={name:fullobs[:,[fs.index(f) for f in fields]].any(axis=1) for name,fields in cfg['domains'].items()}
        keep=(fullobs.sum(axis=1)>=6)&(np.column_stack(list(full_domains.values())).sum(axis=1)>=4)&full_domains['circulation']&full_domains['respiratory']
        eligible_idx=np.flatnonzero(keep);patternbits=fullobs[eligible_idx].astype('uint16')@(2**np.arange(14,dtype='uint16'));unique=np.unique(patternbits)
        compatible=np.array([not np.any((b&unique)==0) for b in patternbits]);expected_idx=eligible_idx[compatible]
        assert np.array_equal(idx,expected_idx) and np.array_equal(hide,fullhide[idx]);assert int(keep.sum())==row.feature_eligible_n and int((~compatible).sum())==row.pair_incompatible_removed
        observed=pd.read_parquet(p/'masked_local.parquet')[fs].to_numpy();assert np.array_equal(np.isnan(observed),hide);assert np.array_equal(observed[~hide],truth[~hide])
        assert len(idx)==row.analyzed_n and len(np.unique(idx))==len(idx);close(hide.mean(),row.retained_hidden_fraction)
        nd=np.column_stack([(~hide)[:,[fs.index(f) for f in fields]].any(axis=1) for fields in cfg['domains'].values()]);names=list(cfg['domains'])
        assert ((~hide).sum(axis=1)>=6).all() and (nd.sum(axis=1)>=4).all() and nd[:,names.index('circulation')].all() and nd[:,names.index('respiratory')].all()
        pattern=np.unique((~hide).astype('uint16')@(2**np.arange(14,dtype='uint16')))
        assert np.all((pattern[:,None]&pattern[None,:])>0)
        zlist=[pd.read_parquet(p/f'imputed_{i:02d}.parquet')[fs].to_numpy() for i in range(1,4)]
        for mi,z in enumerate(zlist,1):
            assert np.isfinite(z).all() and np.array_equal(z[~hide],truth[~hide])
            for j,f in enumerate(fs):
                assert np.isin(z[hide[:,j],j],observed[~hide[:,j],j]).all();imputation_checks+=1
                ee=errs[errs.scenario.eq(row.scenario)&errs.replicate.eq(row.replicate)&errs.imputation.eq(mi)&errs.feature.eq(f)].iloc[0]
                actual=np.abs(np.clip(z[:,j],lo[j],hi[j])-np.clip(truth[:,j],lo[j],hi[j]))/(hi[j]-lo[j])
                if hide[:,j].any():close(ee.scaled_hidden_mae,actual[hide[:,j]].mean())
        frame=detail[detail.scenario.eq(row.scenario)&detail.replicate.eq(row.replicate)]
        for k in range(2,7):
            grids={name:np.load(p/name/f'k{k}.npz') for name in ['complete_baseline','observed_refit','complete_alternate_seeds']+[f'pmm_refit_{i:02d}' for i in range(1,4)]}
            for name,g in grids.items():
                xx=truth if name.startswith('complete') else observed if name=='observed_refit' else zlist[int(name[-2:])-1]
                d=dis(xx,xx[g['medoids']]);assert len(np.unique(g['medoids']))==k;assert np.array_equal(d.argmin(axis=1),g['labels']);close(d.min(axis=1).sum(dtype=np.float64),float(g['loss']));fit_checks+=1
                mm=read(p/name/'manifest.json');assert all(sha(p/name/q)==h for q,h in mm['sha256'].items())
            reference=grids['complete_baseline']['labels'];anchors=grids['complete_baseline']['medoids']
            rowopt=opt[opt.scenario.eq(row.scenario)&opt.replicate.eq(row.replicate)&opt.k.eq(k)].iloc[0];verify_metric(rowopt,reference,grids['complete_alternate_seeds']['labels'],k)
            for mi,xx in [(0,observed)]+list(enumerate(zlist,1)):
                prefix='observed' if mi==0 else 'pmm';fixed={'query':dis(xx,truth[anchors]).argmin(axis=1),'both':dis(xx,xx[anchors]).argmin(axis=1)}
                saved=np.load(p/(f'observed_k{k}_fixed.npz' if mi==0 else f'pmm_{mi:02d}_k{k}_fixed.npz'))
                for key in fixed:assert np.array_equal(saved[key],fixed[key])
                labs={prefix+'_query_fixed_complete_anchors':fixed['query'],prefix+'_both_fixed_identities':fixed['both'],prefix+'_refit':grids['observed_refit' if mi==0 else f'pmm_refit_{mi:02d}']['labels']}
                for name,lab in labs.items():
                    rr=frame[frame.imputation.eq(mi)&frame.k.eq(k)&frame.contrast.eq(name)].iloc[0];verify_metric(rr,reference,lab,k);controlled_checks+=1
                if mi:
                    rr=frame[frame.imputation.eq(mi)&frame.k.eq(k)&frame.contrast.eq('pmm_fixed_to_refitted')].iloc[0];verify_metric(rr,fixed['both'],labs['pmm_refit'],k);controlled_checks+=1
    # Output summaries must reflect means within mask before aggregation across masks.
    per=detail.groupby(['scenario','replicate','k','contrast']).ari.mean().reset_index();expected=per.groupby(['scenario','k','contrast']).ari.mean();summary=pd.read_csv(O/'controlled_summary.csv').set_index(['scenario','k','contrast'])
    for key,value in expected.items():close(value,summary.loc[key,'ari_mean'])
    result={'status':'pass','date':'2026-10-28','frozen_files_verified':len(bm['sha256']),'natural_contrast_metrics_recomputed':natural_checks,
      'natural_patient_pair_and_cross_scores_recomputed':17810*5,'factorial_distance_cells_independently_recomputed':factor_distance_cells,
      'controlled_mask_replicates_verified':len(flow),'controlled_comparison_metrics_recomputed':controlled_checks,'grid_label_and_loss_checks':fit_checks,
      'imputation_feature_donor_checks':imputation_checks,'full_complete_query_invariance':True,'clinical_ground_truth_claimed':False,
      'limitations':'Does not establish clinical truth, correct missingness mechanism or independent external validation; no statistical independence of repeated masks assumed.',
      'script_sha256':sha(__file__),'input_protocol_sha256':sha(O/'protocol.json')}
    (O/'independent_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
