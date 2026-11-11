"""Registered bounded TRAIN diagnostic; immutable v5 sources, aggregate outputs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib
import json
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
m=importlib.import_module('18_physiology_v5')

def observed_limits(raw,quantiles):
    return {f:[float(raw[f].quantile(quantiles[0])),float(raw[f].quantile(quantiles[1]))] for f in m.FEATURES}

def compare(a,b,k,tag):
    joined=a.merge(b,on=m.IDS,validate='one_to_one',suffixes=('_a','_b'))
    alignment,_,_=m.c.aligned(joined.cluster_a.to_numpy()-1,joined.cluster_b.to_numpy()-1,k)
    return {'comparison':tag,'k':k,'n_common':len(joined),
        'ari':float(adjusted_rand_score(joined.cluster_a,joined.cluster_b)),
        'aligned_agreement':alignment}

def main():
    started=time.time();cfgpath=ROOT/'config/clustering_v6.json'
    cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
    old=json.loads((ROOT/'logs/clustering_v5_latest.json').read_text(encoding='utf-8'))
    _,frames=m.load(old);raw=frames['main'][m.eligible(frames['main'],8)].reset_index(drop=True)
    olddata=Path(old['data_path']);oldtables=Path(old['tables_path']);oldreports=Path(old['report_path'])
    oldmanifest=json.loads((oldreports/'run_manifest.json').read_text(encoding='utf-8'))
    assert old['status']=='completed_development_clinical_gate_pending'
    assert cfg['features']==m.FEATURES
    # Verify every filled view and reference artifact before reuse.
    old_unimputed=pd.read_parquet(olddata/'multiple_imputation/coverage8_unimputed.parquet')
    assert raw.equals(old_unimputed)
    filled=pd.read_parquet(olddata/'baseline/coverage8_imputed.parquet')
    expected,_,limits=m.c.fit_transform(raw,m.FEATURES)
    pd.testing.assert_frame_equal(filled,expected)
    scales=observed_limits(raw,cfg['robust_quantiles'])
    run=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    data=olddata.parent/'clustering_v6'/run;data.mkdir(parents=True)
    tables=ROOT/'outputs/tables/clustering_v6'/run;tables.mkdir(parents=True)
    reports=ROOT/'outputs/reports/clustering_v6'/run;reports.mkdir(parents=True)
    loc={'run_id':run,'status':'running','data_path':str(data),'tables_path':str(tables),'report_path':str(reports),
        'v5_data_path':old['data_path'],'v5_report_path':old['report_path'],
        'outcomes_loaded':False,'validation_test_used':False,'clinical_k_frozen':False}
    m.dump(ROOT/'logs/clustering_v6_latest.json',loc)
    hashes={};manifest=dict(loc,config_sha256=m.c.sha(cfgpath),script_sha256=m.c.sha(__file__),
        helper_sha256={n:m.c.sha(ROOT/'analysis'/n) for n in ['18_physiology_v5.py','14_cluster_training.py','15_gender_diagnostic.py']},
        n_train=len(raw),n_complete=int(raw[m.FEATURES].notna().all(axis=1).sum()),
        source_input_sha256=oldmanifest['input_sha256'])
    for f in [Path(__file__),cfgpath,ROOT/'protocol/physiology_v6_diagnostic_plan.md']:(reports/f.name).write_bytes(f.read_bytes())
    m.dump(reports/'run_manifest.json',manifest)
    raw.to_parquet(data/'fixed_training_unimputed.parquet',index=False)
    m.dump(data/'distance_limits.json',{'original':limits,'robust':scales})
    rows=[]
    for f in m.FEATURES:
        s=raw[f];lo,hi=scales[f]
        rows.append({'feature':f,'observed_n':int(s.notna().sum()),'missing_fraction':float(s.isna().mean()),
            'original_min':limits[f][0],'original_max':limits[f][1],'p01':lo,'p99':hi,
            'below_p01':int(s.lt(lo).sum()),'above_p99':int(s.gt(hi).sum())})
    pd.DataFrame(rows).to_csv(tables/'representation_limits_and_counts.csv',index=False)
    correlations=[]
    for i,a in enumerate(m.FEATURES):
        for b in m.FEATURES[i+1:]:
            pair=raw[[a,b]].dropna()
            correlations.append({'feature_a':a,'feature_b':b,'n_paired':len(pair),'spearman_r':float(pair.corr(method='spearman').iloc[0,1])})
    pd.DataFrame(correlations).to_csv(tables/'observed_feature_correlations.csv',index=False)
    complete_mask=raw[m.FEATURES].notna().all(axis=1);complete=raw[complete_mask].reset_index(drop=True)
    descriptions=[]
    for name,mask in [('complete16',complete_mask),('incomplete',~complete_mask)]:
        group=raw[mask]
        for f in ['age']+m.FEATURES:
            s=group[f];descriptions.append({'group':name,'feature':f,'n':len(group),'n_observed':int(s.notna().sum()),
                'median':float(s.median()),'q1':float(s.quantile(.25)),'q3':float(s.quantile(.75))})
        descriptions.append({'group':name,'feature':'female_fraction','n':len(group),'median':float(group.gender.eq('F').mean())})
    pd.DataFrame(descriptions).to_csv(tables/'complete_vs_incomplete_description.csv',index=False)
    fit_b,D=m.fit_grid(filled,raw,scales,'B_median_robust',data,tables,cfg);del D
    pooled=None;individual={}
    original=raw[m.FEATURES].to_numpy(dtype=float);obs=np.isfinite(original)
    for i in range(1,cfg['imputations']+1):
        path=olddata/f'multiple_imputation/imputation{i:02d}_filled.parquet';view=pd.read_parquet(path)
        assert view.shape==original.shape and np.isfinite(view.to_numpy()).all()
        np.testing.assert_array_equal(view.to_numpy()[obs],original[obs])
        for j,f in enumerate(m.FEATURES):assert np.isin(view[f].to_numpy()[~obs[:,j]],original[obs[:,j],j]).all()
        hashes[str(path)]=m.c.sha(path)
        fits,D=m.fit_grid(view,raw,scales,f'D_imputation{i:02d}',data,tables,cfg);individual[i]=fits
        if pooled is None:pooled=D/cfg['imputations']
        else:pooled+=D/cfg['imputations']
        del D
    np.fill_diagonal(pooled,0);np.save(data/'D_mean_gower_distance.npy',pooled)
    fit_d,_=m.fit_grid(raw[m.FEATURES],raw,scales,'D_pooled_robust',data,tables,cfg,distance=pooled)
    del pooled
    agreement=[];uncertainty=[]
    for k in cfg['candidate_k']:
        support=np.zeros((len(raw),k),dtype=int)
        for i,fits in individual.items():
            align,mapped,_=m.c.aligned(fit_d[k]['labels'],fits[k]['labels'],k)
            support[np.arange(len(raw)),mapped]+=1
            agreement.append({'k':k,'imputation':i,'ari':float(adjusted_rand_score(fit_d[k]['labels'],fits[k]['labels'])),
                'aligned_agreement':align})
        probabilities=support/cfg['imputations'];assigned=probabilities[np.arange(len(raw)),fit_d[k]['labels']]
        entropy=-(probabilities*np.log(np.maximum(probabilities,1e-30))).sum(axis=1)/np.log(k)
        out=raw[m.IDS].copy();out['cluster']=fit_d[k]['labels']+1;out['assigned_label_support']=assigned;out['entropy']=entropy
        out.to_parquet(data/f'D_k{k}_uncertainty.parquet',index=False)
        uncertainty.append({'k':k,'mean_assigned_support':float(assigned.mean()),'fraction_support_below_0_8':float((assigned<.8).mean()),'mean_entropy':float(entropy.mean())})
    pd.DataFrame(agreement).to_csv(tables/'D_individual_vs_pooled.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(tables/'D_assignment_uncertainty.csv',index=False)
    for tag,scale in [('CC_original',limits),('CC_robust',scales)]:
        _,D=m.fit_grid(complete[m.FEATURES],complete,scale,tag,data,tables,cfg);del D
    comparisons=[]
    for k in cfg['candidate_k']:
        paths={'A':olddata/f'baseline/coverage8_k{k}_labels.parquet',
            'B':data/f'B_median_robust_k{k}_labels.parquet',
            'C':olddata/f'multiple_imputation/pooled10_k{k}_labels.parquet',
            'D':data/f'D_pooled_robust_k{k}_labels.parquet',
            'CC_original':data/f'CC_original_k{k}_labels.parquet','CC_robust':data/f'CC_robust_k{k}_labels.parquet'}
        labels={name:pd.read_parquet(path) for name,path in paths.items()}
        for name in ['A','B','C','D']:assert labels[name][m.IDS].equals(raw[m.IDS])
        for a,b in [('A','B'),('A','C'),('C','D'),('B','D'),('CC_original','CC_robust')]+[(cc,tag) for cc in ['CC_original','CC_robust'] for tag in ['A','B','C','D']]:
            comparisons.append(compare(labels[a],labels[b],k,a+'_vs_'+b))
        for name in ['A','C']:hashes[str(paths[name])]=m.c.sha(paths[name])
    pd.DataFrame(comparisons).to_csv(tables/'paired_condition_comparisons.csv',index=False)
    # Save the existing aggregate references with explicit provenance.
    for tag,source in [('A',oldtables/'baseline/coverage8_metrics.csv'),('C',oldtables/'multiple_imputation/pooled10_metrics.csv')]:
        table=pd.read_csv(source);table['analysis']=tag;table.to_csv(tables/f'{tag}_reference_metrics.csv',index=False)
        assert m.c.sha(source)==oldmanifest['output_table_sha256'][str(source.relative_to(oldtables))]
    # Recompute an actual matrix sub-block and all pooled assignments.
    saved=np.load(data/'D_mean_gower_distance.npy',mmap_mode='r');assert saved.shape==(len(raw),len(raw))
    idx=np.linspace(0,len(raw)-1,100,dtype=int);check=np.zeros((100,100),dtype=np.float32)
    for i in range(1,cfg['imputations']+1):
        view=pd.read_parquet(olddata/f'multiple_imputation/imputation{i:02d}_filled.parquet').iloc[idx]
        check+=m.g.numeric_gower(view,view,m.FEATURES,scales)/cfg['imputations']
    np.testing.assert_allclose(saved[np.ix_(idx,idx)],check,atol=1e-7)
    for k in cfg['candidate_k']:
        np.testing.assert_array_equal(np.argmin(saved[:,fit_d[k]['medoids']],axis=1),fit_d[k]['labels'])
    for path,value in hashes.items():assert m.c.sha(path)==value
    m.load(old)
    frozen=json.loads((ROOT/'config/frozen_split_v2.json').read_text(encoding='utf-8'))
    assert m.c.sha(frozen['dataset_path'])==frozen['dataset_sha256']
    assert m.c.sha(cfgpath)==manifest['config_sha256']
    loc['status']='computation_verified_report_pending';m.dump(ROOT/'logs/clustering_v6_latest.json',loc)
    manifest.update(loc,elapsed_seconds=time.time()-started,reused_artifact_sha256=hashes,
        verification={'source_observations_unchanged':True,'original_cohort_unchanged':True,
            'fixed_population_and_row_order':True,'observed_values_preserved_in_reused_views':True,
            'donor_support_verified':True,'D_distance_subblock_recomputed':100,
            'all_D_assignments_recomputed':True,'full_pipeline_resampling_done':False})
    m.dump(reports/'run_manifest.json',manifest)
    print('V6 computation and actual-artifact verification completed.',flush=True)

if __name__=='__main__':main()
