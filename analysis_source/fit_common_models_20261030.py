"""Patient-disjoint refit for the new identifiable common-definition branch."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,shutil,time
import importlib.metadata,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import run_revised_phenotypes_20261027 as engine
from build_common_identified_20261030 import ROOT,WORK,DATA,OUT,dump,sha

MODEL=WORK/'models_20261030';TABLES=OUT/'model_tables';FROZEN=WORK/'frozen_20261030'
FEATS=engine.FEATS;KS=list(range(2,7));SEED=20261009

def patient_partition(x):
    people=np.array(sorted(x.person_id.unique()));np.random.default_rng(SEED).shuffle(people)
    n=len(people);a=int(.7*n);b=a+int(.15*n)
    part={p:('train' if i<a else 'validation' if i<b else 'test') for i,p in enumerate(people)}
    x=x.copy();x['split']=x.person_id.map(part)
    assert x.groupby('person_id').split.nunique().max()==1
    return x

def main():
    for name in ['independent_common_cohort_qa.json','remote_chart_aggregation_qa.json']:
        assert json.loads((OUT/name).read_text())['status']=='pass',name
    MODEL.mkdir(exist_ok=True);TABLES.mkdir(exist_ok=True);engine.OUT=MODEL
    dump('software_environment.json',{'python':platform.python_version(),'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','statsmodels','kmedoids','pyarrow','google-cloud-bigquery']}})
    source=DATA/'mimic_common_features_local.parquet';x=pd.read_parquet(source)
    x=patient_partition(x);x['subject_id']=x.person_id;x['hadm_id']=x.admission_id
    x.to_parquet(DATA/'mimic_common_partitioned_features_local.parquet',index=False)
    train=x[x.split.eq('train')&x.feature_eligible].sort_values('stay_id').reset_index(drop=True)
    before=len(train);present=train[FEATS].notna().to_numpy();patterns=(present.astype(np.uint32)*(1<<np.arange(14,dtype=np.uint32))).sum(axis=1)
    unique=np.unique(patterns);bad=unique[((unique[:,None]&unique[None,:])==0).any(axis=1)]
    train=train[~np.isin(patterns,bad)].reset_index(drop=True);raw=train[engine.IDS+FEATS]
    assert len(raw)>100 and raw.stay_id.is_unique
    limits=engine.limits(raw);check=engine.check_distance(raw,limits)
    contract={'source_sha256':sha(source),'protocol_sha256':sha(OUT/'common_protocol_v2.json'),'features':FEATS,'k':KS,'restarts':5,'max_iter':200,
        'training_rows_before_compatibility':before,'training_rows':len(train),'training_people':train.person_id.nunique(),
        'incompatible_excluded':before-len(train),'source_split':'patient-disjoint70/15/15, before feature eligibility',
        'resampling':'10 independent80-percent patient subsamples; keep all admissions from selected patients; refit limits',
        'primary':'10 MICE-PMM full chains plus10 independently imputed patient subsamples;20 updates,10donors,bootstrap parameter perturbation',
        'external_model':'Observed-value equal-feature Gower, no external imputer; all k2..6 frozen without selecting k',
        'no_outcomes_in_model':True,'eicu_used_for_fit':False,'clinical_k_frozen':False,'distance_check':check}
    cp=OUT/'new_training_contract.json'
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:dump(cp.name,contract)
    x.groupby('split').agg(admissions=('stay_id','size'),people=('person_id','nunique'),feature_eligible=('feature_eligible','sum')).reset_index().to_csv(TABLES/'partition_flow.csv',index=False)
    raw.to_parquet(MODEL/'training_reference_local.parquet',index=False)
    dump('new_model_status.json',{'status':'running','stage':'observed_equal','training_rows':len(raw),'training_people':train.person_id.nunique()})
    print('NEW TRAINING',len(raw),'admissions',train.person_id.nunique(),'people',flush=True)
    st=[];mi_detail=[];mi_resample=[];mi_labels={k:[] for k in KS};rng=np.random.default_rng(SEED)
    people=np.array(sorted(train.person_id.unique()))
    with ProcessPoolExecutor(max_workers=3) as pool:
        base=engine.fit_grid(raw,limits,'observed_equal',SEED+170000,pool=pool)
        for rep in range(10):
            picked=rng.choice(people,int(.8*len(people)),replace=False)
            ix=np.flatnonzero(train.person_id.isin(picked));sub=raw.iloc[ix].reset_index(drop=True);lim=engine.limits(sub)
            tag=f'patient_resample_{rep+1:02d}';dump('new_model_status.json',{'status':'running','stage':tag,'training_rows':len(raw)})
            labels=engine.fit_grid(sub,lim,tag,SEED+160000+rep*100,pool=pool)
            sub[engine.IDS].to_parquet(MODEL/tag/'subsample_ids.parquet',index=False)
            for k in KS:
                med=pd.read_parquet(MODEL/tag/f'k{k}_medoids.parquet');pred=np.argmin(engine.eq_distance(raw,med,lim),axis=1)
                st.append({'rep':rep+1,'k':k,'n_subsample':len(sub),'n_people':len(picked),'ari':adjusted_rand_score(base[k],pred)})
            pd.DataFrame(st).to_csv(TABLES/'observed_patient_resampling_detail.csv',index=False)
            completed=engine.impute(raw,rep,f'mice_{rep+1:02d}');ml=engine.fit_grid(completed,limits,f'mice_{rep+1:02d}',SEED+150000+rep*100,pool=pool)
            for k in KS:
                mi_labels[k].append(ml[k]);mi_detail.append({'imputation':rep+1,'k':k,'ari':adjusted_rand_score(base[k],ml[k])})
            pd.DataFrame(mi_detail).to_csv(TABLES/'mice_vs_observed_detail.csv',index=False)
            cs=engine.impute(sub,rep,f'mice_patient_resample_{rep+1:02d}')
            sl=engine.fit_grid(cs,lim,f'mice_patient_resample_{rep+1:02d}',SEED+160000+rep*100,pool=pool)
            for k in KS:mi_resample.append({'rep':rep+1,'k':k,'n_subsample':len(sub),'n_people':len(picked),'ari':adjusted_rand_score(ml[k][ix],sl[k])})
            pd.DataFrame(mi_resample).to_csv(TABLES/'mice_joint_patient_resampling_detail.csv',index=False)
        comparisons=[]
        for kind,off in [('domain',0),('indicator',61000),('availability',62000)]:
            labels=engine.fit_grid(raw,limits,'sensitivity_'+kind,SEED+170000+off,kind=kind,pool=pool)
            comparisons.extend(engine.compare(base,labels,kind))
        complete=raw[raw[FEATS].notna().all(axis=1)].reset_index(drop=True)
        if len(complete)>max(KS):engine.fit_grid(complete,engine.limits(complete),'complete14',SEED+170000,pool=pool)
        pd.DataFrame(comparisons).to_csv(TABLES/'sensitivity_agreement.csv',index=False)
    for name in ['observed_patient_resampling','mice_joint_patient_resampling','mice_vs_observed']:
        d=pd.read_csv(TABLES/f'{name}_detail.csv');d.groupby('k').ari.agg(['count','mean','min','max']).reset_index().to_csv(TABLES/f'{name}_summary.csv',index=False)
    between=[]
    for k in KS:
        vals=[adjusted_rand_score(mi_labels[k][i],mi_labels[k][j]) for i in range(10) for j in range(i+1,10)]
        between.append({'k':k,'pair_count':45,'ari_mean':np.mean(vals),'ari_min':min(vals),'ari_max':max(vals)})
    pd.DataFrame(between).to_csv(TABLES/'between_imputation_summary.csv',index=False)
    metrics=[]
    for p in MODEL.glob('*/metrics.csv'):
        d=pd.read_csv(p);d.insert(0,'analysis',p.parent.name);metrics.append(d)
    pd.concat(metrics).to_csv(TABLES/'all_clustering_metrics.csv',index=False)
    # Freeze only after every planned chain and patient resample has completed.
    FROZEN.mkdir(exist_ok=True)
    transform={'version':'common_identified_v2_20261030','features':FEATS,'domains':engine.CFG['domains'],'eligibility':engine.CFG['eligibility'],
       'limits':limits,'candidate_k':KS,'distance':'equal-feature missing-aware Gower, clip train P1-P99, divide by number of jointly observed features',
       'tie_rule':'first cluster in ascending stored medoid order','all_medoid_distances_required':True,
       'limits_fit_scope':'eligible compatible MIMIC training rows only','external_imputation':False,'clinical_k_frozen':False,
       'scope':'New partially identified common-definition branch, not old-model strict validation'}
    engine.dump(FROZEN/'core14_primary_transform.json',transform)
    shutil.copy2(OUT/'common_protocol_v2.json',FROZEN/'common_protocol_v2.json')
    shutil.copy2(cp,FROZEN/'new_training_contract.json');raw.to_parquet(FROZEN/'training_reference.parquet',index=False)
    qa=[]
    for k in KS:
        med=pd.read_parquet(MODEL/'observed_equal'/f'k{k}_medoids.parquet')
        med.to_parquet(FROZEN/f'equal_feature_k{k}_medoids.parquet',index=False)
        dist=engine.eq_distance(raw,med,limits);assert np.array_equal(dist.argmin(axis=1),base[k])
        # Independent scalar implementation checks each feature, denominator, clip and tie rule.
        for idx in np.linspace(0,len(raw)-1,min(60,len(raw)),dtype=int):
            expected=[]
            for _,anchor in med.iterrows():
                parts=[]
                for f in FEATS:
                    a=raw.at[idx,f];b=anchor[f]
                    if pd.notna(a) and pd.notna(b):
                        lo,hi=limits[f];parts.append(abs(min(hi,max(lo,a))-min(hi,max(lo,b)))/(hi-lo))
                expected.append(sum(parts)/len(parts))
            assert np.allclose(dist[idx],expected,atol=2e-7)
        qa.append({'k':k,'full_train_assignment_reproduced':True,'independent_scalar_distances_checked':60*k})
        for rep in range(10):
            p=MODEL/f'mice_{rep+1:02d}'/f'k{k}_medoids.parquet';dest=FROZEN/f'mice_{rep+1:02d}';dest.mkdir(exist_ok=True);shutil.copy2(p,dest/p.name)
    engine.dump(FROZEN/'freeze_qa.json',{'status':'pass','checks':qa,'patient_overlap_between_splits':0,'original_bundle_untouched':True})
    tracked=json.loads((ROOT/'outputs/common_definition_20261029/original_artifacts_sha256.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in tracked.items())
    provenance=FROZEN/'provenance';provenance.mkdir(exist_ok=True)
    scripts=['extract_common_core_20261030.py','compact_common_chart_20261030.py','build_common_identified_20261030.py','common_features_20261030.py','verify_common_core_20261030.py','verify_compact_chart_20261030.py','fit_common_models_20261030.py','run_revised_phenotypes_20261027.py','start_common_definition_20261029.py','common_sofa_lab_bounds_20261030.py','run_common_pipeline_20261030.py','report_common_1to3_20261030.py']
    for name in scripts:shutil.copy2(ROOT/name,provenance/name)
    for name in ['26_domain_balanced_v7.py','14_cluster_training.py','39_missingness_measurement_sensitivity.py']:
        shutil.copy2(ROOT/'work/mimic_rebuilt_20261026/analysis'/name,provenance/name)
    for name in ['analysis_v2.json','phenotype_primary_v1.json']:
        shutil.copy2(ROOT/'work/mimic_rebuilt_20261026/config'/name,provenance/name)
    for name in ['independent_common_cohort_qa.json','remote_chart_aggregation_qa.json','cohort_summary.json','feature_summary.json','extraction_manifest.json','software_environment.json']:
        shutil.copy2(OUT/name,provenance/name)
    sql=provenance/'sql';sql.mkdir(exist_ok=True)
    for p in (WORK/'sql').glob('*.sql'):shutil.copy2(p,sql/p.name)
    source_paths=[DATA/f'{s}_common_{suffix}_local.parquet' for s in ['mimic','eicu'] for suffix in ['identified_cohort','features','event_bounds','order_audit','component_lower','feature_provenance']]
    engine.dump(provenance/'prepared_input_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in source_paths})
    manifest={str(p.relative_to(FROZEN)):sha(p) for p in FROZEN.rglob('*') if p.is_file() and p.name!='bundle_manifest.json'}
    engine.dump(FROZEN/'bundle_manifest.json',{'version':transform['version'],'sha256':manifest,'source_feature_sha256':sha(source),'original_files_checked':len(tracked)})
    dump('new_model_status.json',{'status':'complete_and_frozen','training_rows':len(raw),'training_people':train.person_id.nunique(),'MICE_draws':10,'patient_resamples':10,'MICE_patient_resamples':10,'k':KS,'clinical_k_frozen':False,'new_external_validation_complete':False,'frozen_directory':str(FROZEN)})
    print('NEW COMMON MODELS COMPLETE AND FROZEN',flush=True)

if __name__=='__main__':main()
