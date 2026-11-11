"""Accept completed analysis and export verified frozen diagnostic/MI ensemble artifacts."""
from pathlib import Path
import hashlib,json,shutil,importlib.util
import numpy as np
import pandas as pd
from run_revised_phenotypes_20261027 import B,OUT,TABLES,REPORT,ROOT,CFG,FEATS,KS,IDS,sha,dump,limits

def main():
    status=json.loads((REPORT/'run_status.json').read_text())
    assert status['status'] in ['computations_complete_freeze_pending','complete_and_frozen']
    raw=pd.read_parquet(B/'data/core14_preparation/eligible_train_unimputed.parquet').sort_values('stay_id').reset_index(drop=True)
    lim=limits(raw)
    contract=json.loads((REPORT/'execution_contract.json').read_text())
    assert contract['source_sha256']==sha(B/'data/core14_preparation/eligible_train_unimputed.parquet')
    assert contract['cohort_sha256']==sha(B/'data/analysis_dataset.parquet')
    assert contract['config_sha256']==sha(B/'config/phenotype_primary_v1.json')
    groups=['observed_equal','complete14']+[f'{prefix}_{i:02d}' for prefix in ['resample','mice','mice_resample'] for i in range(1,11)]+['sensitivity_domain','sensitivity_indicator','sensitivity_availability']
    assert len(groups)==35
    restarts=0
    for group in groups:
        d=OUT/group;m=json.loads((d/'grid_manifest.json').read_text());assert m['status']=='complete'
        for name,h in m['sha256'].items():assert sha(d/name)==h,(group,name)
        for k in KS:
            r=pd.read_csv(d/f'k{k}_restarts.csv');assert len(r)==5 and r.iterations.lt(200).all()
            labels=pd.read_parquet(d/f'k{k}_labels.parquet');assert len(labels)==m['n'] and labels.cluster.nunique()==k
            restarts+=len(r)
    assert restarts==875
    for prefix in ['mice','mice_resample']:
        for i in range(1,11):
            m=json.loads((OUT/f'{prefix}_{i:02d}/imputation.json').read_text())
            assert m['observed_values_unchanged'] and m['all_imputations_are_observed_training_donors']
    dest=B/'frozen/20261027';dest.mkdir(parents=True,exist_ok=True)
    config={'version':'reconstructed_mimic31_core14_20261027','features':FEATS,'limits':lim,'domains':CFG['domains'],'eligibility':CFG['eligibility'],
      'candidate_k':KS,'n_training_eligible':len(raw),'source_cohort_n':31422,'limits_fit_scope':'eligible training set only',
      'distance':'equal-feature Gower; clip each observed value to training P1-P99 before scaling; average over shared observed variables',
      'missingness':'no imputation in this external diagnostic; undefined distances remain undefined',
      'assignment':'require finite distances to every medoid; argmin in stored cluster order; ties choose first',
      'role':'frozen external transportability diagnostic matching v9; NOT the MICE-PMM primary ensemble',
      'clinical_k_frozen':False,'primary_MICE_ensemble':'primary_mice_ensemble/; all 10 draws retained; no selected draw',
      't0':'max(sofa_time, antibiotic_time, culture_time, icu_intime)','feature_windows_hours':{'vitals':6,'labs':24},
      'unit_and_time_semantics':'upstream audited extraction required; observation available by t0',
      'training_source_sha256':sha(B/'data/core14_preparation/eligible_train_unimputed.parquet')}
    extraction=json.loads((B/'config/analysis_v2.json').read_text())
    config['units']={f:(extraction['vitals'].get(f) or extraction['labs'][f])['unit'] for f in FEATS}
    dump(dest/'core14_primary_transform.json',config)
    for k in KS:shutil.copy2(OUT/f'observed_equal/k{k}_medoids.parquet',dest/f'equal_feature_k{k}_medoids.parquet')
    shutil.copy2(ROOT/'frozen_external_assignment_20261027.py',dest/'assign_frozen.py')
    shutil.copy2(TABLES/'training_missingness.csv',dest/'training_missingness.csv')
    pd.DataFrame([{'feature':f,'p01':lim[f][0],'p99':lim[f][1]} for f in FEATS]).to_csv(dest/'training_P1_P99.csv',index=False)
    raw[IDS+FEATS].to_parquet(dest/'training_reference.parquet',index=False)
    ensemble=dest/'primary_mice_ensemble';ensemble.mkdir(exist_ok=True)
    for i in range(1,11):
        sub=ensemble/f'm{i:02d}';sub.mkdir(exist_ok=True)
        for k in KS:
            med=pd.read_parquet(OUT/f'mice_{i:02d}/k{k}_medoids.parquet')
            med.to_parquet(sub/f'k{k}_medoids.parquet',index=False)
            observed=raw.set_index('stay_id').loc[med.stay_id,FEATS].notna().reset_index()
            observed['cluster']=med.cluster.to_numpy();observed.to_parquet(sub/f'k{k}_medoid_observed_masks.parquet',index=False)
        shutil.copy2(OUT/f'mice_{i:02d}/imputation.json',sub/'imputation.json')
    dump(ensemble/'ensemble_contract.json',{'role':'primary training-only MICE-PMM analysis','draws':10,'k':KS,'features':FEATS,'limits':lim,'selected_imputation':None,'pooling':'report label/co-clustering uncertainty; do not concatenate patient rows or average medoid values','validation_test_used':False,'clinical_k_frozen':False,'ready_for_external_MICE_imputation':False,'reason':'No external imputation model has been defined or fitted; external diagnostic uses observed-value bundle.'})
    spec=importlib.util.spec_from_file_location('frozen_assignment',dest/'assign_frozen.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    checks=[]
    for k in KS:
        med=pd.read_parquet(dest/f'equal_feature_k{k}_medoids.parquet')
        labels,computable,d=runner.assign(raw,med,config)
        expected=pd.read_parquet(OUT/f'observed_equal/k{k}_labels.parquet')
        assert raw[IDS].equals(expected[IDS])
        assert computable.all() and np.array_equal(labels,expected.cluster.to_numpy())
        assert np.isfinite(d).all() and (d>=0).all() and (d<=1.000001).all()
        originals=raw.set_index('stay_id').loc[med.stay_id,FEATS].reset_index(drop=True)
        assert originals.equals(med[FEATS].reset_index(drop=True))
        checks.append({'k':k,'n':len(raw),'all_computable':True,'label_roundtrip_exact':True,'medoids_are_unmodified_observed_training_rows':True})
    mi_checks=[]
    for i in range(1,11):
        completed=pd.read_parquet(OUT/f'mice_{i:02d}/imputed.parquet')
        for k in KS:
            med=pd.read_parquet(ensemble/f'm{i:02d}/k{k}_medoids.parquet')
            lab,ok,_=runner.assign(completed,med,config)
            expected=pd.read_parquet(OUT/f'mice_{i:02d}/k{k}_labels.parquet')
            assert ok.all() and np.array_equal(lab,expected.cluster.to_numpy())
            mi_checks.append({'imputation':i,'k':k,'exact_training_label_roundtrip':True})
    # Boundary cases: clipping, no common values, incomplete distance set and deterministic ties.
    a=pd.DataFrame([{f:lim[f][0]-100*(lim[f][1]-lim[f][0]) for f in FEATS}])
    b=pd.DataFrame([{f:lim[f][1]+100*(lim[f][1]-lim[f][0]) for f in FEATS}]);b['cluster']=1
    assert np.isclose(runner.distance_to_medoids(a,b,config)[0,0],1)
    b.loc[:,FEATS]=np.nan
    label,computable,d=runner.assign(a,b,config);assert not computable.any() and np.isnan(label).all() and np.isnan(d).all()
    b=pd.concat([a,a],ignore_index=True);b['cluster']=[1,2]
    label,_,_=runner.assign(a,b,config);assert label[0]==1
    b.loc[1,FEATS]=np.nan
    label,ok,_=runner.assign(a,b,config);assert not ok[0] and np.isnan(label[0])
    original=json.loads((ROOT/'outputs/source_lineage_20261026/frozen_input_recheck.json').read_text())
    # The native shell can read Desktop artifacts which the Python process does
    # not see in its filesystem view; consume today's explicit native hash check.
    native=json.loads((REPORT/'original_artifact_hashes_20261027.json').read_text(encoding='utf-8-sig'))
    expected={Path(r['file']).name:r for r in original}
    assert len(native)==len(expected) and all(r['sha256']==expected[Path(r['file']).name]['sha256'] for r in native)
    oldcheck=[{'file':expected[Path(r['file']).name]['file'],'sha256':r['sha256'],'unchanged':r['unchanged']} for r in native]
    assert all(r['unchanged'] for r in oldcheck)
    qa={'status':'passed','grids':len(groups),'candidate_fits':len(groups)*len(KS),'PAM_restarts':restarts,'MICE_full_draws':10,'MICE_subsample_refits':10,'observed_resamples':10,'training_roundtrip':checks,'primary_MICE_roundtrip':mi_checks,'boundary_checks':['P1-P99_clipping','undefined_pairs_preserved','all_medoids_required','first_label_tie_break'],'old_frozen_artifacts':oldcheck,'external_validation_run':False}
    dump(REPORT/'freeze_qa.json',qa)
    provenance=dest/'provenance';provenance.mkdir(exist_ok=True)
    for p in [ROOT/'run_revised_phenotypes_20261027.py',ROOT/'accept_rebuilt_cohort_20261027.py',ROOT/'freeze_revised_inputs_20261027.py',REPORT/'execution_contract.json',REPORT/'source_code_hashes.json',REPORT/'freeze_qa.json',B/'reports/source_acceptance_20261027/acceptance.json',B/'config/phenotype_primary_v1.json',B/'config/analysis_v2.json',B/'config/preprocessing_review_v3.json',B/'reports/original_source_snapshot.zip']:
        shutil.copy2(p,provenance/p.name)
    dump(dest/'bundle_manifest.json',{'status':'frozen_diagnostic_inputs_verified','date':'2026-10-27','clinical_k_frozen':False,'external_validation_complete':False,'qa_report_sha256':sha(REPORT/'freeze_qa.json'),
      'sha256':{str(p.relative_to(dest)).replace('\\','/'):sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='bundle_manifest.json'}})
    progress={'date':'2026-10-27','plan':'latest user requested first three: cohort acceptance, main rerun, frozen inputs','steps':[{'step':1,'status':'complete','cohort_n':31422,'evidence':str(B/'reports/source_acceptance_20261027/acceptance.json')},{'step':2,'status':'complete','training_eligible_n':17810,'MICE_draws':10,'observed_resamples':10,'MICE_joint_resamples':10,'candidate_k':KS,'evidence':str(REPORT/'freeze_qa.json')},{'step':3,'status':'complete_diagnostic_bundle_and_primary_ensemble_archived','bundle':str(dest),'clinical_k_frozen':False}],'eicu_external_validation_complete':False}
    dump(ROOT/'outputs/first_three_steps_20261027.json',progress)
    active_path=ROOT/'outputs/eicu_external_validation_manifest_v9_20261019.json'
    active=json.loads(active_path.read_text(encoding='utf-8'))
    archive=REPORT/'external_manifest_before_new_freeze.json'
    if not archive.exists():shutil.copy2(active_path,archive)
    active.update(mimic_frozen_dir=str(dest),mimic_medoids=[{'k':k,'path':str(dest/f'equal_feature_k{k}_medoids.parquet'),'sha256':sha(dest/f'equal_feature_k{k}_medoids.parquet'),'exists':True} for k in KS],
      mimic_limits_path=str(dest/'core14_primary_transform.json'),mimic_limits_sha256=sha(dest/'core14_primary_transform.json'),mimic_limits_keys=FEATS,
      source_cohort_reproducible=True,revised_source_cohort_reproducible=True,revised_mimic_medoids_ready=True,
      primary_source_decision_file=str(B/'reports/source_acceptance_20261027/acceptance.json'),
      latest_source_evidence=str(B/'reports/source_acceptance_20261027'),revised_frozen_bundle_manifest=str(dest/'bundle_manifest.json'),
      frozen_assignment_runner=str(dest/'assign_frozen.py'),primary_MICE_ensemble=str(ensemble),
      frozen_input_role='observed-value external diagnostic; primary MICE ensemble archived separately; no clinical k selected',
      external_validation_complete=False,blocking_reason='Revised MIMIC inputs accepted and frozen; eICU infection/SOFA/t0 semantic mappings and formal feature table still required.',
      previous_manifest_archive=str(archive))
    dump(active_path,active)
    status.update(status='complete_and_frozen',bundle=str(dest),freeze_qa_sha256=sha(REPORT/'freeze_qa.json'))
    dump(REPORT/'run_status.json',status)
    print(json.dumps({'status':'FIRST_THREE_COMPLETE','frozen_bundle':str(dest),'grids':35,'PAM_restarts':875,'training_roundtrip_n':len(raw),'external_validation_complete':False},indent=2),flush=True)

if __name__=='__main__':main()
