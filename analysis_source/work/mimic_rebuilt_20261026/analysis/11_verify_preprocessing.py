"""Verify immutable partitions, train-derived imputation and saved artifacts."""
from pathlib import Path
import hashlib
import importlib
import json
import subprocess
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
p=importlib.import_module('10_prepare_reviewed_features')

def main():
    loc=json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'logs/observation_review_latest.json').read_text(encoding='utf-8'))
    dest=Path(loc['data_path']); reports=Path(loc['report_path'])
    base=Path(source['input_path']); original=pd.read_parquet(base)
    pre=pd.read_parquet(dest/'reviewed_unimputed.parquet')
    post=pd.read_parquet(dest/'reviewed_imputed.parquet')
    strict=pd.read_parquet(dest/'strict_screen_sensitivity_unimputed.parquet')
    strict_post=pd.read_parquet(dest/'strict_screen_sensitivity_imputed.parquet')
    features=p.build.load_config()['primary_features']
    params=json.loads((dest/'imputer.json').read_text(encoding='utf-8'))
    assert params['primary']==p.fit_imputer(pre,features)
    assert params['strict_sensitivity']==p.fit_imputer(strict,features)
    for subset in [pre,post,strict,strict_post]:
        assert subset[['subject_id','hadm_id','stay_id','split']].equals(original[['subject_id','hadm_id','stay_id','split']])
    for name in ['train','validation','test']:
        ids=pd.read_parquet(base.parent/(name+'_ids.parquet'))
        assert set(ids.subject_id)==set(post.loc[post.split.eq(name),'subject_id'])
    for before,after,par in [(pre,post,params['primary']),(strict,strict_post,params['strict_sensitivity'])]:
        assert not after[features].isna().any().any()
        for col in features:
            assert after[col].equals(before[col].fillna(par[col]))
            assert after[col+'_missing'].equals(before[col].isna())
        both=before.sbp.notna()&before['map'].notna()
        assert before.loc[both,'map'].le(before.loc[both,'sbp']).all()
    frozen=json.loads((ROOT/'config/frozen_split_v2.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(base.read_bytes()).hexdigest()==frozen['dataset_sha256']
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','analysis','-p','test_preprocessing_review.py'],cwd=ROOT,capture_output=True,text=True)
    (reports/'synthetic_tests.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    assert test.returncode==0
    result={'status':'passed','synthetic_tests':6,'frozen_cohort_rows':len(post),'main_training_phenotype_eligible':int((pre.split.eq('train')&pre.phenotype_eligible).sum()),'no_physiology_rows_all_partitions':int((~pre.phenotype_eligible).sum()),'frozen_partition_files_match':True,'imputer_refit_matches_saved_parameters':True,'missing_flags_match_unimputed_data':True,'paired_bp_aggregate_order_valid':True,'clinical_authenticity_certified':False,'model_evaluation_performed':False}
    (reports/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
