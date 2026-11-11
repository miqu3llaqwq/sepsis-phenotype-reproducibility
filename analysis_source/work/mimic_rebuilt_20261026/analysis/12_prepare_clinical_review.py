"""Prepare TRAIN-only, outcome-blinded review material in controlled data/."""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
p=importlib.import_module('10_prepare_reviewed_features')

def main():
    loc=json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))
    observations=json.loads((ROOT/'logs/observation_review_latest.json').read_text(encoding='utf-8'))
    config=p.build.load_config()
    rules=json.loads((ROOT/'config/preprocessing_review_v3.json').read_text(encoding='utf-8'))
    base=Path(loc['data_path'])/'reviewed_unimputed.parquet'
    df=pd.read_parquet(base)
    train=df[df.split.eq('train')].copy()
    cases=[]
    for feature,(lo,hi) in rules['strict_sensitivity_ranges'].items():
        selected=train[train[feature].notna()&~train[feature].between(lo,hi)]
        for row in selected.sort_values('stay_id').itertuples():
            cases.append({'subject_id':row.subject_id,'stay_id':row.stay_id,'hadm_id':row.hadm_id,'t0':row.t0,'feature':feature,'aggregated_value':getattr(row,feature),'screen_lower':lo,'screen_upper':hi})
    cases=pd.DataFrame(cases)
    if cases.empty:raise ValueError('No flagged training cases to review')
    cases.insert(0,'case_id',[f'R{i:03d}' for i in range(1,len(cases)+1)])
    assert cases.stay_id.isin(train.stay_id).all()
    dest=Path(loc['data_path'])/'clinical_review';dest.mkdir(exist_ok=True)
    reports=ROOT/'outputs/reports/clinical_review';reports.mkdir(exist_ok=True)
    raw_by_domain={}; source_hashes={}
    dictionary=pd.read_csv(Path(observations['report_path'])/'item_dictionary.csv')
    labels=dictionary.set_index('itemid').label
    for domain in ['vitals','labs']:
        path=Path(observations['data_path'])/(domain+'.parquet')
        raw=pd.read_parquet(path)
        raw=raw[raw.stay_id.isin(set(cases.stay_id))].copy()
        _,screened,_,_,pairs=p.aggregate(raw,config,domain)
        screened['contributes_to_source_rule']=screened.valid.copy()
        if domain=='vitals':
            valid=pairs[pairs.valid_pair]
            arterial=set(valid.loc[valid.source.eq('arterial'),'stay_id'])
            chosen=valid[valid.source.eq('arterial')|~valid.stay_id.isin(arterial)]
            keys=set(zip(chosen.stay_id,chosen.charttime,chosen.source))
            bp=screened.feature.isin(['sbp','map','dbp'])
            source=screened.itemid.map(lambda i:'noninvasive' if i in [220179,220180,220181] else 'arterial')
            screened.loc[bp,'contributes_to_source_rule']=screened.loc[bp,'valid']&pd.Series([(s,t,x) in keys for s,t,x in zip(screened.stay_id,screened.charttime,source)],index=screened.index)[bp]
            preferred=set(screened.loc[screened.itemid.eq(220210)&screened.valid,'stay_id'])
            screened.loc[screened.itemid.eq(224690)&screened.stay_id.isin(preferred),'contributes_to_source_rule']=False
        raw_by_domain[domain]=screened
        source_hashes[domain]=hashlib.sha256(path.read_bytes()).hexdigest()
    details=[]; references=[]
    for case in cases.itertuples():
        domain='vitals' if case.feature in config['vitals'] else 'labs'
        raw=raw_by_domain[domain]
        related=['sbp','map','dbp'] if case.feature in ['sbp','map'] else [case.feature]
        records=raw[raw.stay_id.eq(case.stay_id)&raw.feature.isin(related)].copy()
        contributing=records[records.feature.eq(case.feature)&records.contributes_to_source_rule]
        if contributing.empty:raise ValueError('No contributing record for aggregated flagged value')
        reference=contributing.loc[(contributing.value-case.aggregated_value).abs().idxmin(),'charttime']
        references.append(reference)
        records.insert(0,'case_id',case.case_id)
        records['minutes_from_reference']=(pd.to_datetime(records.charttime)-reference).dt.total_seconds()/60
        records['within_30min_reference']=records.minutes_from_reference.abs().le(30)
        records['source_label']=records.itemid.map(labels)
        records['outside_strict_screen']=(records.value.notna()&~records.value.between(case.screen_lower,case.screen_upper)).astype('boolean')
        # Related DBP has no strict DBP threshold in this review configuration.
        records.loc[~records.feature.eq(case.feature),'outside_strict_screen']=pd.NA
        details.append(records[['case_id','feature','source_label','charttime','storetime','itemid','specimen_id','valuenum','valueuom','value','unit','valid','contributes_to_source_rule','minutes_from_reference','within_30min_reference','outside_strict_screen']])
    cases['reference_charttime']=references
    for column in ['review_classification','clinical_evidence','suggested_action','reviewer','review_date']:
        cases[column]=''
    sheet=dest/'待复核清单.csv'
    if sheet.exists():
        previous=pd.read_csv(sheet,keep_default_na=False)
        columns=['review_classification','clinical_evidence','suggested_action','reviewer','review_date']
        if previous[columns].astype(str).apply(lambda col:col.str.strip().ne('')).any().any():
            raise ValueError('Review decisions already entered; refusing to overwrite')
    cases.to_csv(sheet,index=False,encoding='utf-8-sig')
    pd.concat(details,ignore_index=True).to_csv(dest/'来源与邻近观测.csv',index=False,encoding='utf-8-sig')
    counts=cases.groupby('feature').size().rename('review_cases').reset_index()
    counts.to_csv(reports/'training_flagged_counts.csv',index=False)
    manifest={'status':'review_material_prepared_not_adjudicated','partition':'train','review_case_count':len(cases),'distinct_training_patients':int(cases.stay_id.nunique()),'selection':'all training aggregated nonmissing values outside registered strict sensitivity bounds','outcomes_in_review_files':False,'frozen_data_modified':False,'input_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'raw_source_sha256':source_hashes,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'output_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.glob('*.csv')}}
    (reports/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (ROOT/'logs/clinical_review_latest.json').write_text(json.dumps({'data_path':str(dest),'report_path':str(reports),'status':'awaiting_clinical_review'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Prepared {len(cases)} review cases in {cases.stay_id.nunique()} training patients. No patient rows printed.')

if __name__=='__main__':main()
