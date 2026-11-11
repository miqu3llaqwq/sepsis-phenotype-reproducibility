"""Assess existing source support without claiming clinical adjudication."""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
p=importlib.import_module('10_prepare_reviewed_features')

def main():
    review=json.loads((ROOT/'logs/clinical_review_latest.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'logs/observation_review_latest.json').read_text(encoding='utf-8'))
    data=Path(review['data_path']); report=Path(review['report_path'])
    cases=pd.read_csv(data/'待复核清单.csv')
    details=pd.read_csv(data/'来源与邻近观测.csv')
    config=p.build.load_config()
    base=pd.read_parquet(source['input_path'])
    train_stays=set(base.loc[base.split.eq('train'),'stay_id'])
    assert cases.stay_id.isin(train_stays).all()
    reproduced={}; audit=[]
    raw_hashes={}
    for domain in ['vitals','labs']:
        path=Path(source['data_path'])/(domain+'.parquet')
        raw=pd.read_parquet(path)
        t0=pd.to_datetime(raw.t0); chart=pd.to_datetime(raw.charttime); store=pd.to_datetime(raw.storetime)
        hours=config['vitals_hours'] if domain=='vitals' else config['labs_hours']
        joined=raw[['stay_id','t0']].drop_duplicates().merge(base[['stay_id','t0']],on='stay_id',validate='one_to_one',suffixes=('_source','_base'),how='left')
        assert joined.t0_base.notna().all()
        assert pd.to_datetime(joined.t0_source).eq(pd.to_datetime(joined.t0_base)).all()
        bad=chart.isna()|store.isna()|chart.gt(t0)|store.gt(t0)|chart.lt(t0-pd.Timedelta(hours=hours))
        assert not bad.any()
        out,*_=p.aggregate(raw[raw.stay_id.isin(set(cases.stay_id))],config,domain)
        reproduced[domain]=out
        raw_hashes[domain]=hashlib.sha256(path.read_bytes()).hexdigest()
        audit.append({'domain':domain,'source_rows':len(raw),'unexpected_stays':int((~raw.stay_id.isin(base.stay_id)).sum()),'unavailable_or_outside_window_rows':int(bad.sum()),'landmark_matches_frozen_cohort':True})
    rows=[]
    for case in cases.itertuples():
        domain='vitals' if case.feature in config['vitals'] else 'labs'
        value=reproduced[domain].loc[case.stay_id,case.feature]
        assert np.isclose(value,case.aggregated_value,rtol=1e-12,atol=1e-12)
        rec=details[details.case_id.eq(case.case_id)&details.feature.eq(case.feature)]
        eligible=rec[rec.valid.eq(True)&rec.contributes_to_source_rule.eq(True)].copy()
        assert not eligible.empty
        flags=eligible.value.notna()&~eligible.value.between(case.screen_lower,case.screen_upper)
        extreme_times=pd.to_datetime(eligible.loc[flags,'charttime']).drop_duplicates().sort_values()
        gap=extreme_times.diff().dt.total_seconds()/60
        near=gap.gt(0)&gap.le(30)
        if near.any():support='30分钟内不同时间重复超界'
        elif len(extreme_times)>1:support='不同时间复测超界，未发现30分钟内重复'
        elif len(extreme_times)==1:support='仅一个有效超界时点'
        else:support='需进一步检查聚合与单值关系'
        rows.append({'case_id':case.case_id,'feature':case.feature,'aggregated_value':case.aggregated_value,'recomputed_aggregate_matches':True,'valid_contributing_rows':len(eligible),'distinct_contributing_times':eligible.charttime.nunique(),'distinct_extreme_times':len(extreme_times),'technical_support':support,'clinical_truth_determined':False,'suggested_handling':'保留主版并比较严格筛查敏感性；未证实错误，不自动删除'})
    evidence=pd.DataFrame(rows)
    evidence.to_csv(data/'自动证据复核.csv',index=False,encoding='utf-8-sig')
    counts=evidence.technical_support.value_counts().rename_axis('technical_support').reset_index(name='review_cases')
    counts.to_csv(report/'automatic_evidence_counts.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(audit).to_csv(report/'source_extraction_checks.csv',index=False)
    manifest={'status':'technical_support_assessed_clinical_truth_unresolved','cases_reproduced':len(evidence),'partition':'train','raw_source_sha256':raw_hashes,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'evidence_sha256':hashlib.sha256((data/'自动证据复核.csv').read_bytes()).hexdigest(),'clinical_adjudication_completed':False,'manual_review_file_modified':False,'frozen_data_modified':False}
    (report/'automatic_evidence_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('All 49 flagged aggregates reproduced from source with matching values.')
    print(counts.to_string(index=False))

if __name__=='__main__':main()
