"""Quantify unresolved order and infusion-time mappings from original sources."""
from pathlib import Path
import json
import os
import hashlib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from start_common_definition_20261029 import ROOT, OUT, WORK, sha, save


def main():
    client = bigquery.Client(project=os.environ.get('GCP_PROJECT'))
    base = (WORK/'sql/mimic_orders.sql').read_text(encoding='utf-8')
    queries = {
        'mimic_order_lifecycle': f"""WITH candidate AS ({base})
SELECT c.*, p.ordertime, p.order_status, p.transaction_type,
 p.discontinued_by_poe_id, d.ordertime AS discontinue_order_time,
 d.transaction_type AS discontinue_transaction_type
FROM candidate c
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.poe` p
 ON c.poe_id=p.poe_id AND c.subject_id=p.subject_id
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.poe` d
 ON p.discontinued_by_poe_id=d.poe_id AND p.subject_id=d.subject_id""",
        'mimic_infusion_time_profile': """SELECT itemid,statusdescription,rateuom,
 COUNT(*) AS row_count,COUNTIF(rate IS NOT NULL) AS rows_with_rate,
 COUNTIF(rate>0) AS positive_rate_rows,
 COUNTIF(storetime<starttime) AS stored_before_start,
 COUNTIF(storetime BETWEEN starttime AND endtime) AS stored_during_interval,
 COUNTIF(storetime>endtime) AS stored_after_end,
 COUNTIF(starttime IS NULL OR endtime IS NULL OR storetime IS NULL) AS incomplete_times,
 APPROX_QUANTILES(DATETIME_DIFF(storetime,starttime,MINUTE),100)[OFFSET(50)] AS store_minus_start_p50_min,
 APPROX_QUANTILES(DATETIME_DIFF(storetime,endtime,MINUTE),100)[OFFSET(50)] AS store_minus_end_p50_min
FROM `physionet-data.mimiciv_3_1_icu.inputevents`
WHERE itemid IN (221289,221653,221662,221906,229617)
GROUP BY itemid,statusdescription,rateuom""",
        'mimic_gas_time_pair_coverage': """WITH bg AS (
 SELECT hadm_id,charttime,
 COUNTIF(itemid=50821 AND valuenum>0) AS po2_rows,
 COUNTIF(itemid=50816 AND ((valuenum>0.2 AND valuenum<=1) OR (valuenum>20 AND valuenum<=100))) AS fio2_rows,
 COUNT(DISTINCT IF(itemid=50821 AND valuenum>0,valuenum,NULL)) AS n_po2_values,
 COUNT(DISTINCT IF(itemid=50816,CASE WHEN valuenum>0.2 AND valuenum<=1 THEN valuenum
 WHEN valuenum>20 AND valuenum<=100 THEN valuenum/100 END,NULL)) AS n_fio2_values,
 COUNT(DISTINCT specimen_id) AS specimens,
 COUNTIF(itemid=52033 AND value='ART.') AS explicit_arterial_labels,
 COUNTIF(itemid=52033 AND value IS NOT NULL AND value!='ART.') AS other_specimen_labels
 FROM `physionet-data.mimiciv_3_1_hosp.labevents`
 WHERE itemid IN (50821,50816,52033) AND hadm_id IS NOT NULL AND charttime IS NOT NULL
 GROUP BY hadm_id,charttime)
SELECT COUNT(*) AS all_time_groups,COUNTIF(po2_rows>0) AS po2_groups,
 COUNTIF(po2_rows>0 AND fio2_rows>0) AS co_timestamp_groups,
 COUNTIF(n_po2_values=1 AND n_fio2_values=1) AS unique_value_pairs,
 COUNTIF(n_po2_values=1 AND n_fio2_values=1 AND explicit_arterial_labels>0 AND other_specimen_labels=0) AS unique_time_arterial_pairs,
 COUNTIF(n_po2_values=1 AND n_fio2_values=1 AND specimens>1) AS unique_values_multiple_specimens
FROM bg""",
    }
    mp = OUT/'source_semantics_query_manifest.json'
    manifest = json.loads(mp.read_text(encoding='utf-8')) if mp.exists() else {}
    total = 0
    for name,q in queries.items():
        (WORK/'sql'/f'{name}.sql').write_text(q,encoding='utf-8')
        digest=hashlib.sha256(q.encode()).hexdigest(); dest=WORK/'data'/f'{name}_local.parquet'
        old=manifest.get(name,{})
        if old.get('complete') and old.get('sql_sha256')==digest and dest.exists() and old.get('sha256')==sha(dest):
            continue
        dry=client.query(q,job_config=bigquery.QueryJobConfig(dry_run=True,use_query_cache=False),retry=None,timeout=30)
        total+=dry.total_bytes_processed
        assert total<=40_000_000_000
        manifest[name]={'estimated_bytes':dry.total_bytes_processed,'sql_sha256':digest,'complete':False}
        save(mp.name,manifest); print(name,'estimated bytes',dry.total_bytes_processed,flush=True)
    for name,q in queries.items():
        dest=WORK/'data'/f'{name}_local.parquet'
        if manifest[name].get('complete'):continue
        job=client.query(q,job_config=bigquery.QueryJobConfig(maximum_bytes_billed=40_000_000_000),retry=None,timeout=30)
        manifest[name]['job_id']=job.job_id;save(mp.name,manifest)
        df=job.result(timeout=900).to_dataframe(create_bqstorage_client=False);df.to_parquet(dest,index=False)
        manifest[name].update(complete=True,rows=len(df),actual_bytes=job.total_bytes_processed,sha256=sha(dest))
        save(mp.name,manifest); print(name,'completed',len(df),'rows',flush=True)
    lifecycle=pd.read_parquet(WORK/'data/mimic_order_lifecycle_local.parquet')
    original=pd.read_parquet(WORK/'data/mimic_orders_local.parquet')
    assert len(lifecycle)==len(original),'POE join multiplied prescription records'
    lc={'candidate_prescription_rows':len(lifecycle),
        'missing_order_time':int(lifecycle.ordertime.isna().sum()),
        'inactive_order_rows':int(lifecycle.order_status.eq('Inactive').sum()),
        'stop_before_start':int(lifecycle.stoptime.lt(lifecycle.starttime).sum()),
        'zero_duration':int(lifecycle.stoptime.eq(lifecycle.starttime).sum()),
        'stop_missing':int(lifecycle.stoptime.isna().sum()),
        'order_after_start':int(lifecycle.ordertime.gt(lifecycle.starttime).sum()),
        'discontinue_order_before_start':int(lifecycle.discontinue_order_time.lt(lifecycle.starttime).sum()),
        'discontinue_order_after_start':int(lifecycle.discontinue_order_time.gt(lifecycle.starttime).sum()),
        'caution':'Inactive indicates inactivation, not pre-start cancellation. Flags overlap and are not final exclusions.'}
    lifecycle.groupby(['route','order_status'],dropna=False).size().rename('n_rows').to_csv(OUT/'mimic_route_order_status_counts.csv',encoding='utf-8-sig')
    dose=pd.read_parquet(WORK/'data/mimic_infusion_time_profile_local.parquet')
    dose.to_csv(OUT/'mimic_infusion_time_profile.csv',index=False,encoding='utf-8-sig')
    valid=dose[~dose.statusdescription.eq('Rewritten')]
    columns=['row_count','stored_before_start','stored_during_interval','stored_after_end','incomplete_times']
    ds={c:int(valid[c].sum()) for c in columns}
    ds['stored_after_end_fraction']=ds['stored_after_end']/ds['row_count']
    ds['scope']='all five matched item IDs, all units, excluding Rewritten; not a dose-valid cohort'
    gas=pd.read_parquet(WORK/'data/mimic_gas_time_pair_coverage_local.parquet').iloc[0].to_dict()
    result={'order_lifecycle':lc,'infusion_documentation_timing':ds,'bloodgas_time_pair_audit':gas,
            'six_domain_common_sofa_complete':False,'original_model_strict_equivalence':False}
    save('source_semantics_results.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str),flush=True)


if __name__=='__main__':main()
