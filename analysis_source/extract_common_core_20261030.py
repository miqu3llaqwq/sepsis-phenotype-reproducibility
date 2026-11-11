"""New common branch: source extraction only, no old cohort restriction."""
from pathlib import Path
import json, hashlib, os
from google.cloud import bigquery
from start_common_definition_20261029 import ROOT, WORK, sha

OUT=ROOT/'outputs/common_core_20261030'

def run_query(name, sql, parameters=None, parameter_digest='', download=True):
    OUT.mkdir(exist_ok=True); (WORK/'sql'/f'{name}.sql').write_text(sql,encoding='utf-8')
    dest=WORK/'data'/f'{name}_local.parquet'; mp=OUT/'extraction_manifest.json'
    m=json.loads(mp.read_text()) if mp.exists() else {}
    digest=hashlib.sha256((sql+parameter_digest).encode()).hexdigest()
    prior=m.get(name,{})
    if prior.get('complete') and prior.get('sql_sha256')==digest and dest.exists() and sha(dest)==prior.get('sha256'):
        print(name,'verified reuse',flush=True); return dest
    c=bigquery.Client(project=os.environ.get('GCP_PROJECT'))
    if not download and prior.get('sql_sha256')==digest and prior.get('job_id'):
        previous=c.get_job(prior['job_id'])
        if previous.state=='DONE' and previous.error_result is None:
            try:
                c.get_table(previous.destination)
                m[name].update(remote_complete=True,local_full_download_completed=False,status='server_result_complete_local_uses_verified_compaction')
                mp.write_text(json.dumps(m,indent=2),encoding='utf-8');print(name,'verified remote result reuse',flush=True);return None
            except Exception:
                pass  # Expired temporary result: regenerate from versioned source SQL.
    dry=c.query(sql,job_config=bigquery.QueryJobConfig(dry_run=True,use_query_cache=False,query_parameters=parameters or []),retry=None,timeout=30)
    # Source chartevents scans can exceed the earlier small lab-audit cap.
    # This cap equals the existing authorized source-analysis configuration.
    assert dry.total_bytes_processed<=300_000_000_000
    print(name,'bytes',dry.total_bytes_processed,flush=True)
    j=c.query(sql,job_config=bigquery.QueryJobConfig(maximum_bytes_billed=300_000_000_000,query_parameters=parameters or []),retry=None,timeout=30)
    m[name]={'complete':False,'sql_sha256':digest,'job_id':j.job_id,'estimated_bytes':dry.total_bytes_processed}
    mp.write_text(json.dumps(m,indent=2),encoding='utf-8')
    result=j.result(timeout=1800)
    if not download:
        m[name].update(remote_complete=True,local_full_download_completed=False,status='server_result_complete_local_uses_verified_compaction',actual_bytes=j.total_bytes_processed)
        mp.write_text(json.dumps(m,indent=2),encoding='utf-8');return None
    x=result.to_dataframe(create_bqstorage_client=False)
    x.to_parquet(dest,index=False)
    m[name].update(complete=True,rows=len(x),sha256=sha(dest),actual_bytes=j.total_bytes_processed)
    mp.write_text(json.dumps(m,indent=2),encoding='utf-8');print(name,'rows',len(x),flush=True)
    return dest

def main():
    base=(WORK/'sql/mimic_orders.sql').read_text(encoding='utf-8')
    cte=f'WITH ab AS ({base}), s AS (SELECT DISTINCT subject_id,hadm_id,stay_id,intime,outtime,admittime FROM ab)'
    run_query('common_stay_metadata',cte+'''
SELECT s.*,p.anchor_age+EXTRACT(YEAR FROM s.admittime)-p.anchor_year AS age,
 a.dischtime,a.hospital_expire_flag,p.gender
FROM s JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING(subject_id)
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING(hadm_id)''')
    run_query('common_sofa_chart',cte+'''
SELECT s.stay_id,DATETIME_DIFF(e.charttime,s.intime,SECOND)/60.0 AS offset_min,
 e.itemid,e.valueuom,COUNT(*) AS source_rows,COUNTIF(e.valuenum IS NULL) AS null_rows,
 COUNT(DISTINCT e.valuenum) AS distinct_values,MIN(e.valuenum) AS value,
 COUNTIF(e.itemid=223900 AND (e.value='No Response-ETT' OR e.valuenum=0)) AS unscorable_rows
FROM s JOIN `physionet-data.mimiciv_3_1_icu.chartevents` e USING(stay_id)
WHERE e.itemid IN (220052,220181,220739,223900,223901)
AND e.charttime BETWEEN GREATEST(s.admittime,DATETIME_SUB(s.intime,INTERVAL 48 HOUR)) AND s.outtime
GROUP BY s.stay_id,e.charttime,s.intime,e.itemid,e.valueuom''',download=False)
    run_query('common_sofa_gas',cte+'''
SELECT s.stay_id,DATETIME_DIFF(e.charttime,s.intime,SECOND)/60.0 AS offset_min,
 e.itemid,e.valuenum,e.value,e.valueuom,e.specimen_id,e.labevent_id
FROM s JOIN `physionet-data.mimiciv_3_1_hosp.labevents` e ON s.hadm_id=e.hadm_id AND s.subject_id=e.subject_id
WHERE e.itemid IN (50821,50816,52033)
AND e.charttime BETWEEN GREATEST(s.admittime,DATETIME_SUB(s.intime,INTERVAL 48 HOUR)) AND s.outtime''')

if __name__=='__main__':main()
