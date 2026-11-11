"""Download available source observations for frozen cohort; review TRAIN only.

Patient data never goes to outputs. Existing dataset and partitions are immutable.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
build=importlib.import_module('08_rebuild')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local',action='store_true')
    args=parser.parse_args()
    c=build.load_config()
    loc=json.loads((ROOT/'logs/v2_latest.json').read_text(encoding='utf-8'))
    base=Path(loc['data_path'])/'analysis_dataset.parquet'
    frozen=json.loads((ROOT/'config/frozen_split_v2.json').read_text(encoding='utf-8'))
    if hashlib.sha256(base.read_bytes()).hexdigest()!=frozen['dataset_sha256']:
        raise ValueError('Frozen input hash mismatch')
    dest=Path(loc['data_path'])/'observation_review'
    dest.mkdir(exist_ok=True)
    queries={}
    core=[x for s in c['vitals'].values() for x in s['itemids']]
    vital_ids=sorted(set(core+[220051,220180,225309,225310,225312,224690]))
    lab_ids=sorted({x for s in c['labs'].values() for x in s['itemids']})
    for name,table,ids,join,hours in [
        ('vitals',f"{c['icu']}.chartevents",vital_ids,'e.stay_id=c.stay_id',c['vitals_hours']),
        ('labs',f"{c['hosp']}.labevents",lab_ids,'e.hadm_id=c.hadm_id AND e.subject_id=c.subject_id',c['labs_hours'])]:
        specimen='e.specimen_id' if name=='labs' else 'CAST(NULL AS INT64) AS specimen_id'
        queries[name]=f"""WITH {build.cohort_cte(c)}
 SELECT c.subject_id,c.stay_id,c.hadm_id,c.t0,e.charttime,e.storetime,
 e.itemid,e.valuenum,e.valueuom,{specimen}
 FROM `{table}` e JOIN cohort c ON {join}
 WHERE e.itemid IN ({','.join(map(str,ids))})
 AND e.valuenum IS NOT NULL
 AND e.charttime BETWEEN DATETIME_SUB(c.t0,INTERVAL {hours} HOUR) AND c.t0
 AND e.storetime IS NOT NULL AND e.storetime<=c.t0
 ORDER BY c.stay_id,e.charttime,e.itemid
"""
        (ROOT/'sql'/f'v2_review_{name}.sql').write_text(queries[name],encoding='utf-8')
    manifest={'date':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'input_sha256':frozen['dataset_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sql_sha256':{n:hashlib.sha256(q.encode()).hexdigest() for n,q in queries.items()},'jobs':{},'status':'started','selection_partition':'train_only'}
    report=ROOT/'outputs/reports/observation_review_20261009'
    report.mkdir(exist_ok=True)
    try:
        if not args.local:
            from google.cloud import bigquery
            client=bigquery.Client(project=os.environ.get('GCP_PROJECT'))
            for name,sql in queries.items():
                dry=client.query(sql,job_config=bigquery.QueryJobConfig(dry_run=True,use_query_cache=False),retry=None,timeout=30)
                if dry.total_bytes_processed>c['maximum_bytes_billed']:
                    raise ValueError('Configured billing limit exceeded')
                print('Downloading source observations: '+name,flush=True)
                job=client.query(sql,job_config=bigquery.QueryJobConfig(maximum_bytes_billed=c['maximum_bytes_billed']),retry=None,timeout=30)
                df=job.result().to_dataframe(create_bqstorage_client=False)
                df.to_parquet(dest/f'{name}.parquet',index=False)
                manifest['jobs'][name]={'job_id':job.job_id,'bytes_processed':job.total_bytes_processed,'rows':len(df)}
            item_sql=f"SELECT 'vitals' AS domain,itemid,label FROM `{c['icu']}.d_items` WHERE itemid IN ({','.join(map(str,vital_ids))}) UNION ALL SELECT 'labs',itemid,label FROM `{c['hosp']}.d_labitems` WHERE itemid IN ({','.join(map(str,lab_ids))})"
            dictionary=client.query(item_sql).result().to_dataframe(create_bqstorage_client=False)
            dictionary.to_csv(report/'item_dictionary.csv',index=False)
        manifest.update(status='source_observations_downloaded',data_path=str(dest))
        (ROOT/'logs/observation_review_latest.json').write_text(json.dumps({'data_path':str(dest),'report_path':str(report),'input_path':str(base)},indent=2,ensure_ascii=False),encoding='utf-8')
        print('Source observations saved in controlled data directory; no patient rows printed.')
    except Exception as exc:
        manifest.update(status='blocked',error_type=type(exc).__name__)
        if isinstance(exc,ValueError):manifest['reason']=str(exc)
        print('Stopped: '+type(exc).__name__)
        raise
    finally:
        (report/'download_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
