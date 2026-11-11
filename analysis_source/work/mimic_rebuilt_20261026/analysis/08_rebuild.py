"""Rebuild corrected retrospective landmark cohort, features and internal split.

python analysis/08_rebuild.py --local-check   # render SQL, no network
python analysis/08_rebuild.py                # metadata check, dry run, extract, QC
Patient-level files remain in data/v2_20261009/<run timestamp>/.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/analysis_v2.json'


def load_config():
    c = json.loads(CONFIG.read_text(encoding='utf-8'))
    for key in ('hosp', 'icu', 'derived'):
        if not re.fullmatch(r'[\w-]+\.\w+', c[key]):
            raise ValueError('Invalid configured dataset name')
    if not np.isclose(sum(c['split_fractions']), 1):
        raise ValueError('Split fractions must sum to one')
    for key in ('vitals_hours', 'labs_hours', 'fio2_pair_minutes'):
        if not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError('Window parameters must be positive integers')
    return c


def cohort_cte(c):
    h, i, d = c['hosp'], c['icu'], c['derived']
    return f"""candidate AS (
 SELECT s.subject_id, s.stay_id, i.hadm_id, s.sepsis3,
 s.sofa_time, s.suspected_infection_time, s.antibiotic_time, s.culture_time,
 p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group,
 p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS age,
 i.intime AS icu_intime, i.outtime AS icu_outtime,
 a.admittime, a.dischtime, a.deathtime,
 a.hospital_expire_flag AS hosp_mortality,
 GREATEST(s.sofa_time, s.antibiotic_time, s.culture_time, i.intime) AS t0
 FROM `{d}.sepsis3` s
 LEFT JOIN `{i}.icustays` i ON s.stay_id = i.stay_id AND s.subject_id = i.subject_id
 LEFT JOIN `{h}.admissions` a ON i.hadm_id = a.hadm_id AND s.subject_id = a.subject_id
 LEFT JOIN `{h}.patients` p ON s.subject_id = p.subject_id
), eligibility AS (
 SELECT *, CASE
 WHEN sepsis3 IS NULL OR NOT sepsis3 THEN 'not_sepsis3'
 WHEN hadm_id IS NULL OR gender IS NULL THEN 'unlinked_record'
 WHEN age IS NULL OR age < 18 THEN 'age_ineligible'
 WHEN t0 IS NULL OR admittime IS NULL OR dischtime IS NULL
   OR icu_intime IS NULL OR icu_outtime IS NULL THEN 'missing_time'
 WHEN hosp_mortality IS NULL OR hosp_mortality NOT IN (0,1) THEN 'missing_outcome'
 WHEN t0 < admittime OR t0 >= dischtime OR t0 < icu_intime
   OR t0 >= icu_outtime THEN 'landmark_outside_stay'
 WHEN hosp_mortality = 1 AND deathtime IS NULL THEN 'missing_death_time'
 WHEN hosp_mortality = 1 AND (deathtime <= t0 OR deathtime > dischtime)
   THEN 'invalid_death_time'
 WHEN hosp_mortality = 0 AND deathtime IS NOT NULL THEN 'inconsistent_death_flag'
 ELSE NULL END AS exclusion_reason
 FROM candidate
), eligible_ranked AS (
 SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY t0, hadm_id, stay_id) AS eligible_rank
 FROM eligibility WHERE exclusion_reason IS NULL
), cohort AS (
 SELECT * EXCEPT(exclusion_reason, eligible_rank),
 IF(hosp_mortality = 1, 1, 2) AS event_type,
 IF(hosp_mortality = 1, deathtime, dischtime) AS endpoint,
 DATETIME_DIFF(IF(hosp_mortality = 1, deathtime, dischtime), t0, SECOND)/86400.0 AS follow_up_days
 FROM eligible_ranked WHERE eligible_rank = 1
)"""


def median(expr, name):
    return f'APPROX_QUANTILES({expr}, 2)[SAFE_OFFSET(1)] AS {name}'


def observation_cte(c, domain):
    specs = c[domain]
    is_vital = domain == 'vitals'
    table = f"{c['icu']}.chartevents" if is_vital else f"{c['hosp']}.labevents"
    join = 'e.stay_id = c.stay_id' if is_vital else 'e.hadm_id = c.hadm_id AND e.subject_id = c.subject_id'
    hours = c['vitals_hours'] if is_vital else c['labs_hours']
    cases, range_cases, unit_cases, ids = [], [], [], []
    for name, sp in specs.items():
        condition = 'e.itemid IN (' + ','.join(map(str, sp['itemids'])) + ')'
        cases.append(f"WHEN {condition} THEN '{name}'")
        lo, hi = sp['range']
        range_cases.append(f'WHEN feature = \'{name}\' THEN value BETWEEN {lo} AND {hi}')
        aliases={'beats/min':['beats/min','bpm'], 'breaths/min':['breaths/min','insp/min'],
                 'mmHg':['mmhg'], 'C':['c','deg.c','deg.celsius','°c','?c'],
                 '%':['%'], 'K/uL':['k/ul'], 'mmol/L':['mmol/l'],
                 'mg/dL':['mg/dl'], 'g/dL':['g/dl'], 'mEq/L':['meq/l','mmol/l']}
        units=aliases[sp['unit']]
        accepted=','.join("'"+u+"'" for u in units)
        if name=='temp_c':
            unit_cases.append("WHEN feature='temp_c' AND itemid=223761 THEN unit IN ('f','deg.f','deg.fahrenheit','°f','?f')")
        unit_cases.append(f"WHEN feature='{name}' THEN unit IN ({accepted})")
        ids.extend(sp['itemids'])
    # All values are screened at observation level before aggregation.
    value = 'CASE WHEN e.itemid = 223761 THEN (e.valuenum-32)*5.0/9.0 ELSE e.valuenum END' if is_vital else 'e.valuenum'
    return f"""{domain}_raw AS (
 SELECT c.stay_id, e.charttime, e.storetime, e.itemid,
 CASE {' '.join(cases)} END AS feature, {value} AS value,
 LOWER(REPLACE(REPLACE(REPLACE(e.valueuom,' ',''),'µ','u'),'μ','u')) AS unit,
 e.storetime IS NOT NULL AND e.storetime <= c.t0 AS available
 FROM `{table}` e JOIN cohort c ON {join}
 WHERE e.itemid IN ({','.join(map(str, ids))}) AND e.valuenum IS NOT NULL
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL {hours} HOUR) AND c.t0
), {domain}_screened AS (
 SELECT *, CASE {' '.join(range_cases)} ELSE FALSE END AS valid,
 CASE {' '.join(unit_cases)} ELSE FALSE END AS unit_valid
 FROM {domain}_raw
), {domain}_wide AS (
 SELECT stay_id,
 {', '.join(median(f"IF(feature = '{name}' AND available AND valid AND unit_valid, value, NULL)", name) for name in specs)},
 {', '.join(f"COUNTIF(feature = '{name}' AND available AND NOT valid) AS {name}_invalid_n, COUNTIF(feature = '{name}' AND NOT available) AS {name}_unavailable_n, COUNTIF(feature = '{name}' AND available AND (unit_valid IS NULL OR NOT unit_valid)) AS {name}_unit_unknown_n" for name in specs)}
 FROM {domain}_screened GROUP BY stay_id
)"""


def feature_sql(c):
    h, i, d = c['hosp'], c['icu'], c['derived']
    vh, lh, tolerance = c['vitals_hours'], c['labs_hours'], c['fio2_pair_minutes']
    return f"""-- Generated from config/analysis_v2.json by analysis/08_rebuild.py.
-- Retrospective evidence-complete landmark; NOT a real-time sepsis alert.
WITH {cohort_cte(c)},
{observation_cte(c, 'vitals')},
{observation_cte(c, 'labs')},
gcs_components AS (
 SELECT c.stay_id, e.charttime,
 MAX(IF(e.itemid=220739 AND e.valuenum BETWEEN 1 AND 4, e.valuenum, NULL)) AS eyes,
 MAX(IF(e.itemid=223900 AND e.valuenum BETWEEN 1 AND 5, e.valuenum, NULL)) AS verbal,
 MAX(IF(e.itemid=223901 AND e.valuenum BETWEEN 1 AND 6, e.valuenum, NULL)) AS motor,
 COUNTIF(e.itemid=223900 AND (e.value='No Response-ETT' OR e.valuenum=0))>0 AS unable
 FROM `{i}.chartevents` e JOIN cohort c ON e.stay_id=c.stay_id
 WHERE e.itemid IN (220739,223900,223901)
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL {vh} HOUR) AND c.t0
 AND e.storetime IS NOT NULL AND e.storetime <= c.t0
 GROUP BY c.stay_id, e.charttime
), gcs_wide AS (
 SELECT stay_id, MIN(IF(NOT unable, eyes+verbal+motor, NULL)) AS gcs_min,
 COUNTIF(unable) AS gcs_unable_n
 FROM gcs_components GROUP BY stay_id
), fio2_observations AS (
 SELECT c.stay_id, e.charttime,
 CASE WHEN e.valuenum BETWEEN 0.21 AND 1 THEN e.valuenum*100
 WHEN e.valuenum BETWEEN 21 AND 100 THEN e.valuenum ELSE NULL END AS fio2
 FROM `{i}.chartevents` e JOIN cohort c ON e.stay_id=c.stay_id
 WHERE e.itemid=223835
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL {lh} HOUR) AND c.t0
 AND e.storetime IS NOT NULL AND e.storetime<=c.t0
), bloodgas_raw AS (
 SELECT c.stay_id, e.specimen_id, e.charttime,
 MAX(IF(e.itemid=52033, e.value, NULL)) AS specimen,
 MAX(IF(e.itemid=50821 AND e.valuenum>0 AND e.valuenum<=1000
 AND LOWER(REPLACE(e.valueuom,' ',''))='mmhg',e.valuenum,NULL)) AS po2,
 MAX(IF(e.itemid=50816,
 CASE WHEN e.valuenum BETWEEN 0.21 AND 1 THEN e.valuenum*100
 WHEN e.valuenum BETWEEN 21 AND 100 THEN e.valuenum ELSE NULL END,NULL)) AS fio2
 FROM `{h}.labevents` e JOIN cohort c
 ON e.hadm_id=c.hadm_id AND e.subject_id=c.subject_id
 WHERE e.itemid IN (52033,50821,50816) AND e.specimen_id IS NOT NULL
 AND e.charttime BETWEEN DATETIME_SUB(c.t0, INTERVAL {lh} HOUR) AND c.t0
 AND e.storetime IS NOT NULL AND e.storetime<=c.t0
 GROUP BY c.stay_id,e.specimen_id,e.charttime
), arterial_samples AS (
 SELECT stay_id,specimen_id,charttime,po2 AS pao2,fio2 AS sample_fio2
 FROM bloodgas_raw bg WHERE bg.specimen LIKE 'ART%' AND bg.po2 IS NOT NULL
), pairs AS (
 SELECT b.*, COALESCE(b.sample_fio2, f.fio2) AS paired_fio2,
 IF(b.sample_fio2 IS NOT NULL, 0, DATETIME_DIFF(b.charttime,f.charttime,SECOND)) AS pair_lag_seconds
 FROM arterial_samples b LEFT JOIN fio2_observations f
 ON b.stay_id=f.stay_id AND f.fio2 IS NOT NULL
 AND f.charttime BETWEEN DATETIME_SUB(b.charttime, INTERVAL {tolerance} MINUTE) AND b.charttime
 QUALIFY ROW_NUMBER() OVER (PARTITION BY b.stay_id,b.specimen_id ORDER BY f.charttime DESC, f.fio2 DESC)=1
), oxygen_wide AS (
 SELECT stay_id, {median('pao2', 'pao2')},
 {median('paired_fio2', 'fio2')},
 {median('SAFE_DIVIDE(pao2,paired_fio2/100.0)', 'pao2_fio2_ratio')},
 COUNT(*) AS arterial_samples_n, COUNTIF(paired_fio2 IS NOT NULL) AS oxygen_pairs_n,
 MAX(IF(paired_fio2 IS NOT NULL, pair_lag_seconds, NULL)) AS pair_max_lag_seconds
 FROM pairs GROUP BY stay_id
)
SELECT c.*,
 v.* EXCEPT(stay_id), l.* EXCEPT(stay_id), g.* EXCEPT(stay_id), o.* EXCEPT(stay_id)
FROM cohort c
LEFT JOIN vitals_wide v USING(stay_id)
LEFT JOIN labs_wide l USING(stay_id)
LEFT JOIN gcs_wide g USING(stay_id)
LEFT JOIN oxygen_wide o USING(stay_id)
ORDER BY c.subject_id
"""


def validate_dataset(df, c):
    required = set(c['primary_features']) | {'subject_id','stay_id','hadm_id','t0','admittime','dischtime','icu_intime','icu_outtime','endpoint','follow_up_days','event_type','hosp_mortality','antibiotic_time','culture_time','sofa_time','sepsis3'}
    if required - set(df):
        raise ValueError('Missing columns: ' + ','.join(sorted(required-set(df))))
    if df.empty:
        raise ValueError('Empty eligible cohort')
    if any(df[x].isna().any() or not df[x].is_unique for x in ['subject_id','stay_id','hadm_id']):
        raise ValueError('Missing or duplicated patient/event keys')
    if not df['sepsis3'].eq(True).all():
        raise ValueError('Sepsis-3 filter failed')
    if not df.age.ge(18).all() or not df.event_type.isin([1,2]).all():
        raise ValueError('Invalid age or competing event')
    t = pd.to_datetime(df.t0)
    for earlier in ['admittime','icu_intime','antibiotic_time','culture_time','sofa_time']:
        x = pd.to_datetime(df[earlier], errors='coerce')
        if x.isna().any() or (x>t).any():
            raise ValueError('Landmark evidence/time inconsistent: '+earlier)
    for later in ['dischtime','icu_outtime','endpoint']:
        x = pd.to_datetime(df[later], errors='coerce')
        if x.isna().any() or (x<=t).any():
            raise ValueError('Invalid endpoint/time: '+later)
    actual = (pd.to_datetime(df.endpoint)-t).dt.total_seconds()/86400
    if not np.allclose(actual, df.follow_up_days.astype(float), rtol=0, atol=1e-8):
        raise ValueError('Follow-up does not match endpoint')
    if not df.event_type.eq(np.where(df.hosp_mortality.eq(1),1,2)).all():
        raise ValueError('Outcome flag mismatch')
    if df.pair_max_lag_seconds.dropna().gt(c['fio2_pair_minutes']*60).any():
        raise ValueError('Oxygen pairing tolerance exceeded')
    return True


def split_ids(df, c):
    # Independent of outcomes and input row order. This is internal, not temporal validation.
    ids = np.sort(df.subject_id.astype('int64').to_numpy())
    ids = np.random.default_rng(c['seed']).permutation(ids)
    n1 = int(len(ids)*c['split_fractions'][0])
    n2 = n1+int(len(ids)*c['split_fractions'][1])
    parts = np.split(ids,[n1,n2])
    if any(len(p)==0 for p in parts):
        raise ValueError('Too few patients for three partitions')
    mapping = pd.concat([pd.DataFrame({'subject_id':p,'split':label}) for label,p in zip(['train','validation','test'],parts)],ignore_index=True)
    if not mapping.subject_id.is_unique or len(mapping)!=len(df):
        raise ValueError('Partition overlap or missing patients')
    return df.merge(mapping,on='subject_id',validate='one_to_one')


def save_summaries(df, audit, c, dest):
    dest.mkdir(parents=True,exist_ok=True)
    counts = audit.exclusion_reason.fillna('eligible_before_dedup').value_counts()
    remaining=len(audit)
    rows=[{'step':'candidate_records','n_excluded':0,'n_remaining':remaining}]
    for reason in ['not_sepsis3','unlinked_record','age_ineligible','missing_time','missing_outcome','landmark_outside_stay','missing_death_time','invalid_death_time','inconsistent_death_flag']:
        excluded=int(counts.get(reason,0)); remaining-=excluded
        rows.append({'step':reason,'n_excluded':excluded,'n_remaining':remaining})
    repeats=remaining-len(df)
    if repeats<0:
        raise ValueError('Feature rows exceed eligible event rows')
    rows.append({'step':'repeat_eligible_events_removed','n_excluded':repeats,'n_remaining':len(df)})
    rows.append({'step':'final_patients','n_excluded':0,'n_remaining':len(df)})
    pd.DataFrame(rows).to_csv(dest/'cohort_flow_counts.csv',index=False)
    summary = df.groupby('split',sort=False).agg(n=('subject_id','size'),deaths=('hosp_mortality','sum'))
    summary['alive_discharges']=summary.n-summary.deaths
    summary.to_csv(dest/'split_summary.csv')
    rows=[]
    for part in ['all','train','validation','test']:
        sub=df if part=='all' else df[df.split==part]
        for name in c.get('candidate_features',c['primary_features']):
            rows.append({'partition':part,'feature':name,'n':len(sub),'n_missing':int(sub[name].isna().sum()),'pct_missing':round(sub[name].isna().mean()*100,2)})
    pd.DataFrame(rows).to_csv(dest/'feature_missingness.csv',index=False)
    numeric = [x for x in c.get('candidate_features',c['primary_features']) if x!='gender']
    df[numeric].describe().T.to_csv(dest/'feature_summary.csv')
    flags=[x for x in df if x.endswith('_invalid_n') or x.endswith('_unavailable_n') or x.endswith('_unit_unknown_n') or x in ['gcs_unable_n','arterial_samples_n','oxygen_pairs_n']]
    pd.DataFrame([{'metric':x,'observations':int(df[x].fillna(0).sum())} for x in flags]).to_csv(dest/'observation_qc.csv',index=False)
    # Do not choose variables automatically from full/test missingness.
    train=df[df.split=='train']
    high=[x for x in c['primary_features'] if train[x].isna().mean()>0.5]
    return high


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local-check',action='store_true')
    parser.add_argument('--preflight',action='store_true',help='Metadata and dry runs only; no extraction')
    args=parser.parse_args()
    c=load_config()
    sql_dir=ROOT/'sql'; sql_dir.mkdir(exist_ok=True)
    queries={'cohort_audit':f'WITH {cohort_cte(c)} SELECT * FROM eligibility', 'features':feature_sql(c)}
    for name,sql in queries.items():
        (sql_dir/f'v2_{name}.sql').write_text(sql,encoding='utf-8')
    if args.local_check:
        print('Local configuration and SQL generation passed. No cloud query performed.')
        return 0
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    reports=ROOT/'outputs/reports'/c['version']/stamp
    reports.mkdir(parents=True,exist_ok=False)
    manifest={'date':stamp,'version':c['version'],'status':'started','python':platform.python_version(), 'config':c, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'sql_sha256':{n:hashlib.sha256(q.encode()).hexdigest() for n,q in queries.items()}, 'packages':{}, 'jobs':{}}
    for package in ['pandas','numpy','google-cloud-bigquery','pyarrow']:
        manifest['packages'][package]=importlib.metadata.version(package)
    try:
        from google.cloud import bigquery
        client=bigquery.Client(project=os.environ.get('GCP_PROJECT'))
        requirements={
            f"{c['derived']}.sepsis3":{'subject_id','stay_id','sepsis3','sofa_time','antibiotic_time','culture_time','suspected_infection_time'},
            f"{c['hosp']}.patients":{'subject_id','gender','anchor_age','anchor_year','anchor_year_group'},
            f"{c['hosp']}.admissions":{'subject_id','hadm_id','admittime','dischtime','deathtime','hospital_expire_flag'},
            f"{c['icu']}.icustays":{'subject_id','hadm_id','stay_id','intime','outtime'},
            f"{c['icu']}.chartevents":{'stay_id','itemid','charttime','storetime','value','valuenum','valueuom'},
            f"{c['hosp']}.labevents":{'subject_id','hadm_id','specimen_id','itemid','charttime','storetime','value','valuenum','valueuom'}
        }
        schemas={}
        for table,fields in requirements.items():
            schema=client.get_table(table,retry=None,timeout=20).schema
            schemas[table]=[{'name':f.name,'type':f.field_type} for f in schema]
            missing=fields-{f.name for f in schema}
            if missing:
                raise ValueError('Schema mismatch: '+table+' '+','.join(sorted(missing)))
        (reports/'schema_result.json').write_text(json.dumps(schemas,indent=2),encoding='utf-8')
        for name,sql in queries.items():
            job=client.query(sql,job_config=bigquery.QueryJobConfig(dry_run=True,use_query_cache=False),retry=None,timeout=30)
            manifest['jobs'][name]={'dry_run_bytes':job.total_bytes_processed}
            if job.total_bytes_processed>c['maximum_bytes_billed']:
                raise ValueError('Dry-run scan exceeds configured billing limit')
        if args.preflight:
            manifest['status']='schema_and_dry_runs_passed'
            print('Cloud metadata and SQL dry runs passed; no patient extraction performed.')
            return 0
        data_dir=ROOT/'data'/c['version']/stamp
        data_dir.mkdir(parents=True,exist_ok=False)
        frames={}
        for name,sql in queries.items():
            print('Extracting '+name+' ...',flush=True)
            job=client.query(sql,job_config=bigquery.QueryJobConfig(maximum_bytes_billed=c['maximum_bytes_billed']),retry=None,timeout=30)
            frame=job.result().to_dataframe(create_bqstorage_client=False)
            manifest['jobs'][name].update(job_id=job.job_id,bytes_processed=job.total_bytes_processed,rows=len(frame))
            frame.to_parquet(data_dir/f'{name}.parquet',index=False)
            frames[name]=frame
        validate_dataset(frames['features'],c)
        dataset=split_ids(frames['features'],c)
        dataset.to_parquet(data_dir/'analysis_dataset.parquet',index=False)
        for part in ['train','validation','test']:
            dataset.loc[dataset.split==part,['subject_id','hadm_id','stay_id']].to_parquet(data_dir/f'{part}_ids.parquet',index=False)
        tables=ROOT/'outputs/tables'/c['version']/stamp
        high=save_summaries(dataset,frames['cohort_audit'],c,tables)
        manifest.update(status='qc_passed_clustering_not_started',data_path=str(data_dir),tables_path=str(tables),training_missingness_over_50=high)
        (ROOT/'logs/v2_latest.json').write_text(json.dumps({'manifest':str(reports/'run_manifest.json'),'data_path':str(data_dir),'tables_path':str(tables)},indent=2),encoding='utf-8')
        print('Cohort and feature QC completed; partitions saved. Clustering has not started.')
        if high:
            print('Training variables requiring missingness review: '+', '.join(high))
    except Exception as exc:
        manifest['status']='blocked'
        # Credentials, patient records and arbitrary exception text are not logged.
        manifest['error_type']=type(exc).__name__
        if isinstance(exc,ValueError):
            manifest['reason']=str(exc)
        else:
            manifest['reason']='Cloud authentication, access or network failed; inspect the local environment.'
        (reports/'BLOCKED.md').write_text('# Rebuild stopped\n\n'+manifest['reason']+'\n\nNo old data files were overwritten. No clustering or model evaluation was performed.\n',encoding='utf-8')
        print('Stopped: '+manifest['error_type']+'. '+manifest['reason'])
        return 1
    finally:
        (reports/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return 0


if __name__=='__main__':
    sys.exit(main())
