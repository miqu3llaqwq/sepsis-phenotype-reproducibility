"""Audit every rebuilt event against source dependencies, then independently select cohort."""
from pathlib import Path
import json, hashlib, os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from rebuild_corrected_mimic_branch import BRANCH, ROOT

OUT=BRANCH/'reports/source_acceptance_20261027'
OUT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    prefix=(BRANCH/'sql/cohort_audit.sql').read_text().split(', candidate AS (')[0]
    sql=prefix+''', micro AS (
 SELECT micro_specimen_id, MAX(subject_id) subject_id, MAX(charttime) charttime,
 CAST(MAX(chartdate) AS DATE) chartdate FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents` GROUP BY 1
)
SELECT s.*,
 EXISTS(SELECT 1 FROM `physionet-data.mimiciv_3_1_derived.antibiotic` a
 WHERE a.subject_id=s.subject_id AND a.stay_id=s.stay_id AND a.starttime=s.antibiotic_time) antibiotic_match,
 EXISTS(SELECT 1 FROM `physionet-data.mimiciv_3_1_hosp.prescriptions` p
 JOIN `physionet-data.mimiciv_3_1_icu.icustays` i ON p.hadm_id=i.hadm_id AND p.subject_id=i.subject_id
 WHERE i.stay_id=s.stay_id AND p.subject_id=s.subject_id AND p.starttime=s.antibiotic_time) prescription_match,
 EXISTS(SELECT 1 FROM `physionet-data.mimiciv_3_1_derived.suspicion_of_infection` q
 WHERE q.subject_id=s.subject_id AND q.stay_id=s.stay_id AND q.antibiotic_time=s.antibiotic_time
 AND q.culture_time=s.culture_time AND q.suspected_infection_time=s.suspected_infection_time
 AND q.suspected_infection=1) infection_match,
 EXISTS(SELECT 1 FROM `physionet-data.mimiciv_3_1_derived.sofa` f
 WHERE f.stay_id=s.stay_id AND f.endtime=s.sofa_time AND f.sofa_24hours=s.sofa_score
 AND f.respiration_24hours=s.respiration AND f.coagulation_24hours=s.coagulation
 AND f.liver_24hours=s.liver AND f.cardiovascular_24hours=s.cardiovascular
 AND f.cns_24hours=s.cns AND f.renal_24hours=s.renal) sofa_match,
 EXISTS(SELECT 1 FROM micro me WHERE me.subject_id=s.subject_id
 AND COALESCE(me.charttime,DATETIME(me.chartdate))=s.culture_time
 AND ((me.charttime IS NOT NULL AND
 ((s.antibiotic_time>me.charttime AND s.antibiotic_time<=DATETIME_ADD(me.charttime,INTERVAL 72 HOUR))
 OR (s.antibiotic_time<me.charttime AND s.antibiotic_time>=DATETIME_SUB(me.charttime,INTERVAL 24 HOUR))))
 OR (me.charttime IS NULL AND DATE(s.antibiotic_time) BETWEEN DATE_SUB(me.chartdate,INTERVAL 1 DAY) AND DATE_ADD(me.chartdate,INTERVAL 3 DAY)))) culture_pair_match,
 EXISTS(SELECT 1 FROM micro me WHERE me.subject_id=s.subject_id AND me.charttime IS NULL
 AND DATETIME(me.chartdate)=s.culture_time) has_date_only_culture,
 EXISTS(SELECT 1 FROM `physionet-data.mimiciv_3_1_derived.sepsis3` old WHERE old.stay_id=s.stay_id) in_materialized
FROM rebuilt_sepsis3 s'''
    (OUT/'event_checks.sql').write_text(sql,encoding='utf-8')
    path=OUT/'event_checks.parquet'; meta_path=OUT/'query_manifest.json'
    digest=hashlib.sha256(sql.encode()).hexdigest()
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text()); assert meta['sql_sha256']==digest and meta['sha256']==sha(path)
        events=pd.read_parquet(path)
    else:
        client=bigquery.Client(project=os.environ.get('GCP_PROJECT'))
        dry=client.query(sql,job_config=bigquery.QueryJobConfig(dry_run=True,use_query_cache=False),timeout=30,retry=None)
        print('source audit estimated bytes',dry.total_bytes_processed,flush=True)
        assert dry.total_bytes_processed<5_000_000_000
        job=client.query(sql,job_config=bigquery.QueryJobConfig(maximum_bytes_billed=5_000_000_000),timeout=30,retry=None)
        print('source audit job',job.job_id,flush=True)
        events=job.result(timeout=600).to_dataframe(create_bqstorage_client=False)
        events.to_parquet(path,index=False)
        meta={'job_id':job.job_id,'sql_sha256':digest,'bytes_processed':job.total_bytes_processed,'sha256':sha(path)}
        meta_path.write_text(json.dumps(meta,indent=2))
    audit=pd.read_parquet(BRANCH/'data/cohort_audit.parquet')
    current=pd.read_parquet(BRANCH/'data/analysis_dataset.parquet')
    old=pd.read_parquet(BRANCH/'data/historical_reuse_source/analysis_dataset.parquet')
    for frame in [events,audit,current,old]:
        for c in ['sofa_time','antibiotic_time','culture_time','suspected_infection_time','t0','admittime','dischtime','icu_intime','icu_outtime','deathtime']:
            if c in frame: frame[c]=pd.to_datetime(frame[c])
    audit['independent_t0']=audit[['sofa_time','antibiotic_time','culture_time','icu_intime']].max(axis=1)
    # Reimplement the ordered exclusions rather than use the stored label to choose rows.
    reasons=pd.Series(pd.NA,index=audit.index,dtype='string')
    predicates=[('not_sepsis3',~audit.sepsis3.fillna(False)),
      ('unlinked_record',audit.hadm_id.isna()|audit.gender.isna()),
      ('age_ineligible',audit.age.isna()|audit.age.lt(18)),
      ('missing_time',audit[['t0','admittime','dischtime','icu_intime','icu_outtime']].isna().any(axis=1)),
      ('missing_outcome',~audit.hosp_mortality.isin([0,1])),
      ('landmark_outside_stay',audit.t0.lt(audit.admittime)|audit.t0.ge(audit.dischtime)|audit.t0.lt(audit.icu_intime)|audit.t0.ge(audit.icu_outtime)),
      ('missing_death_time',audit.hosp_mortality.eq(1)&audit.deathtime.isna()),
      ('invalid_death_time',audit.hosp_mortality.eq(1)&(audit.deathtime.le(audit.t0)|audit.deathtime.gt(audit.dischtime))),
      ('inconsistent_death_flag',audit.hosp_mortality.eq(0)&audit.deathtime.notna())]
    for label,mask in predicates: reasons.loc[reasons.isna()&mask.fillna(False)]=label
    assert reasons.fillna('eligible').eq(audit.exclusion_reason.fillna('eligible')).all()
    selected=audit[reasons.isna()].sort_values(['subject_id','t0','hadm_id','stay_id']).drop_duplicates('subject_id')
    assert selected.set_index('subject_id').stay_id.sort_index().equals(current.set_index('subject_id').stay_id.sort_index())
    key=['subject_id','stay_id','sofa_time','antibiotic_time','culture_time','suspected_infection_time']
    joined=audit.merge(events,on=key,validate='one_to_one',indicator=True)
    assert joined['_merge'].eq('both').all()
    checks=['antibiotic_match','prescription_match','infection_match','sofa_match','culture_pair_match']
    joined['sofa_ge2']=joined.sofa_score.ge(2)
    joined['sofa_time_window']=joined.sofa_time.between(joined.suspected_infection_time-pd.Timedelta(hours=48),joined.suspected_infection_time+pd.Timedelta(hours=24))
    joined['component_sum']=joined[['respiration','coagulation','liver','cardiovascular','cns','renal']].sum(axis=1).eq(joined.sofa_score)
    checks+=['sofa_ge2','sofa_time_window','component_sum']
    failures={c:int((~joined[c].fillna(False)).sum()) for c in checks}
    assert not any(failures.values()),failures
    assert audit.t0.eq(audit.independent_t0).all()
    comparison=current[['subject_id','stay_id','t0']].merge(old[['subject_id','stay_id','t0']],on='subject_id',how='outer',suffixes=('_new','_old'),indicator=True)
    conditions=[comparison['_merge'].eq('left_only'),comparison['_merge'].eq('right_only'),comparison.stay_id_new.ne(comparison.stay_id_old),comparison.t0_new.ne(comparison.t0_old)]
    comparison['group']=np.select([x.fillna(False).to_numpy(dtype=bool) for x in conditions],['new_subject','old_only','changed_stay','changed_t0'],default='unchanged')
    counts=comparison.group.value_counts().to_dict()
    assert counts=={'unchanged':25187,'new_subject':6230,'changed_stay':5},counts
    changed=comparison[comparison.group.isin(['new_subject','changed_stay'])].copy()
    evidence=changed.merge(joined,left_on=['subject_id','stay_id_new'],right_on=['subject_id','stay_id'],validate='one_to_one')
    assert len(evidence)==6235 and evidence.exclusion_reason.isna().all()
    assert not evidence.in_materialized.any()
    swaps=comparison[comparison.group.eq('changed_stay')]
    assert swaps.t0_new.le(swaps.t0_old).all()
    evidence.to_parquet(OUT/'all_changed_subjects_evidence.parquet',index=False)
    # A reproducible 30-case audit file includes all changed stays and timed/date-only/new-year strata.
    samples=[evidence[evidence.group.eq('changed_stay')]]
    for col in ['has_date_only_culture','anchor_year_group']:
        for _,part in evidence.groupby(col,dropna=False): samples.append(part.sort_values('stay_id').head(3))
    sample=pd.concat(samples).drop_duplicates('stay_id').head(30)
    sample.to_parquet(OUT/'stratified_case_checks.parquet',index=False)
    byyear=evidence.groupby(['group','anchor_year_group'],dropna=False).size().reset_index(name='n_subjects')
    byyear.to_csv(OUT/'changed_subjects_by_year_group.csv',index=False)
    result={'date':'2026-10-27','status':'accepted_for_revised_analysis','cohort_subjects':len(current),'candidate_stays':len(events),'all_candidate_source_checks_failures':failures,
      'changed_subject_counts':counts,'changed_subjects_all_source_checks_passed':len(evidence),'new_stays_absent_from_materialized':bool((~evidence.in_materialized).all()),
      'changed_stays_are_earlier_or_equal_landmarks':bool(swaps.t0_new.le(swaps.t0_old).all()),'independent_exclusions_and_first_eligible_selection_match':True,
      't0_four_component_max_verified':True,'traceable_case_sample_n':len(sample),'source_query':meta,
      'dataset_sha256':sha(BRANCH/'data/analysis_dataset.parquet'),'sofa_scope':'current versioned derived SOFA; six component values/time/score checked, not raw SOFA rebuilt',
      'historical_build_root_cause':'unproven; not necessary to define reproducible revised cohort','original_frozen_outputs_unchanged':True}
    (OUT/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
