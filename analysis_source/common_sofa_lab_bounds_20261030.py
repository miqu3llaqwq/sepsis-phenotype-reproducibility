"""Shared laboratory SOFA lower bounds and event-level t0 identifiability audit.

Not a complete six-domain SOFA, cohort, or external validation. Unimplemented
components remain unknown, not normal. No infusion timestamp substitution.
"""
from pathlib import Path
import json, hashlib, os, math
import numpy as np
import pandas as pd
from start_common_definition_20261029 import ROOT, WORK, OUT, sha

DEST=ROOT/'outputs/common_sofa_20261030'
DOMAINS={'platelet':'coagulation','bilirubin':'liver','creatinine':'renal'}
RANGES={'platelet':(0,2000),'bilirubin':(0,100),'creatinine':(.01,50)}

def dump(name,x):
    (DEST/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')

def score(variable,x):
    if variable=='platelet':return 4 if x<20 else 3 if x<50 else 2 if x<100 else 1 if x<150 else 0
    cuts=[1.2,2,6,12] if variable=='bilirubin' else [1.2,2,3.5,5]
    return sum(x>=c for c in cuts)

def t0_bounds(hours,lower,upper,window_start,window_end,antibiotic,culture):
    use=(hours>=window_start)&(hours<=window_end)
    if not use.any():return None,None,None,None,'no_hour_in_infection_window'
    possible=hours[use&(upper>=2)]; certain=hours[use&(lower>=2)]
    if not len(possible):return None,None,None,None,'no_possible_SOFA2'
    first_possible=float(possible[0])
    if not len(certain):return first_possible,None,max(first_possible,antibiotic,culture,0),None,'no_certain_SOFA2_from_labs'
    first_certain=float(certain[0])
    lo=max(first_possible,antibiotic,culture,0);hi=max(first_certain,antibiotic,culture,0)
    return first_possible,first_certain,lo,hi,'t0_identified_conditional_on_event' if lo==hi else 't0_interval_unresolved'

def tests():
    assert [score('platelet',x) for x in [19,20,49,50,99,100,149,150]]==[4,3,3,2,2,1,1,0]
    assert [score('bilirubin',x) for x in [1.19,1.2,2,6,12]]==[0,1,2,3,4]
    assert [score('creatinine',x) for x in [1.19,1.2,2,3.5,5]]==[0,1,2,3,4]
    # Check all completions of short score intervals against the diagnostic.
    import itertools
    n=0
    for lo in itertools.product([0,2],repeat=3):
      for hi in itertools.product([0,2],repeat=3):
        if any(a>b for a,b in zip(lo,hi)):continue
        for ab in [0,60,120,180]:
          h=np.array([0,60,120]);res=t0_bounds(h,np.array(lo),np.array(hi),0,120,ab,0)
          choices=[range(a,b+1) for a,b in zip(lo,hi)];out=[]
          for vals in itertools.product(*choices):
            idx=np.where(np.array(vals)>=2)[0]
            out.append(max(float(h[idx[0]]),ab,0) if len(idx) else None)
          if res[-1]=='t0_identified_conditional_on_event':assert set(out)=={res[2]}
          if res[3] is not None:
            assert None not in out and min(out)==res[2] and max(out)==res[3]
          n+=1
    assert t0_bounds(np.array([0]),np.array([0]),np.array([24]),1,2,3,0)[-1]=='no_hour_in_infection_window'
    return {'threshold_boundaries':True,'exhaustive_short_interval_cases':n,'empty_window_not_no_dysfunction':True,'hour_assignment_ceil':np.ceil(np.array([-60,-1,0,1,60])/60).tolist()==[-1,0,0,1,1]}

def extract():
    from google.cloud import bigquery
    base=(WORK/'sql/mimic_orders.sql').read_text(encoding='utf-8')
    query=f'''WITH abx AS ({base}), stays AS (
 SELECT DISTINCT subject_id,hadm_id,stay_id,intime,outtime,admittime FROM abx)
SELECT s.stay_id,s.hadm_id,le.labevent_id,le.itemid,le.valuenum,le.valueuom,
 DATETIME_DIFF(le.charttime,s.intime,SECOND)/60.0 AS offset_min,
 DATETIME_DIFF(le.storetime,s.intime,SECOND)/60.0 AS entry_min
FROM stays s JOIN `physionet-data.mimiciv_3_1_hosp.labevents` le
 ON s.hadm_id=le.hadm_id AND s.subject_id=le.subject_id
WHERE le.itemid IN (51265,50885,50912)
 AND le.charttime>=GREATEST(s.admittime,DATETIME_SUB(s.intime,INTERVAL 48 HOUR))
 AND le.charttime<=s.outtime'''
    qpath=WORK/'sql/common_three_domain_labs.sql';qpath.write_text(query,encoding='utf-8')
    path=WORK/'data/common_three_domain_labs_local.parquet';mp=DEST/'query_manifest.json'
    digest=hashlib.sha256(query.encode()).hexdigest()
    if mp.exists():
        old=json.loads(mp.read_text(encoding='utf-8'))
        if old.get('complete') and old.get('sql_sha256')==digest and path.exists() and sha(path)==old.get('output_sha256'):return path
    c=bigquery.Client(project=os.environ.get('GCP_PROJECT'))
    dry=c.query(query,job_config=bigquery.QueryJobConfig(dry_run=True,use_query_cache=False),retry=None,timeout=30)
    assert dry.total_bytes_processed<=40_000_000_000
    m={'complete':False,'sql_sha256':digest,'estimated_bytes':dry.total_bytes_processed}
    print('Three-domain raw extraction estimated bytes',dry.total_bytes_processed,flush=True)
    j=c.query(query,job_config=bigquery.QueryJobConfig(maximum_bytes_billed=40_000_000_000),retry=None,timeout=30)
    m['job_id']=j.job_id;dump(mp.name,m)
    x=j.result(timeout=1200).to_dataframe(create_bqstorage_client=False);x.to_parquet(path,index=False)
    m.update(complete=True,rows=len(x),actual_bytes=j.total_bytes_processed,output_sha256=sha(path));dump(mp.name,m)
    print('Raw laboratory rows',len(x),flush=True);return path

def inputs(source,mimic_path):
    events=pd.read_parquet(WORK/'data'/f'{source}_infection_candidates_local.parquet')
    events['stay_id']=events.stay_id.astype(str)
    if source=='mimic':
        x=pd.read_parquet(mimic_path);x['stay_id']=x.stay_id.astype(str);x=x[x.stay_id.isin(events.stay_id)].copy()
        x['variable']=x.itemid.map({51265:'platelet',50885:'bilirubin',50912:'creatinine'})
        x['value']=pd.to_numeric(x.valuenum,errors='coerce');u=x.valueuom.fillna('').str.lower().str.replace(' ','')
        x['unit_ok']=(x.variable.eq('platelet')&u.isin(['k/ul','10^3/ul','10*3/ul']))|(~x.variable.eq('platelet')&u.eq('mg/dl'))
        x['source_id']=x.labevent_id.astype(str)
        z=pd.read_parquet(WORK/'data/mimic_orders_local.parquet').drop_duplicates('stay_id').copy()
        z['stay_id']=z.stay_id.astype(str);z=z[z.stay_id.isin(events.stay_id)]
        z['icu_start_hospital_min']=(z.intime-z.admittime).dt.total_seconds()/60
        z['icu_end_min']=(z.outtime-z.intime).dt.total_seconds()/60
        z=z.set_index('stay_id')
    else:
        rawp=pd.read_parquet(ROOT/'outputs/eicu_culture_linked_raw_20261026/patient_local.parquet')
        for col in ['patientunitstayid','patienthealthsystemstayid']:rawp[col]=rawp[col].astype(str)
        rawp['hospitaladmitoffset']=pd.to_numeric(rawp.hospitaladmitoffset)
        rawp['unitdischargeoffset']=pd.to_numeric(rawp.unitdischargeoffset)
        z=rawp[rawp.patientunitstayid.isin(events.stay_id)].copy().rename(columns={'patientunitstayid':'stay_id'})
        z['icu_start_hospital_min']=-z.hospitaladmitoffset;z['icu_end_min']=z.unitdischargeoffset;z=z.set_index('stay_id')
        labs=pd.read_parquet(ROOT/'outputs/sofa_observation_audit_20261026/lab_sofa_evidence_local.parquet')
        labs=labs[labs.variable.isin(DOMAINS)].copy();labs['patientunitstayid']=labs.patientunitstayid.astype(str)
        labs=labs.merge(rawp[['patientunitstayid','patienthealthsystemstayid','hospitaladmitoffset']],on='patientunitstayid',validate='many_to_one')
        labs['hospital_min']=labs.offset_min-labs.hospitaladmitoffset
        x=labs.merge(z.reset_index()[['stay_id','patienthealthsystemstayid','icu_start_hospital_min','icu_end_min']],on='patienthealthsystemstayid')
        x['offset_min']=x.hospital_min-x.icu_start_hospital_min
        x=x[x.offset_min.ge(-2880)&x.hospital_min.ge(0)&x.offset_min.le(x.icu_end_min)].copy()
        x['unit_ok']=x.unit_supported;x['source_id']=x.labid.astype(str)
    x['valid']=x.unit_ok & np.isfinite(x.value)
    for var,(lo,hi) in RANGES.items():x.loc[x.variable.eq(var),'valid'] &= x.loc[x.variable.eq(var),'value'].between(lo,hi)
    x['valid']=x.valid.fillna(False)
    # A contradictory timestamp cannot be chosen as a definite scalar observation.
    group=x.groupby(['stay_id','offset_min','variable'],dropna=False)
    clean=group.agg(value=('value','first'),all_valid=('valid','all'),nvalues=('value','nunique'),source_rows=('source_id','size')).reset_index()
    clean['usable']=clean.all_valid&clean.nvalues.eq(1)&clean.offset_min.notna()
    clean['lower']=0
    for var in DOMAINS:
        ix=clean.usable&clean.variable.eq(var)
        clean.loc[ix,'lower']=clean.loc[ix,'value'].map(lambda v:score(var,v))
    clean.to_parquet(WORK/'data'/f'{source}_common_lab_evidence_local.parquet',index=False)
    clean.groupby(['variable','usable']).agg(groups=('stay_id','size'),stays=('stay_id','nunique')).reset_index().to_csv(DEST/f'{source}_laboratory_coverage.csv',index=False)
    events['antibiotic_min']=events.antibiotic_seconds/60-events.stay_id.map(z.icu_start_hospital_min)
    events['culture_min']=events.culture_seconds/60-events.stay_id.map(z.icu_start_hospital_min)
    return clean,events,z

def diagnostic(source,clean,events,stays):
    groups={s:g for s,g in clean.groupby('stay_id')};records=[]
    for idx,(stay,ev) in enumerate(events.groupby('stay_id')):
        end=float(stays.at[stay,'icu_end_min'])
        # No hour endpoint after actual discharge; fixed ICU-relative grid.
        hours=np.arange(-24,math.floor(end/60)+1,dtype=float)*60
        extended=np.arange(-48,math.floor(end/60)+1,dtype=float)*60
        domain=np.zeros((len(extended),3),dtype=np.int8)
        g=groups.get(stay)
        if g is not None:
            for j,var in enumerate(DOMAINS):
                rows=g[(g.variable==var)&g.usable];pos=np.searchsorted(extended,rows.offset_min.to_numpy(float),side='left')
                good=(pos<len(extended))&(rows.offset_min.to_numpy(float)>extended[0]-60)
                np.maximum.at(domain[:,j],pos[good],rows.lower.to_numpy(np.int8)[good])
        rolled=pd.DataFrame(domain).rolling(24,min_periods=1).max().to_numpy(np.int8)
        low=rolled[24:].sum(axis=1);high=np.repeat(24,len(hours))
        assert len(low)==len(hours)
        for r in ev.itertuples():
            suspected=min(r.antibiotic_min,r.culture_min)
            values=t0_bounds(hours,low,high,suspected-2880,suspected+1440,r.antibiotic_min,r.culture_min)
            records.append((r.order_key,stay,*values))
        if idx and idx%10000==0:print(source,'stays scored',idx,flush=True)
    df=pd.DataFrame(records,columns=['order_key','stay_id','first_possible_hour','first_certain_hour','t0_lower','t0_upper','status'])
    df.to_parquet(WORK/'data'/f'{source}_lab_only_t0_diagnostic_local.parquet',index=False)
    counts=df.groupby('status').agg(events=('order_key','size'),stays=('stay_id','nunique')).reset_index()
    counts.to_csv(DEST/f'{source}_conditional_t0_counts.csv',index=False)
    return counts.to_dict('records')

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    protocol={'stage':'partial_component_and_identifiability_audit','hour_grid':'ICU-relative 60-minute endpoints from -24h through last endpoint <= discharge; no prior nurse-heart-rate anchoring',
        'lab_window':'24 hourly bins, each (hour-60min,hour]; extract preceding 48h to support earliest grid endpoint',
        'unknown_policy':'Unknown domains do not receive normal scores. Aggregate upper bound remains 24; laboratory observations only establish a conservative lower bound.',
        'renal':'Creatinine contribution only; urine unimplemented, so renal total is not treated as exact.',
        'time':'Retrospective chart-time observations; no storetime substitution. This is a new common-grid diagnostic, not the original model definition.',
        't0_logic':'For a given candidate infection event, t0 is identified when max(first_possible_SOFA2,AB,culture,ICU) equals max(first_certain_SOFA2,AB,culture,ICU). This does not prove the order valid or first hospital event.',
        'not_complete':['drug route/cancellation alignment','full six-domain scoring','clinical eligibility','first hospital event selection','features','training','external validation']}
    dump('protocol.json',protocol);dump('tests.json',tests())
    mp=extract();result={'stage':'partial_lab_components_and_conditional_t0_audit','complete_six_domain_sofa':False,'formal_cohort':False,'external_validation_complete':False}
    for source in ['eicu','mimic']:
        clean,ev,z=inputs(source,mp);result[source]=diagnostic(source,clean,ev,z);dump('results.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
