"""Retrospective common windows; all raw values and chosen sources traceable."""
import json,hashlib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from extract_common_core_20261030 import run_query
from build_common_identified_20261030 import ROOT,WORK,DATA,OUT,frame,metadata,dump

CFG=json.loads((ROOT/'work/mimic_rebuilt_20261026/config/analysis_v2.json').read_text())
PCFG=json.loads((ROOT/'work/mimic_rebuilt_20261026/config/phenotype_primary_v1.json').read_text())
FEATURES=PCFG['features_primary']

def remote(c):
    params=[bigquery.ArrayQueryParameter('ids','INT64',c.stay_id.astype(int).tolist()),bigquery.ArrayQueryParameter('mins','FLOAT64',c.t0_min.astype(float).tolist())]
    digest=hashlib.sha256(c[['stay_id','t0_min']].to_csv(index=False).encode()).hexdigest()
    ct='''WITH q AS (SELECT id AS stay_id, @mins[OFFSET(i)] AS t0_min FROM UNNEST(@ids) id WITH OFFSET i),
c AS (SELECT s.*,q.t0_min,DATETIME_ADD(s.intime,INTERVAL CAST(ROUND(q.t0_min*60) AS INT64) SECOND) AS t0,a.admittime FROM q
JOIN `physionet-data.mimiciv_3_1_icu.icustays` s USING(stay_id)
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING(hadm_id))'''
    run_query('common_feature_vitals',ct+'''
SELECT c.stay_id,e.itemid,e.valuenum,e.valueuom,
DATETIME_DIFF(e.charttime,c.intime,SECOND)/60.0 AS offset_min,
DATETIME_DIFF(e.storetime,c.intime,SECOND)/60.0 AS entry_min
FROM c JOIN `physionet-data.mimiciv_3_1_icu.chartevents` e USING(stay_id)
WHERE e.itemid IN (220045,220050,220051,220052,220179,220180,220181,220210,223761,223762,220277)
AND e.charttime BETWEEN GREATEST(c.admittime,DATETIME_SUB(c.t0,INTERVAL 6 HOUR)) AND c.t0''',params,digest)
    run_query('common_feature_labs',ct+'''
SELECT c.stay_id,e.labevent_id,e.itemid,e.valuenum,e.valueuom,
DATETIME_DIFF(e.charttime,c.intime,SECOND)/60.0 AS offset_min,
DATETIME_DIFF(e.storetime,c.intime,SECOND)/60.0 AS entry_min
FROM c JOIN `physionet-data.mimiciv_3_1_hosp.labevents` e ON c.hadm_id=e.hadm_id AND c.subject_id=e.subject_id
WHERE e.itemid IN (50813,51301,51265,51222,50912,50983,50971,50882,50931)
AND e.charttime BETWEEN GREATEST(c.admittime,DATETIME_SUB(c.t0,INTERVAL 24 HOUR)) AND c.t0''',params,digest)

def prepare_vitals(source,c):
    if source=='mimic':
        v=frame(DATA/'common_feature_vitals_local.parquet')
        mapping={220045:'heart_rate',220050:'sbp',220051:'dbp',220052:'map',220179:'sbp',220180:'dbp',220181:'map',220210:'resp_rate',223761:'temp_c',223762:'temp_c',220277:'spo2'}
        v['feature']=v.itemid.map(mapping);v['value']=pd.to_numeric(v.valuenum,errors='coerce')
        v['bp_source']=np.where(v.itemid.isin([220179,220180,220181]),'noninvasive','arterial');v['measurement_source']='chart'
        u=v.valueuom.fillna('').str.replace(' ','').str.lower();v['unit_ok']=False
        whitelist={'heart_rate':['bpm','beats/min'],'resp_rate':['insp/min','breaths/min'],'spo2':['%'],'sbp':['mmhg'],'dbp':['mmhg'],'map':['mmhg']}
        for f,units in whitelist.items():v.loc[v.feature.eq(f),'unit_ok']=u[v.feature.eq(f)].isin(units)
        fi=v.itemid.eq(223761);ce=v.itemid.eq(223762)
        v.loc[fi,'unit_ok']=u[fi].isin(['°f','f','degf']);v.loc[ce,'unit_ok']=u[ce].isin(['°c','c','degc'])
        v.loc[fi,'value']=(v.loc[fi,'value']-32)*5/9
        v['temperature_scale']=np.where(fi,'F',np.where(ce,'C',''))
        v['source_key']=v.itemid.astype(str);v['source_id']=v.itemid.astype(str)+':'+v.offset_min.astype(str)
    else:
        v=frame(ROOT/'outputs/eicu_monitor_inputs_20261028/expanded_vital_evidence_local.parquet').rename(columns={'patientunitstayid':'stay_id','chart_min':'offset_min'})
        v=v[v.stay_id.isin(c.stay_id)].copy();v['unit_ok']=True
        v['temperature_scale']=np.where(v.unit.eq('F'),'F',np.where(v.feature.eq('temp_c'),'C',''))
        v['source_key']=v.nursingchartcelltypevalname;v['source_id']=v.nursingchartid.astype(str)
    v['valid']=v.unit_ok&np.isfinite(v.value)
    for f in v.feature.unique():
        bounds=CFG['vitals'].get(f,{'range':[1,300]})['range'];ix=v.feature.eq(f);v.loc[ix,'valid'] &=v.loc[ix,'value'].between(*bounds)
    v['t0_min']=v.stay_id.map(c.set_index('stay_id').t0_min)
    use=v.offset_min.between(v.t0_min-360,v.t0_min)
    use&=(v.offset_min+v.stay_id.map(c.set_index('stay_id').icu_start_hospital_min)).ge(0)
    # A monitor value is a 5-minute summary, so retain a full guard at each edge.
    use&=~v.measurement_source.eq('periodic')|v.offset_min.between(v.t0_min-355,v.t0_min-5)
    v=v[use].copy()
    keys=['stay_id','offset_min','feature','measurement_source','bp_source','source_key']
    validgroup=v.groupby(keys,dropna=False).agg(all_valid=('valid','all'),n=('value','nunique')).reset_index()
    v=v.merge(validgroup,on=keys,validate='many_to_one');v=v[v.all_valid&v.n.eq(1)]
    # Fahrenheit conversion accepted; conflict with Celsius holds both values.
    temp=v[v.feature.eq('temp_c')].groupby(['stay_id','offset_min','measurement_source']).value.agg(['min','max'])
    bad=set(temp.index[(temp['max']-temp['min']).gt(.06)])
    badmask=[f=='temp_c' and (s,t,m) in bad for s,t,m,f in zip(v.stay_id,v.offset_min,v.measurement_source,v.feature)]
    v=v.loc[~np.array(badmask)].sort_values('temperature_scale')
    v=v.drop_duplicates(['stay_id','offset_min','feature','measurement_source','bp_source'])
    return v

def prepare_labs(source,c):
    if source=='mimic':
        l=frame(DATA/'common_feature_labs_local.parquet');mapping={i:f for f,d in CFG['labs'].items() for i in d['itemids']}
        l['feature']=l.itemid.map(mapping);l['value']=pd.to_numeric(l.valuenum,errors='coerce');u=l.valueuom.fillna('').str.lower().str.replace(' ','')
        l['unit_ok']=False
        for f in FEATURES[5:]:
            target=CFG['labs'][f]['unit'].lower();units=[target]
            if target=='meq/l':units+=['mmol/l'] # Na,K,HCO3 are monovalent.
            if target=='k/ul':units+=['10^3/ul','10*3/ul']
            l.loc[l.feature.eq(f),'unit_ok']=u[l.feature.eq(f)].isin(units)
        l['source_id']=l.labevent_id.astype(str)
    else:
        l=frame(ROOT/'outputs/eicu_mapped_observations_20261027/lab_observation_evidence_local.parquet');m=metadata('eicu')
        l['admission_id']=l.patientunitstayid.map(m.admission_id)
        l['hospital_min']=l.draw_min+l.patientunitstayid.map(m.icu_start_hospital_min)
        l['hospital_entry']=l.entry_min+l.patientunitstayid.map(m.icu_start_hospital_min)
        l=l[l.feature.isin(FEATURES[5:])].merge(c[['stay_id','admission_id','icu_start_hospital_min']],on='admission_id')
        l['offset_min']=l.hospital_min-l.icu_start_hospital_min;l['entry_min']=l.hospital_entry-l.icu_start_hospital_min
        l['unit_ok']=l.unit_supported;l['source_id']=l.labid.astype(str)
    l['valid']=l.unit_ok&np.isfinite(l.value)
    for f in FEATURES[5:]:
        ix=l.feature.eq(f);l.loc[ix,'valid'] &=l.loc[ix,'value'].between(*CFG['labs'][f]['range'])
    l['t0_min']=l.stay_id.map(c.set_index('stay_id').t0_min)
    l=l[l.offset_min.between(l.t0_min-1440,l.t0_min)&(l.offset_min+l.stay_id.map(c.set_index('stay_id').icu_start_hospital_min)).ge(0)].copy()
    # Cross-stay same-hospital timestamps cannot be medianed across conflicting values.
    keys=['stay_id','offset_min','feature'];g=l.groupby(keys).agg(all_valid=('valid','all'),n=('value','nunique')).reset_index()
    l=l.merge(g,on=keys,validate='many_to_one');l=l[l.all_valid&l.n.eq(1)].drop_duplicates(keys)
    return l

def extract(source):
    c=frame(DATA/f'{source}_common_identified_cohort_local.parquet')
    if source=='mimic':remote(c)
    vit=prepare_vitals(source,c);lab=prepare_labs(source,c);vg={s:g for s,g in vit.groupby('stay_id')};lg={s:g for s,g in lab.groupby('stay_id')}
    rows=[];proof=[]
    for r in c.itertuples():
        row={k:getattr(r,k) for k in ['stay_id','person_id','admission_id','hospitalid','t0_min']};row.update({f:np.nan for f in FEATURES})
        v=vg.get(r.stay_id,vit.iloc[:0]);l=lg.get(r.stay_id,lab.iloc[:0]);chosen=[]
        bp=v[v.feature.isin(['map','sbp','dbp'])].pivot_table(index=['offset_min','measurement_source','bp_source'],columns='feature',values='value',aggfunc='first').reindex(columns=['map','sbp','dbp'])
        bp=bp[bp['map'].notna()&bp.sbp.notna()&bp['map'].le(bp.sbp)&(bp.dbp.isna()|bp.dbp.le(bp['map']))].reset_index()
        if len(bp):
            if bp.bp_source.eq('arterial').any():bp=bp[bp.bp_source.eq('arterial')]
            if bp.measurement_source.eq('nurse').any():bp=bp[bp.measurement_source.eq('nurse')]
            row['map']=float(bp['map'].median())
            chosen.append(v[v.feature.eq('map')].merge(bp[['offset_min','measurement_source','bp_source']],on=['offset_min','measurement_source','bp_source']))
        for f in ['heart_rate','resp_rate','temp_c','spo2']:
            z=v[v.feature.eq(f)]
            if z.measurement_source.eq('nurse').any():z=z[z.measurement_source.eq('nurse')]
            if len(z):row[f]=float(z.value.median());chosen.append(z)
        for f in FEATURES[5:]:
            z=l[l.feature.eq(f)]
            if len(z):row[f]=float(z.value.median());chosen.append(z.assign(measurement_source='lab'))
        for z in chosen:
            proof.append(z[['stay_id','feature','offset_min','entry_min','t0_min','value','measurement_source','source_id']])
        rows.append(row)
    x=pd.DataFrame(rows);x['n_features']=x[FEATURES].notna().sum(axis=1)
    x['n_domains']=sum(x[fs].notna().any(axis=1) for fs in PCFG['domains'].values())
    x['feature_eligible']=x.n_features.ge(6)&x.n_domains.ge(4)
    for d in PCFG['eligibility']['required_domains']:x['feature_eligible'] &=x[PCFG['domains'][d]].notna().any(axis=1)
    x.to_parquet(DATA/f'{source}_common_features_local.parquet',index=False)
    p=pd.concat(proof,ignore_index=True);assert p.offset_min.le(p.t0_min).all()
    assert p.offset_min.ge(p.t0_min-np.where(p.feature.isin(FEATURES[:5]),360,1440)).all()
    p.to_parquet(DATA/f'{source}_common_feature_provenance_local.parquet',index=False)
    p.groupby(['feature','measurement_source']).agg(records=('stay_id','size'),stays=('stay_id','nunique')).reset_index().to_csv(OUT/f'{source}_feature_source_counts.csv',index=False)
    missing=x[FEATURES].isna().mean().rename_axis('feature').rename('missing_fraction').reset_index();missing.to_csv(OUT/f'{source}_feature_missingness.csv',index=False)
    return {'identified_admissions':len(x),'feature_eligible':int(x.feature_eligible.sum()),'feature_eligible_people':x.loc[x.feature_eligible,'person_id'].nunique(),'hospitals':x.loc[x.feature_eligible,'hospitalid'].nunique(),'source_records':len(p),'retrospective_entry_after_t0_records':int(p.entry_min.gt(p.t0_min).sum())}

if __name__=='__main__':
    result={}
    for source in ['eicu','mimic']:
        result[source]=extract(source);dump('feature_summary.json',result);print(source,result[source],flush=True)
