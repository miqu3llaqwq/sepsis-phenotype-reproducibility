"""Common-definition, partially identified cohort. Never normal-impute SOFA.

All six clinical domains retain their possible range. Dose timing and urine
collection-duration gaps are explicitly unresolved in BOTH source adapters.
This is not a reconstruction of an exact scalar six-domain SOFA.
"""
from pathlib import Path
import json, math, ast
import numpy as np
import pandas as pd
from start_common_definition_20261029 import ROOT,WORK,sha
from common_sofa_lab_bounds_20261030 import score,t0_bounds

OUT=ROOT/'outputs/common_core_20261030'; DATA=WORK/'data'; CACHE=ROOT/'outputs/eicu_culture_linked_raw_20261026'
DOMAINS=['respiration','coagulation','liver','cardiovascular','cns','renal']

def dump(name,x):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')

def frame(path):
    x=pd.read_parquet(path)
    for c in ['stay_id','person_id','admission_id','patientunitstayid','patienthealthsystemstayid','uniquepid','order_key']:
        if c in x:x[c]=x[c].astype(str)
    return x

def protocol():
    p={'version':'common_identified_v2_20261030','posthoc':True,'previous_external_results_seen':True,
       'scope':'Identifiable subset under six-domain SOFA bounds; not exact six-domain scalar SOFA and not original-model strict external validation.',
       'hour_grid':'ICU-relative endpoints -24h through floor(ICU discharge/60); each score uses (h-1440,h] observations within hospital admission.',
       'baseline':'Operational absolute SOFA >=2 under assumed baseline zero, as in reference extraction; not documented patient-specific delta SOFA.',
       'infection':'Frozen reference antimicrobial name dictionary, including its antifungals; explicit systemic routes; prescription-start time, not actual administration. Same-admission exact culture times. Prior [AB-72h,AB) preferred, else (AB,AB+24h].',
       'orders':'Known systemic route, known non-cancelled/valid POE lifecycle, stop strictly after start. Before-start stop/discontinuation/cancellation exclude. Unknown route/lifecycle, zero or missing duration held possible and can block an earlier hospital event. Order entry after start is retained as retrospective metadata, not prospective availability.',
       'culture':'Deduplicate exact collection timestamp for pairing, retain specimen provenance; no midnight imputation, restrict to hospital period.',
       'sofa_bounds':{
         'coagulation':'Valid unique-time platelet; standard thresholds; unknown has lower0 upper4.',
         'liver':'Valid unique-time bilirubin mg/dL; standard thresholds; unknown0..4.',
         'renal':'Valid creatinine lower score. Urine duration and complete collection cannot be certified in both sources; urine remains0..4 in BOTH adapters, never treated as normal.',
         'respiration':'Same-time unique arterial-category PaO2/FiO2 observation pairing, not specimen identity equivalence. PF<300 gives lower2, <400 lower1. Ventilation unaligned: possible3/4 retained, never infer off ventilation from missing record.',
         'cardiovascular':'Valid recorded MAP<70 lower1. Infusion entry timestamp does not certify clinical dose interval; dose contribution stays0..4 in BOTH sources. No substitution of MIMIC store/start time for eICU infusionoffset.',
         'cns':'Unique valid numeric total or complete same-time numeric Eyes+Verbal+Motor; conflicting, intubation-unscorable and invalid records unknown. No fill-forward or default15.'},
       'upper_bound':'Conservative total upper24 at each hour, including unrecorded domains. This deliberately overbounds possible score paths.',
       't0':'max(first qualifying SOFA hour,AB,culture,ICU). Accept event only if first-possible and first-certain SOFA completions yield same t0.',
       'analysis_unit':'Earliest identified qualifying t0 per hospital admission; any earlier/tied potentially qualifying unresolved event in another stay or earlier unknown order blocks selection. First event is selected before feature missingness filtering.',
       'eligibility':'Age>=18 (eICU >89 is adult without exact imputation), known hospital discharge status, 0<=t0<ICU discharge and hospital discharge. No death-time equality claim.',
       'features':'Fixed14. Retrospective observed-time, 6h vitals and24h labs, unique-time nonconflicting values; no entry<=t0 claim. Entry/revision times retained in evidence. Median; source priorities explicit; technical ranges frozen from existing config.',
       'training':'Patient-disjoint70/15/15; train-only limits; observed Gower k2..6 plus10 MICE-PMM draws and resampling; no eICU training/k selection.',
       'limitations':['Selective measurement-complete/identifiable population, not all sepsis patients.','Unidentified urine/dose/ventilation retained as uncertainty, not semantic equivalence.','No real-time deployment claim.','Earlier clinically unrecorded infection events cannot be recovered.']}
    f=OUT/'common_protocol_v2.json'
    if f.exists():assert json.loads(f.read_text(encoding='utf-8'))==p
    else:dump(f.name,p)

def metadata(source):
    if source=='mimic':
        x=frame(DATA/'common_stay_metadata_local.parquet')
        x['admission_id']=x.hadm_id.astype(str);x['person_id']=x.subject_id.astype(str)
        x['icu_start_hospital_min']=(x.intime-x.admittime).dt.total_seconds()/60
        x['icu_end_min']=(x.outtime-x.intime).dt.total_seconds()/60
        x['hospital_end_min']=(x.dischtime-x.intime).dt.total_seconds()/60
        x['known_discharge']=x.hospital_expire_flag.isin([0,1]);x['hospitalid']='MIMIC'
    else:
        x=frame(CACHE/'patient_local.parquet').rename(columns={'patientunitstayid':'stay_id','patienthealthsystemstayid':'admission_id','uniquepid':'person_id'})
        x['icu_start_hospital_min']=-pd.to_numeric(x.hospitaladmitoffset)
        x['icu_end_min']=pd.to_numeric(x.unitdischargeoffset);x['hospital_end_min']=pd.to_numeric(x.hospitaldischargeoffset)
        age=x.age.astype(str).str.strip();x['age']=pd.to_numeric(age,errors='coerce');x.loc[age.isin(['> 89','>89']),'age']=90
        x['known_discharge']=x.hospitaldischargestatus.isin(['Alive','Expired'])
    x['clinical_static_eligible']=x.age.ge(18)&x.known_discharge&x.icu_end_min.gt(0)&x.hospital_end_min.gt(0)
    return x.drop_duplicates('stay_id').set_index('stay_id',drop=False)

def orders(source,meta):
    from start_common_definition_20261029 import pair_events
    # Restrict culture timestamps BEFORE pairing, so an out-of-admission row
    # cannot hide a later eligible culture in the same matching window.
    if source=='mimic':
        pr=frame(DATA/'mimic_order_provenance_local.parquet')
        norm=pd.DataFrame({'order_key':pr.order_key,'person_id':pr.subject_id.astype(str),'admission_id':pr.hadm_id.astype(str),'stay_id':pr.stay_id,'antibiotic_seconds':pr.antibiotic_seconds.astype(float)})
        mc=frame(DATA/'mimic_cultures_local.parquet')
        cultures=pd.DataFrame({'admission_id':mc.hadm_id.astype(str),'culture_seconds':mc.culture_seconds.astype(float)})
    else:
        pr=frame(ROOT/'outputs/infection_mapping_refined_20261026/order_evidence_local.parquet')
        pr=pr[pr.reference_syntax_candidate]
        norm=pd.DataFrame({'order_key':pr.medicationid.astype(str),'person_id':pr.patientunitstayid.map(meta.person_id),'admission_id':pr.patienthealthsystemstayid,'stay_id':pr.patientunitstayid,'antibiotic_seconds':pr.hospital_relative_min.astype(float)*60})
        mc=frame(ROOT/'outputs/infection_candidates_20261026/culture_evidence_local.parquet')
        cultures=pd.DataFrame({'admission_id':mc.patienthealthsystemstayid,'culture_seconds':mc.hospital_relative_min.astype(float)*60})
    end=meta.assign(hospital_end=(meta.icu_start_hospital_min+meta.hospital_end_min)*60).groupby('admission_id').hospital_end.min()
    cultures=cultures[cultures.culture_seconds.ge(0)&cultures.culture_seconds.le(cultures.admission_id.map(end))]
    ev=pair_events(norm,cultures)
    # Extract only literal audited route sets; do not execute the historical script.
    tr=ast.parse((ROOT/'refine_eicu_infection_events.py').read_text(encoding='utf-8'))
    routes=set()
    for n in tr.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['iv','enteral'] for t in n.targets):routes |= ast.literal_eval(n.value)
    routes|={'IM','INTRAMUSCULAR','IP','INTRAPERITONEAL','I-PERITONEAL'}
    local={'PR','RECTAL','NEB','IRRIGATION','IRR','IRRG','IRRIGATE','INTRANASAL','EACH NARE','CATHETER LOCK','LOCK','IH','INHALATION','INTRAVITREAL','IT','IN','IR','PL'}
    if source=='mimic':
        pr=frame(DATA/'mimic_order_provenance_local.parquet');lc=frame(DATA/'mimic_order_lifecycle_local.parquet')
        keys=[c for c in pr if c in lc]
        lc=lc.drop_duplicates()
        assert not lc.duplicated(keys).any(),'Conflicting POE lifecycle requires record audit'
        z=pr.merge(lc,on=keys,how='left',validate='many_to_one').set_index('order_key').loc[ev.order_key].reset_index()
        r=z.route.fillna('').str.strip().str.upper()
        excluded=z.stoptime.lt(z.starttime)|z.discontinue_order_time.le(z.starttime)|r.isin(local)|z.drug_type.eq('BASE')
        life=z.ordertime.notna()&z.order_status.isin(['Inactive','Active'])&z.transaction_type.isin(['New','Change'])
        # An absent linked discontinue row cannot establish its timing.
        life&=z.discontinued_by_poe_id.isna()|z.discontinue_order_time.notna()
        known=life&z.stoptime.gt(z.starttime)&r.isin(routes)&~excluded
        why=np.select([excluded,~r.isin(routes),~life,~z.stoptime.gt(z.starttime)],['excluded_route_or_lifecycle','unknown_route','unknown_lifecycle','unknown_or_zero_duration'],default='accepted')
    else:
        z=frame(ROOT/'outputs/infection_mapping_refined_20261026/order_evidence_local.parquet');z['order_key']=z.medicationid.astype(str)
        z=z.set_index('order_key').loc[ev.order_key].reset_index();r=z.routeadmin.fillna('').str.strip().str.upper()
        cancel=z.drugordercancelled.fillna('').str.strip().str.lower()
        start=pd.to_numeric(z.drugstartoffset);stop=pd.to_numeric(z.drugstopoffset,errors='coerce')
        excluded=cancel.isin(['yes','true','1'])|stop.lt(start)|r.isin(local)
        life=cancel.isin(['no','false','0'])
        known=life&stop.gt(start)&r.isin(routes)&~excluded
        why=np.select([excluded,~r.isin(routes),~life,~stop.gt(start)],['excluded_route_or_lifecycle','unknown_route','unknown_lifecycle','unknown_or_zero_duration'],default='accepted')
    ev['order_state']=np.where(excluded,'excluded',np.where(known,'accepted','unresolved'));ev['order_reason']=why
    ev['antibiotic_min']=ev.antibiotic_seconds/60-ev.stay_id.map(meta.icu_start_hospital_min)
    ev['culture_min']=ev.culture_seconds/60-ev.stay_id.map(meta.icu_start_hospital_min)
    ev['suspected_min']=ev[['antibiotic_min','culture_min']].min(axis=1)
    # Invalid hospital culture times are outside the common event definition.
    valid=(ev.culture_seconds.ge(0)&ev.culture_min.le(ev.stay_id.map(meta.hospital_end_min))&ev.antibiotic_seconds.ge(0)&ev.antibiotic_min.ge(0)&ev.antibiotic_min.lt(ev.stay_id.map(meta.icu_end_min)))
    ev.loc[~valid,['order_state','order_reason']]=['excluded','outside_hospital_or_icu']
    ev.to_parquet(DATA/f'{source}_common_order_audit_local.parquet',index=False)
    ev.groupby(['order_state','order_reason']).agg(events=('order_key','size'),stays=('stay_id','nunique')).reset_index().to_csv(OUT/f'{source}_order_counts.csv',index=False)
    return ev

def observations(source,meta,ids):
    parts=[]
    def add(d,domain,t='offset_min',s='lower'):
        z=d[['stay_id',t,s]].rename(columns={t:'offset_min',s:'lower'}).copy();z['domain']=domain;parts.append(z)
    labs=frame(DATA/f'{source}_common_lab_evidence_local.parquet');labs=labs[labs.usable&labs.stay_id.isin(ids)]
    for var,domain in [('platelet','coagulation'),('bilirubin','liver'),('creatinine','renal')]:add(labs[labs.variable.eq(var)],domain)
    if source=='mimic':
        ch=frame(DATA/'common_sofa_chart_compact_local.parquet');ch=ch[ch.stay_id.isin(ids)].copy()
        for domain in ['cardiovascular','cns']:add(ch[ch.domain.eq(domain)],domain)
        g=pd.DataFrame(columns=['stay_id','offset_min','valid','component','value'])
        gas=frame(DATA/'common_sofa_gas_local.parquet');gas=gas[gas.stay_id.isin(ids)]
        keys=['stay_id','offset_min'];p=gas[gas.itemid.eq(50821)].copy();f=gas[gas.itemid.eq(50816)].copy();sp=gas[gas.itemid.eq(52033)].copy()
        p['valid']=p.valuenum.between(.01,1000)&p.valueuom.fillna('').str.lower().eq('mmhg')
        fv=pd.to_numeric(f.valuenum,errors='coerce');f['fraction']=np.where(fv.gt(.2)&fv.le(1),fv,np.where(fv.gt(20)&fv.le(100),fv/100,np.nan));f['valid']=f.fraction.notna()
        sp['valid']=sp.value.eq('ART.')
        pg=p.groupby(keys).agg(pao2=('valuenum','first'),pn=('valuenum','nunique'),pok=('valid','all'))
        fg=f.groupby(keys).agg(fio2=('fraction','first'),fn=('fraction','nunique'),fok=('valid','all'))
        sg=sp.groupby(keys).agg(arterial=('valid','all'))
        paired=pg.join(fg,how='inner').join(sg,how='inner');paired=paired[paired.pok&paired.fok&paired.pn.eq(1)&paired.fn.eq(1)&paired.arterial]
        pf=paired.pao2/paired.fio2;paired['lower']=np.where(pf<300,2,np.where(pf<400,1,0))
        add(paired.reset_index(),'respiration')
    else:
        v=frame(ROOT/'outputs/eicu_monitor_inputs_20261028/expanded_vital_evidence_local.parquet').rename(columns={'patientunitstayid':'stay_id'})
        mp=v[v.stay_id.isin(ids)&v.feature.eq('map')&v.technical_range_pass&v.same_source_time_distinct_values.eq(1)].copy()
        mp['lower']=mp.value.lt(70).astype(int);add(mp,'cardiovascular','chart_min')
        g=frame(ROOT/'outputs/sofa_observation_audit_20261026/gcs_evidence_local.parquet').rename(columns={'patientunitstayid':'stay_id','nursingchartcelltypevalname':'component'})
        g=g[g.stay_id.isin(ids)].copy();g['valid']=g.numeric_valid&~g.medication_unscorable&g.n_distinct.eq(1)&g.value.mod(1).eq(0)
        gas=frame(ROOT/'outputs/eicu_pairing_audit_20261028/bloodgas_exact_time_pair_evidence_local.parquet')
        gas=gas[gas.unique_numeric_unit_pair&gas.pao2_value.between(.01,1000)].copy()
        # Cross-ICU lab linkage remains within the same hospital admission.
        gas['admission_id']=gas.patientunitstayid.map(meta.admission_id);gas['hospital_min']=gas.offset_min+gas.patientunitstayid.map(meta.icu_start_hospital_min)
        gas=gas.merge(meta.loc[list(ids),['stay_id','admission_id','icu_start_hospital_min']].reset_index(drop=True),on='admission_id')
        gas['offset_min']=gas.hospital_min-gas.icu_start_hospital_min
        pf=gas.pao2_value/gas.fio2_fraction;gas['lower']=np.where(pf<300,2,np.where(pf<400,1,0));add(gas,'respiration')
    rows=[]
    for (stay,t),a in g.groupby(['stay_id','offset_min']):
        if not a.valid.all():continue
        if a.groupby('component').value.nunique().gt(1).any():continue
        vals=a.groupby('component').value.first().to_dict();total=vals.get('GCS Total',np.nan)
        comp=sum(vals[k] for k in ['Eyes','Verbal','Motor']) if all(k in vals for k in ['Eyes','Verbal','Motor']) else np.nan
        if np.isfinite(total) and np.isfinite(comp) and total!=comp:continue
        value=total if np.isfinite(total) else comp
        if not np.isfinite(value) or not 3<=value<=15:continue
        sc=4 if value<6 else 3 if value<10 else 2 if value<13 else 1 if value<15 else 0
        rows.append((stay,t,sc))
    add(pd.DataFrame(rows,columns=['stay_id','offset_min','lower']),'cns')
    obs=pd.concat(parts,ignore_index=True)
    obs=obs[obs.offset_min.ge(-2880)&obs.offset_min.le(obs.stay_id.map(meta.icu_end_min))&(obs.offset_min+obs.stay_id.map(meta.icu_start_hospital_min)).ge(0)]
    obs=obs.groupby(['stay_id','offset_min','domain'],as_index=False).lower.max()
    obs.to_parquet(DATA/f'{source}_common_component_lower_local.parquet',index=False)
    obs.groupby('domain').agg(observations=('stay_id','size'),stays=('stay_id','nunique'),positive_lower=('lower',lambda s:int(s.gt(0).sum()))).reset_index().to_csv(OUT/f'{source}_domain_coverage.csv',index=False)
    return obs

def event_scores(source,obs,ev,meta):
    groups={s:g for s,g in obs.groupby('stay_id')};rows=[];hours_n=0
    for n,(stay,e) in enumerate(ev.groupby('stay_id')):
        end=float(meta.at[stay,'icu_end_min']);ext=np.arange(-48,math.floor(end/60)+1)*60;hours=ext[24:]
        x=np.zeros((len(ext),6),np.int8);g=groups.get(stay)
        if g is not None:
            for j,d in enumerate(DOMAINS):
                z=g[g.domain.eq(d)];ix=np.searchsorted(ext,z.offset_min,side='left');good=(ix<len(ext))&(z.offset_min.to_numpy()>ext[0])
                np.maximum.at(x[:,j],ix[good],z.lower.to_numpy(np.int8)[good])
        low=pd.DataFrame(x).rolling(24,min_periods=1).max().to_numpy(np.int8)[24:].sum(axis=1);upper=np.repeat(24,len(hours));hours_n+=len(hours)
        for r in e.itertuples():
            if r.order_state=='excluded':v=(None,None,None,None,'excluded_order')
            else:v=t0_bounds(hours,low,upper,r.suspected_min-2880,r.suspected_min+1440,r.antibiotic_min,r.culture_min)
            rows.append((r.order_key,*v))
        if n and n%10000==0:print(source,'scored',n,flush=True)
    d=pd.DataFrame(rows,columns=['order_key','first_possible_hour','first_certain_hour','t0_lower','t0_upper','score_status'])
    result=ev.merge(d,on='order_key',validate='one_to_one')
    result['identified']=result.order_state.eq('accepted')&result.score_status.eq('t0_identified_conditional_on_event')
    result['possible_hospital_t0']=result.t0_lower+result.stay_id.map(meta.icu_start_hospital_min)
    result['certain_hospital_t0']=result.t0_upper+result.stay_id.map(meta.icu_start_hospital_min)
    result['possible_eligible']=result.stay_id.map(meta.clinical_static_eligible)&result.t0_lower.notna()&result.t0_lower.lt(result.stay_id.map(meta.icu_end_min))&result.t0_lower.lt(result.stay_id.map(meta.hospital_end_min))
    result['identified'] &=result.possible_eligible&result.t0_upper.lt(result.stay_id.map(meta.icu_end_min))&result.t0_upper.lt(result.stay_id.map(meta.hospital_end_min))
    result.to_parquet(DATA/f'{source}_common_event_bounds_local.parquet',index=False)
    result.groupby(['order_state','score_status']).agg(events=('order_key','size'),stays=('stay_id','nunique')).reset_index().to_csv(OUT/f'{source}_event_counts.csv',index=False)
    selected=[];audit=[]
    for admission,g in result.groupby('admission_id'):
        cert=g[g.identified].sort_values(['certain_hospital_t0','stay_id','order_key'])
        possible=g[g.possible_eligible]
        if cert.empty:reason='no_identified_event'
        else:
            a=cert.iloc[0]; earlier=possible[possible.possible_hospital_t0.lt(a.certain_hospital_t0)]
            tied_other=possible[possible.possible_hospital_t0.eq(a.certain_hospital_t0)&possible.stay_id.ne(a.stay_id)]
            if len(earlier) or len(tied_other):reason='earlier_or_tied_event_unresolved'
            else:selected.append(a.to_dict());reason='identified_first_hospital_t0'
        audit.append((admission,reason,len(g),len(cert)))
    cohort=pd.DataFrame(selected)
    if len(cohort):
        cohort['t0_min']=cohort.t0_upper
        for col in ['age','hospitalid','icu_start_hospital_min','icu_end_min','hospital_end_min']:cohort[col]=cohort.stay_id.map(meta[col])
        assert cohort.admission_id.is_unique and cohort.stay_id.is_unique
    cohort.to_parquet(DATA/f'{source}_common_identified_cohort_local.parquet',index=False)
    pd.DataFrame(audit,columns=['admission_id','status','candidate_events','identified_events']).to_parquet(DATA/f'{source}_hospital_selection_audit_local.parquet',index=False)
    return {'candidate_stays':ev.stay_id.nunique(),'candidate_events':len(ev),'hours_scored':hours_n,'identified_events':int(result.identified.sum()),'selected_admissions':len(cohort),'selected_people':cohort.person_id.nunique() if len(cohort) else 0,'selection_counts':pd.Series([x[1] for x in audit]).value_counts().to_dict()}

def main():
    protocol();result={'scope':'common partially identified cohort; not complete scalar SOFA','model_fitted':False,'external_validation_complete':False}
    for source in ['eicu','mimic']:
        meta=metadata(source);ev=orders(source,meta);ids=set(ev.stay_id)
        print(source,'orders reviewed',len(ev),flush=True)
        obs=observations(source,meta,ids);print(source,'source observations',len(obs),flush=True)
        result[source]=event_scores(source,obs,ev,meta);dump('cohort_summary.json',result)
    print(json.dumps(result,indent=2,default=str),flush=True)

if __name__=='__main__':main()
