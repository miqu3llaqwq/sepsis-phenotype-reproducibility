"""Independent interval/selection and delivered feature checks on real records."""
import itertools,json,math
import numpy as np
import pandas as pd
from build_common_identified_20261030 import ROOT,WORK,OUT,DATA,frame,metadata,dump
from common_sofa_lab_bounds_20261030 import t0_bounds
from common_features_20261030 import FEATURES

def interval_exhaustive():
    cases=0
    for low in itertools.product([0,1,2],repeat=3):
        for high in itertools.product([1,2,3],repeat=3):
            if any(a>b for a,b in zip(low,high)):continue
            for ab,cu in [(0,0),(60,0),(0,120),(180,0)]:
                hours=np.array([0.,60.,120.]);r=t0_bounds(hours,np.array(low),np.array(high),0,120,ab,cu)
                values=[]
                for path in itertools.product(*[range(a,b+1) for a,b in zip(low,high)]):
                    hit=[h for h,s in zip(hours,path) if s>=2]
                    values.append(max(hit[0],ab,cu,0) if hit else None)
                if r[-1]=='t0_identified_conditional_on_event':assert set(values)=={r[2]}
                cases+=1
    return cases

def verify_source(source):
    e=frame(DATA/f'{source}_common_event_bounds_local.parquet');o=frame(DATA/f'{source}_common_component_lower_local.parquet');m=metadata(source)
    ids=e.stay_id.unique()
    if source=='mimic':ids=np.random.default_rng(20261030).choice(ids,min(150,len(ids)),replace=False)
    grouped={s:g for s,g in o[o.stay_id.isin(ids)].groupby('stay_id')};checked=0;windows=0
    for stay,g in e[e.stay_id.isin(ids)].groupby('stay_id'):
        obs=grouped.get(stay,o.iloc[:0]);h=np.arange(-24,math.floor(m.at[stay,'icu_end_min']/60)+1)*60
        low=[]
        for hour in h:
            z=obs[obs.offset_min.gt(hour-1440)&obs.offset_min.le(hour)]
            low.append(z.groupby('domain').lower.max().sum());windows+=1
        low=np.array(low)
        for r in g.itertuples():
            if r.order_state=='excluded':assert r.score_status=='excluded_order';continue
            eligible=(h>=r.suspected_min-2880)&(h<=r.suspected_min+1440)
            possible=h[eligible];certain=h[eligible&(low>=2)]
            if not len(possible):assert r.score_status=='no_hour_in_infection_window'
            else:
                assert r.first_possible_hour==possible[0]
                assert r.t0_lower==max(possible[0],r.antibiotic_min,r.culture_min,0)
                if len(certain):
                    assert r.first_certain_hour==certain[0]
                    assert r.t0_upper==max(certain[0],r.antibiotic_min,r.culture_min,0)
                else:assert pd.isna(r.first_certain_hour)
            checked+=1
    c=frame(DATA/f'{source}_common_identified_cohort_local.parquet');assert c.stay_id.is_unique and c.admission_id.is_unique
    assert c.t0_min.eq(c.t0_lower).all() and c.t0_min.eq(c.t0_upper).all() and c.order_state.eq('accepted').all()
    eligible_events=e[e.possible_eligible]
    admission_groups={k:g for k,g in eligible_events.groupby('admission_id')}
    for r in c.itertuples():
        g=admission_groups[r.admission_id]
        assert not g.possible_hospital_t0.lt(r.certain_hospital_t0).any()
        assert not (g.possible_hospital_t0.eq(r.certain_hospital_t0)&g.stay_id.ne(r.stay_id)).any()
    f=frame(DATA/f'{source}_common_features_local.parquet');p=frame(DATA/f'{source}_common_feature_provenance_local.parquet')
    assert set(f.stay_id)==set(c.stay_id)
    grouped=p.groupby(['stay_id','feature']).value.median()
    count=0
    for r in f.itertuples():
        for feature in FEATURES:
            v=getattr(r,feature);expected=grouped.get((r.stay_id,feature),np.nan)
            assert (pd.isna(v) and pd.isna(expected)) or np.isclose(v,expected);count+=1
    assert p.offset_min.le(p.t0_min).all()
    assert p.offset_min.ge(p.t0_min-np.where(p.feature.isin(FEATURES[:5]),360,1440)).all()
    assert (p.offset_min+p.stay_id.map(m.icu_start_hospital_min)).ge(0).all()
    return {'candidate_events_independently_checked':checked,'hour_windows_checked':windows,'all_selected_admissions_checked':len(c),'feature_cells_recomputed':count,'t0_not_assumed_first_sofa_exact':True}

if __name__=='__main__':
    result={'status':'pass','scope':'six-domain partial identification, event selection and actual feature values; NOT proof of scalar SOFA equivalence','exhaustive_interval_cases':interval_exhaustive()}
    for source in ['eicu','mimic']:
        result[source]=verify_source(source);print(source,result[source],flush=True)
    dump('independent_common_cohort_qa.json',result)
