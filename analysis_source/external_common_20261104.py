import os
from pathlib import Path
import json, shutil
from collections import Counter
import numpy as np
import pandas as pd
import build_common_identified_20261030 as b
import common_features_20261030 as cf
import verify_common_core_20261030 as verify
from frozen_external_assignment_20261027 import assign
from start_common_definition_20261029 import sha
root=b.ROOT
original=b.DATA
frozen=b.WORK/'frozen_20261030'
out=root/'outputs/external_common_20261104'
local=root/'work/external_common_20261104'
out.mkdir(exist_ok=True);local.mkdir(exist_ok=True,parents=True)
def save(name,data):
    (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
manifest=json.loads((frozen/'bundle_manifest.json').read_text())
assert all(sha(frozen/p)==h for p,h in manifest['sha256'].items())
cfg=json.loads((frozen/'core14_primary_transform.json').read_text()); fs=cfg['features']
events=b.frame(original/'eicu_common_order_audit_local.parquet')
obs=b.frame(original/'eicu_common_component_lower_local.parquet')
target=events[events.order_reason.eq('unknown_route')].order_key
raw=(Path(os.environ['EICU_DATA_DIR']) / 'medication.csv.gz')
counts=Counter();rows=[]
for d in pd.read_csv(raw,usecols=['medicationid','routeadmin','drugname','drugivadmixture','drugordercancelled','drugstartoffset','drugstopoffset'],chunksize=250000):
    r=d.routeadmin.fillna('').str.strip().str.upper();counts.update(r.value_counts().to_dict())
    rows.append(d[d.medicationid.isin(target.astype(int))])
orders=pd.concat(rows);orders['order_key']=orders.medicationid.astype(str)
orders=orders.set_index('order_key').loc[target]
aliases={'INTRAVENOU':'intravenous','ORAL GASTR':'enteral'}
labels=orders.routeadmin.str.strip().str.upper()
assert set(labels)==set(aliases) and orders.index.is_unique
matches={a:{k:v for k,v in counts.items() if k.startswith(a)} for a in aliases}
assert all(k.startswith('INTRAVENOUS') for k in matches['INTRAVENOU'] if k!='INTRAVENOU')
assert set(matches['ORAL GASTR'])=={'ORAL GASTR','ORAL GASTRIC TUBE'}
save('route_evidence.json',{'source_sha256':sha(raw),'documentation':'https://eicu.mit.edu/eicutables/medication/','status':'Inferred route family from complete raw picklist prefixes. Exploratory sensitivity, no certified local codebook.','full_raw_prefix_frequencies':matches,'paired_label_counts':labels.value_counts().to_dict(),'iv_admixture_yes_with_iv_label':int((labels.eq('INTRAVENOU')&orders.drugivadmixture.eq('Yes')).sum()),'dictionary':aliases})
clean=orders.drugordercancelled.str.strip().str.lower().eq('no')&orders.drugstopoffset.gt(orders.drugstartoffset)
revised=events.copy().set_index('order_key');revised.loc[orders.index[clean],['order_state','order_reason']]=['accepted','mapped_route_and_valid_lifecycle']
revised=revised.reset_index();assert set(revised.order_key)==set(events.order_key)
b.DATA=cf.DATA=verify.DATA=local;b.OUT=cf.OUT=out
shutil.copy2(original/'eicu_common_component_lower_local.parquet',local/'eicu_common_component_lower_local.parquet')
revised.to_parquet(local/'eicu_common_order_audit_local.parquet',index=False)
cohort=b.event_scores('eicu',obs,revised,b.metadata('eicu')); feat=cf.extract('eicu')
qa=verify.verify_source('eicu')
train=pd.read_parquet(frozen/'training_reference.parquet')
tables=[];distrows=[];result={}
cohorts={}
for variant,folder in [('frozen_routes',original),('route_sensitivity',local)]:
    x=b.frame(folder/'eicu_common_features_local.parquet');x=x[x.feature_eligible].reset_index(drop=True);cohorts[variant]=x
    local_labels=x[['stay_id','admission_id','person_id','hospitalid','t0_min']].copy()
    for k in range(2,7):
        med=pd.read_parquet(frozen/f'equal_feature_k{k}_medoids.parquet')
        labels,ok,dist=assign(x,med,cfg)
        scalar=np.full_like(dist,np.nan,dtype=float)
        for i,(_,r) in enumerate(x.iterrows()):
            for j,(_,m) in enumerate(med.iterrows()):
                v=[]
                for f in fs:
                    if pd.notna(r[f]) and pd.notna(m[f]):
                        lo,hi=cfg['limits'][f];v.append(abs(np.clip(r[f],lo,hi)-np.clip(m[f],lo,hi))/(hi-lo))
                if v:scalar[i,j]=np.mean(v)
        assert np.allclose(scalar,dist,atol=2e-7,equal_nan=True) and np.array_equal(scalar[ok].argmin(axis=1)+1,labels[ok])
        local_labels[f'k{k}']=labels
        for j in range(k):
            tables.append(dict(variant=variant,k=k,cluster=j+1,eligible=len(x),assigned=int(ok.sum()),external_n=int((labels==j+1).sum()),training_n=int((pd.read_parquet(b.WORK/f'models_20261030/observed_equal/k{k}_labels.parquet').cluster==j+1).sum()),median_nearest_distance=float(np.median(np.min(dist[ok],axis=1)))))
    local_labels.to_parquet(local/f'{variant}_assignments_local.parquet',index=False)
    result[variant]={'eligible':len(x),'people':x.person_id.nunique(),'hospital_sizes':x.hospitalid.value_counts().tolist()}
    for f in fs:
        s=x[f];t=train[f];lo,hi=cfg['limits'][f]
        distrows.append(dict(variant=variant,feature=f,external_missing=s.isna().mean(),training_missing=t.isna().mean(),external_median=s.median(),training_median=t.median(),external_below_P1=int((s<lo).sum()),external_above_P99=int((s>hi).sum())))
a,c=cohorts.values();over=a.merge(c,on='admission_id',suffixes=('_a','_c'))
save('results.json',{'status':'executed_independently_verified','model':'common_identified_v2_20261030','cohorts':result,'sensitivity_flow':cohort,'sensitivity_features':feat,'additional_orders_meeting_all_conditions':int(clean.sum()),'shared_eligible_admissions':len(over),'shared_admissions_changed_t0_or_stay':int((over.stay_id_a.ne(over.stay_id_c)|over.t0_min_a.ne(over.t0_min_c)).sum()),'sensitivity_qa':qa,'frozen_hashes_verified':len(manifest['sha256']),'clinical_outcome_tested':False})
pd.DataFrame(tables).to_csv(out/'assignments.csv',index=False);pd.DataFrame(distrows).to_csv(out/'features.csv',index=False)
print((out/'results.json').read_text(encoding='utf-8'));print(pd.DataFrame(tables).query('k==2').to_string(index=False))
