"""Apply the frozen observed-value diagnostic; no imputation, fitting or k selection."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

def distance_to_medoids(rows,medoids,config):
    features=config['features'];lim=config['limits']
    a=rows[features].to_numpy(float);b=medoids[features].to_numpy(float)
    va=np.isfinite(a);vb=np.isfinite(b)
    numerator=np.zeros((len(a),len(b)),np.float32);denominator=np.zeros_like(numerator)
    for j,f in enumerate(features):
        lo,hi=lim[f]
        if not np.isfinite([lo,hi]).all() or hi<=lo:raise ValueError('Invalid frozen bounds: '+f)
        av=(np.clip(np.nan_to_num(a[:,j],nan=0.),lo,hi)-lo)/(hi-lo)
        bv=(np.clip(np.nan_to_num(b[:,j],nan=0.),lo,hi)-lo)/(hi-lo)
        common=va[:,j,None]&vb[None,:,j]
        numerator+=np.where(common,np.abs(av[:,None]-bv[None,:]),0.).astype(np.float32)
        denominator+=common
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)

def assign(rows,medoids,config):
    if not medoids.cluster.tolist()==list(range(1,len(medoids)+1)):raise ValueError('Invalid medoid/label ordering')
    distance=distance_to_medoids(rows,medoids,config)
    # Unknown distances could conceal the true nearest medoid; do not silently rank a subset.
    complete=np.isfinite(distance).all(axis=1)
    labels=np.full(len(rows),np.nan)
    labels[complete]=np.argmin(distance[complete],axis=1)+1
    return labels,complete,distance

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    config=json.loads((a.bundle/'core14_primary_transform.json').read_text(encoding='utf-8'))
    manifest=json.loads((a.bundle/'bundle_manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['sha256'].items():
        if hashlib.sha256((a.bundle/name).read_bytes()).hexdigest()!=h:raise ValueError('Frozen artifact changed: '+name)
    x=pd.read_parquet(a.features) if a.features.suffix=='.parquet' else pd.read_csv(a.features)
    times=['sofa_time_min','antibiotic_time_min','culture_time_min','icu_intime_min']
    required=['patientunitstayid']+times+config['features']
    missing=set(required)-set(x.columns)
    if missing:raise ValueError('Missing prepared fields: '+str(sorted(missing)))
    if x.patientunitstayid.isna().any() or x.patientunitstayid.duplicated().any():raise ValueError('ICU IDs missing or duplicated')
    for f in times+config['features']:
        x[f]=pd.to_numeric(x[f],errors='raise')
        if np.isinf(x[f].to_numpy(float)).any():raise ValueError('Infinite input: '+f)
    x['t0_min']=x[times].max(axis=1,skipna=False)
    good=x['t0_min'].notna()&x['t0_min'].ge(x.icu_intime_min)
    e=config['eligibility'];domains=config['domains']
    good &= x[config['features']].notna().sum(axis=1).ge(e['minimum_features_observed'])
    good &= sum(x[fs].notna().any(axis=1) for fs in domains.values()).ge(e['minimum_domains_observed'])
    for domain in e['required_domains']:good &= x[domains[domain]].notna().any(axis=1)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    result={'status':'fixed_diagnostic_assignment_complete','n_prepared':len(x),'n_feature_eligible':int(good.sum()),'refit':False,'k_selected':None,
       'scope':'Requires upstream Sepsis-3, time availability and unit audit; this script does not establish those semantics.',
       'features_sha256':hashlib.sha256(a.features.read_bytes()).hexdigest(),'bundle_manifest_sha256':hashlib.sha256((a.bundle/'bundle_manifest.json').read_bytes()).hexdigest(),'k':{}}
    use=x.loc[good].reset_index(drop=True)
    for k in config['candidate_k']:
        med=pd.read_parquet(a.bundle/f'equal_feature_k{k}_medoids.parquet')
        label,computable,d=assign(use,med,config)
        out=use[['patientunitstayid']].copy();out['cluster']=label;out['all_medoid_distances_computable']=computable
        for j in range(k):out[f'distance_cluster_{j+1}']=d[:,j]
        out.to_csv(a.out_dir/f'assignment_k{k}.csv',index=False)
        result['k'][str(k)]={'n_computable':int(computable.sum()),'cluster_counts':out.cluster.value_counts().to_dict()}
    (a.out_dir/'run_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
