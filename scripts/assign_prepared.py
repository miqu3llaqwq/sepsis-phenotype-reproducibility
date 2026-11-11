"""Assign only a separately audited prepared cohort; no clinical derivation."""
from pathlib import Path
import sys,argparse,json,hashlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sepsis_repro import assign

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True,type=Path);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--out-dir',required=True,type=Path)
    a=p.parse_args();bundle=a.bundle.resolve();manifest=json.loads((bundle/'bundle_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['sha256'].items():
        file=(bundle/name).resolve()
        if not file.is_relative_to(bundle):raise ValueError('Manifest path escapes frozen bundle')
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen artifact hash mismatch: '+name)
    cfg=json.loads((bundle/'core14_primary_transform.json').read_text(encoding='utf-8'))
    x=pd.read_parquet(a.features) if a.features.suffix=='.parquet' else pd.read_csv(a.features)
    missing=set(cfg['features']+['feature_eligible'])-set(x.columns)
    if missing:raise ValueError('Missing fields: '+str(sorted(missing)))
    if x.feature_eligible.isna().any() or not x.feature_eligible.isin([True,False,0,1]).all():raise ValueError('feature_eligible must contain booleans, not missing values or strings')
    for f in cfg['features']:
        x[f]=pd.to_numeric(x[f],errors='raise')
        if np.isinf(x[f].to_numpy(float)).any():raise ValueError('Infinite feature: '+f)
    rows=x.loc[x.feature_eligible.astype(bool)].copy();rows['input_row']=rows.index
    if a.out_dir.exists() and any(a.out_dir.iterdir()):raise ValueError('Use a new empty output directory to preserve prior runs')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    result={'prepared':len(x),'eligible':len(rows),'refitted':False,'k_selected':None,'clinical_semantics_checked_here':False,'k':{}}
    for k in [2,3,4,5,6]:
        med=pd.read_parquet(bundle/f'equal_feature_k{k}_medoids.parquet')
        if len(med)!=k:raise ValueError('Unexpected number of medoids')
        labels,ok,dist=assign(rows,med,cfg)
        pd.DataFrame({'input_row':rows.input_row.to_numpy(),'cluster':labels,'all_distances_computable':ok}).to_csv(a.out_dir/f'assignments_k{k}.csv',index=False)
        result['k'][str(k)]={'assigned':int(ok.sum())}
    (a.out_dir/'run_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
