"""Pre-specified comparison of the 14-feature and historical 16-feature sets.

Outcome-blinded; equal-feature missing-aware Gower is used only for this
variable-set diagnostic. No k is selected from outcomes.
"""
from pathlib import Path
import importlib, json
import numpy as np, pandas as pd
from sklearn.metrics import silhouette_samples

ROOT=Path(__file__).resolve().parents[1]
v=importlib.import_module('26_domain_balanced_v7')

def run(raw, feats, tag, tables):
    eq={f:[f] for f in feats}; mask=raw[feats].notna().sum(axis=1).ge(max(6, int(np.ceil(.5*len(feats)))))
    mask &= raw[['heart_rate','resp_rate']].notna().any(axis=1)
    work=raw.loc[mask, ['subject_id','stay_id','hadm_id','split','age','gender']+feats].reset_index(drop=True)
    bad=v.incompatible_positions(work,feats,eq)
    work=work.drop(index=bad).reset_index(drop=True)
    lim=v.domain_limits(work,eq);D=v.missing_domain_gower(work,feats,eq,lim);rows=[]
    for k in [2,3,4,5,6]:
        fit,_=v.fit_parallel(D,k,{'pam_restarts':5,'max_iter':200},20261010+90000+len(feats)*100+k)
        sil=silhouette_samples(D,fit['labels'],metric='precomputed');sizes=np.bincount(fit['labels'],minlength=k)
        rows.append({'variable_set':tag,'n_eligible':len(work),'n_pairwise_excluded':len(bad),'p':len(feats),'k':k,'silhouette_mean':float(sil.mean()),'negative_fraction':float((sil<0).mean()),'min_cluster_fraction':float(sizes.min()/len(work))})
    return rows

def main():
    rel='data/v2_20261009/20261009_203156/observation_review/derived_v3/clustering_v5/clustering_v6/clustering_v7/20261010_125027/eligible_train_unimputed.parquet'
    raw=pd.read_parquet(ROOT/rel)
    primary=['heart_rate','map','resp_rate','temp_c','spo2','lactate','wbc','platelet','hemoglobin','creatinine','sodium','potassium','bicarbonate','glucose']
    expanded=primary+['gcs_min','bilirubin']
    result=pd.DataFrame(run(raw,expanded,'historical16',ROOT/'outputs/tables/clustering_v7/20261010_125027'))
    old=pd.read_csv(ROOT/'outputs/tables/clustering_v7/20261010_125027/equal_feature_gower_metrics.csv').rename(columns={'analysis':'variable_set','n':'n_eligible'})
    old['negative_fraction']=np.nan;old=old[['variable_set','n_eligible','k','silhouette_mean','negative_fraction','min_cluster_fraction']];old['p']=14;old['n_pairwise_excluded']=np.nan
    out=pd.concat([old,result[old.columns]],ignore_index=True);out.to_csv(ROOT/'outputs/tables/clustering_v7/20261010_125027/variable_set_comparison.csv',index=False);print(out.to_string(index=False))

if __name__=='__main__': main()
