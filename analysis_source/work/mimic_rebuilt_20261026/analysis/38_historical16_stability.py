"""Equal-feature Gower stability for the pre-specified historical 16-feature set."""
from pathlib import Path
import importlib, json
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parents[1];v=importlib.import_module('26_domain_balanced_v7')
def main():
    rel='data/v2_20261009/20261009_203156/observation_review/derived_v3/clustering_v5/clustering_v6/clustering_v7/20261010_125027/eligible_train_unimputed.parquet';raw=pd.read_parquet(ROOT/rel)
    feats=['heart_rate','map','resp_rate','temp_c','spo2','lactate','wbc','platelet','hemoglobin','creatinine','sodium','potassium','bicarbonate','glucose','gcs_min','bilirubin'];eq={f:[f] for f in feats};mask=raw[feats].notna().sum(axis=1).ge(8)&raw[['heart_rate','resp_rate']].notna().any(axis=1);raw=raw.loc[mask].reset_index(drop=True);bad=v.incompatible_positions(raw,feats,eq);raw=raw.drop(index=bad).reset_index(drop=True);lim=v.domain_limits(raw,eq);D=v.missing_domain_gower(raw,feats,eq,lim);fits={};rows=[];cfg={'pam_restarts':5,'max_iter':200}
    for k in [2,3,4,5,6]:fits[k],_=v.fit_parallel(D,k,cfg,20261010+91000+k)
    rng=np.random.default_rng(20261010+92000)
    for rep in range(10):
        ids=np.sort(rng.choice(len(raw),int(.8*len(raw)),replace=False));sub=raw.iloc[ids].reset_index(drop=True);slim=v.domain_limits(sub,eq);Ds=v.missing_domain_gower(sub,feats,eq,slim)
        for k in [2,3,4,5,6]:
            fit,_=v.fit_parallel(Ds,k,cfg,20261010+93000+rep*100+k);assigned=np.argmin(v.missing_domain_gower_rect(raw,sub.iloc[fit['medoids']],feats,eq,slim),axis=1);_,mapped,_=v.m.c.aligned(fits[k]['labels'],assigned,k);rows.append({'rep':rep+1,'k':k,'ari':float(adjusted_rand_score(fits[k]['labels'],assigned)),'agreement':float(np.mean(fits[k]['labels']==mapped))})
    out=pd.DataFrame(rows);out.to_csv(ROOT/'outputs/tables/clustering_v7/20261010_125027/historical16_equal_feature_stability_detail.csv',index=False);summary=out.groupby('k').agg(ari_mean=('ari','mean'),ari_median=('ari','median'),ari_min=('ari','min'),agreement_mean=('agreement','mean'),agreement_min=('agreement','min')).reset_index();summary.to_csv(ROOT/'outputs/tables/clustering_v7/20261010_125027/historical16_equal_feature_stability_summary.csv',index=False);print(f'n={len(raw)}');print(summary.to_string(index=False))
if __name__=='__main__':main()
