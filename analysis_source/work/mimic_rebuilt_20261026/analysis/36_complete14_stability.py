"""Bootstrap stability for the pre-specified complete14 sensitivity population."""
from pathlib import Path
import importlib, json
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
v=importlib.import_module('26_domain_balanced_v7')
m=importlib.import_module('18_physiology_v5')

def main():
    p=json.loads((ROOT/'logs/clustering_v7_latest.json').read_text(encoding='utf-8')); data=Path(p['data_path']); tables=Path(p['tables_path'])
    cfg=json.loads((ROOT/'config/phenotype_primary_v1.json').read_text(encoding='utf-8')); feats=cfg['features_primary']
    raw=pd.read_parquet(data/'core14_sensitivity/eligible_train_unimputed.parquet'); raw=raw[raw[feats].notna().all(axis=1)].reset_index(drop=True)
    eq={f:[f] for f in feats}; limits=v.domain_limits(raw,eq); D=v.missing_domain_gower(raw,feats,eq,limits); rng=np.random.default_rng(cfg['clustering']['seed']+36000); rows=[]
    fits={}
    for k in cfg['clustering']['candidate_k']:
        fits[k],_=v.fit_parallel(D,k,{'pam_restarts':cfg['clustering']['pam_restarts'],'max_iter':200},cfg['clustering']['seed']+70000+k)
    for rep in range(10):
        ids=np.sort(rng.choice(len(raw),int(.8*len(raw)),replace=False)); sub=raw.iloc[ids].reset_index(drop=True); sub_limits=v.domain_limits(sub,eq); Ds=v.missing_domain_gower(sub,feats,eq,sub_limits)
        for k in cfg['clustering']['candidate_k']:
            fit,_=v.fit_parallel(Ds,k,{'pam_restarts':cfg['clustering']['pam_restarts'],'max_iter':200},cfg['clustering']['seed']+80000+rep*100+k)
            assigned=np.argmin(v.missing_domain_gower_rect(raw,sub.iloc[fit['medoids']],feats,eq,sub_limits),axis=1)
            _,mapped,_=m.c.aligned(fits[k]['labels'],assigned,k)
            rows.append({'rep':rep+1,'k':k,'ari':float(adjusted_rand_score(fits[k]['labels'],assigned)),'aligned_agreement':float(np.mean(fits[k]['labels']==mapped))})
    detail=pd.DataFrame(rows);detail.to_csv(tables/'complete14_equal_feature_stability_detail.csv',index=False)
    summary=detail.groupby('k').agg(ari_mean=('ari','mean'),ari_median=('ari','median'),ari_min=('ari','min'),agreement_mean=('aligned_agreement','mean'),agreement_min=('aligned_agreement','min')).reset_index();summary.to_csv(tables/'complete14_equal_feature_stability_summary.csv',index=False);print(summary.to_string(index=False))

if __name__=='__main__': main()
