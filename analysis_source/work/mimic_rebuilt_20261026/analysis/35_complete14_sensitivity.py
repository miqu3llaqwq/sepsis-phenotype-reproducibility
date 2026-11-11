"""Pre-specified complete-observation sensitivity for the 14 primary features."""
from pathlib import Path
import importlib, json
import numpy as np, pandas as pd
from sklearn.metrics import silhouette_samples

ROOT=Path(__file__).resolve().parents[1]
v=importlib.import_module('26_domain_balanced_v7')
m=importlib.import_module('18_physiology_v5')

def main():
    pointer=json.loads((ROOT/'logs/clustering_v7_latest.json').read_text(encoding='utf-8'))
    data=Path(pointer['data_path']); tables=Path(pointer['tables_path']); reports=Path(pointer['report_path'])
    cfg=json.loads((ROOT/'config/phenotype_primary_v1.json').read_text(encoding='utf-8'))
    feats=cfg['features_primary']; raw=pd.read_parquet(data/'core14_sensitivity/eligible_train_unimputed.parquet')
    complete=raw[raw[feats].notna().all(axis=1)].reset_index(drop=True)
    eq={f:[f] for f in feats}; limits=v.domain_limits(complete,eq)
    D=v.missing_domain_gower(complete,feats,eq,limits)
    rows=[]
    for k in cfg['clustering']['candidate_k']:
        fit,_=v.fit_parallel(D,k,{'pam_restarts':cfg['clustering']['pam_restarts'],'max_iter':200},cfg['clustering']['seed']+70000+k)
        sil=silhouette_samples(D,fit['labels'],metric='precomputed');sizes=np.bincount(fit['labels'],minlength=k)
        rows.append({'k':k,'n_complete14':len(complete),'silhouette_mean':float(sil.mean()),'silhouette_negative_fraction':float((sil<0).mean()),'min_cluster_fraction':float(sizes.min()/len(complete)),'loss':float(fit['loss'])})
        out=complete[m.IDS].copy();out['cluster']=fit['labels']+1;out.to_parquet(data/f'complete14_equal_feature_k{k}_labels.parquet',index=False)
    result=pd.DataFrame(rows);result.to_csv(tables/'complete14_equal_feature_metrics.csv',index=False)
    report=reports/'complete14_sensitivity.md'
    report.write_text('# 完整14项观测敏感性分析\n\n```\n'+result.to_string(index=False)+'\n```\n\n该分析只用于判断缺失对聚类稳定性的影响，不替代主分析，也不使用结局。\n',encoding='utf-8')
    print(f'n_complete14={len(complete)}');print(result.to_string(index=False))

if __name__=='__main__': main()
