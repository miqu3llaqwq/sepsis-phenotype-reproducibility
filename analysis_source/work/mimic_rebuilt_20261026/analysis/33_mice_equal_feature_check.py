"""Reuse completed MICE-PMM datasets under the primary equal-feature Gower."""
from pathlib import Path
import importlib,json,time
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parents[1]; v=importlib.import_module('26_domain_balanced_v7'); m=importlib.import_module('18_physiology_v5')
def main():
 p=json.loads((ROOT/'logs/clustering_v7_latest.json').read_text(encoding='utf-8')); data=Path(p['data_path']);tables=Path(p['tables_path']);cfg=json.loads((ROOT/'config/phenotype_primary_v1.json').read_text(encoding='utf-8'));raw=pd.read_parquet(data/'core14_sensitivity/eligible_train_unimputed.parquet');feats=cfg['features_primary'];eq={f:[f] for f in feats};lim=v.domain_limits(raw,eq);base={k:pd.read_parquet(data/f'equal_feature_k{k}_labels.parquet')['cluster'].to_numpy()-1 for k in [2,3]};rows=[];labels={2:[],3:[]};start=time.time()
 for mi in range(1,11):
  completed=pd.read_parquet(data/f'mice_pmm_sensitivity/imputed_m{mi:02d}.parquet');D=v.missing_domain_gower(completed,feats,eq,lim)
  for k in [2,3]:
   fit,_=v.fit_parallel(D,k,{'pam_restarts':5,'max_iter':200},20261010+50000+mi*100+k); labels[k].append(fit['labels']); _,mapped,_=m.c.aligned(base[k],fit['labels'],k); rows.append({'imputation':mi,'k':k,'ari_vs_equal_feature_missing_aware':float(adjusted_rand_score(base[k],fit['labels'])),'aligned_agreement':float(np.mean(base[k]==mapped)),'loss':float(fit['loss'])})
  print(f'equal-feature MICE {mi}/10',flush=True)
 det=pd.DataFrame(rows);det.to_csv(tables/'mice_equal_feature_k23_detail.csv',index=False);summary=[]
 for k in [2,3]:
  vals=[adjusted_rand_score(labels[k][i],labels[k][j]) for i in range(10) for j in range(i+1,10)];x=det[det.k.eq(k)];summary.append({'k':k,'ari_vs_equal_feature_mean':float(x.ari_vs_equal_feature_missing_aware.mean()),'ari_vs_equal_feature_min':float(x.ari_vs_equal_feature_missing_aware.min()),'imputation_pairwise_ari_mean':float(np.mean(vals)),'imputation_pairwise_ari_min':float(np.min(vals))})
 pd.DataFrame(summary).to_csv(tables/'mice_equal_feature_k23_summary.csv',index=False);print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
