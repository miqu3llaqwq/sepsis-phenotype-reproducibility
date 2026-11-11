"""Outcome-blinded 80% resampling stability for the frozen core14 candidate."""
from pathlib import Path
import importlib,json,time
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parents[1]; v=importlib.import_module('26_domain_balanced_v7'); m=importlib.import_module('18_physiology_v5')
def main():
 started=time.time(); ptr=json.loads((ROOT/'logs/clustering_v7_latest.json').read_text(encoding='utf-8')); cfg=json.loads((ROOT/'config/phenotype_primary_v1.json').read_text(encoding='utf-8')); data=Path(ptr['data_path']); tables=Path(ptr['tables_path']); raw=pd.read_parquet(data/'core14_sensitivity/eligible_train_unimputed.parquet'); feats=cfg['features_primary']; domains=cfg['domains']; local={'pam_restarts':cfg['clustering']['pam_restarts'],'max_iter':200}; full={k:pd.read_parquet(data/f'core14_sensitivity/k{k}_labels.parquet')['cluster'].to_numpy()-1 for k in cfg['clustering']['candidate_k']}; rng=np.random.default_rng(cfg['clustering']['seed']); rows=[]
 for rep in range(10):
  ids=np.sort(rng.choice(len(raw),int(len(raw)*.8),replace=False)); sub=raw.iloc[ids].reset_index(drop=True); lim=v.domain_limits(sub,domains); Ds=v.missing_domain_gower(sub,feats,domains,lim)
  for k in cfg['clustering']['candidate_k']:
   fit,_=v.fit_parallel(Ds,k,local,cfg['clustering']['seed']+60000+rep*100+k); Dall=v.missing_domain_gower_rect(raw,sub.iloc[fit['medoids']],feats,domains,lim); assigned=np.argmin(Dall,axis=1); _,mapped,_=m.c.aligned(full[k],assigned,k); rows.append({'rep':rep+1,'k':k,'ari':float(adjusted_rand_score(full[k],assigned)),'aligned_agreement':float(np.mean(full[k]==mapped))})
  print(f'core14 stability {rep+1}/10',flush=True)
 out=pd.DataFrame(rows); out.to_csv(tables/'core14_stability.csv',index=False); out.groupby('k').agg(repetitions=('rep','count'),ari_mean=('ari','mean'),ari_median=('ari','median'),ari_min=('ari','min'),agreement_mean=('aligned_agreement','mean'),agreement_median=('aligned_agreement','median'),agreement_min=('aligned_agreement','min')).reset_index().to_csv(tables/'core14_stability_summary.csv',index=False); print(json.dumps({'status':'complete','n':len(raw),'elapsed_seconds':time.time()-started},ensure_ascii=False))
if __name__=='__main__':main()
