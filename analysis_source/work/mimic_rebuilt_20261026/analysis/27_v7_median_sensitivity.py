"""Same-population median-imputation sensitivity for v7; no outcomes."""
from pathlib import Path
import importlib, json, time
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
v=importlib.import_module('26_domain_balanced_v7')
m=importlib.import_module('18_physiology_v5')

def main():
    started=time.time(); pointer=json.loads((ROOT/'logs/clustering_v7_latest.json').read_text(encoding='utf-8'))
    cfg=json.loads((ROOT/'config/clustering_v7.json').read_text(encoding='utf-8')); data=Path(pointer['data_path']); tables=Path(pointer['tables_path'])
    raw=pd.read_parquet(data/'eligible_train_unimputed.parquet'); feats=cfg['features']; limits=v.domain_limits(raw,cfg['domains'])
    medians={f:float(raw[f].median()) for f in feats}; filled=raw.copy(); filled[feats]=filled[feats].fillna(pd.Series(medians))
    filled.to_parquet(data/'median_imputed_sensitivity_train.parquet',index=False)
    dist=v.missing_domain_gower(filled,feats,cfg['domains'],limits); rows=[]
    for k in cfg['candidate_k']:
        fit,restarts=v.fit_parallel(dist,k,cfg,cfg['seed']+30000+k)
        labels=fit['labels']+1; pd.DataFrame({'cluster':labels}).to_parquet(data/f'median_sensitivity_k{k}_labels.parquet',index=False)
        pd.DataFrame(restarts).to_csv(tables/f'median_sensitivity_k{k}_restarts.csv',index=False)
        primary=pd.read_parquet(data/f'primary_domain_balanced_k{k}_labels.parquet')['cluster'].to_numpy()-1
        ari=float(adjusted_rand_score(primary,fit['labels']))
        _,mapped,_=m.c.aligned(primary,fit['labels'],k)
        rows.append({'analysis':'median_imputation_same_population','k':k,'n':len(raw),'ari_vs_missing_aware':ari,'aligned_agreement':float(np.mean(primary==mapped)),'loss':float(fit['loss']),'medians':json.dumps(medians,ensure_ascii=False)})
    pd.DataFrame(rows).to_csv(tables/'median_sensitivity_metrics.csv',index=False)
    summary={'status':'complete','run_id':pointer['run_id'],'n':len(raw),'outcomes_loaded':False,'validation_test_used':False,'imputation':'single deterministic training-eligible-population median; sensitivity only, not a claim of proper multiple imputation','elapsed_seconds':time.time()-started,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    m.dump(Path(pointer['report_path'])/'median_sensitivity_manifest.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
