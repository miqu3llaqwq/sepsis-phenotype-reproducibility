"""Equal-feature missing-aware Gower control for the core14 population."""
from pathlib import Path
import importlib,json,time
import numpy as np,pandas as pd
from sklearn.metrics import silhouette_samples,adjusted_rand_score
ROOT=Path(__file__).resolve().parents[1]; v=importlib.import_module('26_domain_balanced_v7'); m=importlib.import_module('18_physiology_v5')
def main():
 p=json.loads((ROOT/'logs/clustering_v7_latest.json').read_text(encoding='utf-8')); data=Path(p['data_path']);tables=Path(p['tables_path']);cfg=json.loads((ROOT/'config/phenotype_primary_v1.json').read_text(encoding='utf-8'));raw=pd.read_parquet(data/'core14_sensitivity/eligible_train_unimputed.parquet');feats=cfg['features_primary'];eq={f:[f] for f in feats};lim=v.domain_limits(raw,eq);D=v.missing_domain_gower(raw,feats,eq,lim);rows=[]
 for k in cfg['clustering']['candidate_k']:
  fit,restarts=v.fit_parallel(D,k,{'pam_restarts':cfg['clustering']['pam_restarts'],'max_iter':200},cfg['clustering']['seed']+70000+k);sil=silhouette_samples(D,fit['labels'],metric='precomputed');sizes=np.bincount(fit['labels'],minlength=k);dom=pd.read_parquet(data/f'core14_sensitivity/k{k}_labels.parquet')['cluster'].to_numpy()-1;agree,mapped,_=m.c.aligned(dom,fit['labels'],k);rows.append({'analysis':'core14_equal_feature_missing_aware_gower','k':k,'n':len(raw),'silhouette_mean':float(sil.mean()),'min_cluster_fraction':float(sizes.min()/len(raw)),'ari_vs_domain_balanced':float(adjusted_rand_score(dom,fit['labels'])),'aligned_agreement_vs_domain_balanced':float(agree),'loss':float(fit['loss'])});pd.DataFrame({'cluster':fit['labels']+1}).to_parquet(data/f'equal_feature_k{k}_labels.parquet',index=False);raw.iloc[fit['medoids']].assign(cluster=np.arange(1,k+1)).to_parquet(data/f'equal_feature_k{k}_medoids.parquet',index=False);pd.DataFrame(restarts).to_csv(tables/f'equal_feature_k{k}_restarts.csv',index=False)
 pd.DataFrame(rows).to_csv(tables/'equal_feature_gower_metrics.csv',index=False);(Path(p['report_path'])/'equal_feature_gower_manifest.json').write_text(json.dumps({'status':'complete','features':feats,'n':len(raw),'outcomes_loaded':False,'validation_test_used':False,'distance':'each primary variable equal weight; pairwise observed denominator'},ensure_ascii=False,indent=2),encoding='utf-8');print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
