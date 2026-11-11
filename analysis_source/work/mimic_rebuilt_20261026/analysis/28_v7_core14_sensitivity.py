"""Outcome-blinded sensitivity excluding high-missing GCS and bilirubin."""
from pathlib import Path
import importlib,json,time,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_samples
ROOT=Path(__file__).resolve().parents[1]
v=importlib.import_module('26_domain_balanced_v7')
m=importlib.import_module('18_physiology_v5')
def main():
    started=time.time(); ptr=json.loads((ROOT/'logs/clustering_v7_latest.json').read_text(encoding='utf-8')); cfg=json.loads((ROOT/'config/clustering_v7.json').read_text(encoding='utf-8')); data=Path(ptr['data_path']); tables=Path(ptr['tables_path'])
    feats=[f for f in cfg['features'] if f not in ['gcs_min','bilirubin']]; domains={k:[f for f in fs if f in feats] for k,fs in cfg['domains'].items() if any(f in feats for f in fs)}
    source=Path(json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))['data_path'])/'reviewed_unimputed.parquet'; cols=m.IDS+['split','age','gender']+feats; frame=pd.read_parquet(source,columns=cols); train=frame[frame.split.eq('train')].sort_values('stay_id').reset_index(drop=True)
    local_cfg=dict(cfg);local_cfg['features']=feats;local_cfg['domains']=domains
    raw_pre=train[v.eligibility(train,local_cfg)].reset_index(drop=True); bad=v.incompatible_positions(raw_pre,feats,domains); raw=raw_pre.drop(index=bad).reset_index(drop=True); limits=v.domain_limits(raw,domains); D=v.missing_domain_gower(raw,feats,domains,limits); out=data/'core14_sensitivity'; out.mkdir(exist_ok=True); raw.to_parquet(out/'eligible_train_unimputed.parquet',index=False)
    rows=[]
    for k in cfg['candidate_k']:
        fit,restarts=v.fit_parallel(D,k,local_cfg,cfg['seed']+40000+k); pd.DataFrame({'cluster':fit['labels']+1}).to_parquet(out/f'k{k}_labels.parquet',index=False); pd.DataFrame(restarts).to_csv(tables/f'core14_sensitivity_k{k}_restarts.csv',index=False)
        v7_ids=pd.read_parquet(data/'eligible_train_unimputed.parquet',columns=['stay_id'])['stay_id'].to_numpy(); primary=pd.read_parquet(data/f'primary_domain_balanced_k{k}_labels.parquet')['cluster'].to_numpy()-1; label_map=pd.Series(primary,index=v7_ids)
        common=raw['stay_id'].isin(label_map.index).to_numpy(); p=label_map.loc[raw.loc[common,'stay_id']].to_numpy()
        pred=fit['labels'][common]; ari=float(adjusted_rand_score(p,pred)); _,mapped,_=m.c.aligned(p,pred,k); sizes=np.bincount(fit['labels'],minlength=k)
        sil=silhouette_samples(D,fit['labels'],metric='precomputed')
        rows.append({'analysis':'core14_excluding_gcs_bilirubin','k':k,'n_core14':len(raw),'n_pairwise_incompatible_excluded':len(bad),'ari_vs_v7_common':ari,'aligned_agreement_vs_v7_common':float(np.mean(p==mapped)),'silhouette_mean':float(sil.mean()),'min_cluster_fraction':float(sizes.min()/len(raw)),'loss':float(fit['loss'])})
    pd.DataFrame(rows).to_csv(tables/'core14_sensitivity_metrics.csv',index=False)
    manifest={'status':'complete','run_id':ptr['run_id'],'features':feats,'n':len(raw),'n_pairwise_incompatible_excluded':len(bad),'outcomes_loaded':False,'validation_test_used':False,'elapsed_seconds':time.time()-started,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; (Path(ptr['report_path'])/'core14_sensitivity_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(manifest,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
