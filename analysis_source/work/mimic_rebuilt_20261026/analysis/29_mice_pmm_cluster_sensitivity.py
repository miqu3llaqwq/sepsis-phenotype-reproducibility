"""Training-only MICE-PMM clustering sensitivity for phenotype_primary_v1."""
from pathlib import Path
import hashlib, importlib, json, time
import numpy as np, pandas as pd
from statsmodels.imputation.mice import MICEData
from sklearn.metrics import adjusted_rand_score, silhouette_samples

ROOT=Path(__file__).resolve().parents[1]
v=importlib.import_module('26_domain_balanced_v7'); m=importlib.import_module('18_physiology_v5')

def main():
    started=time.time(); cfg=json.loads((ROOT/'config/phenotype_primary_v1.json').read_text(encoding='utf-8')); ptr=json.loads((ROOT/'logs/clustering_v7_latest.json').read_text(encoding='utf-8'))
    v7data=Path(ptr['data_path']); raw=pd.read_parquet(v7data/'core14_sensitivity/eligible_train_unimputed.parquet'); feats=cfg['features_primary']; domains=cfg['domains']; limits=v.domain_limits(raw,domains)
    out=v7data/'mice_pmm_sensitivity'; out.mkdir(exist_ok=True); tables=Path(ptr['tables_path']); rows=[]; labels_by_k={k:[] for k in cfg['clustering']['candidate_k']}; cluster_cfg={'pam_restarts':cfg['clustering']['pam_restarts'],'max_iter':200}
    base_labels={k:pd.read_parquet(v7data/f'core14_sensitivity/k{k}_labels.parquet')['cluster'].to_numpy()-1 for k in cfg['clustering']['candidate_k']}
    source=raw[feats].astype(float)
    for mi in range(cfg['imputation_sensitivity']['datasets']):
        chain=MICEData(source.copy(),perturbation_method='boot',k_pmm=cfg['imputation_sensitivity']['donor_count'],rng=np.random.default_rng(cfg['clustering']['seed']+50000+mi))
        chain.update_all(cfg['imputation_sensitivity']['burn_in_cycles'])
        completed=None
        for _ in range(cfg['imputation_sensitivity']['between_draw_cycles']): completed=chain.next_sample().copy()
        completed=completed[feats].astype(float)
        original=source.to_numpy(); imputed=completed.to_numpy(); missing=np.isnan(original)
        if not np.array_equal(imputed[~missing],original[~missing]) or not np.isfinite(imputed).all(): raise ValueError('MICE-PMM changed observed values or left nonfinite values')
        completed.to_parquet(out/f'imputed_m{mi+1:02d}.parquet',index=False)
        for k in cfg['clustering']['candidate_k']:
            D=v.missing_domain_gower(completed,feats,domains,limits); fit,restarts=v.fit_parallel(D,k,cluster_cfg,cfg['clustering']['seed']+50000+mi*100+k); label=fit['labels']; labels_by_k[k].append(label)
            sil=silhouette_samples(D,label,metric='precomputed')
            ari_base=adjusted_rand_score(base_labels[k],label); rows.append({'imputation':mi+1,'k':k,'n':len(raw),'silhouette_mean':float(sil.mean()),'ari_vs_core14_missing_aware':float(ari_base),'min_cluster_fraction':float(np.bincount(label,minlength=k).min()/len(label)),'loss':float(fit['loss'])})
            pd.DataFrame(restarts).to_csv(tables/f'mice_pmm_m{mi+1:02d}_k{k}_restarts.csv',index=False)
            pd.DataFrame({**{x:raw[x] for x in m.IDS},'cluster':label+1}).to_parquet(out/f'm{mi+1:02d}_k{k}_labels.parquet',index=False)
        print(f'MICE-PMM dataset {mi+1}/{cfg["imputation_sensitivity"]["datasets"]}',flush=True)
    detail=pd.DataFrame(rows); detail.to_csv(tables/'mice_pmm_cluster_metrics.csv',index=False)
    summary=[]
    for k,labels in labels_by_k.items():
        pair=[]
        for i in range(len(labels)):
            for j in range(i+1,len(labels)): pair.append(adjusted_rand_score(labels[i],labels[j]))
        x=detail[detail.k.eq(k)]; summary.append({'k':k,'datasets':len(labels),'silhouette_mean_across_imputations':float(x.silhouette_mean.mean()),'silhouette_min':float(x.silhouette_mean.min()),'silhouette_max':float(x.silhouette_mean.max()),'ari_vs_missing_aware_mean':float(x.ari_vs_core14_missing_aware.mean()),'ari_vs_missing_aware_min':float(x.ari_vs_core14_missing_aware.min()),'imputation_pairwise_ari_mean':float(np.mean(pair)),'imputation_pairwise_ari_min':float(np.min(pair))})
    pd.DataFrame(summary).to_csv(tables/'mice_pmm_cluster_summary.csv',index=False)
    manifest={'status':'complete','method':'statsmodels MICEData; predictive mean matching; bootstrap perturbation','datasets':cfg['imputation_sensitivity']['datasets'],'burn_in_cycles':cfg['imputation_sensitivity']['burn_in_cycles'],'between_draw_cycles':cfg['imputation_sensitivity']['between_draw_cycles'],'donor_count':cfg['imputation_sensitivity']['donor_count'],'n':len(raw),'outcomes_loaded':False,'validation_test_used':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.time()-started}
    (Path(ptr['report_path'])/'mice_pmm_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(manifest,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
