"""Training empirical-null Gap diagnostic and average-linkage sensitivity."""
from pathlib import Path
import importlib,json,sys,time
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score,silhouette_samples

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
c=importlib.import_module('14_cluster_training')

def main():
    started=time.time()
    loc=json.loads((ROOT/'logs/clustering_latest.json').read_text(encoding='utf-8'))
    prep=json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/'config/clustering_v4.json').read_text(encoding='utf-8'))
    extra=json.loads((ROOT/'config/clustering_supplementary.json').read_text(encoding='utf-8'))
    features=json.loads((ROOT/'config/analysis_v2.json').read_text(encoding='utf-8'))['primary_features']
    data=Path(loc['data_path'])/'supplementary';data.mkdir(exist_ok=True)
    tables=Path(loc['tables_path'])/'supplementary';tables.mkdir(exist_ok=True)
    report=Path(loc['report_path'])/'supplementary';report.mkdir(exist_ok=True)
    cols=['subject_id','hadm_id','stay_id','split','phenotype_eligible']+features
    frame=pd.read_parquet(Path(prep['data_path'])/'reviewed_unimputed.parquet',columns=cols)
    frame=frame[frame.split.eq('train')&frame.phenotype_eligible].sort_values('stay_id').reset_index(drop=True)
    filled,_,limits=c.fit_transform(frame,features)
    rng=np.random.default_rng(config['seed']+40000)
    idx=np.sort(rng.choice(len(frame),min(extra['gap_sample_n'],len(frame)),replace=False))
    sample=filled.iloc[idx].reset_index(drop=True)
    frame.iloc[idx][['subject_id','hadm_id','stay_id']].to_parquet(data/'gap_training_sample_ids.parquet',index=False)
    manifest={'status':'running','config':extra,'script_sha256':c.sha(__file__),'outcomes_loaded':False,'partition':'eligible_train_only','n_gap_sample':len(sample)}
    def record():(report/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for file in [Path(__file__),ROOT/'config/clustering_supplementary.json']:(report/file.name).write_bytes(file.read_bytes())
    record()
    try:
        D=c.gower(sample,sample,features,limits);np.fill_diagonal(D,0)
        observed={};reference=[]
        for k in config['candidate_k']:
            fit,_=c.fit_pam(D,k,extra['gap_restarts'],config['seed']+50000+k,config['max_iter'])
            observed[k]=fit['loss']
        for rep in range(extra['gap_reference_repetitions']):
            null=sample.copy()
            for feature in features:null[feature]=rng.permutation(null[feature].to_numpy())
            N=c.gower(null,null,features,limits);np.fill_diagonal(N,0)
            for k in config['candidate_k']:
                fit,_=c.fit_pam(N,k,extra['gap_restarts'],config['seed']+60000+100*rep+k,config['max_iter'])
                reference.append({'rep':rep+1,'k':k,'loss':fit['loss'],'log_loss':float(np.log(fit['loss']))})
            del N
            if (rep+1)%5==0:print(f'Empirical marginal-permutation Gap reference {rep+1}/20',flush=True)
        ref=pd.DataFrame(reference);rows=[]
        for k in config['candidate_k']:
            logs=ref.loc[ref.k.eq(k),'log_loss']
            sd=float(logs.std(ddof=1));rows.append({'k':k,'observed_sample_loss':observed[k],'reference_mean_log_loss':float(logs.mean()),'gap':float(logs.mean()-np.log(observed[k])),'s':sd*np.sqrt(1+1/len(logs)),'sample_n':len(sample),'reference_repetitions':len(logs)})
        gap=pd.DataFrame(rows)
        gap['one_se_crossing']=False;chosen=None
        for i in range(len(gap)-1):
            if gap.loc[i,'gap']>=gap.loc[i+1,'gap']-gap.loc[i+1,'s']:
                gap.loc[i,'one_se_crossing']=True
                if chosen is None:chosen=int(gap.loc[i,'k'])
        if chosen is None:chosen=int(gap.k.max())
        gap.to_csv(tables/'empirical_marginal_gap.csv',index=False);ref.to_csv(tables/'gap_reference_losses.csv',index=False)
        print('Computing full-training average-linkage sensitivity',flush=True)
        full=np.load(Path(loc['data_path'])/'primary_training_gower.npy',mmap_mode='r')
        condensed=squareform(full,checks=False)
        tree=linkage(condensed,method='average');del condensed
        np.save(data/'average_linkage_tree.npy',tree)
        cuts=cut_tree(tree,n_clusters=config['candidate_k'])
        alternate=[]
        for i,k in enumerate(config['candidate_k']):
            labels=cuts[:,i];counts=np.bincount(labels,minlength=k)
            primary=pd.read_parquet(Path(loc['data_path'])/f'primary_k{k}_training_labels.parquet')
            assert primary[['subject_id','hadm_id','stay_id']].equals(frame[['subject_id','hadm_id','stay_id']])
            alternate.append({'k':k,'n':len(frame),'ari_vs_primary_pam':float(adjusted_rand_score(primary.cluster,labels)),'mean_silhouette':float(silhouette_samples(full,labels,metric='precomputed').mean()),'min_cluster_n':int(counts.min()),'min_cluster_fraction':float(counts.min()/len(frame))})
            output=frame[['subject_id','hadm_id','stay_id']].copy();output['cluster']=labels+1
            output.to_parquet(data/f'average_linkage_k{k}_training_labels.parquet',index=False)
        pd.DataFrame(alternate).to_csv(tables/'average_linkage_sensitivity.csv',index=False)
        manifest.update(status='supplementary_diagnostics_complete',gap_diagnostic_k=chosen,gap_boundary_warning=not bool(gap.one_se_crossing.any()),elapsed_seconds=time.time()-started,data_sha256={f.name:c.sha(f) for f in data.iterdir() if f.is_file()})
        record();print('Gap and average-linkage diagnostics complete; diagnostic Gap k='+str(chosen),flush=True)
    except Exception as exc:
        manifest.update(status='failed',failure={'type':type(exc).__name__,'reason':str(exc)});record();raise

if __name__=='__main__':main()
