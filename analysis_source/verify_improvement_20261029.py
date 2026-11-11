"""Independent validation of optimized labels, candidate search and heldout replay."""
from pathlib import Path
import json,hashlib,itertools,sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parent;B=ROOT/'work/mimic_rebuilt_20261026';F=B/'frozen/20261027';OLD=B/'data/revised_analysis_20261027';O=ROOT/'outputs/improvement_20261029';W=ROOT/'work/improvement_20261029'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(F/'core14_primary_transform.json');fs=cfg['features'];ids=['subject_id','stay_id','hadm_id']
def distance(a,b,mode='original'):
    x=a[fs].to_numpy(float);y=b[fs].to_numpy(float);result=[]
    for row in y:
        sums=np.zeros(len(x),np.float32 if mode=='original' else np.float64);n=np.zeros(len(x),int)
        for j,f in enumerate(fs):
            lo,hi=cfg['limits'][f];valid=np.isfinite(x[:,j])&np.isfinite(row[j]);xx=(np.clip(x[:,j],lo,hi)-lo)/(hi-lo);yy=(np.clip(row[j],lo,hi)-lo)/(hi-lo)
            term=np.where(valid,np.abs(xx-yy),0)
            sums+=term.astype(np.float32) if mode=='original' else term;n+=valid
        result.append(np.divide(sums,n,out=np.full(len(x),np.nan),where=n>0).astype('float32'))
    return np.column_stack(result)
def align(a,b,k):
    tab=np.zeros((k,k),int)
    for i,j in zip(a,b):tab[i,j]+=1
    r,c=linear_sum_assignment(-tab);m=np.empty(k,int);m[c]=r;return m[b],m
def eligible(frame):
    present=frame[fs].notna();domains={k:present[v].any(axis=1) for k,v in cfg['domains'].items()}
    return (present.sum(axis=1)>=6)&(pd.DataFrame(domains).sum(axis=1)>=4)&domains['circulation']&domains['respiratory']
def close(a,b):assert np.allclose(a,b,rtol=2e-7,atol=2e-7,equal_nan=True),(a,b)
def main():
    protocol=read(O/'protocol.json');assert read(O/'test_gate.json')['technical_pass']
    bm=read(F/'bundle_manifest.json');assert all(sha(F/p)==h for p,h in bm['sha256'].items())
    train=pd.read_parquet(B/'data/core14_preparation/eligible_train_unimputed.parquet',columns=ids+['split']+fs).sort_values('stay_id').reset_index(drop=True)
    reviewed=B/'data/observation_review/derived_v3/reviewed_unimputed.parquet';assert sha(reviewed)==protocol['inputs']['reviewed_feature_source']
    checktrain=pd.read_parquet(reviewed,columns=ids+['split']+fs,filters=[('split','=','train')]).set_index('stay_id').loc[train.stay_id]
    assert np.array_equal(checktrain[fs].to_numpy(),train[fs].to_numpy(),equal_nan=True)
    complete=train[train[fs].notna().all(axis=1)].reset_index(drop=True);frames={'observed':train}
    for mi in range(1,11):
        z=train[ids].copy();z[fs]=pd.read_parquet(OLD/f'mice_{mi:02d}/imputed.parquet')[fs].to_numpy();frames[f'mice_{mi:02d}']=z
    for tag in protocol['optimization']['controlled_complete_datasets']:
        ix=np.load(ROOT/f'work/missingness_depth_20261028/{tag}/mask_and_order_local.npz')['complete_case_row'];frames['control_'+tag]=complete.iloc[ix].reset_index(drop=True)
    heldout_only='--heldout-only' in sys.argv
    if heldout_only:frames={}
    fit_checks=0;selection_checks=0;convergence=pd.read_csv(O/'optimization_convergence.csv');near=pd.read_csv(O/'near_optimal_agreement.csv')
    if not heldout_only:
        assert len(convergence)==250 and len(near)==140 and len(convergence[['dataset','k']].drop_duplicates())==70
    for name,frame in frames.items():
        path=W/'optimization'/name;mode='depth' if name.startswith('control_') else 'original';manifest=read(path/'manifest.json');assert all(sha(path/p)==h for p,h in manifest['outputs'].items())
        assert manifest['n']==len(frame) and manifest['input_hash']==hashlib.sha256(frame[fs].to_numpy(float).tobytes()).hexdigest()
        for k in range(2,7):
            z=np.load(path/f'k{k}_all_local.npz');selected=np.load(path/f'k{k}_selected_local.npz');seedloss=pd.read_csv(path/f'k{k}_restarts.csv')
            for i,med in enumerate(z['medoids']):
                d=distance(frame,frame.iloc[med],mode);assert np.array_equal(d.argmin(axis=1),z['labels'][i]);loss=d.min(axis=1).sum(dtype=np.float64);close(loss,z['loss'][i]);close(loss,seedloss.iloc[i].loss);fit_checks+=1
            d=distance(frame,frame.iloc[selected['medoids']],mode);assert np.array_equal(d.argmin(axis=1),selected['labels']);close(d.min(axis=1).sum(dtype=np.float64),selected['loss']);selection_checks+=1
            cc=convergence[convergence.dataset.eq(name)&convergence.k.eq(k)]
            for rr in cc.itertuples():close(rr.best_loss,min(rr.old_loss,float(z['loss'][:rr.new_starts].min())))
            assert float(selected['loss'])<=float(z['loss'].min())+1e-7
            for rr in near[near.dataset.eq(name)&near.k.eq(k)].itertuples():
                which=np.flatnonzero(z['loss']<=float(selected['loss'])*(1+rr.relative_loss_tolerance)+1e-10);assert len(which)==rr.eligible_new_starts
                assert rr.pairs==len(which)*(len(which)-1)//2
                baseline=float(cc.old_loss.iloc[0]);close(rr.last_batch_improvement,(min(baseline,float(z['loss'][:-10].min()))-float(selected['loss']))/max(baseline,1e-15))
                if len(which)>1:
                    scores=[adjusted_rand_score(z['labels'][a],z['labels'][b]) for a,b in itertools.combinations(which,2)];close(min(scores),rr.ari_min);close(np.mean(scores),rr.ari_mean)
    if not heldout_only:
        cross=pd.read_csv(O/'enhanced_cross_path_detail.csv');assert len(cross)==50
        for row in cross.itertuples():
            a=np.load(W/f'optimization/observed/k{row.k}_selected_local.npz')['labels'];b=np.load(W/f'optimization/mice_{row.imputation:02d}/k{row.k}_selected_local.npz')['labels'];close(row.enhanced_cross_path_ari,adjusted_rand_score(a,b))
    d=np.load(W/'optimization/observed/distance.npy',mmap_mode='r');candidate_checks=0;single_swap_checks=0
    for variant in ['all','complete']:
        candidates=np.arange(len(train)) if variant=='all' else np.flatnonzero(train[fs].notna().all(axis=1))
        for k in range(2,7):
            z=np.load(W/f'candidates/{variant}_k{k}_local.npz');med=z['medoids'];assert np.isin(med,candidates).all();rect=distance(train,train.iloc[med]);assert np.array_equal(rect.argmin(axis=1),z['labels']);close(rect.min(axis=1).sum(dtype=np.float64),z['loss']);candidate_checks+=1
            history=pd.read_csv(W/f'candidates/{variant}_k{k}_history.csv');assert (history.loss.diff().dropna()<0).all() and history.iloc[-1].best_delta>=-1e-8
            probe=np.random.default_rng(7029+k).choice(candidates,size=min(80,len(candidates)),replace=False)
            # Independently evaluate direct loss for sampled admissible swaps, not the producer's delta formula.
            for incoming in probe:
                if incoming in med:continue
                column=distance(train,train.iloc[[incoming]])[:,0]
                for slot in range(k):
                    test=rect.copy();test[:,slot]=column;assert test.min(axis=1).sum(dtype=np.float64)>=float(z['loss'])-1e-6;single_swap_checks+=1
    lock=read(W/'locked_models/manifest.json');assert all(sha(W/'locked_models'/n)==h for n,h in lock['files'].items())
    cohorts={};hold_checks=0;mask_checks=0;anchor_checks=0;lookup={int(s):i for i,s in enumerate(train.stay_id)}
    for split in ['validation','test']:
        src=pd.read_parquet(reviewed,columns=ids+['split']+fs,filters=[('split','=',split)]).sort_values('stay_id').reset_index(drop=True)
        use=src.loc[eligible(src)].reset_index(drop=True);saved=pd.read_parquet(W/split/'eligible_local.parquet');assert saved.equals(use);cohorts[split]=set(use.subject_id)
        assert not cohorts[split]&set(train.subject_id);gate=read(O/f'{split}_gate.json');assert gate['locked_model_manifest_sha256']==sha(W/'locked_models/manifest.json') and gate['n_eligible']==len(use)
        coverage=pd.read_csv(O/f'{split}_coverage.csv');anchorstats=pd.read_csv(O/f'{split}_anchor_perturbation.csv');models={}
        for k in range(2,7):
            for variant in lock['variants']:
                med=pd.read_parquet(W/f'locked_models/{variant}_k{k}.parquet');models[variant,k]=med;rr=coverage[coverage.variant.eq(variant)&coverage.k.eq(k)].iloc[0]
                saved=np.load(W/f'{split}/{variant}_k{k}_assignment_local.npz');dd=distance(use,med);good=np.isfinite(dd).all(axis=1);close(dd,saved['distance']);assert np.array_equal(good,saved['computable']) and np.array_equal(dd[good].argmin(axis=1),saved['labels'][good]);assert good.sum()==rr.n_computable;hold_checks+=1
            original=models['original_frozen',k];ix=[lookup[int(s)] for s in original.stay_id];base_train=distance(train,original).argmin(axis=1);saved=np.load(W/f'{split}/anchor_perturbation_k{k}_local.npz');lab=[]
            for mi in range(1,11):
                med=pd.read_parquet(OLD/f'mice_{mi:02d}/imputed.parquet')[fs].iloc[ix];_,mapping=align(base_train,distance(train,med).argmin(axis=1),k)
                labels=mapping[distance(use,med).argmin(axis=1)];assert np.array_equal(mapping,saved['training_maps'][mi-1]) and np.array_equal(labels,saved['mapped_labels'][mi-1]);lab.append(labels);anchor_checks+=1
            disagreement=np.mean([lab[i]!=lab[j] for i,j in itertools.combinations(range(10),2)],axis=0);allcomplete=use[fs].notna().all(axis=1).to_numpy()
            for group,select in [('all',np.ones(len(use),bool)),('complete',allcomplete),('incomplete',~allcomplete)]:close(anchorstats[anchorstats.k.eq(k)&anchorstats.group.eq(group)].iloc[0].anchor_pair_disagreement_training_aligned,disagreement[select].mean())
        full=use[use[fs].notna().all(axis=1)].reset_index(drop=True);detail=pd.read_csv(O/f'{split}_mask_detail.csv')
        for scenario in ['mcar20','mcar40','empirical_incomplete_pattern']:
            for rep in range(1,6):
                zz=np.load(W/f'{split}/{scenario}_r{rep:02d}_local.npz');masked=full.copy();masked[fs]=masked[fs].mask(zz['hidden']);ix=np.flatnonzero(eligible(masked));assert np.array_equal(ix,zz['eligible_index'])
                for k in range(2,7):
                    common=np.ones(len(ix),bool);base={};perturbed={}
                    for variant in lock['variants']:
                        aa=distance(full.iloc[ix],models[variant,k]);bb=distance(masked.iloc[ix],models[variant,k]);common &= np.isfinite(aa).all(axis=1)&np.isfinite(bb).all(axis=1);base[variant]=aa.argmin(axis=1);perturbed[variant]=bb.argmin(axis=1)
                    for variant in lock['variants']:
                        a=base[variant][common];b=perturbed[variant][common];r=detail[detail.scenario.eq(scenario)&detail.replicate.eq(rep)&detail.k.eq(k)&detail.variant.eq(variant)].iloc[0];assert r.paired_n==len(a);close(r.ari_vs_own_unmasked,adjusted_rand_score(a,b));close(r.identity_disagreement,np.mean(a!=b));close(r.aligned_disagreement,np.mean(a!=align(a,b,k)[0]));mask_checks+=1
    assert not cohorts['validation']&cohorts['test']
    ext=read(O/'eicu_source_coverage_audit.json');assert all(sha(n)==h for n,h in ext['input_sha256'].items())
    result={'status':'pass','date':'2026-10-29','new_pam_starts_replayed':fit_checks,'selected_solutions_replayed':selection_checks,'candidate_solutions_replayed':candidate_checks,'candidate_direct_swaps_sampled':single_swap_checks,'heldout_assignments_replayed':hold_checks,'heldout_anchor_draw_assignments_replayed':anchor_checks,'heldout_mask_metrics_recomputed':mask_checks,'frozen_files_hash_checked':len(bm['sha256']),'train_validation_test_overlap':0,'no_clinical_truth_claim':True,'script_sha256':sha(__file__),'limitations':'Candidate local optimality sampled independently, not global optimality; repeated starts/masks are not independent patients; observed-value holdout diagnostics do not validate a deployment MICE imputer.'}
    result['scope']='candidate_and_heldout_only' if heldout_only else 'full'
    if not heldout_only:assert fit_checks==2000 and selection_checks==70
    (O/('independent_holdout_qa.json' if heldout_only else 'independent_qa.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
