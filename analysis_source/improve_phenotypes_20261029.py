"""Outcome-blind optimization, constrained medoid candidates, locked holdout audit."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,time,sys,itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from numba import njit,prange,set_num_threads
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import run_revised_phenotypes_20261027 as R
import run_missingness_depth_20261028 as DEP
ROOT=Path(__file__).resolve().parent;B=R.B;F=B/'frozen/20261027';OLD=R.OUT
O=ROOT/'outputs/improvement_20261029';W=ROOT/'work/improvement_20261029';KS=list(range(2,7));FS=R.FEATS;IDS=R.IDS;SEED=202610290
CFG=json.loads((F/'core14_primary_transform.json').read_text());LIM=CFG['limits']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def fitworker(task):
    path,k,nstarts,seed,output=task;d=np.load(path,mmap_mode='r');rng=np.random.default_rng(seed+k);labels=[];medoids=[];rows=[];start=time.time()
    import kmedoids
    for rep in range(nstarts):
        init=rng.choice(len(d),k,replace=False).astype(np.uintp);result=kmedoids.fastpam1(d,init,max_iter=200)
        if result.n_iter>=200:raise RuntimeError('PAM iteration cap reached')
        med=np.sort(np.asarray(result.medoids,dtype=int));lab=np.argmin(d[:,med],axis=1);loss=float(d[np.arange(len(d)),med[lab]].sum(dtype=np.float64))
        assert len(np.unique(lab))==k
        labels.append(lab.astype('uint8'));medoids.append(med);rows.append({'restart':rep+1,'loss':loss,'iterations':int(result.n_iter),'seconds_cumulative':time.time()-start})
    output=Path(output);np.savez_compressed(output/f'k{k}_all_local.npz',labels=np.array(labels),medoids=np.array(medoids),loss=np.array([r['loss'] for r in rows]))
    pd.DataFrame(rows).to_csv(output/f'k{k}_restarts.csv',index=False)
    return k,time.time()-start
def summarize_grid(dest,old):
    rows=[];near=[]
    for k in KS:
        z=np.load(dest/f'k{k}_all_local.npz');oldlabel,oldmed,oldloss=old[k];loss=z['loss'];best_idx=int(loss.argmin())
        labels=z['labels'][best_idx];med=z['medoids'][best_idx];bestloss=float(loss[best_idx]);oldwins=oldloss<=bestloss
        if oldwins:labels=oldlabel;med=oldmed;bestloss=oldloss
        np.savez_compressed(dest/f'k{k}_selected_local.npz',labels=labels,medoids=med,loss=bestloss)
        for n in [5,10,20,30,50]:
            if n>len(loss):continue
            current=min(oldloss,float(loss[:n].min()));rows.append({'k':k,'new_starts':n,'historical_starts':5,'old_loss':oldloss,'best_loss':current,'relative_loss_improvement':(oldloss-current)/max(oldloss,1e-15),'ari_old_to_selected':float(adjusted_rand_score(oldlabel,labels)) if n==len(loss) else None})
        for tolerance in [.001,.01]:
            ids=np.flatnonzero(loss<=bestloss*(1+tolerance)+1e-10);aris=[adjusted_rand_score(z['labels'][i],z['labels'][j]) for i,j in itertools.combinations(ids,2)]
            near.append({'k':k,'relative_loss_tolerance':tolerance,'eligible_new_starts':len(ids),'pairs':len(aris),'ari_min':float(min(aris)) if aris else None,'ari_mean':float(np.mean(aris)) if aris else None,'old_solution_selected':bool(oldwins),
              'last_batch_improvement':float((min(oldloss,float(loss[:-10].min()))-bestloss)/max(oldloss,1e-15))})
    pd.DataFrame(rows).to_csv(dest/'convergence.csv',index=False);pd.DataFrame(near).to_csv(dest/'near_optimal_agreement.csv',index=False)
def load_training():
    raw=pd.read_parquet(B/'data/core14_preparation/eligible_train_unimputed.parquet',columns=IDS+['split']+FS).sort_values('stay_id').reset_index(drop=True)
    assert len(raw)==17810 and raw.split.eq('train').all() and raw.subject_id.is_unique
    return raw
def establish_protocol():
    O.mkdir(exist_ok=True,parents=True);W.mkdir(exist_ok=True,parents=True)
    m=read(F/'bundle_manifest.json');assert all(sha(F/n)==h for n,h in m['sha256'].items())
    protocol={'date':'2026-10-29','posthoc':True,'seed':SEED,'unit':'patient; starts, draws and masks are repeated computational perturbations, not independent patients',
      'training_n':17810,'features':FS,'k':KS,'scales':'original training P1-P99; no refit on holdout',
      'optimization':{'observed_new_starts_per_k':50,'each_of_10_mice_draws_new_starts_per_k':20,'controlled_complete_datasets':['mcar20_r01','mcar40_r01','empirical_incomplete_pattern_r01'],'controlled_new_starts_per_k':50,'historical_best_of_5_retained':True,'near_optimal_relative_tolerance':[.001,.01],'no_global_optimality_claim':True,'reason_different_budgets':'50-start diagnostic on observed/controlled data; all ten MICE draws receive twenty additional starts, with last-batch improvement explicitly reported'},
      'candidate_completeness':{'same_training_patients':True,'candidates':['all_17810','complete_8338'],'algorithm':'same deterministic greedy BUILD plus exhaustive best single SWAP for both candidate sets; fixed observed-value distance matrix','selection':'minimum training dissimilarity only; no outcomes/holdout','comparison_limitation':'candidate restriction changes admissible solutions and search path; lower distance objective is not clinical accuracy'},
      'heldout':{'validation_then_test':True,'all_rules_fixed_before_validation':True,'test_requires_validation_gate':True,'gate':'no ID overlap, frozen artifact hashes, finite eligible distances handled, no fit or re-estimation; never gate on favorable effect direction',
        'variants':['original_frozen','optimized_observed','greedy_all_candidates','greedy_complete_candidates'],'no_heldout_imputation':True,
        'anchor_perturbation':'ten saved training MICE completions at original observed medoid identities; fixed heldout observed values; label maps determined using training queries only',
        'masking':'complete heldout patients, 20% and 40% MCAR and empirical incomplete training masks, five repetitions each; frozen feature eligibility; each variant compared against its own unmasked assignment; paired results restricted to common computable patients',
        'interpretation':'internal observed-value diagnostic, not validation of a deployable MICE imputer or clinical subtypes; unmasked labels are not clinical truth'},
      'eicu':'original 39-case adapted analysis not replaced; source-wide attrition audit separate',
      'inputs':{'training_source':sha(B/'data/core14_preparation/eligible_train_unimputed.parquet'),'reviewed_feature_source':sha(B/'data/observation_review/derived_v3/reviewed_unimputed.parquet'),'frozen_bundle_manifest':sha(F/'bundle_manifest.json'),'prior_depth_protocol':sha(ROOT/'outputs/missingness_depth_20261028/protocol.json')}}
    p=O/'protocol.json'
    if p.exists():assert read(p)==protocol
    else:dump(p,protocol)
    return protocol
def optimize(raw):
    truth=raw[raw[FS].notna().all(axis=1)].reset_index(drop=True);allrows=[];nearrows=[]
    tasks=[('observed',raw,50,0)]
    for mi in range(1,11):
        dest=OLD/f'mice_{mi:02d}';m=read(dest/'imputation.json');assert sha(dest/'imputed.parquet')==m['sha256']
        z=raw[IDS].copy();z[FS]=pd.read_parquet(dest/'imputed.parquet')[FS].to_numpy();tasks.append((f'mice_{mi:02d}',z,20,mi))
    for ci,tag in enumerate(['mcar20_r01','mcar40_r01','empirical_incomplete_pattern_r01'],1):
        ix=np.load(ROOT/f'work/missingness_depth_20261028/{tag}/mask_and_order_local.npz')['complete_case_row']
        tasks.append(('control_'+tag,truth.iloc[ix].reset_index(drop=True),50,100+ci))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for name,frame,nstarts,si in tasks:
            dest=W/'optimization'/name;dest.mkdir(parents=True,exist_ok=True);old={};started=time.time()
            done=dest/'manifest.json'
            if done.exists():
                mm=read(done);assert all(sha(dest/p)==h for p,h in mm['outputs'].items())
                allrows.append(pd.read_csv(dest/'convergence.csv').assign(dataset=name));nearrows.append(pd.read_csv(dest/'near_optimal_agreement.csv').assign(dataset=name));continue
            dump(O/'run_status.json',{'status':'running','stage':'optimization_'+name})
            x=frame[FS].to_numpy(float)
            if name.startswith('control_'):d=DEP.distance(x,x)
            else:d=R.eq_distance(frame,frame,LIM)
            np.fill_diagonal(d,0);assert np.isfinite(d).all() and np.array_equal(d,d.T)
            matrix=dest/'distance.npy';np.save(matrix,d)
            for k in KS:
                if name.startswith('control_'):
                    z=np.load(ROOT/f'work/missingness_depth_20261028/{name[8:]}/complete_baseline/k{k}.npz');med=z['medoids'];lab=z['labels']
                else:
                    orig=OLD/('observed_equal' if name=='observed' else name)
                    ml=pd.read_parquet(orig/f'k{k}_medoids.parquet');indexes={int(v):i for i,v in enumerate(frame.stay_id)};med=np.array([indexes[int(v)] for v in ml.stay_id]);lab=pd.read_parquet(orig/f'k{k}_labels.parquet').cluster.to_numpy()-1
                assert np.array_equal(d[:,med].argmin(axis=1),lab);old[k]=(lab,med,float(d[np.arange(len(d)),med[lab]].sum(dtype=np.float64)))
            del d
            for fu in as_completed([pool.submit(fitworker,(str(matrix),k,nstarts,SEED+si*1000,str(dest))) for k in KS]):
                k,secs=fu.result();print('optimization',name,'k',k,'starts',nstarts,'seconds',round(secs,1),flush=True)
            summarize_grid(dest,old)
            if name!='observed':matrix.unlink()
            dump(done,{'n':len(frame),'new_starts':nstarts,'seed':SEED+si*1000,'input_hash':hashlib.sha256(x.tobytes()).hexdigest(),'elapsed_seconds':time.time()-started,'outputs':{p.name:sha(p) for p in dest.iterdir() if p.suffix in ['.npz','.csv']}})
            allrows.append(pd.read_csv(dest/'convergence.csv').assign(dataset=name));nearrows.append(pd.read_csv(dest/'near_optimal_agreement.csv').assign(dataset=name))
            pd.concat(allrows).to_csv(O/'optimization_convergence.csv',index=False);pd.concat(nearrows).to_csv(O/'near_optimal_agreement.csv',index=False)
    # Cached grids must be present in the aggregate even when every remaining grid was staged externally.
    pd.concat(allrows).to_csv(O/'optimization_convergence.csv',index=False)
    pd.concat(nearrows).to_csv(O/'near_optimal_agreement.csv',index=False)
    compare=[]
    for k in KS:
        a=np.load(W/f'optimization/observed/k{k}_selected_local.npz')['labels']
        for mi in range(1,11):
            b=np.load(W/f'optimization/mice_{mi:02d}/k{k}_selected_local.npz')['labels'];compare.append({'k':k,'imputation':mi,'enhanced_cross_path_ari':float(adjusted_rand_score(a,b))})
    pd.DataFrame(compare).to_csv(O/'enhanced_cross_path_detail.csv',index=False)
    dump(O/'optimization_manifest.json',{'status':'complete','datasets':len(tasks),'new_starts_total':sum(n*5 for _,_,n,_ in tasks),'outcomes_used':False})
@njit(parallel=True)
def candidate_sums(d,candidates,current):
    scores=np.empty(len(candidates),np.float64)
    for j in prange(len(candidates)):
        total=0.
        for i in range(d.shape[0]):total+=min(float(d[i,candidates[j]]),current[i])
        scores[j]=total
    return scores
@njit(parallel=True)
def best_swap_scores(d,candidates,medoids,labels,first,second):
    out=np.full((len(candidates),len(medoids)),np.inf)
    for j in prange(len(candidates)):
        c=candidates[j];present=False
        for m in medoids:
            if c==m:present=True
        if present:continue
        common=0.;adjust=np.zeros(len(medoids),np.float64)
        for i in range(d.shape[0]):
            dist=float(d[i,c]);base=min(dist,first[i]);common+=base-first[i];adjust[labels[i]]+=min(dist,second[i])-base
        for m in range(len(medoids)):out[j,m]=common+adjust[m]
    return out
def constrained_fit(d,candidates,k):
    set_num_threads(4);med=[];current=np.full(len(d),np.inf)
    for step in range(k):
        score=candidate_sums(d,candidates,current)
        for m in med:score[candidates==m]=np.inf
        med.append(int(candidates[score.argmin()]));current=np.minimum(current,d[:,med[-1]])
    history=[]
    for iteration in range(200):
        med=np.sort(np.array(med));rect=d[:,med].astype(float);lab=rect.argmin(axis=1);ordered=np.sort(rect,axis=1);loss=float(ordered[:,0].sum())
        change=best_swap_scores(d,candidates,med,lab,ordered[:,0],ordered[:,1]);j,slot=np.unravel_index(change.argmin(),change.shape);delta=float(change[j,slot]);history.append({'iteration':iteration,'loss':loss,'best_delta':delta})
        if delta>=-1e-8:return med,lab,loss,history
        old=loss;med[slot]=candidates[j];new=float(np.min(d[:,med],axis=1).sum(dtype=np.float64));assert new<old and abs((new-old)-delta)<1e-4
    raise RuntimeError('Constrained search did not finish within cap')
def validate_swap_implementation():
    rng=np.random.default_rng(509);x=rng.normal(size=(30,4));d=np.abs(x[:,None]-x[None,:]).mean(axis=2).astype('float32');c=np.arange(0,30,2);m=np.array([0,8,12]);rect=d[:,m].astype(float);lab=rect.argmin(axis=1);sort=np.sort(rect,axis=1);delta=best_swap_scores(d,c,m,lab,sort[:,0],sort[:,1])
    count=0
    for j,candidate in enumerate(c):
        if candidate in m:continue
        for slot in range(3):
            changed=m.copy();changed[slot]=candidate;actual=d[:,changed].min(axis=1).sum(dtype=np.float64)-sort[:,0].sum();assert abs(actual-delta[j,slot])<1e-8;count+=1
    return count
def candidate_audit(raw):
    set_num_threads(4);nchecks=validate_swap_implementation();rows=[];dest=W/'candidates';dest.mkdir(exist_ok=True)
    colfile=dest/'distance_column_major.npy'
    if not colfile.exists():np.save(colfile,np.asfortranarray(np.load(W/'optimization/observed/distance.npy',mmap_mode='r')))
    d=np.load(colfile,mmap_mode='r')
    for variant,candidates in [('all',np.arange(len(raw))),('complete',np.flatnonzero(raw[FS].notna().all(axis=1).to_numpy()))]:
        for k in KS:
            path=dest/f'{variant}_k{k}_local.npz';history=dest/f'{variant}_k{k}_history.csv';started=time.time()
            if path.exists():z=np.load(path);med=z['medoids'];labels=z['labels'];loss=float(z['loss'])
            else:
                dump(O/'run_status.json',{'status':'running','stage':f'candidate_{variant}_k{k}'})
                med,labels,loss,h=constrained_fit(d,candidates,k);np.savez_compressed(path,medoids=med,labels=labels,loss=loss);pd.DataFrame(h).to_csv(history,index=False)
            assert np.isin(med,candidates).all();missing=raw.iloc[med][FS].isna().sum(axis=1).to_numpy()
            rows.append({'variant':variant,'k':k,'n_fitting':len(raw),'candidate_n':len(candidates),'loss':loss,'medoid_missing_min':int(missing.min()),'medoid_missing_max':int(missing.max()),'iterations':len(pd.read_csv(history)),'last_best_swap_delta':float(pd.read_csv(history).iloc[-1].best_delta)})
            print('candidate',variant,'k',k,'seconds',round(time.time()-started,1),flush=True)
    pd.DataFrame(rows).to_csv(O/'candidate_constraint_summary.csv',index=False)
    model=W/'locked_models';model.mkdir(exist_ok=True);model_files=[];lookup={int(s):i for i,s in enumerate(raw.stay_id)}
    for k in KS:
        for variant in ['original_frozen','optimized_observed','greedy_all_candidates','greedy_complete_candidates']:
            if variant=='original_frozen':medframe=pd.read_parquet(F/f'equal_feature_k{k}_medoids.parquet').sort_values('cluster');indices=np.array([lookup[int(s)] for s in medframe.stay_id])
            elif variant=='optimized_observed':indices=np.load(W/f'optimization/observed/k{k}_selected_local.npz')['medoids']
            else:indices=np.load(dest/f'{"all" if variant=="greedy_all_candidates" else "complete"}_k{k}_local.npz')['medoids']
            p=model/f'{variant}_k{k}.parquet';raw.iloc[indices][IDS+FS].assign(cluster=np.arange(1,k+1)).to_parquet(p,index=False);model_files.append(p)
    freeze={'status':'locked_before_validation','date':'2026-10-29','variants':['original_frozen','optimized_observed','greedy_all_candidates','greedy_complete_candidates'],'k':KS,'features':FS,'limits':LIM,'protocol_sha256':sha(O/'protocol.json'),'files':{p.name:sha(p) for p in model_files},'candidate_toy_bruteforce_checks':nchecks,'clinical_k_selected':None,'outcomes_used':False}
    dump(model/'manifest.json',freeze);dump(O/'candidate_manifest.json',freeze)
def rect(raw,med):
    # Do not replace undefined distances. Use precisely the old frozen float32 accumulation.
    from importlib.util import spec_from_file_location,module_from_spec
    sp=spec_from_file_location('locked_assign',F/'assign_frozen.py');mod=module_from_spec(sp);sp.loader.exec_module(mod)
    return mod.distance_to_medoids(raw,med,CFG)
def mapped(a,b,k):
    counts=np.bincount(a*k+b,minlength=k*k).reshape(k,k);r,c=linear_sum_assignment(-counts);mapping=np.empty(k,int);mapping[c]=r;return mapping[b],mapping
def eligibility(frame):
    e=CFG['eligibility'];ok=frame[FS].notna().sum(axis=1)>=e['minimum_features_observed'];ok &= sum(frame[ff].notna().any(axis=1) for ff in CFG['domains'].values())>=e['minimum_domains_observed']
    for name in e['required_domains']:ok &= frame[CFG['domains'][name]].notna().any(axis=1)
    return ok.to_numpy()
def holdout(raw,split):
    lock=read(W/'locked_models/manifest.json');assert all(sha(W/'locked_models'/p)==h for p,h in lock['files'].items())
    if split=='test':
        gate=read(O/'validation_gate.json');assert gate['technical_pass'] and gate['locked_model_manifest_sha256']==sha(W/'locked_models/manifest.json')
    cols=IDS+['split']+FS;source=B/'data/observation_review/derived_v3/reviewed_unimputed.parquet'
    assert sha(source)==read(O/'protocol.json')['inputs']['reviewed_feature_source']
    frame=pd.read_parquet(source,columns=cols,filters=[('split','=',split)]).sort_values('stay_id').reset_index(drop=True)
    assert frame.split.eq(split).all() and frame.subject_id.is_unique and not set(frame.subject_id)&set(raw.subject_id)
    assert len(frame)==(4713 if split=='validation' else 4714)
    use=frame.loc[eligibility(frame)].reset_index(drop=True);dest=W/split;dest.mkdir(exist_ok=True);use.to_parquet(dest/'eligible_local.parquet',index=False)
    models={};assignments={};coverage=[];profiles=[]
    for k in KS:
        common=np.ones(len(use),bool)
        for variant in lock['variants']:
            med=pd.read_parquet(W/f'locked_models/{variant}_k{k}.parquet');models[variant,k]=med;dd=rect(use,med);good=np.isfinite(dd).all(axis=1);lab=np.full(len(use),-1);lab[good]=dd[good].argmin(axis=1)
            assignments[variant,k]=(lab,good);common &= good
            coverage.append({'split':split,'variant':variant,'k':k,'n_split':len(frame),'n_eligible':len(use),'n_computable':int(good.sum()),'uncomputable':int((~good).sum()),'minimum_cluster_n':int(np.bincount(lab[good],minlength=k).min())})
            np.savez_compressed(dest/f'{variant}_k{k}_assignment_local.npz',labels=lab,computable=good,distance=dd)
        for variant in lock['variants'][1:]:
            profiles.append({'split':split,'k':k,'variant':variant,'paired_n':int(common.sum()),'ari_vs_original_on_common':float(adjusted_rand_score(assignments['original_frozen',k][0][common],assignments[variant,k][0][common]))})
    pd.DataFrame(coverage).to_csv(O/f'{split}_coverage.csv',index=False);pd.DataFrame(profiles).to_csv(O/f'{split}_variant_agreement.csv',index=False)
    # Original observed anchor identities completed using only saved training draws. Heldout query values stay fixed.
    lookup={int(s):i for i,s in enumerate(raw.stay_id)};anchor_rows=[]
    for k in KS:
        med=models['original_frozen',k];indices=[lookup[int(s)] for s in med.stay_id];base_train=rect(raw,med).argmin(axis=1);labelsets=[];training_maps=[]
        for mi in range(1,11):
            completion=pd.read_parquet(OLD/f'mice_{mi:02d}/imputed.parquet')[FS].iloc[indices].reset_index(drop=True)
            trainlab=rect(raw,completion).argmin(axis=1);_,mapping=mapped(base_train,trainlab,k);held=rect(use,completion).argmin(axis=1);labelsets.append(mapping[held]);training_maps.append(mapping)
        labelsets=np.array(labelsets);np.savez_compressed(dest/f'anchor_perturbation_k{k}_local.npz',mapped_labels=labelsets,training_maps=np.array(training_maps))
        change=np.mean(np.array([labelsets[i]!=labelsets[j] for i,j in itertools.combinations(range(10),2)]),axis=0)
        full=use[FS].notna().all(axis=1).to_numpy()
        for group,select in [('all',np.ones(len(use),bool)),('complete',full),('incomplete',~full)]:anchor_rows.append({'split':split,'k':k,'group':group,'n':int(select.sum()),'anchor_pair_disagreement_training_aligned':float(change[select].mean())})
    pd.DataFrame(anchor_rows).to_csv(O/f'{split}_anchor_perturbation.csv',index=False)
    # Stress test only complete holdout query rows, no fitting, hidden reference is comparator only.
    full=use.loc[use[FS].notna().all(axis=1)].reset_index(drop=True);mask_source=raw[FS].isna().to_numpy();mask_source=mask_source[mask_source.any(axis=1)];maskrows=[];flow=[]
    for si,scenario in enumerate(['mcar20','mcar40','empirical_incomplete_pattern']):
        for rep in range(1,6):
            rng=np.random.default_rng(SEED+20000000+(0 if split=='validation' else 1000000)+si*10000+rep)
            hide=mask_source[rng.integers(0,len(mask_source),len(full))] if si==2 else rng.random((len(full),14))<(.2 if si==0 else .4)
            masked=full.copy();masked[FS]=masked[FS].mask(hide);eligible=eligibility(masked);ix=np.flatnonzero(eligible)
            masksfile=dest/f'{scenario}_r{rep:02d}_local.npz';np.savez_compressed(masksfile,hidden=hide,eligible_index=ix)
            flow.append({'split':split,'scenario':scenario,'replicate':rep,'complete_n':len(full),'eligible_n':len(ix)})
            for k in KS:
                refs={};preds={};common=np.ones(len(ix),bool)
                for variant in lock['variants']:
                    med=models[variant,k];base=rect(full.iloc[ix],med);perturbed=rect(masked.iloc[ix],med);good=np.isfinite(base).all(axis=1)&np.isfinite(perturbed).all(axis=1);common &= good
                    refs[variant]=base.argmin(axis=1);preds[variant]=perturbed.argmin(axis=1)
                for variant in lock['variants']:
                    a=refs[variant][common];b=preds[variant][common];aligned,_=mapped(a,b,k)
                    maskrows.append({'split':split,'scenario':scenario,'replicate':rep,'k':k,'variant':variant,'paired_n':int(common.sum()),'ari_vs_own_unmasked':float(adjusted_rand_score(a,b)),'identity_disagreement':float(np.mean(a!=b)),'aligned_disagreement':float(np.mean(a!=aligned))})
    pd.DataFrame(maskrows).to_csv(O/f'{split}_mask_detail.csv',index=False);pd.DataFrame(flow).to_csv(O/f'{split}_mask_flow.csv',index=False)
    pd.DataFrame(maskrows).groupby(['scenario','k','variant']).agg(ari_mean=('ari_vs_own_unmasked','mean'),ari_min=('ari_vs_own_unmasked','min'),ari_max=('ari_vs_own_unmasked','max'),identity_disagreement_mean=('identity_disagreement','mean'),paired_n_min=('paired_n','min'),paired_n_max=('paired_n','max')).reset_index().to_csv(O/f'{split}_mask_summary.csv',index=False)
    gate={'split':split,'technical_pass':True,'n_split':len(frame),'n_eligible':len(use),'n_complete':len(full),'training_overlap':0,'locked_model_manifest_sha256':sha(W/'locked_models/manifest.json'),'selection_based_on_results':False,'rule_changes_after_validation':False,'outcomes_loaded':False,'no_heldout_model_fitting':True}
    dump(O/f'{split}_gate.json',gate);print('holdout',split,'eligible',len(use),'complete',len(full),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['all','optimize','candidates','holdout'],default='all');args=ap.parse_args()
    establish_protocol();raw=load_training();start=time.time()
    if args.phase in ['all','optimize']:optimize(raw)
    if args.phase in ['all','candidates']:candidate_audit(raw)
    if args.phase in ['all','holdout']:
        holdout(raw,'validation');holdout(raw,'test')
    dump(O/'run_status.json',{'status':'computations_complete_independent_qa_pending','phase':args.phase,'elapsed_seconds':time.time()-start,'script_sha256':sha(__file__)})
if __name__=='__main__':main()
