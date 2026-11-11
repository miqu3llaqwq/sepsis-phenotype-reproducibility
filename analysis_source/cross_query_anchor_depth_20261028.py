"""Cross all 10 saved query/anchor draws, preserving anchor identities and labels."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import run_missingness_depth_20261028 as R
O=R.O;W=R.W/'natural';F=R.F;OLD=R.OLD
def pair_disagreement(labels,k,axis):
    # A mean over all unordered draw pairs, calculated through cluster counts.
    counts=np.stack([(labels==c).sum(axis=axis) for c in range(k)],axis=0)
    return 1-(counts*(counts-1)).sum(axis=0)/(10*9)
def main():
    t=time.time();raw=R.load_source();x=raw[R.FEATS].to_numpy(float);obs=np.isfinite(x);complete=obs.all(axis=1)
    contract={'date':'2026-10-28','status':'post-hoc_additional_design_before_factorial_run',
      'motivation':'Observed training medoid identities have 4-8 missing features. Factorially separate query draw and anchor draw rather than attributing joint changes to patient missingness.',
      'design':'all 10x10 saved imputation combinations for each k2-6; same observed medoid patient identities, fixed P1-P99; no new fit',
      'labels':'identity-preserving labels; no Hungarian remapping because center patient identities are fixed',
      'query_effect':'mean pairwise assignment disagreement across 10 query draws at each fixed anchor draw, averaged over 10 anchor draws',
      'anchor_effect':'mean pairwise assignment disagreement across 10 anchor draws at each fixed query draw, averaged over 10 query draws',
      'interpretation':'descriptive factorial sensitivity, not variance decomposition or additive causal shares; not an outcome probability',
      'source_protocol_sha256':R.sha(O/'protocol.json')}
    p=O/'query_anchor_factorial_protocol.json'
    if p.exists():assert R.read(p)==contract
    else:R.dump(p,contract)
    alignment_contract={'design':'additional label-permutation sensitivity before aligned computation','reason':'Anchor identities are fixed, but cluster labels remain statistical labels. Report both identity-preserving and independently optimal Hungarian-aligned pair disagreement so identity switches are not misread as structural changes.','all_pairs':'10 held-fixed levels x 45 pairs per factor per k; average, not independent trials.'}
    ap=O/'query_anchor_alignment_protocol.json'
    if ap.exists():assert R.read(ap)==alignment_contract
    else:R.dump(ap,alignment_contract)
    draws=[pd.read_parquet(OLD/f'mice_{i:02d}/imputed.parquet')[R.FEATS].to_numpy(float) for i in range(1,11)]
    ids={int(v):i for i,v in enumerate(raw.stay_id)};rows=[];profiles=[];patients=[]
    for k in R.KS:
        med=pd.read_parquet(F/f'equal_feature_k{k}_medoids.parquet').sort_values('cluster');ind=np.array([ids[int(s)] for s in med.stay_id])
        labels=np.empty((10,10,len(raw)),np.int8)
        for i in range(10):
            for j in range(10):labels[i,j]=R.distance(draws[i],draws[j][ind]).argmin(axis=1)
        query=pair_disagreement(labels,k,axis=0).mean(axis=0)
        anchor=pair_disagreement(labels,k,axis=1).mean(axis=0)
        assert np.all(query[complete]==0)
        aq=np.zeros(len(raw));aa=np.zeros(len(raw))
        for fixed in range(10):
            for i in range(10):
                for j in range(i):
                    aq+=(R.align(labels[i,fixed],labels[j,fixed],k)[0]!=labels[i,fixed])
                    aa+=(R.align(labels[fixed,i],labels[fixed,j],k)[0]!=labels[fixed,i])
        aq/=450;aa/=450
        assert np.all(aq[complete]==0)
        diagonal=np.array([labels[i,i] for i in range(10)])
        joint=pair_disagreement(diagonal,k,axis=0)
        np.savez_compressed(W/f'factorial_k{k}_labels_local.npz',labels=labels,medoid_row=ind)
        patients.append(raw[R.IDS].assign(k=k,complete=complete,query_draw_disagreement=query,anchor_draw_disagreement=anchor,joint_draw_disagreement=joint,query_aligned_disagreement=aq,anchor_aligned_disagreement=aa))
        for group,select in [('all',np.ones(len(raw),bool)),('complete',complete),('incomplete',~complete)]:
            rows.append({'k':k,'group':group,'n':int(select.sum()),'query_draw_disagreement_mean':float(query[select].mean()),'anchor_draw_disagreement_mean':float(anchor[select].mean()),'joint_draw_disagreement_mean':float(joint[select].mean()),'query_invariant_n':int((query[select]==0).sum()),'anchor_invariant_n':int((anchor[select]==0).sum()),'query_aligned_disagreement_mean':float(aq[select].mean()),'anchor_aligned_disagreement_mean':float(aa[select].mean())})
        for c,idx in enumerate(ind):
            for j,f in enumerate(R.FEATS):
                z=np.array([d[idx,j] for d in draws]);profiles.append({'k':k,'cluster':c+1,'feature':f,'originally_missing':bool(~obs[idx,j]),'n_missing_features_at_medoid':int((~obs[idx]).sum()),'completed_scaled_sd':float(np.std((np.clip(z,R.LO[j],R.HI[j])-R.LO[j])/(R.HI[j]-R.LO[j]),ddof=1))})
        print('crossed query/anchor k',k,flush=True)
    pd.DataFrame(rows).to_csv(O/'query_anchor_factorial_summary.csv',index=False)
    pd.DataFrame(profiles).to_csv(O/'observed_medoid_missingness_and_variation.csv',index=False)
    pd.concat(patients,ignore_index=True).to_parquet(W/'query_anchor_patient_sensitivity_local.parquet',index=False)
    R.dump(O/'query_anchor_factorial_manifest.json',{'status':'complete','k':R.KS,'combinations_per_k':100,'n':len(raw),'complete_n':int(complete.sum()),'complete_query_invariance_passed':True,'seconds':time.time()-t,'script_sha256':R.sha(__file__)})
if __name__=='__main__':main()
