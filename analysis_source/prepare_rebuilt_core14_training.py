"""Apply existing training eligibility; save provisional limits without fitting k."""
from pathlib import Path
import ast,hashlib,json,types
import numpy as np
import pandas as pd
from rebuild_corrected_mimic_branch import BRANCH
# Load only the three unchanged pure data functions. Importing the full module
# also imports the PAM package, which is unnecessary for eligibility checks.
tree=ast.parse((BRANCH/'analysis/26_domain_balanced_v7.py').read_text(encoding='utf-8'))
names={'eligibility','incompatible_positions','domain_limits'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(nodes)==len(names)
namespace={'np':np,'pd':pd}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'historical_pure_functions','exec'),namespace)
v=types.SimpleNamespace(**{n:namespace[n] for n in names})
c=json.loads((BRANCH/'config/phenotype_primary_v1.json').read_text())
features=c['features_primary'];cfg={'features':features,'domains':c['domains'],'eligibility':c['eligibility']}
source=BRANCH/'data/observation_review/derived_v3/reviewed_unimputed.parquet'
frame=pd.read_parquet(source,columns=['subject_id','stay_id','hadm_id','split']+features)
train=frame[frame.split.eq('train')].sort_values('stay_id').reset_index(drop=True)
mask=v.eligibility(train,cfg)
eligible=train[mask].reset_index(drop=True)
observed=np.isfinite(eligible[features].to_numpy(dtype=float))
patterns=(observed.astype(np.uint32)*(1<<np.arange(len(features),dtype=np.uint32))).sum(axis=1)
unique=np.unique(patterns)
bad_patterns=unique[((unique[:,None]&unique[None,:])==0).any(axis=1)]
bad=np.isin(patterns,bad_patterns)
# Independent full comparison over unique missingness patterns, using the exact
# historical implementation, validates the compressed-pattern compatibility test.
_,positions=np.unique(patterns,return_index=True)
representatives=eligible.iloc[positions].reset_index(drop=True)
original_bad=v.incompatible_positions(representatives,features,cfg['domains'])
assert set(patterns[positions][original_bad])==set(bad_patterns)
final=eligible[~bad].reset_index(drop=True)
limits=v.domain_limits(final,cfg['domains'])
assert all(np.isfinite(x).all() and x[1]>x[0] for x in limits.values())
dest=BRANCH/'data/core14_preparation';dest.mkdir(exist_ok=True)
final.to_parquet(dest/'eligible_train_unimputed.parquet',index=False)
manifest={'status':'eligibility_and_limits_prepared_model_fitting_pending','n_train':len(train),'n_with_any_core14':int(train[features].notna().any(axis=1).sum()),'n_eligible_before_pair_compatibility':len(eligible),'n_pairwise_incompatible_excluded':int(bad.sum()),'n_eligible':len(final),'features':features,'eligibility':cfg['eligibility'],'limits':limits,'limits_fit_scope':'eligible_training_rows_only','compatibility_crosscheck':'exact_historical_implementation_on_all_unique_patterns','outcomes_loaded':False,'validation_test_used_for_fit':False,'MICE_PMM_completed':False,'new_medoids_frozen':False,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(dest/'preparation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
pd.DataFrame([{'step':'training_split','n':len(train)},{'step':'at_least_one_core14','n':int(train[features].notna().any(axis=1).sum())},{'step':'feature_and_domain_eligible','n':len(eligible)},{'step':'pairwise_compatible','n':len(final)}]).to_csv(BRANCH/'tables/core14_training_flow.csv',index=False)
print(json.dumps({k:manifest[k] for k in ['status','n_train','n_with_any_core14','n_eligible_before_pair_compatibility','n_pairwise_incompatible_excluded','n_eligible','compatibility_crosscheck']},indent=2))
