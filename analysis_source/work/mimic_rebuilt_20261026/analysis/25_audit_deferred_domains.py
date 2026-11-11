"""TRAIN-only aggregate audit of existing deferred organ features and flags."""
from pathlib import Path
import importlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
m=importlib.import_module('18_physiology_v5')
DEFERRED=['gcs_min','bilirubin','pao2_fio2_ratio']
FLAGS=['gcs_unable_n','arterial_samples_n','oxygen_pairs_n','bilirubin_invalid_n','bilirubin_unavailable_n','bilirubin_unit_unknown_n']

def main():
    v6=json.loads((ROOT/'logs/clustering_v6_latest.json').read_text(encoding='utf-8'))
    v3=json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))
    v2=json.loads((ROOT/'logs/v2_latest.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'config/frozen_split_v2.json').read_text(encoding='utf-8'))
    assert m.c.sha(frozen['dataset_path'])==frozen['dataset_sha256']
    path=Path(v3['data_path'])/'reviewed_unimputed.parquet'
    original=json.loads((Path(v3['report_path'])/'run_manifest.json').read_text(encoding='utf-8'))
    assert m.c.sha(path)==original['output_sha256']['reviewed_unimputed.parquet']
    frame=pd.read_parquet(path,columns=m.IDS+['split']+m.FEATURES+DEFERRED)
    frame=frame[frame.split.eq('train')].sort_values('stay_id').reset_index(drop=True)
    base=pd.read_parquet(Path(v2['data_path'])/'analysis_dataset.parquet',columns=m.IDS+['split']+DEFERRED+FLAGS)
    base=base[base.split.eq('train')].sort_values('stay_id').reset_index(drop=True)
    assert frame[m.IDS].equals(base[m.IDS])
    # v3 is reviewed/cleaned; v2 is retained as the original extraction
    # reference. Values may differ after source-priority and conflict rules.
    selected=m.eligible(frame,8);sub=frame[selected];flags=base.loc[selected]
    rows=[];differences=[]
    for f in DEFERRED:
        left=frame[f].to_numpy(dtype=float);right=base[f].to_numpy(dtype=float)
        comparable=np.isfinite(left)&np.isfinite(right)
        differences.append({'feature':f,'both_observed':int(comparable.sum()),
            'different_value_n':int(np.sum(comparable&(~np.isclose(left,right,rtol=1e-10,atol=1e-12)))),
            'v3_only_observed':int(np.sum(np.isfinite(left)&~np.isfinite(right))),
            'v2_only_observed':int(np.sum(~np.isfinite(left)&np.isfinite(right)))})
    for scope,a in [('original_train',frame),('fixed_v6_train',sub)]:
        for f in DEFERRED:rows.append({'scope':scope,'feature':f,'n':len(a),
            'n_observed':int(a[f].notna().sum()),'n_missing':int(a[f].isna().sum()),'missing_pct':float(100*a[f].isna().mean())})
    tables=Path(v6['tables_path']);reports=Path(v6['report_path'])
    pd.DataFrame(rows).to_csv(tables/'deferred_domain_coverage.csv',index=False)
    pd.DataFrame(differences).to_csv(tables/'deferred_domain_v2_v3_value_differences.csv',index=False)
    reasons=[]
    def count(name,mask):reasons.append({'overlapping_flag':name,'n_fixed_v6_train':int(mask.sum()),'denominator':len(sub)})
    count('GCS missing with at least1 unassessable recorded time',sub.gcs_min.isna()&flags.gcs_unable_n.fillna(0).gt(0))
    count('GCS missing without positive unassessable flag;cause unresolved',sub.gcs_min.isna()&~flags.gcs_unable_n.fillna(0).gt(0))
    count('P/F missing with no qualifying arterial sample in extracted window',sub.pao2_fio2_ratio.isna()&flags.arterial_samples_n.fillna(0).eq(0))
    count('P/F missing with qualifying arterial sample but no paired ratio',sub.pao2_fio2_ratio.isna()&flags.arterial_samples_n.fillna(0).gt(0))
    for name in ['bilirubin_invalid_n','bilirubin_unavailable_n','bilirubin_unit_unknown_n']:
        count('bilirubin missing with positive '+name,sub.bilirubin.isna()&flags[name].fillna(0).gt(0))
    count('all19 variables observed',sub[m.FEATURES+DEFERRED].notna().all(axis=1))
    pd.DataFrame(reasons).to_csv(tables/'deferred_domain_existing_flags.csv',index=False)
    report='# 已有神经、肝脏与氧合字段的可获得性审查\n\n仅使用原固定训练数据和v6合格人群，不查看结局或验证/测试表现。三项在v3中均保留；v3是经来源优先、去重/冲突处理和范围筛查后的复核表示，v2为原始提取参考，二者数值差异统计于deferred_domain_v2_v3_value_differences.csv。缺失率分母分别列明，不能混用。覆盖及既有标志见deferred_domain_coverage.csv和deferred_domain_existing_flags.csv。\n\n'
    report+='GCS至少一次不可评分标志不是全部缺失的唯一解释；没有该标志也不证明从未评估。P/F无合格动脉样本指当前窗口/存储可获得性/标本条件下未形成合格样本，不等于临床没做血气；有动脉样本而无配对比值需核查FiO2可用性。胆红素迟录、技术范围与未知单位标志可重叠，不能将计数相加当互斥患者排除流程。没有重查全部原始候选观测，因此不能认证来源完整性。\n\n'
    report+='合理的后续修订是分领域评审这些字段的来源与缺失机制，明确不完整神经/肝脏/氧合的推断限制，另行登记纳入方案与对应插补验证。不能将不可评分GCS填为正常、无P/F填为正常或使用t0后的观测补齐；也不能以死亡率差异选择是否纳入。19项全实测比例仅作覆盖诊断，不自动改为主完整病例队列。\n'
    (reports/'deferred_domain_audit.md').write_text(report,encoding='utf-8')
    audit={'status':'aggregate_audit_completed','outcomes_loaded':False,'validation_test_used':False,
        'original_cohort_hash_unchanged':True,'v3_source_hash_unchanged':True,
        'deferred_values_exactly_match_original_v2':False,'v2_v3_differences_quantified':True,'source_completeness_certified':False,
        'source_flag_semantics':'overlapping,not mutually exclusive reasons',
        'script_sha256':m.c.sha(__file__),'n_fixed_train':len(sub)}
    m.dump(reports/'deferred_domain_audit.json',audit)
    (reports/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'coverage':rows,'flags':reasons,'verification':audit},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
