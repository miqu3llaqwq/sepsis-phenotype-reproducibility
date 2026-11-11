"""Publish aggregate completion notes without altering patient inputs."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    loc=json.loads((ROOT/'logs/clustering_v5_latest.json').read_text(encoding='utf-8'))
    assert loc['status']=='completed_development_clinical_gate_pending'
    tables=Path(loc['tables_path']);reports=Path(loc['report_path'])
    coverage=pd.read_csv(tables/'baseline/coverage_counts.csv')
    baseline=pd.read_csv(tables/'baseline8_summary.csv')
    pooled=pd.read_csv(tables/'pooled10_summary.csv')
    compare=pd.read_csv(tables/'median_vs_pooled_comparison.csv')
    k2=compare[compare.k.eq(2)].iloc[0]
    uncertainty=pooled[pooled.k.eq(3)].iloc[0]
    verification=json.loads((reports/'verification.json').read_text(encoding='utf-8'))
    assert verification['original_cohort_hash_unchanged'] and verification['synthetic_tests']==10
    body='# v5已完成：覆盖门槛、插补与尺度敏感性\n\n'
    body+='本轮只用固定训练数据，计算和核验完成，最终临床表型k尚未冻结。原队列/分区与输入哈希未变，验证/测试未参与开发。8/16是登记主操作标准，不是已证实最佳数字；6/10仅开展中位数基线敏感性。\n\n'
    body+='| 插补前至少实测 | 合格训练人数 | 原训练保留比例 |\n| --- | ---: | ---: |\n'
    for row in coverage.itertuples():body+=f'| {row.minimum_observed}/16 | {row.n_eligible:,} | {100*row.fraction_retained:.2f}% |\n'
    body+='\n覆盖标准均要求至少各1项生命体征和化验。8项标准排除3,536名训练患者；符合与不符合者的年龄中位67/70岁，女性比例41.47%/45.25%。仅描述训练记录，不证明无筛选偏倚。\n\n'
    body+='已完成的主要发现：\n\n'
    body+='- 二簇共同患者8vs6 ARI=0.519，13.9%对齐标签改变；8vs10 ARI=0.449，16.5%改变。门槛会影响结构，未确认某一个最优。\n'
    body+=f'- 8项中位数基线二簇/三簇平均重采样ARI分别{baseline.loc[baseline.k.eq(2),"stability_ari_mean"].iloc[0]:.3f}/{baseline.loc[baseline.k.eq(3),"stability_ari_mean"].iloc[0]:.3f}；三簇最差一次约0.430，平均稳定不等于全部重复稳定。\n'
    body+=f'- 同样8项资格，中位数与10套RF-PMM平均距离二簇ARI={k2.median_vs_pooled_ari:.3f}，约{100*(1-k2.median_vs_pooled_aligned):.1f}%对齐标签改变；说明插补选择也是重要依赖。\n'
    body+=f'- 三簇的个别插补对平均距离分群平均ARI={uncertainty.imputation_ari_mean:.3f}；{100*uncertainty.fraction_support_below_0_8:.1f}%患者的分配标签在10套插补中支持度不足80%。这些是经验一致性指标，不是置信概率。\n'
    body+='- 严格筛查但保留主尺度时二簇标签不变；重拟合严格尺度则ARI=0.594，约11.5%改变。定向检查呼吸频率、血氧、白细胞的23项重点记录，来源重算一致，未将其自动判错或删去。\n'
    body+='- 人工遮蔽实测值后，3次RF-PMM均值的16变量等权IQR归一化MAE约0.681，中位数0.690；RMSE约1.012与1.073。改善有限，且不代表真实临床缺失的数值已验证。\n\n'
    body+='当前结论：尚无足够证据确认唯一最佳完整性门槛或稳健离散生理表型。保留8为本轮已登记操作标准，不凭更好看的结果改阈值或k。下一研究步骤应针对观测缺失、插补及距离缩放进行有依据的验证和透明修订；未经冻结，不进入有利的亚群预测结论。\n\n'
    body+=f'详细报告：{reports / "stage_report.md"}\n\n方法与结果边界：protocol/physiology_v5.md、config/clustering_v5.json；数值核验10项合成测试及真实产物核对通过，不是临床真实性认证。\n'
    (ROOT/'outputs/reports/v5_completion_summary.md').write_text(body,encoding='utf-8')
    (reports/'completion_summary.md').write_text(body,encoding='utf-8')
    notice=f'> **最新完成 v5（{loc["run_id"]}）：16项生理训练开发、6/8/10覆盖比较、10套随机插补和定向尺度回查已完成。** 8项不是已确认最佳，门槛、插补及尺度均影响分群；临床k和亚群预测未冻结。当前入口 `logs/clustering_v5_latest.json`，摘要 `outputs/reports/v5_completion_summary.md`，修订规则 `protocol/physiology_v5.md`。下文v3/v4进度为历史记录。\n\n'
    for name in ['research tasklist.md','protocol/analysis_v2.md','outputs/reports/v5_method_recommendation.md']:
        p=ROOT/name;text=p.read_text(encoding='utf-8')
        if not text.startswith(notice):p.write_text(notice+text,encoding='utf-8')
    log=ROOT/'logs/analysis_log.md'
    marker=f'## v5开发数值完成 {loc["run_id"]}'
    if marker not in log.read_text(encoding='utf-8'):
        with log.open('a',encoding='utf-8') as f:
            f.write('\n\n'+marker+'\n\n'+body.split('已完成的主要发现：')[1])
    (reports/'21_finalize_v5.py').write_bytes(Path(__file__).read_bytes())
    print('Saved aggregate completion summary and current project entry points.',flush=True)

if __name__=='__main__':main()
