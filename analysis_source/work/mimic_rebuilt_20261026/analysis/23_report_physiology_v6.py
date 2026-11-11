"""Aggregate v6 diagnostic report and transparent variable-role audit."""
from pathlib import Path
import importlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
m=importlib.import_module('18_physiology_v5')

def table(frame):
    def v(x):return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(v(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    loc=json.loads((ROOT/'logs/clustering_v6_latest.json').read_text(encoding='utf-8'))
    assert loc['status']=='computation_verified_report_pending'
    tables=Path(loc['tables_path']);reports=Path(loc['report_path'])
    metrics=pd.concat([pd.read_csv(tables/name) for name in ['A_reference_metrics.csv','B_median_robust_metrics.csv','C_reference_metrics.csv','D_pooled_robust_metrics.csv','CC_original_metrics.csv','CC_robust_metrics.csv']],ignore_index=True)
    metrics.to_csv(tables/'all_conditions_metrics.csv',index=False)
    comparisons=pd.read_csv(tables/'paired_condition_comparisons.csv')
    agreement=pd.read_csv(tables/'D_individual_vs_pooled.csv').groupby('k').agg(
        mean_ari=('ari','mean'),min_ari=('ari','min'),mean_alignment=('aligned_agreement','mean')).reset_index()
    agreement=agreement.merge(pd.read_csv(tables/'D_assignment_uncertainty.csv'),on='k')
    agreement.to_csv(tables/'D_imputation_agreement_summary.csv',index=False)
    complete_comparisons=comparisons[comparisons.comparison.str.startswith('CC')]
    correlations=pd.read_csv(tables/'observed_feature_correlations.csv')
    pairs=correlations[(correlations.feature_a.eq('sbp')&correlations.feature_b.eq('map'))|(correlations.feature_a.eq('creatinine')&correlations.feature_b.eq('bun'))]
    feature_review=ROOT/'outputs/reports/v6_feature_selection_review.md'
    # Roles are version-specific and do not overwrite v2/v3 source definitions.
    dictionary=pd.read_csv(ROOT/'feature_dictionary.csv')
    roles=[]
    for feature in m.FEATURES+['age','gender','gcs_min','bilirubin','pao2_fio2_ratio']:
        selected=feature in m.FEATURES
        original=dictionary[dictionary.analysis_name.eq(feature)]
        source=original.iloc[0] if len(original) else {}
        note='unchanged v6 distance input; clinical coverage and dependence reviewed'
        if feature in ['sbp','map']:note='correlated BP measures; same domain may have larger combined influence; no automatic removal'
        if feature in ['creatinine','bun']:note='related but clinically nonidentical renal/metabolic measures; correlation alone does not justify removal'
        if feature in ['age','gender']:note='description only in v6; future prediction role not frozen'
        if feature in ['gcs_min','bilirubin','pao2_fio2_ratio']:note='retained in source dataset, not v6 distance; high training missingness is not proof of low clinical importance'
        roles.append({'feature':feature,'source_domain':source.get('domain',''),
            'source_unit':source.get('unit',''),'v6_distance_input':selected,
            'v6_role':'fixed diagnostic physiology input' if selected else 'descriptive/deferred for separately registered review',
            'notes':note})
    (ROOT/'data_dictionary').mkdir(exist_ok=True)
    pd.DataFrame(roles).to_csv(ROOT/'data_dictionary/phenotype_v6_variable_roles.csv',index=False)
    figdir=ROOT/'outputs/figures/clustering_v6'/loc['run_id'];figdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names={'A':'A: median/original','B_median_robust':'B: median/P1-P99','C':'C: RF-PMM/original','D_pooled_robust':'D: RF-PMM/P1-P99'}
    for tag,label in names.items():
        s=metrics[metrics.analysis.eq(tag)];axes[0].plot(s.k,s.silhouette_mean,'o-',label=label)
    axes[0].set(xlabel='k',ylabel='Mean silhouette');axes[0].set_xticks([2,3,4,5,6]);axes[0].legend(fontsize=8)
    for tag,label in [('A_vs_B','Scale, median'),('C_vs_D','Scale, RF-PMM'),('B_vs_D','Imputation, P1-P99')]:
        s=comparisons[comparisons.comparison.eq(tag)];axes[1].plot(s.k,s.ari,'o-',label=label)
    axes[1].set(xlabel='k',ylabel='Adjusted Rand index',ylim=(-.05,1.05));axes[1].set_xticks([2,3,4,5,6]);axes[1].legend(fontsize=8)
    fig.suptitle('Fixed TRAIN diagnostic; no outcome or holdout selection')
    fig.tight_layout();fig.savefig(figdir/'scale_and_imputation.png',dpi=180);fig.savefig(figdir/'scale_and_imputation.pdf');plt.close(fig)
    body='# v6固定人群诊断：计算与解释\n\n'
    body+='已完成固定14,098训练患者的四条件对照，以及6,291名16项全实测患者的无插补诊断。仅训练预测变量与必要标识，原队列/分区/观测不变；未用结局或验证/测试决定变量、尺度或k。此方案登记于看到v5结果之后，是诊断开发，临床k未冻结。\n\n'
    body+='A/C为核验后引用的v5原尺度结果，B/D分别为相同中位数/相同10套随机插补的P1–P99截断距离表示。端点由固定人群的未插补实测值估计，原始数据不截断。分位选择是压力测试，不是最佳界限。\n\n'
    body+='## 条件内的分群指标\n\n'+table(metrics)+'\n\n'
    body+='各条件使用相同种子、5次随机初始化、FastPAM1、k2–6。不同表示下loss的量级不能直接比较优劣，轮廓高低也不能单独认证自然离散簇。\n\n'
    body+='## 同一患者的条件比较\n\n'+table(comparisons[~comparisons.comparison.str.startswith('CC')])+'\n\n'
    body+='这些比较隔离固定人群中的尺度与插补依赖；差异不证明任一插补的临床真值更准确。D复用v5全训练插补，不是子抽样内重新拟合的MI稳定性验证。\n\n'
    body+='## D跨插补的一致性\n\n'+table(agreement)+'\n\n'
    body+='支持度为10套插补下标签对齐经验频率，不是校准概率。平均距离可能掩盖各套分群的不确定性。\n\n'
    body+='## 全部实测者的对照\n\n'+table(complete_comparisons)+'\n\n'
    body+='完整病例拟合使用固定原/稳健尺度，没有重新估计端点。完整病例无缺失，可以排除该对照内部的插补依赖；其分群与全人群的差异仍可能来自样本人群与中心变化。该群体的检查并不能证明缺失机制可忽略、没有选择偏倚或可推广到全部脓毒症患者；基线与观测差异见complete_vs_incomplete_description.csv。\n\n'
    body+='## 指标选择是否有问题\n\n'+table(pairs)+'\n\n'
    body+='血压两项及肌酐/尿素氮有较高相关，等权变量可让相关领域有较大综合影响；两指标并非完全同义，相关性不是自动删除依据。v2的训练缺失>50%筛选是可获得性的操作选择；GCS、胆红素、氧合比退出后，当前表示对神经、肝脏和氧合功能覆盖不完整。故16项可用于限定的生理诊断，不能宣称全面器官表型或复制某篇论文的变量体系。详细文献及角色审查见outputs/reports/v6_feature_selection_review.md；当前角色表data_dictionary/phenotype_v6_variable_roles.csv。\n\n'
    body+='下一阶段需按研究目标及可获得性明确器官领域与变量理由，分别登记冗余变量/领域权重及高缺失功能项的处理比较，再进行严格训练开发。不能在本轮逐个删变量找好看的分群，不因不稳定就判定原始数据或拉取错误。若缺乏可靠结构，保留未识别稳健离散表型的结论。完整病例或稳健表示更高的指标不足以自动选择主模型。\n'
    (reports/'stage_report.md').write_text(body,encoding='utf-8')
    (ROOT/'outputs/reports/clustering_v6_latest.md').write_text(body,encoding='utf-8')
    manifest=json.loads((reports/'run_manifest.json').read_text(encoding='utf-8'))
    manifest.update(status='completed_diagnostic_clinical_gate_pending',report_script_sha256=m.c.sha(__file__),
        aggregate_table_sha256={str(p.relative_to(tables)):m.c.sha(p) for p in tables.glob('*.csv')},
        feature_review_available=feature_review.exists(),proper_mi_or_convergence_certified=False,
        full_pipeline_resampling_done=False,clinical_phenotype_certified=False)
    (reports/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    m.dump(reports/'run_manifest.json',manifest)
    loc['status']=manifest['status'];m.dump(ROOT/'logs/clustering_v6_latest.json',loc)
    notice=f'> **v6诊断已完成（{loc["run_id"]}）：固定人群的尺度×插补及全部实测者对照。** 入口 `logs/clustering_v6_latest.json`，报告 `outputs/reports/clustering_v6_latest.md`；变量理由仍需器官覆盖/冗余审查，临床k未冻结，未开展留出或亚群预测。\n\n'
    for name in ['research tasklist.md','analysis/03_feature_list.md']:
        p=ROOT/name;text=p.read_text(encoding='utf-8');p.write_text(notice+text,encoding='utf-8')
    with (ROOT/'logs/analysis_log.md').open('a',encoding='utf-8') as f:f.write('\n\n## v6诊断完成 '+loc['run_id']+'\n\n'+notice+'\n'+table(comparisons[comparisons.k.eq(2)&~comparisons.comparison.str.startswith('CC')])+'\n')
    print(json.dumps({'status':loc['status'],'main_comparisons':comparisons[~comparisons.comparison.str.startswith('CC')].to_dict('records'),
        'D_agreement':agreement.to_dict('records'),'complete_case_metrics':metrics[metrics.analysis.str.startswith('CC')].to_dict('records')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
