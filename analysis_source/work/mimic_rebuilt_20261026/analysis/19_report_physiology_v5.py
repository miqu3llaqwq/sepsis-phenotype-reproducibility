"""Verify and report completed v5 development, without clinical adjudication."""
from pathlib import Path
import importlib
import json
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
m=importlib.import_module('18_physiology_v5')
review=importlib.import_module('16_report_clustering')

def table(df):
    def value(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(value(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    loc=json.loads((ROOT/'logs/clustering_v5_latest.json').read_text(encoding='utf-8'))
    reports=Path(loc['report_path']);tables=Path(loc['tables_path']);data=Path(loc['data_path'])
    initial_manifest=json.loads((reports/'run_manifest.json').read_text(encoding='utf-8'))
    assert m.c.sha(ROOT/'config/clustering_v5.json')==initial_manifest['config_sha256']
    for stage in ['baseline','multiple_imputation']:
        status=json.loads((reports/(stage+'_status.json')).read_text(encoding='utf-8'))
        assert status['status']=='complete'
    cfg,frames=m.load(loc);raw=frames['main'][m.eligible(frames['main'],8)].reset_index(drop=True)
    frozen=json.loads((ROOT/'config/frozen_split_v2.json').read_text(encoding='utf-8'))
    assert m.c.sha(frozen['dataset_path'])==frozen['dataset_sha256']
    btab=tables/'baseline';mtab=tables/'multiple_imputation'
    bdata=data/'baseline';mdata=data/'multiple_imputation'
    coverage=pd.read_csv(btab/'coverage_counts.csv')
    stability=pd.read_csv(btab/'coverage8_stability.csv').groupby('k').agg(
        stability_ari_mean=('ari','mean'),stability_ari_min=('ari','min')).reset_index()
    baseline=pd.read_csv(btab/'coverage8_metrics.csv').merge(stability,on='k')
    pooled=pd.read_csv(mtab/'pooled10_metrics.csv')
    mi_agreement=pd.read_csv(mtab/'individual_vs_pooled_agreement.csv').groupby('k').agg(
        imputation_ari_mean=('ari','mean'),imputation_ari_min=('ari','min'),
        imputation_aligned_mean=('aligned_agreement','mean')).reset_index()
    pooled=pooled.merge(mi_agreement,on='k').merge(pd.read_csv(mtab/'assignment_uncertainty.csv'),on='k')
    comparisons=[];diagnostic=[];checks={}
    for k in cfg['candidate_k']:
        bl=pd.read_parquet(bdata/f'coverage8_k{k}_labels.parquet')
        pl=pd.read_parquet(mdata/f'pooled10_k{k}_labels.parquet')
        assert bl[m.IDS].equals(raw[m.IDS]) and pl[m.IDS].equals(raw[m.IDS])
        a=bl.cluster.to_numpy()-1;b=pl.cluster.to_numpy()-1
        aligned,_,_=m.c.aligned(a,b,k)
        comparisons.append({'k':k,'median_vs_pooled_ari':float(adjusted_rand_score(a,b)),
            'median_vs_pooled_aligned':aligned})
        for tag,directory in [('median',bdata),('pooled10',mdata)]:
            prefix='coverage8' if tag=='median' else 'pooled10'
            centres=pd.read_parquet(directory/f'{prefix}_k{k}_observed_medoids.parquet')
            assert m.eligible(centres,8).all()
        if k==2:
            observed_count=raw[m.FEATURES].notna().sum(axis=1).to_numpy()
            median=pd.read_parquet(bdata/'coverage8_imputed.parquet')
            averaged=sum(pd.read_parquet(mdata/f'imputation{i:02d}_filled.parquet') for i in range(1,11))/10
            for tag,labels,filled in [('median',a,median),('pooled10',b,averaged)]:
                for feature in m.FEATURES+['age','observed_count']:
                    values=(observed_count if feature=='observed_count' else raw.age.to_numpy() if feature=='age' else filled[feature].to_numpy())
                    threshold,agreement=review.best_binary_threshold_agreement(values,labels)
                    majority=float(max(np.mean(labels==0),np.mean(labels==1)))
                    internal_agreement=agreement
                    if agreement<=majority:threshold=None;agreement=majority
                    diagnostic.append({'analysis':tag,'k':2,'feature':feature,
                        'threshold':threshold,'agreement':agreement,
                        'internal_cut_agreement':internal_agreement,
                        'majority_baseline':majority,'gain_over_majority':agreement-majority})
                female=raw.gender.eq('F').to_numpy().astype(int)
                diagnostic.append({'analysis':tag,'k':2,'feature':'gender',
                    'agreement':float(max(np.mean(female==labels),np.mean(1-female==labels)))})
    comparison=pd.DataFrame(comparisons);diagnostics=pd.DataFrame(diagnostic)
    comparison.to_csv(tables/'median_vs_pooled_comparison.csv',index=False)
    diagnostics.to_csv(tables/'k2_single_variable_diagnostics.csv',index=False)
    baseline.to_csv(tables/'baseline8_summary.csv',index=False)
    pooled.to_csv(tables/'pooled10_summary.csv',index=False)
    focus=json.loads((reports/'scale_focus/status.json').read_text(encoding='utf-8'))
    assert focus['status']=='complete'
    # Verify all draws preserve true observed inputs and empirical donor support.
    original=raw[m.FEATURES].to_numpy(dtype=float);observed=np.isfinite(original)
    for i in range(1,11):
        filled=pd.read_parquet(mdata/f'imputation{i:02d}_filled.parquet').to_numpy()
        assert filled.shape==original.shape and np.isfinite(filled).all()
        np.testing.assert_array_equal(filled[observed],original[observed])
        for j in range(len(m.FEATURES)):
            assert np.isin(filled[~observed[:,j],j],original[observed[:,j],j]).all()
    limits=json.loads((mdata/'limits.json').read_text(encoding='utf-8'))
    saved=np.load(mdata/'mean_gower_distance.npy',mmap_mode='r')
    assert saved.shape==(len(raw),len(raw))
    indices=np.linspace(0,len(raw)-1,100,dtype=int)
    distance=np.zeros((len(indices),len(indices)),dtype=np.float32)
    for i in range(1,11):
        filled=pd.read_parquet(mdata/f'imputation{i:02d}_filled.parquet').iloc[indices]
        distance+=m.g.numeric_gower(filled,filled,m.FEATURES,limits)/10
    np.testing.assert_allclose(saved[np.ix_(indices,indices)],distance,atol=1e-7)
    np.testing.assert_allclose(saved[np.ix_(indices,indices)],saved[np.ix_(indices,indices)].T,atol=1e-7)
    np.testing.assert_array_equal(saved[indices,indices],np.zeros(len(indices)))
    for k in cfg['candidate_k']:
        centres=pd.read_parquet(mdata/f'pooled10_k{k}_observed_medoids.parquet')
        positions=pd.Index(raw.stay_id).get_indexer(centres.stay_id)
        assert (positions>=0).all() and len(np.unique(positions))==k
        assigned=np.argmin(saved[:,positions],axis=1)+1
        np.testing.assert_array_equal(assigned,pd.read_parquet(mdata/f'pooled10_k{k}_labels.parquet').cluster)
    verification=subprocess.run([sys.executable,'-m','unittest','discover','-s','analysis','-p','test_physiology_v5.py','-v'],cwd=ROOT,capture_output=True,text=True)
    (reports/'synthetic_tests.txt').write_text(verification.stdout+verification.stderr,encoding='utf-8')
    assert verification.returncode==0
    checks.update(synthetic_tests=10,all_imputation_observed_values_preserved=True,
        all_imputed_values_in_fitting_donor_support=True,pooled_distance_recomputed_100_train=True,
        all_pooled_assignments_recomputed=True,input_preprocessing_hashes_unchanged=True,
        original_cohort_hash_unchanged=True,pooled_distance_symmetry_and_diagonal_100_train_verified=True,
        medoids_coverage8_and_both_domains_verified=True,outcome_columns_loaded=False,
        validation_test_used=False,heldout_assignment_procedure_validated=False,
        multiple_imputation_population_only_coverage8=True,
        coverage6_10_comparison_median_only=True,proper_bayesian_mi_certified=False,
        focused_source_review_cases=focus['n_cases'],focused_primary_cases=focus['priority_cases'],
        focused_source_aggregates_match=focus['all_selected_source_aggregates_match'],
        imputation_model_convergence_certified=False,clinical_phenotype_truth_certified=False)
    m.dump(reports/'verification.json',checks)
    # Keep a clinical gate: computational diagnostics do not certify clusters.
    loc.update(status='completed_development_clinical_gate_pending',clinical_k_frozen=False)
    manifest=json.loads((reports/'run_manifest.json').read_text(encoding='utf-8'))
    manifest.update(loc,verification=checks,report_script_sha256=m.c.sha(__file__),
        rf_tie_fixed_script_sha256=m.c.sha(ROOT/'analysis/18_physiology_v5.py'),
        auxiliary_script_sha256={name:m.c.sha(ROOT/'analysis'/name) for name in ['14_cluster_training.py','15_gender_diagnostic.py','16_report_clustering.py','test_physiology_v5.py']},
        kmedoids_version=m.c.metadata.version('kmedoids'),
        output_table_sha256={str(path.relative_to(tables)):m.c.sha(path) for path in tables.rglob('*.csv') if 'incomplete' not in str(path)},
        protocol_sha256=m.c.sha(ROOT/'protocol/physiology_v5.md'),
        input_sha256={f:m.c.sha(Path(loc['source_data_path'])/f) for f in ['reviewed_unimputed.parquet','strict_screen_sensitivity_unimputed.parquet']})
    (reports/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    figures=ROOT/'outputs/figures/clustering_v5'/loc['run_id'];figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].bar(coverage.minimum_observed.astype(str),100*coverage.fraction_retained)
    axes[0].set(xlabel='Minimum observed physiology features',ylabel='Original TRAIN retained (%)',ylim=(0,100))
    axes[1].plot(baseline.k,baseline.silhouette_mean,'o-',label='Median, coverage8')
    axes[1].plot(pooled.k,pooled.silhouette_mean,'o-',label='Mean Gower, 10 RF-PMM')
    axes[1].set(xlabel='k',ylabel='Mean silhouette');axes[1].legend(fontsize=8)
    axes[1].set_xticks(cfg['candidate_k'])
    axes[2].plot(baseline.k,baseline.stability_ari_mean,'o-',label='Median: 80% subsamples')
    axes[2].plot(pooled.k,pooled.imputation_ari_mean,'o-',label='RF-PMM: between imputations')
    axes[2].set(xlabel='k',ylabel='Mean ARI',ylim=(-.05,1.05));axes[2].legend(fontsize=8)
    axes[2].set_xticks(cfg['candidate_k'])
    fig.suptitle('TRAIN development only; sampling and imputation ARIs are different diagnostics')
    fig.tight_layout();fig.savefig(figures/'coverage_and_imputation.png',dpi=180);fig.savefig(figures/'coverage_and_imputation.pdf');plt.close(fig)
    coverage_comparison=pd.read_csv(btab/'coverage_common_patient_comparison.csv')
    errors=pd.read_csv(mtab/'masked_observation_errors.csv')
    error_summary=errors[(errors.stratum=='all')&errors.method.isin(['median','rf_pmm_mean'])].groupby('method').agg(
        mean_feature_mae_over_iqr=('mae_over_iqr','mean'),mean_feature_rmse_over_iqr=('rmse_over_iqr','mean'),
        n_features_scored=('mae_over_iqr','count')).reset_index()
    report='''# 生理表型开发 v5：实际运行报告

计算已完成，临床表型与k尚未冻结。仅使用固定训练集，未读取结局列，未开发验证/测试表现。该修订登记于看到v4人口学驱动结果之后，不是前瞻性预注册。

## 覆盖门槛与适用人群

16项生理指标、等权数值Gower+PAM；年龄/性别仅描述。插补前至少各1项生命体征/化验；8项为本轮主操作标准，6/10为中位数基线敏感性。原队列和分区不改变，适用对象限定为有充分记录的患者。

'''+table(coverage)+'''

门槛不同会改变人群、插补负担与训练尺度。不能把高silhouette自动解释为最佳门槛；共同患者一致性见下表，排除者的基线与观测分布见baseline/retained_excluded_descriptions.csv。此次RF-PMM仅在8项人群执行，因此未完成6/8/10与全部插补方法的因子比较。

## 8项门槛的中位数基线

'''+table(baseline.drop(columns=['analysis']))+'''

稳定性为10次80%无放回训练子样本，各自重新拟合中位数和尺度后将全体合格训练患者分配至子样本中心；不是bootstrap置信区间。

## 10次随机RF-PMM插补比较

'''+table(pooled.drop(columns=['analysis']))+'''

每次6轮条件随机森林，每变量仅用原始实测值作为响应/供体，随机抽取最近5名候选之一，第5近绝对预测距离的并列候选在两侧整个并列组随机处理。人口学和结局不进入插补。10个Gower矩阵取平均后PAM，分配与个别插补分群对齐比较。它是灵活的随机插补压力测试，不声称proper Bayesian MI、与潜在真实聚类相容或已验证收敛；轨迹仅描述迭代行为。个别插补与共识ARI不等于同时扰动样本和插补的稳定性。共识中心为患者身份，其缺失项仍具有跨插补不确定性。未来持出集分配流程尚未验证，不可直接进入最终评价。

## 门槛与插补选择对结果的影响

'''+table(coverage_comparison)+'\n\n'+table(comparison)+'''

## 隐藏实测值的恢复误差

训练合格者的10%实测单元格随机遮蔽，先遮蔽再拟合。分别报告3个RF-PMM随机抽值和它们的均值；下表为可评分变量IQR归一化误差的等权均值，实际变量数单列，变量及覆盖层详情见multiple_imputation/masked_observation_errors.csv。供体随机抽值并非最优点预测，故不能只凭其单次MAE给插补方法排名；均值结果也不证明真实缺失项准确。该随机遮蔽不能重现临床非随机缺失。

'''+table(error_summary)+'''

## 剩余限制与下一步判据

主8项中位数与平均距离聚类的中心按主数据计均至少8/16项实测且生命体征/化验均有记录，但这仍不等于生理真值。严格筛查敏感性沿用主资格，不保证筛查后仍达到8项。k2_single_variable_diagnostics.csv检查二簇是否近乎被一个数值、年龄、性别或观测量解释；另列多数类常量基线，阈值诊断不足以证明临床解释。生理画像使用未插补数据，并报告各变量缺失率。严格范围对照目前限8项中位数基线，尚未开展RF-PMM严格筛查对照。

已按研究者要求完成尺度定向回查。scale_focus/报告和表格记录单尺度替换、圈定的原始观测证据及来源哈希核验。重点为呼吸频率、血氧和白细胞23项记录，其聚合值能由下载来源重算一致；这不证明临床真值。尚未据此删病例或调整主权重。

最终可以支持的是“这个预先固定的操作标准下结果是否稳健”，而非证明8、6或10是唯一最佳。需要共同患者分群一致、跨插补不确定性可接受、中心与画像有多变量依据、排除偏倚可说明，才可冻结下一阶段。若这些条件不成立，应报告门槛/插补依赖或未识别可靠离散表型。不会凭死亡率差异选择门槛或k。

参考：[聚类与插补模型相容性研究](https://arxiv.org/abs/2106.04424)、[跨插补分群汇总研究](https://arxiv.org/abs/2011.13694)、[RandomForestRegressor官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)。
'''
    (reports/'stage_report.md').write_text(report,encoding='utf-8')
    (ROOT/'outputs/reports/clustering_v5_latest.md').write_text(report,encoding='utf-8')
    m.dump(reports/'run_manifest.json',manifest)
    m.dump(ROOT/'logs/clustering_v5_latest.json',loc)
    print(json.dumps({'status':loc['status'],'coverage':coverage.to_dict('records'),
        'baseline':baseline.to_dict('records'),'pooled':pooled.to_dict('records'),
        'comparison':comparisons,'masked_error_summary':error_summary.to_dict('records')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
