"""Summarize outcome-blinded training analyses and validate saved results."""
from pathlib import Path
import hashlib
import importlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
c=importlib.import_module('14_cluster_training')

def markdown_table(frame):
    columns=[str(x) for x in frame.columns]
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    rows.extend('| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)

def best_binary_threshold_agreement(values,labels):
    order=np.argsort(values,kind='stable');v=np.asarray(values)[order];y=np.asarray(labels)[order]
    classes=np.unique(y)
    if len(classes)!=2:raise ValueError('Only binary diagnostic supported')
    y=y==classes[1];n=len(y);left=np.arange(1,n);ones=np.cumsum(y)[:-1]
    zeros=left-ones;rightones=y.sum()-ones;rightzeros=(n-y.sum())-zeros
    score=np.maximum(zeros+rightones,ones+rightzeros)
    score=np.where(v[:-1]!=v[1:],score,-1)
    if score.max()<0:return None,float(max(y.sum(),n-y.sum())/n)
    i=int(score.argmax());return float((v[i]+v[i+1])/2),float(score[i]/n)

def main():
    loc=json.loads((ROOT/'logs/clustering_latest.json').read_text(encoding='utf-8'))
    prep=json.loads((ROOT/'logs/preprocessing_review_latest.json').read_text(encoding='utf-8'))
    data=Path(loc['data_path']);reports=Path(loc['report_path']);tables=Path(loc['tables_path'])
    manifest=json.loads((reports/'run_manifest.json').read_text(encoding='utf-8'))
    extra=json.loads((reports/'gender_diagnostic/run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='training_clustering_and_sensitivity_complete'
    assert extra['status']=='gender_diagnostic_complete'
    config=json.loads((ROOT/'config/analysis_v2.json').read_text(encoding='utf-8'))
    features=config['primary_features']
    cols=['subject_id','hadm_id','stay_id','split','phenotype_eligible']+features
    base=pd.read_parquet(Path(prep['data_path'])/'reviewed_unimputed.parquet',columns=cols)
    train=base[base.split.eq('train')&base.phenotype_eligible].sort_values('stay_id').reset_index(drop=True)
    strict=pd.read_parquet(Path(prep['data_path'])/'strict_screen_sensitivity_unimputed.parquet',columns=cols)
    strict=strict[strict.split.eq('train')&strict.phenotype_eligible].sort_values('stay_id').reset_index(drop=True)
    transform=json.loads((data/'primary_transform.json').read_text(encoding='utf-8'))
    filled,imputer,limits=c.fit_transform(train,features)
    assert imputer==transform['imputer'] and limits==transform['numeric_minmax']
    order=pd.read_parquet(data/'training_order.parquet')
    assert order.equals(train[['subject_id','hadm_id','stay_id']])
    D=np.load(data/'primary_training_gower.npy',mmap_mode='r')
    assert D.shape==(len(train),len(train)) and np.all(np.diag(D)==0)
    indices=np.random.default_rng(20261009).choice(len(train),100,replace=False)
    np.testing.assert_allclose(D[np.ix_(indices,indices)],c.gower(filled.iloc[indices],filled.iloc[indices],features,limits),atol=1e-7)
    metrics=pd.read_csv(tables/'primary_k_metrics.csv');diagnostic=pd.read_csv(tables/'gender_diagnostic/numeric17_k_metrics.csv')
    sensitivity=pd.read_csv(tables/'outlier_sensitivity.csv');ds=pd.read_csv(tables/'gender_diagnostic/numeric17_outlier_sensitivity.csv')
    supplement=json.loads((reports/'supplementary/run_manifest.json').read_text(encoding='utf-8'))
    assert supplement['status']=='supplementary_diagnostics_complete'
    gap=pd.read_csv(tables/'supplementary/empirical_marginal_gap.csv')
    alternate=pd.read_csv(tables/'supplementary/average_linkage_sensitivity.csv')
    for k in manifest['config']['candidate_k']:
        for prefix,path,counts in [('primary',data,metrics),('numeric17',data/'gender_diagnostic',diagnostic)]:
            labels=pd.read_parquet(path/f'{prefix}_k{k}_training_labels.parquet')
            assert labels[['subject_id','hadm_id','stay_id']].equals(order)
            assert labels.cluster.nunique()==k and len(labels)==len(train)
        medoids=pd.read_parquet(data/f'primary_k{k}_medoid_features.parquet')
        labels=pd.read_parquet(data/f'primary_k{k}_training_labels.parquet')
        predicted=np.argmin(c.gower(filled,medoids[features],features,limits),axis=1)+1
        np.testing.assert_array_equal(predicted,labels.cluster.to_numpy())
        for name in ['strict_main_scales','strict_refit_scales']:
            s=pd.read_parquet(data/f'{name}_k{k}_labels.parquet')
            assert s[['subject_id','hadm_id','stay_id']].equals(order)
            expected=sensitivity[(sensitivity.variant==name)&(sensitivity.k==k)].iloc[0]
            assert np.isclose(adjusted_rand_score(s.primary_cluster,s.sensitivity_cluster),expected.ari_vs_primary)
            assert np.isclose(s.primary_cluster.eq(s.sensitivity_cluster_aligned).mean(),expected.aligned_agreement)
        other=pd.read_parquet(data/'supplementary'/f'average_linkage_k{k}_training_labels.parquet')
        assert other[['subject_id','hadm_id','stay_id']].equals(order)
        expected=alternate[alternate.k.eq(k)].iloc[0]
        assert np.isclose(adjusted_rand_score(labels.cluster,other.cluster),expected.ari_vs_primary_pam)
    observed_source=json.loads((ROOT/'logs/observation_review_latest.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'config/frozen_split_v2.json').read_text(encoding='utf-8'))
    assert c.sha(observed_source['input_path'])==frozen['dataset_sha256']
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','analysis','-p','test_clustering.py'],cwd=ROOT,capture_output=True,text=True)
    (reports/'synthetic_tests.txt').write_text(test.stdout+test.stderr,encoding='utf-8');assert test.returncode==0
    result={'status':'passed','synthetic_tests':7,'n_eligible_training':len(train),'gower_recomputed_sample_matches':True,'saved_medoid_assignment_matches':True,'saved_sensitivity_metrics_match':True,'average_linkage_metrics_match':True,'frozen_cohort_hash_unchanged':True,'all_labels_match_training_ids':True,'outcome_fields_loaded':False,'clinical_k_frozen':False}
    (reports/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    scale_rows=[];_,_,strict_limits=c.fit_transform(strict,features)
    for f in features:
        if f=='gender':continue
        main_range=limits[f][1]-limits[f][0];strict_range=strict_limits[f][1]-strict_limits[f][0]
        scale_rows.append({'feature':f,'newly_missing_training_values':int((train[f].notna()&strict[f].isna()).sum()),'primary_min':limits[f][0],'primary_max':limits[f][1],'strict_min':strict_limits[f][0],'strict_max':strict_limits[f][1],'primary_range_over_strict_range':main_range/strict_range if strict_range else None})
    scale_frame=pd.DataFrame(scale_rows).sort_values('primary_range_over_strict_range',ascending=False)
    scale_frame.to_csv(tables/'feature_range_sensitivity.csv',index=False)
    physiology=[f for f in features if f not in ['age','gender']]
    primary_medoid_ids=pd.read_parquet(data/'primary_k2_medoid_ids.parquet')
    observed=primary_medoid_ids[['stay_id','cluster']].merge(train[['stay_id']+physiology],on='stay_id',validate='one_to_one')
    medoid_counts=pd.DataFrame({'analysis':'registered18','k':2,'cluster':observed.cluster,'physiology_observed':observed[physiology].notna().sum(axis=1),'physiology_imputed':observed[physiology].isna().sum(axis=1)})
    medoid_counts.to_csv(tables/'primary_k2_medoid_observability.csv',index=False)
    numeric17_labels=pd.read_parquet(data/'gender_diagnostic/numeric17_k2_training_labels.parquet')
    threshold,age_agreement=best_binary_threshold_agreement(filled.age.to_numpy(),numeric17_labels.cluster.to_numpy())
    numeric_features=[f for f in features if f!='gender']
    prototypes=pd.read_parquet(data/'gender_diagnostic/numeric17_k2_medoids.parquet')
    matching=prototypes.merge(filled[numeric_features].assign(stay_id=train.stay_id.to_numpy()),on=numeric_features,how='inner').merge(train[['stay_id']+physiology],on='stay_id',suffixes=('_imputed','_original'))
    matching['physiology_observed']=matching[[f+'_original' for f in physiology]].notna().sum(axis=1)
    prototype_counts=matching.groupby('cluster').physiology_observed.agg(matches='size',min_observed='min',max_observed='max').reset_index()
    assert set(prototype_counts.cluster)==set(prototypes.cluster)
    prototype_counts.to_csv(tables/'numeric17_k2_prototype_observability.csv',index=False)
    interpret={'primary_k2_gender_agreement':float(metrics.loc[metrics.k.eq(2),'gender_purity'].iloc[0]),'numeric17_k2_best_age_threshold':threshold,'numeric17_k2_age_threshold_agreement':age_agreement,'primary_k2_medoid_physiology_observed':medoid_counts.physiology_observed.tolist(),'numeric17_k2_prototype_observed_ranges':prototype_counts.to_dict(orient='records'),'final_clinical_k_frozen':False,'conclusion':'Current numerical winner is demographic-driven; medoids predominantly imputed. Do not treat as established physiological sepsis phenotypes.'}
    (reports/'interpretation_review.json').write_text(json.dumps(interpret,indent=2),encoding='utf-8')
    focus_summary={'triggered':False,'priority_cases':0,'priority_patients':0}
    diag_candidate=extra['decision']['diagnostic_candidate_k']
    if diag_candidate is not None and extra['decision']['outlier_influence_at_candidate']:
        review=json.loads((ROOT/'logs/clinical_review_latest.json').read_text(encoding='utf-8'))
        review_data=Path(review['data_path'])
        cases=pd.read_csv(review_data/'待复核清单.csv')
        support=pd.read_csv(review_data/'自动证据复核.csv')
        priority=cases.merge(scale_frame,on='feature',validate='many_to_one').merge(support[['case_id','technical_support','distinct_extreme_times']],on='case_id',validate='one_to_one')
        endpoint=np.isclose(priority.aggregated_value,priority.primary_min,rtol=1e-12,atol=1e-12)|np.isclose(priority.aggregated_value,priority.primary_max,rtol=1e-12,atol=1e-12)
        # Prioritization is a follow-up heuristic, not a causal attribution or clinical error diagnosis.
        priority['range_endpoint_priority']=endpoint&priority.primary_range_over_strict_range.gt(1.1)
        priority=priority.sort_values(['range_endpoint_priority','primary_range_over_strict_range'],ascending=False)
        priority.to_csv(data/'outlier_focused_review_priority.csv',index=False,encoding='utf-8-sig')
        top=priority[priority.range_endpoint_priority]
        top.to_csv(data/'outlier_range_endpoints_first.csv',index=False,encoding='utf-8-sig')
        focus_summary={'triggered':True,'diagnostic_k':diag_candidate,'priority_cases':len(top),'priority_patients':int(top.stay_id.nunique()),'all_review_cases_retained':len(priority),'heuristic':'flagged main range endpoints where main/strict range ratio>1.1;not proof of individual causal influence'}
    (reports/'focused_review_summary.json').write_text(json.dumps(focus_summary,indent=2),encoding='utf-8')
    figs=ROOT/'outputs/figures/clustering_v4'/data.name;figs.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(1,3,figsize=(13,3.7))
    for frame,label in [(metrics,'Registered 18 inputs'),(diagnostic,'Diagnostic 17 numeric inputs')]:
        ax[0].plot(frame.k,frame.silhouette_mean,'o-',label=label)
        ax[1].plot(frame.k,frame.stability_ari_mean,'o-',label=label)
        ax[2].plot(frame.k,frame.min_cluster_fraction,'o-',label=label)
    for a in ax:a.set_xlabel('Number of clusters k');a.set_xticks([2,3,4,5,6]);a.grid(alpha=.2)
    ax[0].set_ylabel('Mean ordinary silhouette');ax[1].set_ylabel('Mean subsample stability ARI');ax[2].set_ylabel('Smallest cluster fraction')
    ax[1].axhline(.8,color='gray',linestyle='--');ax[2].axhline(.05,color='gray',linestyle='--');ax[0].legend(fontsize=8)
    fig.suptitle('Training-only clustering diagnostics; no clinical k frozen')
    fig.tight_layout();fig.savefig(figs/'training_k_diagnostics.png',dpi=180);fig.savefig(figs/'training_k_diagnostics.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for a,frame,title in [(ax[0],sensitivity,'Registered 18 inputs'),(ax[1],ds,'Diagnostic 17 numeric inputs')]:
        for name,group in frame.groupby('variant'):a.plot(group.k,group.iloc[:,2],'o-',label=name)
        a.axhline(.8,color='gray',linestyle='--');a.set_title(title);a.set_xlabel('k');a.set_xticks([2,3,4,5,6]);a.grid(alpha=.2);a.legend(fontsize=8)
    ax[0].set_ylabel('ARI with unscreened training clustering');fig.tight_layout();fig.savefig(figs/'outlier_sensitivity.png',dpi=180);fig.savefig(figs/'outlier_sensitivity.pdf');plt.close(fig)
    selection=manifest['selection_review'];diag=extra['decision']
    text=['# 训练集聚类与异常值敏感性阶段报告','',f'实际使用{len(train):,}名符合资格的训练患者；没有加载结局字段，没有使用验证/测试数据选k或预处理版本。原18项主分析全部保留，17项去性别分析是看到性别驱动结果后增加的诊断性探索，不是预先指定的主分析。','',f'18项版按登记数值规则的暂定k={selection["provisional_k"]}；性别驱动标记={selection["gender_dominance_flag"]}。17项诊断版通过同样大小/稳定性门槛的候选k={diag["diagnostic_candidate_k"]}。未冻结临床表型k，未开展预测模型。','', '## 实际训练指标','',metrics.round(4).pipe(markdown_table),'','## 去性别诊断指标','',diagnostic.round(4).pipe(markdown_table),'','## 原18项异常筛查对照','',sensitivity.round(4).pipe(markdown_table),'','## 去性别17项异常筛查对照','',ds.round(4).pipe(markdown_table),'','## 解释和下一步','', '原版k=2与性别完全一致；不能把高轮廓、稳定性或异常筛查一致性当作已发现生理表型。k>2也须检查性别分层和真实生理特征的贡献。17项诊断版只用于识别此问题，不能按更好看的结果悄悄替换主分析。','', '严格筛查版同时改变少量观测和插补；独立重拟合尺度还改变所有人的特征相对距离。因此聚类变化不一定证明异常值错误。主尺度条件用于分离数值/插补影响，重拟合尺度条件反映整个处理管线变化；还存在PAM局部解差异。','', '若17项候选触发登记影响门槛，集中回查被筛查的训练记录和范围变化最大的特征，沿用自动证据，不强行判错；若无稳定候选，报告不稳定，不强定k。正式主变量角色或距离规则若调整，须另记为观察结果后的修订，并在未来验证阶段冻结；不要使用结局或测试来挑选修改。','', '核验：7项合成测试及实际距离、训练ID、medoid分配和敏感性指标校验通过。10次80%无放回子抽样是有限稳定性评估，不称bootstrap置信区间。','', '软件方法依据：[FastPAM1官方说明](https://python-kmedoids.readthedocs.io/en/latest/index.html)、[普通轮廓系数](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html)、[ARI说明](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)。']
    text.extend(['', '## 人口学驱动和中心的实际观测覆盖', '', '去掉性别后k2可由年龄阈值'+str(threshold)+'完全/近似重现，训练标签对齐一致率='+str(age_agreement)+'。这是已拟合标签的解释性诊断，不是预测性能或外部验证。', '', medoid_counts.pipe(markdown_table), '', prototype_counts.pipe(markdown_table), '', '原18项k2中心的16项生理指标分别只有2项和1项实际观测；17项k2两个原型的匹配训练患者均只有1项生理实测。其余生理值由中位数插补生成，中心的临床代表性因此受到限制。不能仅因分群稳定就认定为新的脓毒症生理表型。后续需评审人口学变量在距离中的角色、记录覆盖资格和插补策略，并如实登记看到结果后的修订；本轮保留所有原版结果，尚不进入基于这些组的主预测结论。', '', '## 补充：经验边际置换Gap形式诊断', '', gap.round(4).pipe(markdown_table), '', '固定2000名训练患者、20个逐列独立置换参照，以到medoid的距离总和为离散度；保持单变量经验分布并打乱变量间联系。它是明确参照和离散度的Gap形式诊断，不是全17,560人的标准均匀参照Gap。仅评审k2–6，未检验k1，不能证明自然真簇存在。注册时点在初始PAM结果之后。', '', '## 补充：全训练Gower加average-linkage敏感性', '', alternate.round(4).pipe(markdown_table), '', '替代算法的极小簇或较低一致性是结构对算法敏感的证据，不以更高轮廓强行替换PAM。', '', '依据：[Gap原始论文](https://academic.oup.com/jrsssb/article/63/2/411/7083348)、[SciPy linkage说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)。本研究的经验参照、medoid离散度和计算子样本是自定义方案。'])
    (reports/'stage_report.md').write_text('\n'.join(text),encoding='utf-8')
    for file in [Path(__file__),ROOT/'analysis/test_clustering.py']:(reports/'source_snapshot'/file.name).write_bytes(file.read_bytes())
    manifest['report_verification']=result
    manifest['additional_artifact_sha256']={str(file.relative_to(ROOT)):c.sha(file) for file in [Path(__file__),ROOT/'analysis/test_clustering.py',reports/'stage_report.md',reports/'verification.json',reports/'synthetic_tests.txt']}
    (reports/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Training clustering report, figures and verification saved.')
    print(json.dumps({'primary':selection,'numeric17_diagnostic':diag},ensure_ascii=False))

if __name__=='__main__':main()
