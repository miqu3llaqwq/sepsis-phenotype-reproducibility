"""Aggregate-only delivery; patient-level data stay in the local work tree."""
from pathlib import Path
import json,hashlib,zipfile
import pandas as pd
from build_common_identified_20261030 import ROOT,WORK,OUT,DATA,dump,sha

def main():
    cohort=json.loads((OUT/'cohort_summary.json').read_text());features=json.loads((OUT/'feature_summary.json').read_text())
    model=json.loads((OUT/'new_model_status.json').read_text());assert model['status']=='complete_and_frozen'
    qa=json.loads((OUT/'independent_common_cohort_qa.json').read_text());assert qa['status']=='pass'
    chart=json.loads((OUT/'remote_chart_aggregation_qa.json').read_text());assert chart['status']=='pass'
    frozen=WORK/'frozen_20261030';bm=json.loads((frozen/'bundle_manifest.json').read_text())
    assert all(sha(frozen/p)==h for p,h in bm['sha256'].items())
    tables=OUT/'model_tables';metrics=pd.read_csv(tables/'all_clustering_metrics.csv')
    obs=metrics[metrics.analysis.eq('observed_equal')].set_index('k')
    os=pd.read_csv(tables/'observed_patient_resampling_summary.csv').set_index('k')
    ms=pd.read_csv(tables/'mice_joint_patient_resampling_summary.csv').set_index('k')
    mi=pd.read_csv(tables/'mice_vs_observed_summary.csv').set_index('k')
    between=pd.read_csv(tables/'between_imputation_summary.csv').set_index('k')
    fullmi=metrics[metrics.analysis.str.fullmatch(r'mice_\d\d')].groupby('k').silhouette_mean.mean()
    lines=[]
    for k in range(2,7):lines.append(f'| {k} | {obs.at[k,"silhouette_mean"]:.3f} | {os.at[k,"mean"]:.3f} | {fullmi[k]:.3f} | {between.at[k,"ari_mean"]:.3f} | {mi.at[k,"mean"]:.3f} | {ms.at[k,"mean"]:.3f} |')
    flows=[]
    for s in ['mimic','eicu']:
        c=cohort[s];f=features[s]
        flows.append(f'| {s.upper()} | {c["candidate_stays"]:,} | {c["selected_admissions"]:,} | {c["selected_people"]:,} | {f["feature_eligible"]:,} |')
    split=pd.read_csv(tables/'partition_flow.csv')
    splitlines=['| '+str(r.split)+' | '+f'{r.admissions:,}'+' | '+f'{r.people:,}'+' | '+f'{r.feature_eligible:,}'+' |' for r in split.itertuples()]
    text=f'''# 共同定义分支：第1–3项计算与冻结报告

日期：2026-10-30。分支：common_identified_v2_20261030。

## 结论与适用范围

已实际完成两库共同规则下的可识别子集构建、14项特征重新提取、新MIMIC模型重跑及冻结。这里的“完成”适用于**SOFA范围判定与t0可识别子集**，不表示已经得到每名患者完整、确定的六域SOFA标量，也不表示已经完成新版eICU外部分配。

输注时钟、尿量完整采集时长、通气状态仍有无法严格对应的部分。在两库中统一保留为未知范围；没有把未知项设为正常、没有把录入时刻改称给药时刻。仅当这些未知取值都不能改变最早合格t0时，才保留该次医院住院。旧31,422人分析、原模型、39例适配结果及v18论文均保留。

## 1. 共同规则与六域范围

| 项目 | 执行规则 |
|---|---|
| 感染时间 | 使用处方计划开始，不能称首次实际给药；沿用参考抗菌药名称字典，其所含抗真菌药另在适用范围中说明。明确系统给药途径；已取消或开始前停止排除；无法辨认的途径、生命周期或零/缺失持续时间标记未确定。 |
| 培养 | 先限制在本次医院住院时间内，再按唯一精确采集时间配对；前72小时优先，否则后24小时；排除同时刻；不补午夜、不把药敏行数当培养数。 |
| 血小板、胆红素 | 原始化验、明确单位、同刻唯一有效数值，标准评分阈值。 |
| 肾脏 | 肌酐提供确定的评分下界；尿量无法证明共同的完整采集时长，保留未知贡献。 |
| 呼吸 | 同刻唯一动脉血气类别PaO2/FiO2值；同刻不等于同标本；通气状态未对齐时保留评分范围。 |
| 循环 | 有效MAP低于70提供1分下界；升压药的时间区间无法统一时，两库均不用于确定下界，保留其可能贡献。 |
| 神经 | 唯一有效GCS总分或完整同刻三项相加；冲突、无法评分、无完整记录均未知；不默认15分。 |
| 小时钟 | ICU入科相对整小时；前24小时观测窗口；不再使用护士首次心率作为时钟。 |
| t0 | max(SOFA首次达标小时、抗菌药处方开始、培养、ICU入科)。使用最早可能与最早确定达标小时的上下界；只有两端产生相同t0才确定。 |
| 首次事件 | 在医院住院内选最早可确定t0；可能更早或同刻在其他ICU的未确定事件会阻止入组。特征缺失筛选在首次事件选择之后执行。 |

采用参考实现的绝对SOFA≥2及基线0假定，不声称掌握患者个体的真实SOFA增量。所有记录不足的分项均允许0–4，总分上界保守设为24。记录下界之和达到2可以证明达标，但未达2不能证明无器官功能障碍。

依据：[eICU medication](https://eicu.mit.edu/eicutables/medication/)区分医嘱与给药；[infusionDrug](https://eicu.mit.edu/eicutables/infusiondrug/)说明录入偏移；[intakeOutput](https://eicu.mit.edu/eicutables/intakeoutput/)区分逐行值、累计值与缺失。阈值和原始字段来源另保存在协议、SQL和代码快照中。

## 2. 实际队列与特征

| 数据库 | 配对候选ICU住院 | 首次t0可识别医院住院 | 涉及患者 | 满足特征资格住院 |
|---|---:|---:|---:|---:|
{chr(10).join(flows)}

各阶段统计单位不同；重复患者可有多次医院住院，不能把住院次数写成人数。eICU特征合格记录来自{features['eicu']['hospitals']}家医院。新队列与旧39例适配版不是同一定义，人数差异不能作为验证效果提高的证据。

固定14项变量；生命体征前6小时、化验前24小时，中位数；均限制在本次医院住院内。MAP要求同来源同刻SBP配对和血压次序合理，优先动脉；eICU同一窗口优先护士记录、无可用记录才用监护接口，周期值两端保留5分钟保护。记录来源异质性没有被宣称消除。允许有依据的摄氏/华氏、FiO2比例/百分比，以及单价离子mmol/L与mEq/L换算；冲突值不参与汇总。

这是**回顾性观测时间分析**。eICU用于汇总的记录中有{features['eicu']['retrospective_entry_after_t0_records']}条的录入/修订时间晚于t0；它们不代表t0当时已可获得。MIMIC同样按观测时间处理；本分支不能用于宣称实时预测或无信息延迟的临床部署。

患者层面划分先于特征资格筛选。同一患者的所有住院只进入一个分区：

| 分区 | 医院住院 | 患者 | 特征合格住院 |
|---|---:|---:|---:|
{chr(10).join(splitlines)}

训练最终使用{model['training_rows']:,}次住院、{model['training_people']:,}名患者；P1–P99仅在符合资格且共同观测距离可计算的训练记录中拟合。

## 3. 新模型、稳定性与冻结

已运行所有k=2–6，每次5个初始值；观测值Gower模型、10份训练内MICE-PMM、10次80%患者重抽样，以及10次重抽样内重新插补。MICE每链20轮、10个供者，使用bootstrap参数扰动。重抽样以患者为单位，保留被选患者全部住院；未将重复住院当独立抽样单位。另执行按域加权、缺失指示、可用性及完整14项分支。

| k | 观测轮廓 | 观测重抽样ARI | MICE轮廓均值 | 插补间ARI | MICE与观测ARI | MICE联合重抽样ARI |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

表中ARI为相应重复的描述性均值，重复范围另见明细表；不是独立临床样本的置信区间。没有依据eICU结果重新选择k，也没有冻结临床亚型数量。插补与观测值分配代表不同分析路径；对外转移的输入是观测值Gower代表点，不假称已部署外部MICE插补器。

冻结目录：`{frozen}`。其中包含各k代表点、训练P1–P99、变量顺序、距离/缺失资格规则、10份MICE代表点、代码与SQL快照、输入指纹和逐文件校验清单。

## 独立核对

- {qa['exhaustive_interval_cases']}种短时间序列区间组合验证t0识别逻辑。
- eICU全部候选住院和MIMIC抽样住院，以独立逐窗口程序复核；两库全部入选医院住院核对首次事件条件。
- 全部{qa['eicu']['feature_cells_recomputed']+qa['mimic']['feature_cells_recomputed']:,}个特征单元从实际采用的观测重新计算中位数；核对时间窗与医院边界。
- MIMIC远端汇总与{chart['sample_stays']}次ICU住院、{chart['raw_chart_groups_checked']:,}组原始记录的独立Python评分一致。
- 各k全部训练分配复现，另用逐变量标量距离独立复算；患者跨分区重叠为0。
- 原有{bm['original_files_checked']}个受保护文件指纹保持不变；新版冻结清单复核通过。

## 尚未完成

新版eICU固定代表点分配、覆盖/距离分布分析和论文数字更新尚未执行。本报告完成的是新共同范围规则下的可识别子集计算与MIMIC重训冻结，不能改写成“完整六域SOFA严格等价验证已经完成”。研究定义在见到旧eICU结果后制定，属于事后方法学分析。
'''
    report=ROOT/'outputs/共同定义第1至3项_计算与冻结报告_20261030.md';report.write_text(text,encoding='utf-8')
    progress_path=ROOT/'outputs/common_definition_20261029/progress.json';p=json.loads(progress_path.read_text(encoding='utf-8'))
    p.update(stage='common_identifiable_cohorts_and_new_models_frozen',common_six_domain_bounds_complete=True,common_six_domain_sofa_complete=False,
       new_t0_complete=True,new_t0_scope='earliest hospital t0 identifiable subset',new_model_fitted=True,new_external_validation_complete=False,
       new_feature_summary=features,new_training_summary=model,report=str(report),next_action='Fixed k2..6 assignment of51 eICU eligible admissions with new frozen observed-value medoids; no external refit or k selection.')
    progress_path.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
    status=ROOT/'outputs/最新进度与交付_20261030.md'
    old=status.read_text(encoding='utf-8');marker='\n## 共同定义分支本轮计算更新\n'
    old=old.split(marker)[0]
    old+=marker+f'\n已完成共同范围规则下的可识别子集构建、两库14项特征提取、新MIMIC重训和冻结。MIMIC可识别医院住院{cohort["mimic"]["selected_admissions"]:,}次；最终训练{model["training_rows"]:,}次住院。eICU特征合格{features["eicu"]["feature_eligible"]}次住院、{features["eicu"]["hospitals"]}家医院，新版外部分配待执行。\n\n这不是完整标量六域SOFA；输注时钟、完整尿量与通气的不足保留为范围。v18仍为最新论文，未将本轮数字混入旧稿。详见[计算与冻结报告]({report.name})。\n'
    status.write_text(old,encoding='utf-8')
    deliveries=[report,OUT/'common_protocol_v2.json',OUT/'cohort_summary.json',OUT/'feature_summary.json',OUT/'new_training_contract.json',OUT/'new_model_status.json',OUT/'independent_common_cohort_qa.json',OUT/'remote_chart_aggregation_qa.json']+list(tables.glob('*.csv'))+list(OUT.glob('*counts.csv'))+list(OUT.glob('*missingness.csv'))+list(OUT.glob('*coverage.csv'))
    package=ROOT/'outputs/共同定义第1至3项_汇总审阅包_20261030.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for f in deliveries:z.write(f,f.name if f==report else str(f.relative_to(OUT)))
    dump('aggregate_delivery_manifest.json',{'files':{str(f.relative_to(ROOT)):sha(f) for f in deliveries},'package_sha256':sha(package),'patient_level_records_in_package':False,'frozen_manifest_files':len(bm['sha256'])})
    print(str(report));print(str(package))

if __name__=='__main__':main()
