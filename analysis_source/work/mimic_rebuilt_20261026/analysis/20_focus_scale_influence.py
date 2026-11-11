"""Post-result, outcome-blinded scale-influence diagnostic and source focus."""
from pathlib import Path
import importlib
import json
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
m=importlib.import_module('18_physiology_v5')

def verify_focus(loc):
    data=Path(loc['data_path'])/'scale_focus';reports=Path(loc['report_path'])/'scale_focus'
    selected=pd.read_csv(data/'聚类尺度影响定向复核.csv')
    priority=selected[selected.focus_priority.eq('single_scale_major_effect')]
    review=json.loads((ROOT/'logs/clinical_review_latest.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'logs/observation_review_latest.json').read_text(encoding='utf-8'))
    original=json.loads((Path(review['report_path'])/'automatic_evidence_manifest.json').read_text(encoding='utf-8'))
    checks={domain:m.c.sha(Path(source['data_path'])/(domain+'.parquet'))==value for domain,value in original['raw_source_sha256'].items()}
    checks['evidence']=m.c.sha(Path(review['data_path'])/'自动证据复核.csv')==original['evidence_sha256']
    assert all(checks.values());m.dump(reports/'source_evidence_hash_verification.json',checks)
    status=json.loads((reports/'status.json').read_text(encoding='utf-8'))
    status.update(priority_cases=len(priority),priority_patients=int(priority.subject_id.nunique()),
        script_sha256=m.c.sha(__file__),source_evidence_hashes_verified=True)
    m.dump(reports/'status.json',status)
    (reports/Path(__file__).name).write_bytes(Path(__file__).read_bytes())

def main():
    loc=json.loads((ROOT/'logs/clustering_v5_latest.json').read_text(encoding='utf-8'))
    cfg,frames=m.load(loc);mask=m.eligible(frames['main'],8)
    raw=frames['main'][mask].reset_index(drop=True);strict=frames['strict'][mask].reset_index(drop=True)
    _,_,main_limits=m.c.fit_transform(raw,m.FEATURES)
    strict_filled,_,strict_limits=m.c.fit_transform(strict,m.FEATURES)
    data=Path(loc['data_path'])/'scale_focus';data.mkdir(exist_ok=True)
    tables=Path(loc['tables_path'])/'scale_focus';tables.mkdir(exist_ok=True)
    reports=Path(loc['report_path'])/'scale_focus';reports.mkdir(exist_ok=True)
    m.dump(reports/'diagnostic_registration.json',{'timing':'after v5 baseline strict-refit result; exploratory explanation,not primary replacement',
        'isolation':'one numeric scale changed at a time,all values fixed to strict-screen median-filled input',
        'k':2,'range_ratio_screen':1.1,'focus_screen':'ARI<0.8 or aligned disagreement>10%;not universal clinical thresholds',
        'outcomes_used':False,'clinical_adjudication':False})
    reference=pd.read_parquet(Path(loc['data_path'])/'baseline/coverage8_k2_labels.parquet')
    assert reference[m.IDS].equals(raw[m.IDS]);labels=reference.cluster.to_numpy()-1
    rows=[]
    for feature in m.FEATURES:
        lo,hi=main_limits[feature];slo,shi=strict_limits[feature]
        ratio=(hi-lo)/(shi-slo) if shi>slo else 1.
        row={'feature':feature,'main_min':lo,'main_max':hi,'strict_min':slo,'strict_max':shi,
            'range_ratio':ratio,'newly_missing':int((raw[feature].notna()&strict[feature].isna()).sum())}
        if ratio>=1.1:
            scales=dict(main_limits);scales[feature]=strict_limits[feature]
            D=m.g.numeric_gower(strict_filled,strict_filled,m.FEATURES,scales);np.fill_diagonal(D,0)
            path=data/(feature+'_temporary_distance.npy');np.save(path,D)
            with ProcessPoolExecutor(4,initializer=m.initialize_pam_worker,initargs=(str(path),)) as pool:
                fit,_=m.parallel_pam(pool,len(D),2,cfg['pam_restarts'],cfg['seed']+2,cfg['max_iter'])
            agreement,_,_=m.c.aligned(labels,fit['labels'],2)
            row.update(ari=float(adjusted_rand_score(labels,fit['labels'])),aligned_agreement=agreement)
            out=raw[m.IDS].copy();out['cluster']=fit['labels']+1;out.to_parquet(data/(feature+'_k2_labels.parquet'),index=False)
            path.unlink();del D
            print('Single-scale influence '+feature+' completed',flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(tables/'single_scale_influence.csv',index=False)
    influence=pd.DataFrame(rows)
    priority=influence.loc[(influence.ari.lt(.8)|influence.aligned_agreement.lt(.9)),'feature'].tolist()
    # Include all substantial scale changes as secondary focus when combined
    # effects are substantial even if no single-feature change crosses screen.
    review=json.loads((ROOT/'logs/clinical_review_latest.json').read_text(encoding='utf-8'))
    original=Path(review['data_path'])
    cases=pd.read_csv(original/'待复核清单.csv')
    selected=cases[cases.stay_id.isin(raw.stay_id)&cases.feature.isin(influence.loc[influence.range_ratio.ge(1.1),'feature'])].copy()
    selected['focus_priority']=np.where(selected.feature.isin(priority),'single_scale_major_effect','combined_scale_secondary')
    selected.to_csv(data/'聚类尺度影响定向复核.csv',index=False,encoding='utf-8-sig')
    evidence=pd.read_csv(original/'自动证据复核.csv')
    evidence=evidence[evidence.case_id.isin(selected.case_id)]
    assert evidence.recomputed_aggregate_matches.all()
    evidence.to_csv(data/'定向复核_来源核验.csv',index=False,encoding='utf-8-sig')
    details=pd.read_csv(original/'来源与邻近观测.csv')
    details[details.case_id.isin(selected.case_id)].to_csv(data/'定向复核_邻近原始观测.csv',index=False,encoding='utf-8-sig')
    summary=selected.groupby(['feature','focus_priority']).agg(n_cases=('case_id','size'),n_patients=('subject_id','nunique')).reset_index()
    summary.to_csv(tables/'focused_review_counts.csv',index=False)
    counts=evidence.groupby('technical_support').size().reset_index(name='n_cases')
    counts.to_csv(tables/'focused_source_support.csv',index=False)
    text='# 极端值与Gower尺度的定向检查\n\n这是看到主8项严格筛查/重拟合尺度结果后的解释性诊断，不替换主方法或按结果调权。主数据与严格数据的固定原尺度k2标签完全一致；重拟合严格尺度ARI约0.594。逐项仅改变尺度、保持严格筛查后的中位数填补值不变，结果见single_scale_influence.csv。变量同时缩放的影响可能非加和，单项诊断不是因果归因。\n\n'
    text+=f'已圈定{len(selected)}项、{selected.subject_id.nunique()}名训练患者作为尺度定向复核，单项重大影响变量：{priority}。来源重算全部一致；配套原始观测和技术证据仅在data/。原人工复核表未改，未删病例。\n\n'
    text+='范围是研究压力测试而非正常/异常真值。重复记录不能证明临床真实，单次记录也不能证明错误。需结合原始来源、单位和邻近观测判断；若仍无法判真，保留主数据并报告尺度敏感性，不为了获得好分群删除极端值。记录缩放可以明显影响Gower的有效权重，即便被筛查患者仅占小比例，不能据此证明拉取过程错误。\n'
    (reports/'focused_review_report.md').write_text(text,encoding='utf-8')
    m.dump(reports/'status.json',{'status':'complete','priority_features':priority,'n_cases':len(selected),
        'n_patients':int(selected.subject_id.nunique()),'all_selected_source_aggregates_match':True,
        'clinical_truth_determined':False,'original_review_file_unchanged':True,
        'script_sha256':m.c.sha(__file__)})
    (reports/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    verify_focus(loc)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-existing',action='store_true');args=parser.parse_args()
    if args.verify_existing:verify_focus(json.loads((ROOT/'logs/clustering_v5_latest.json').read_text(encoding='utf-8')))
    else:main()
