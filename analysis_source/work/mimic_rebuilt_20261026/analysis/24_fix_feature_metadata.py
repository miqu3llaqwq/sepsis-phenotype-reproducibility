"""Correct metadata against executed SQL; do not change measurements or SQL."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    sql=ROOT/'sql/v2_features.sql';query=sql.read_text(encoding='utf-8')
    gcs=query.split('gcs_components AS (',1)[1].split('fio2_observations AS (',1)[0]
    assert 'mimiciv_3_1_icu.chartevents' in gcs and 'INTERVAL 6 HOUR' in gcs
    assert all(str(n) in gcs for n in [220739,223900,223901])
    assert 'mimiciv_3_1_hosp.labevents' in query and 'e.itemid=223835' in query
    loc=json.loads((ROOT/'logs/clustering_v6_latest.json').read_text(encoding='utf-8'))
    dictionary=ROOT/'feature_dictionary.csv';before=sha(dictionary)
    run=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    archive=ROOT/'logs'/('metadata_correction_'+run);archive.mkdir()
    (archive/'feature_dictionary_before.csv').write_bytes(dictionary.read_bytes())
    frame=pd.read_csv(dictionary);assert not frame.analysis_name.duplicated().any()
    g=frame.analysis_name.eq('gcs_min');assert g.sum()==1
    frame.loc[g,['domain','source_table','window_hours']]=['neurologic','icu.chartevents',6]
    frame.loc[g,'notes']='v2 SQL: complete assessable eye/verbal/motor at same charttime; minimum in6h; unassessable ETT excluded; train-missingness rule deferred this variable;metadata corrected,values unchanged'
    p=frame.analysis_name.eq('pao2_fio2_ratio');assert p.sum()==1
    frame.loc[p,'source_table']='hosp.labevents + icu.chartevents'
    frame.loc[p,'notes']='explicit arterial specimen; sample FiO2 preferred,otherwise prior60min chartevents FiO2;median of specimen-paired ratios;both charttime/storetime<=t0;metadata corrected,values unchanged'
    gender=frame.analysis_name.eq('gender')
    frame.loc[gender,'notes']='recorded demographic; description only in v5/v6 physiology diagnostics;future prediction/auxiliary roles require separate freeze'
    frame.to_csv(dictionary,index=False,encoding='utf-8')
    # Correct version-specific role metadata generated before dictionary repair.
    role_path=ROOT/'data_dictionary/phenotype_v6_variable_roles.csv'
    roles=pd.read_csv(role_path)
    roles.loc[roles.feature.eq('gcs_min'),'source_domain']='neurologic'
    roles.to_csv(role_path,index=False,encoding='utf-8')
    report=Path(loc['report_path'])
    changes={'gcs_min':{'domain':'neurologic','source_table':'icu.chartevents','window_hours':6},
        'pao2_fio2_ratio':{'source_table':'hosp.labevents + icu.chartevents'},
        'gender':{'notes':'removed unrelated oxygen note;clarified diagnostic role'}}
    evidence={'date':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'changes':changes,
        'before_sha256':before,'after_sha256':sha(dictionary),'query_sha256':sha(sql),
        'original_backup':str(archive/'feature_dictionary_before.csv'),
        'patient_values_changed':False,'extraction_sql_changed':False,'cluster_feature_set_changed':False,
        'historical_query_snapshots_changed':False}
    (report/'metadata_correction.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    text='# 元数据修正与执行证据\n\n'
    text+='已按sql/v2_features.sql核对并修正根字典：GCS为neurologic、icu.chartevents、6h；P/F由实验室动脉血气与床旁FiO2共同构成；gender备注移除误混入的氧合说明。已保存修改前字典，不改SQL、患者值、队列、分区或本轮16项距离输入。\n\n'
    text+='根字典的primary_role字段仍保留历史v2来源口径；当前v6诊断角色以data_dictionary/phenotype_v6_variable_roles.csv为准，不能把历史cluster_and_prediction读作临床k或预测已经冻结。已有不可评分GCS、氧合配对不完整及来源覆盖需另行审计，元数据修正不证明缺失率下降。\n'
    (report/'metadata_correction.md').write_text(text,encoding='utf-8')
    (report/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    with (ROOT/'logs/analysis_log.md').open('a',encoding='utf-8') as f:f.write('\n\n## 字典标注修正 '+run+'\n\n'+text.split('\n\n',1)[1])
    print(json.dumps(evidence,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
