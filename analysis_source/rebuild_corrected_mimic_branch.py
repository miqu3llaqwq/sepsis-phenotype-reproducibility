"""Rebuild cohort/features in an isolated branch using original eligibility.

The original Desktop project and its frozen inputs are never modified.
This is a versioned rerun, not external validation or a replacement clinical k.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BRANCH=ROOT/'work/mimic_rebuilt_20261026'
OLD=Path(os.environ.get('SEPSIS_LEGACY_SOURCE', str(BRANCH)))
EVIDENCE=ROOT/'outputs/source_lineage_20261026'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    for d in ['analysis','config','data','sql','reports','tables','logs']:
        (BRANCH/d).mkdir(parents=True,exist_ok=True)
    snapshots={}
    for folder,pattern in [('analysis','*.py'),('config','*.json')]:
        for p in (OLD/folder).glob(pattern):
            dest=BRANCH/folder/p.name
            if not dest.exists(): dest.write_bytes(p.read_bytes())
            snapshots[str(dest.relative_to(BRANCH))]={'source':str(p),'sha256':sha(dest)}
    c=json.loads((BRANCH/'config/analysis_v2.json').read_text())
    soi=(EVIDENCE/'declared_build_snapshot/sepsis__suspicion_of_infection.sql').read_text().strip().rstrip(';')
    s3=(EVIDENCE/'declared_build_snapshot/sepsis__sepsis3.sql').read_text().strip().rstrip(';')
    for module in ['hosp','icu','derived']:
        soi=soi.replace('physionet-data.mimiciv_'+module,c[module])
        s3=s3.replace('physionet-data.mimiciv_'+module,c[module])
    s3=s3.replace('`'+c['derived']+'.suspicion_of_infection`','rebuilt_soi')
    prefix=f'WITH rebuilt_soi AS ({soi}), rebuilt_sepsis3 AS ({s3}), candidate AS ('
    queries={}
    for name,file in [('cohort_audit','cohort_audit_historical_query.sql'),('features','features_historical_query.sql')]:
        q=(EVIDENCE/file).read_text()
        assert q.count('WITH candidate AS (')==1
        assert q.count('`'+c['derived']+'.sepsis3`')==1
        q=q.replace('WITH candidate AS (',prefix,1).replace('`'+c['derived']+'.sepsis3`','rebuilt_sepsis3')
        queries[name]=q
        (BRANCH/'sql'/(name+'.sql')).write_text(q,encoding='utf-8')
    (BRANCH/'reports/source_snapshot_manifest.json').write_text(json.dumps(snapshots,indent=2,ensure_ascii=False),encoding='utf-8')
    return c,queries

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--extract',action='store_true');a=parser.parse_args()
    c,queries=prepare()
    c['maximum_bytes_billed']=min(c['maximum_bytes_billed'],40_000_000_000)
    manifest_path=BRANCH/'reports/rebuild_manifest.json'
    m=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'status':'prepared','jobs':{}}
    m.update(branch='reconstructed_sepsis3_primary_rerun_candidate',scope='declared Sepsis-3 and suspicion SQL over v3.1 dependencies; materialized SOFA not rebuilt',original_frozen_inputs_untouched=True,
      sql_sha256={n:hashlib.sha256(q.encode()).hexdigest() for n,q in queries.items()},maximum_bytes_billed=c['maximum_bytes_billed'])
    if not (a.preflight or a.extract):
        manifest_path.write_text(json.dumps(m,indent=2),encoding='utf-8'); print('Prepared isolated SQL and source snapshots.'); return
    from google.cloud import bigquery
    client=bigquery.Client(project=os.environ.get('GCP_PROJECT'))
    for name,q in queries.items():
        dry=client.query(q,job_config=bigquery.QueryJobConfig(dry_run=True,use_query_cache=False),timeout=30,retry=None)
        info=m['jobs'].setdefault(name,{})
        info['estimated_bytes']=dry.total_bytes_processed
        assert dry.total_bytes_processed<=c['maximum_bytes_billed']
        print(name,'estimated_bytes',dry.total_bytes_processed,flush=True)
    m['status']='preflight_passed'
    manifest_path.write_text(json.dumps(m,indent=2),encoding='utf-8')
    if not a.extract:return
    frames={}
    for name,q in queries.items():
        dest=BRANCH/'data'/(name+'.parquet')
        info=m['jobs'][name]
        if info.get('complete') and dest.exists() and info.get('sha256')==sha(dest):
            frame=pd.read_parquet(dest)
            print(name,'reusing completed extraction',len(frame),flush=True)
        else:
            j=client.query(q,job_config=bigquery.QueryJobConfig(maximum_bytes_billed=c['maximum_bytes_billed']),timeout=30,retry=None)
            info['job_id']=j.job_id; m['status']='extracting_'+name
            manifest_path.write_text(json.dumps(m,indent=2),encoding='utf-8')
            frame=j.result(timeout=600).to_dataframe(create_bqstorage_client=False)
            frame.to_parquet(dest,index=False)
            info.update(complete=True,rows=len(frame),bytes_processed=j.total_bytes_processed,sha256=sha(dest))
            manifest_path.write_text(json.dumps(m,indent=2),encoding='utf-8')
            print(name,'extracted_rows',len(frame),flush=True)
        frames[name]=frame
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('source_rebuild',BRANCH/'analysis/08_rebuild.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.validate_dataset(frames['features'],c)
    assert len(frames['features'])==31422 and len(frames['cohort_audit'])==41295
    dataset=module.split_ids(frames['features'],c)
    assert dataset.subject_id.is_unique
    dest=BRANCH/'data/analysis_dataset.parquet'
    dataset.to_parquet(dest,index=False)
    # Original split routine must be insensitive to input row order.
    check=module.split_ids(frames['features'].iloc[::-1],c)
    left=dataset.set_index('subject_id').split.sort_index()
    right=check.set_index('subject_id').split.sort_index()
    assert left.equals(right)
    for split in ['train','validation','test']:
        dataset.loc[dataset.split==split,['subject_id','hadm_id','stay_id']].to_parquet(BRANCH/'data'/(split+'_ids.parquet'),index=False)
    missing=module.save_summaries(dataset,frames['cohort_audit'],c,BRANCH/'tables')
    m.update(status='cohort_features_qc_and_split_complete_clustering_not_started',dataset_sha256=sha(dest),split_counts=dataset.split.value_counts().to_dict(),split_seed=c['seed'],split_row_order_invariance=True,training_missingness_over_50=missing)
    manifest_path.write_text(json.dumps(m,indent=2),encoding='utf-8')
    (BRANCH/'logs/v2_latest.json').write_text(json.dumps({'manifest':str(manifest_path),'data_path':str(BRANCH/'data'),'tables_path':str(BRANCH/'tables')},indent=2),encoding='utf-8')
    # New freeze applies only to this initial rerun dataset, never old limits/medoids.
    (BRANCH/'config/frozen_split_v2.json').write_text(json.dumps({'dataset_sha256':sha(dest),'seed':c['seed'],'scope':'new rerun initial features; phenotype preprocessing and fitting pending'},indent=2),encoding='utf-8')
    print('Cohort/initial features QC passed; splits',m['split_counts'],'phenotype preprocessing and clustering pending.',flush=True)

if __name__=='__main__':main()
