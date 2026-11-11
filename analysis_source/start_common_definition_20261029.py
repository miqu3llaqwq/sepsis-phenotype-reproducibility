"""Start the authorized common-definition branch. No fitting or original writes.

The infection calculation is a candidate-event audit, not a sepsis cohort.
No old Sepsis-3 membership restriction is applied to the MIMIC source query.
"""
from pathlib import Path
import argparse
import bisect
import hashlib
import json
import os
import re
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/common_definition_20261029'
WORK = ROOT / 'work/common_definition_20261029'
RAW = Path(os.environ['EICU_DATA_DIR'])
REF = ROOT / 'outputs/source_lineage_20261026/declared_build_snapshot'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def pair_one(times, antibiotic):
    """Hospital-relative seconds; [a-72h,a) first, then (a,a+24h]."""
    lo = bisect.bisect_left(times, antibiotic - 72*3600)
    mid = bisect.bisect_left(times, antibiotic)
    if lo < mid:
        return times[lo], 'prior72'
    right = bisect.bisect_right(times, antibiotic)
    end = bisect.bisect_right(times, antibiotic + 24*3600)
    if right < end:
        return times[right], 'later24'
    return np.nan, 'none'


def pair_events(orders, cultures):
    times = {key: sorted(set(g.culture_seconds.dropna().tolist()))
             for key, g in cultures.groupby('admission_id')}
    records = []
    for r in orders.itertuples(index=False):
        t, branch = pair_one(times.get(r.admission_id, []), r.antibiotic_seconds)
        if np.isfinite(t):
            records.append((r.order_key, r.person_id, r.admission_id, r.stay_id,
                            r.antibiotic_seconds, t, min(r.antibiotic_seconds, t), branch))
    return pd.DataFrame(records, columns=['order_key', 'person_id', 'admission_id', 'stay_id',
                                         'antibiotic_seconds', 'culture_seconds', 'suspected_seconds', 'branch'])


def check_pairing():
    cases = [([], 0), ([0], 0), ([-259201], 0), ([-259200], 0),
             ([86400], 0), ([86401], 0), ([-1, 1], 0), ([-4, -2, 1], 0)]
    rng = np.random.default_rng(20261029)
    for _ in range(1000):
        cases.append((sorted(set(rng.integers(-400000, 400000, 20).tolist())), int(rng.integers(-100, 100))))
    for ts, a in cases:
        prior = [t for t in ts if a-259200 <= t < a]
        later = [t for t in ts if a < t <= a+86400]
        expected = min(prior) if prior else min(later) if later else np.nan
        result, _ = pair_one(ts, a)
        assert (np.isnan(result) and np.isnan(expected)) or result == expected
    return len(cases)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / 'sql').mkdir(parents=True, exist_ok=True)
    (WORK / 'data').mkdir(exist_ok=True)
    protocol = {
        'version': 'common_definition_20261029_design_v1',
        'authorization': 'User selected option 2: rebuild MIMIC under definitions implementable in both databases, then external assignment in eICU.',
        'stage': 'feasibility_and_infection_candidates_not_frozen_full_protocol',
        'posthoc': True, 'previous_eicu_results_already_seen': True,
        'original_results_retained': True,
        'analysis_unit': 'First eligible ICU event per hospital admission; patient-disjoint source split; repeated admissions handled as correlated observations.',
        'infection_pairing_locked_for_audit': {
            'scope': 'same hospital admission only',
            'culture_time': 'observed collection/chart timestamp only; no date-only midnight imputation',
            'antibiotic_time': 'prescription start, not claimed administration time',
            'windows': '[antibiotic-72h,antibiotic) preferred; otherwise (antibiotic,antibiotic+24h]; earliest culture in chosen window',
            'simultaneous': 'excluded to retain published reference strict inequalities; separately counted if needed',
            'culture_rows': 'unique time for temporal pairing, preserve source rows; not a reconstructed specimen identifier',
            'drug_dictionary': 'Frozen public reference substring dictionary; includes some antifungals. Final route/form/cancellation alignment still requires review.',
        },
        'planned_time_design': 'Retrospective observation-time analysis; storetime is not equated to observation time. Exact clock anchoring and missing-SOFA policy must be implemented identically before cohort freeze.',
        'planned_features': 'Same fixed core14, 6h vitals and 24h labs. Measurement-source heterogeneity must remain explicit; common window summaries do not create equivalent sensors.',
        'planned_model': 'Training-only P1-P99 and preprocessing; missing-aware equal-feature Gower, k=2..6; train and freeze only in MIMIC. No eICU refitting, k selection or outcome-based rule tuning.',
        'release_gates': ['two source adapters plus shared scoring/time kernel', 'six-domain and ambiguity rules audited',
                          'patient-disjoint split', 'training-only transformations', 'stability checks and frozen manifest',
                          'independent assignment recomputation'],
        'not_claimed': ['original-model strict external validation', 'new sepsis cohort size', 'new t0', 'new fitted model', 'clinical validity'],
    }
    save('design_v1.json', protocol)
    frozen = ROOT / 'work/mimic_rebuilt_20261026/frozen/20261027'
    tracked = {str(p.relative_to(ROOT)): sha(p) for p in frozen.rglob('*') if p.is_file()}
    for p in [ROOT/'outputs/中文论文精简稿_v17_20261029.md',
              ROOT/'outputs/eicu_source_sensitivity_20261028/monitor_ordered/run_manifest.json']:
        tracked[str(p.relative_to(ROOT))] = sha(p)
    existing = OUT / 'original_artifacts_sha256.json'
    if existing.exists():
        assert json.loads(existing.read_text(encoding='utf-8')) == tracked, 'Original files changed since branch creation'
    else:
        save(existing.name, tracked)
    sql = (REF / 'medication__antibiotic.sql').read_text(encoding='utf-8').strip().rstrip(';')
    sql = sql.replace('physionet-data.mimiciv_hosp', 'physionet-data.mimiciv_3_1_hosp').replace('physionet-data.mimiciv_icu', 'physionet-data.mimiciv_3_1_icu')
    # Keep original row provenance for the candidate audit.
    marker = 'pr.subject_id, pr.hadm_id'
    assert sql.count(marker) == 1
    sql = sql.replace(marker, marker + ', pr.poe_id, pr.pharmacy_id, pr.drug_type')
    queries = {
        'mimic_orders': f"""WITH antibiotic AS ({sql})
SELECT ab.*, ad.admittime, ie.intime, ie.outtime,
 DATETIME_DIFF(ab.starttime,ad.admittime,SECOND) AS antibiotic_seconds
FROM antibiotic ab
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` ad USING(hadm_id)
JOIN `physionet-data.mimiciv_3_1_icu.icustays` ie USING(stay_id)
WHERE EXISTS (
 SELECT 1 FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
 WHERE m.hadm_id=ab.hadm_id AND m.charttime IS NOT NULL
 AND m.charttime BETWEEN DATETIME_SUB(ab.starttime, INTERVAL 72 HOUR)
 AND DATETIME_ADD(ab.starttime, INTERVAL 24 HOUR))""",
        'mimic_cultures': """SELECT m.hadm_id, m.subject_id, m.micro_specimen_id,
 m.charttime, m.spec_type_desc, COUNT(*) AS raw_rows,
 DATETIME_DIFF(m.charttime,a.admittime,SECOND) AS culture_seconds
FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING(hadm_id)
WHERE m.charttime IS NOT NULL AND EXISTS (
 SELECT 1 FROM `physionet-data.mimiciv_3_1_icu.icustays` i WHERE i.hadm_id=m.hadm_id)
GROUP BY m.hadm_id,m.subject_id,m.micro_specimen_id,m.charttime,m.spec_type_desc,a.admittime""",
        'mimic_culture_coverage': """SELECT COUNT(*) AS row_count,
 COUNTIF(charttime IS NULL) AS rows_without_exact_time,
 COUNT(DISTINCT hadm_id) AS hospital_admissions,
 COUNT(DISTINCT IF(charttime IS NOT NULL,hadm_id,NULL)) AS admissions_with_exact_time
FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents`""",
        'mimic_dose_item_dictionary': """SELECT itemid,label,abbreviation,linksto,unitname,category
FROM `physionet-data.mimiciv_3_1_icu.d_items`
WHERE REGEXP_CONTAINS(LOWER(label),r'norepinephrine|epinephrine|dopamine|dobutamine|levophed|dobutrex')""",
        'mimic_poe_status_dictionary': """SELECT order_type,order_status,transaction_type,COUNT(*) AS row_count
FROM `physionet-data.mimiciv_3_1_hosp.poe`
WHERE order_type='Medications'
GROUP BY order_type,order_status,transaction_type""",
    }
    for key, query in queries.items():
        (WORK/'sql'/f'{key}.sql').write_text(query, encoding='utf-8')
    return queries


def eicu():
    path = ROOT/'outputs/infection_mapping_refined_20261026/order_evidence_local.parquet'
    orders = pd.read_parquet(path)
    mpath = ROOT/'outputs/infection_candidates_20261026/culture_evidence_local.parquet'
    micro = pd.read_parquet(mpath)
    raw_patient = pd.read_csv(RAW/'patient.csv.gz', usecols=['patientunitstayid','uniquepid'])
    z = orders[orders.reference_syntax_candidate].copy()
    z = z.merge(raw_patient, on='patientunitstayid', validate='many_to_one')
    norm = pd.DataFrame({'order_key': z.medicationid.astype(str), 'person_id': z.uniquepid.astype(str),
                         'admission_id': z.patienthealthsystemstayid.astype(str),
                         'stay_id': z.patientunitstayid.astype(str),
                         'antibiotic_seconds': z.hospital_relative_min.astype(float)*60})
    cultures = pd.DataFrame({'admission_id': micro.patienthealthsystemstayid.astype(str),
                             'culture_seconds': micro.hospital_relative_min.astype(float)*60})
    pairs = pair_events(norm, cultures)
    old = pd.read_parquet(ROOT/'outputs/infection_mapping_refined_20261026/infection_event_candidates_local.parquet')
    a = pairs.set_index('order_key')
    b = old.set_index(old.medicationid.astype(str))
    assert set(a.index) == set(b.index)
    patient = orders.set_index(orders.medicationid.astype(str))
    expected = (b.culture_time_candidate-patient.loc[b.index,'hospitaladmitoffset'])*60
    assert np.allclose(a.loc[b.index,'culture_seconds'], expected)
    joined = pairs.merge(z[['medicationid','explicit_route_chronology_screen']].assign(order_key=z.medicationid.astype(str)).drop(columns='medicationid'),on='order_key',validate='one_to_one')
    joined.to_parquet(WORK/'data/eicu_infection_candidates_local.parquet', index=False)
    return {'candidate_orders_before_pairing': len(norm), 'paired_order_rows': len(pairs),
            'candidate_icu_stays': int(pairs.stay_id.nunique()),
            'candidate_hospital_admissions': int(pairs.admission_id.nunique()),
            'candidate_people': int(pairs.person_id.nunique()),
            'passes_previous_eicu_route_cancel_stop_screen': int(joined.explicit_route_chronology_screen.sum()),
            'earlier_audit_time_concordance': True, 'not_final_cohort': True,
            'input_sha256': {str(path.relative_to(ROOT)): sha(path), str(mpath.relative_to(ROOT)): sha(mpath)}}


def remote(queries, execute):
    from google.cloud import bigquery
    client = bigquery.Client(project=os.environ.get('GCP_PROJECT'))
    schemas = {}
    for table in ['hosp.prescriptions','hosp.poe','hosp.poe_detail','hosp.labevents','hosp.microbiologyevents',
                  'icu.chartevents','icu.inputevents','icu.outputevents','icu.icustays']:
        group, name = table.split('.')
        full = f'physionet-data.mimiciv_3_1_{group}.{name}'
        meta = client.get_table(full, timeout=30, retry=None)
        schemas[full] = {'rows': meta.num_rows, 'modified': str(meta.modified),
                         'fields': [{'name': f.name, 'type': f.field_type} for f in meta.schema]}
        save('live_mimic_schemas.json', schemas)
    manifest_path = OUT/'query_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    total = 0
    for key, query in queries.items():
        digest = hashlib.sha256(query.encode()).hexdigest()
        dest = WORK/'data'/f'{key}_local.parquet'
        prior = manifest.get(key, {})
        if prior.get('sql_sha256') == digest and prior.get('complete') and dest.exists() and sha(dest) == prior.get('output_sha256'):
            print(key, 'verified cache reuse', flush=True)
            continue
        dry = client.query(query, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False), retry=None, timeout=30)
        total += dry.total_bytes_processed
        assert total <= 40_000_000_000, 'Aggregate dry-run bytes exceed branch audit budget'
        manifest[key] = {'sql_sha256': digest, 'estimated_bytes': dry.total_bytes_processed, 'complete': False}
        save('query_manifest.json', manifest)
        print(key, 'dry-run bytes', dry.total_bytes_processed, flush=True)
    if not execute:
        return {'status': 'dry_run_complete'}
    for key, query in queries.items():
        dest = WORK/'data'/f'{key}_local.parquet'
        if manifest[key].get('complete') and dest.exists() and sha(dest) == manifest[key].get('output_sha256'):
            continue
        job = client.query(query, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=40_000_000_000), retry=None, timeout=30)
        manifest[key]['job_id'] = job.job_id
        save('query_manifest.json', manifest)
        df = job.result(timeout=900).to_dataframe(create_bqstorage_client=False)
        df.to_parquet(dest, index=False)
        manifest[key].update(complete=True, rows=len(df), actual_bytes=job.total_bytes_processed,
                             output_sha256=sha(dest), cache_hit=job.cache_hit)
        save('query_manifest.json', manifest)
        print(key, 'complete', len(df), 'rows', flush=True)
    return {'status': 'source_queries_complete', 'query_count': len(queries)}


def mimic_pairs():
    source = pd.read_parquet(WORK/'data/mimic_orders_local.parquet')
    m = pd.read_parquet(WORK/'data/mimic_cultures_local.parquet')
    # Row key preserves original extracted multiplicity; identity is not inferred from order time.
    orders = pd.DataFrame({'order_key': np.arange(len(source)).astype(str),
                           'person_id': source.subject_id.astype(str), 'admission_id': source.hadm_id.astype(str),
                           'stay_id': source.stay_id.astype(str), 'antibiotic_seconds': source.antibiotic_seconds.astype(float)})
    cultures = pd.DataFrame({'admission_id': m.hadm_id.astype(str), 'culture_seconds': m.culture_seconds.astype(float)})
    pairs = pair_events(orders, cultures)
    # Independent inequality oracle on actual source data, not the bisect implementation.
    grouped = {k: g.culture_seconds.to_numpy(float) for k, g in cultures.groupby('admission_id')}
    checked = 0
    for r in orders.sample(min(1000,len(orders)), random_state=20261029).itertuples(index=False):
        t = grouped.get(r.admission_id,np.array([])); a = r.antibiotic_seconds
        prior = t[(t >= a-259200) & (t < a)]; later = t[(t > a) & (t <= a+86400)]
        expected = prior.min() if len(prior) else later.min() if len(later) else np.nan
        actual, _ = pair_one(sorted(set(t.tolist())), a)
        assert (np.isnan(expected) and np.isnan(actual)) or expected == actual
        checked += 1
    pairs.to_parquet(WORK/'data/mimic_infection_candidates_local.parquet',index=False)
    source.assign(order_key=np.arange(len(source)).astype(str)).to_parquet(WORK/'data/mimic_order_provenance_local.parquet',index=False)
    return {'prefiltered_order_rows': len(source), 'paired_order_rows': len(pairs),
            'candidate_icu_stays': int(pairs.stay_id.nunique()),
            'candidate_hospital_admissions': int(pairs.admission_id.nunique()),
            'candidate_people': int(pairs.person_id.nunique()), 'independent_actual_record_checks': checked,
            'culture_specimens_with_multiple_exact_times': int(m.groupby('micro_specimen_id').charttime.nunique().gt(1).sum()),
            'not_final_cohort': True, 'old_sepsis_cohort_membership_filter_used': False,
            'route_cancel_stop_alignment': 'pending; broad reference candidates only'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    queries = prepare()
    result = {'stage': 'common_definition_feasibility', 'new_model_fitted': False,
              'new_external_validation_complete': False, 'pairing_test_cases': check_pairing(),
              'eicu': eicu(), 'authorized_option': 2}
    save('progress.json', result)
    if args.remote:
        try:
            result['mimic_access'] = remote(queries, args.execute)
            if args.execute:
                result['mimic'] = mimic_pairs()
        except Exception as exc:
            result['mimic_access'] = {'status': 'error', 'type': type(exc).__name__, 'message': str(exc)[:1000]}
            save('progress.json', result)
            raise
    tracked = json.loads((OUT/'original_artifacts_sha256.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p) == digest for p,digest in tracked.items())
    result['original_artifacts_unchanged_checked'] = len(tracked)
    save('progress.json', result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
