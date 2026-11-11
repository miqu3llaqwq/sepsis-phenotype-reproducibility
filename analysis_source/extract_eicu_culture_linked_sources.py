"""Cache raw evidence for all hospital stays with culture records, not a cohort.

Preserves source rows and raw strings. Does not calculate scores or replace t0.
Completed tables are reusable only after their source fingerprint is checked.
"""
import os
from pathlib import Path
import hashlib
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

raw = Path(os.environ['EICU_DATA_DIR'])
out = Path('outputs/eicu_culture_linked_raw_20261026')
out.mkdir(exist_ok=True, parents=True)
patient = pd.read_csv(raw / 'patient.csv.gz', dtype=str, keep_default_na=False)
micro = pd.read_parquet('outputs/infection_candidates_20261026/culture_evidence_local.parquet')
hospital_stays = set(micro.patienthealthsystemstayid.astype(str))
target = patient[patient.patienthealthsystemstayid.isin(hospital_stays)]
ids = set(target.patientunitstayid)
assert target.patientunitstayid.is_unique
target.to_parquet(out / 'patient_local.parquet', index=False)

specs = {
 'infusionDrug': None,
 'respiratoryCare': ['respcareid','patientunitstayid','respcarestatusoffset','currenthistoryseqnum','airwaytype','ventstartoffset','ventendoffset','priorventstartoffset','priorventendoffset'],
 'lab': None,
 'respiratoryCharting': None,
 'intakeOutput': None,
 'nurseCharting': None,
 'vitalAperiodic': ['vitalaperiodicid','patientunitstayid','observationoffset','noninvasivesystolic','noninvasivediastolic','noninvasivemean'],
 'vitalPeriodic': ['vitalperiodicid','patientunitstayid','observationoffset','temperature','sao2','heartrate','respiration','systemicsystolic','systemicdiastolic','systemicmean'],
}
manifest_path = out / 'extraction_manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'tables': {}}
manifest.update(status='running_raw_cache_not_validation', scope='all ICU stays within hospital stays having a microLab row', target_icu_stays=len(ids), target_hospital_stays=len(hospital_stays), target_hospitals=int(target.hospitalid.nunique()))
print('target_stays', len(ids), 'hospitals', target.hospitalid.nunique(), flush=True)
for name, columns in specs.items():
    src = raw / (name + '.csv.gz')
    dest = out / (name + '_local.parquet')
    stat = src.stat()
    fingerprint = {'path': str(src), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    previous = manifest['tables'].get(name, {})
    if previous.get('status') == 'complete' and previous.get('source') == fingerprint and dest.exists():
        print(name, 'reusing completed cache', flush=True)
        continue
    writer = None
    scanned = kept = 0
    started = time.monotonic()
    last = started
    for chunk in pd.read_csv(src, usecols=columns, dtype=str, keep_default_na=False, chunksize=250000):
        scanned += len(chunk)
        selected = chunk[chunk.patientunitstayid.isin(ids)]
        if len(selected):
            table = pa.Table.from_pandas(selected, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema, compression='zstd')
            writer.write_table(table)
            kept += len(selected)
        if time.monotonic() - last > 25:
            print(name, 'scanned', scanned, 'retained', kept, flush=True)
            last = time.monotonic()
    if writer:
        writer.close()
    else:
        pd.read_csv(src, usecols=columns, nrows=0, dtype=str).to_parquet(dest, index=False)
    info = {'status': 'complete', 'source': fingerprint, 'rows_scanned': scanned, 'rows_retained': kept, 'seconds': round(time.monotonic()-started, 1), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    manifest['tables'][name] = info
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(name, 'complete', scanned, kept, flush=True)
manifest['status'] = 'complete_raw_cache_not_validation'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
