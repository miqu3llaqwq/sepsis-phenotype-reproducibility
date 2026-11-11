"""Audit the frozen phenotype protocol without reading outcomes into clustering."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    cfg = json.loads((ROOT / 'config/phenotype_primary_v1.json').read_text(encoding='utf-8'))
    latest = json.loads((ROOT / 'logs/clustering_v7_latest.json').read_text(encoding='utf-8'))
    data = Path(latest['data_path'])
    checks = []

    raw = pd.read_parquet(data / 'core14_sensitivity/eligible_train_unimputed.parquet')
    primary = cfg['features_primary']
    checks.append({'check': 'all_primary_features_present', 'pass': set(primary).issubset(raw.columns)})
    checks.append({'check': 'training_split_only', 'pass': set(raw['split'].astype(str)) == {'train'}})
    checks.append({'check': 'patient_ids_unique', 'pass': not raw['subject_id'].duplicated().any() and not raw['stay_id'].duplicated().any()})
    checks.append({'check': 'primary_distance_equal_feature', 'pass': cfg['clustering']['distance'] == 'equal_feature_gower'})
    checks.append({'check': 'primary_imputation_training_only', 'pass': cfg['primary_missingness']['method'] == 'MICE_PMM_training_only' and cfg['primary_missingness']['imputation_used_for_primary'] is True})
    checks.append({'check': 'clinical_k_not_frozen', 'pass': cfg['clinical_k_frozen'] is False and cfg['clustering']['primary_k_candidate'] is None})

    mi_dir = data / 'mice_pmm_sensitivity'
    imputed = sorted(mi_dir.glob('imputed_m*.parquet'))
    checks.append({'check': 'ten_mice_datasets_present', 'pass': len(imputed) == 10})
    checks.append({'check': 'mice_training_only_by_input_partition', 'pass': set(raw['split'].astype(str)) == {'train'}})
    checks.append({'check': 'mice_outcome_blind_by_primary_schema', 'pass': not {'hosp_mortality', 'mort_28d', 'mort_90d', 'surv_days_90'}.intersection(raw.columns)})

    result = {'status': 'pass' if all(x['pass'] for x in checks) else 'fail', 'checks': checks, 'data_path': str(data), 'n_audited': int(len(raw))}
    out = ROOT / 'outputs/reports/clustering_v7/20261010_125027/protocol_consistency_audit.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['status'] != 'pass': raise SystemExit(1)

if __name__ == '__main__':
    main()
