# Sepsis phenotype reproducibility

Research code accompanying **Missing-Data Sensitivity of Medoid-Based Sepsis Phenotyping in MIMIC-IV and eICU**.

This repository studies sensitivity of medoid-based clustering to missing-data handling, uncertainty in representative values, and external documentation coverage. It does not provide a clinically validated subtype classifier.

## Contents

- `analysis_source/`: versioned analysis scripts, their local Python dependencies, non-fitted configuration, extraction SQL and the upstream SQL snapshot. Historical modules are retained for dependencies and provenance; their standalone older experiments are not the final manuscript analysis.
- `src/sepsis_repro/assignment.py`: the original observed-value Gower/nearest-medoid functions, extracted without changing their function bodies.
- `scripts/assign_prepared.py`: an explicit-input wrapper for separately authorized, semantically audited feature and frozen-model files.
- `tests/`: small synthetic tests, not clinical evidence.
- `config/`: definition and eligibility contracts, without learned percentiles or patient representatives.
- `docs/REPRODUCING.md`: run order, private prerequisites and remaining portability limitations.
- `docs/DATA_ACCESS.md`: access and redistribution boundaries.
- `provenance/`: original/release code hashes and the recorded historical environment.

## Safe local check (no database required)

```shell
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_source.py
```

These checks assess code packaging, source syntax and synthetic distance behavior. They do **not** reproduce the reported clinical results.

## Reproducing the study

The primary MIMIC branch used 17,810 feature-eligible training patients. The separate post hoc shared-definition branch refitted MIMIC models before fixed assignment of 51 eligible eICU patients; the route sensitivity analysis had 115. These are existing reported results, not outputs produced by cloning this repository. See `docs/REPRODUCING.md` for each branch and its prerequisites.

Source databases, patient-level extracts, medoid patient records, learned transformation files, cloud credentials and manuscript files are intentionally absent. Obtain database access through PhysioNet and generate or obtain restricted derived inputs under the applicable data agreement. No public model weights are supplied. The shared-definition branch is not strict external validation of the original primary model. No k was selected using eICU.

## Review and licensing

The repository is prepared for author review. A separate code-only ZIP excludes Git history and personal repository links for double-blind review. Repository visibility and the conference's handling of supplementary files must be managed separately.

MIT-LCP SQL is used under its MIT license; see `third_party/MIMIC_CODE_LICENSE.txt` and `NOTICE.md`. A blanket license for newly authored study code has not been selected. Do not infer that the upstream MIT license licenses the source clinical data.
