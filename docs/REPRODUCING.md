# Reproduction guide and honest limits

## 1. Scope of the release

This is a source release of the existing analyses plus a standalone assignment kernel. It is not a tested one-command reconstruction from raw databases on a new machine. Packaging did not rerun the original analyses. Exact cached inputs and historical audit artifacts are not distributed. Several archived scripts read files at import time, perform work when executed, or require historical QA sentinels; do not bulk-import the archive.

Keep the archived tree intact in a separate authorized workspace. Run archived scripts from `analysis_source/`. Set `EICU_DATA_DIR` to the directory holding the licensed eICU CSV files and `GCP_PROJECT` to your own billing project. Use standard local cloud authentication; never commit credentials. `SEPSIS_LEGACY_SOURCE` is optional and defaults to the supplied legacy analysis/config directory.

Non-fitted definitions are supplied. Primary P1/P99, patient-level prepared data, imputation realizations and fitted representatives must be recreated/obtained through an authorized route. The recorded package versions describe the original environment and are not a verified fresh-environment lockfile. Historical SQL references specific BigQuery schema aliases; confirm that those aliases refer to MIMIC-IV v3.1 in your environment before extraction. Querying clinical data may incur cloud charges; this release performs no cloud queries by default.

## 2. Primary branch

1. Inspect `rebuild_corrected_mimic_branch.py` and `work/mimic_rebuilt_20261026/sql/`. The source reconstruction reused the derived hourly SOFA table; it did not independently rebuild every SOFA component. The reference infection implementation permits patient-level culture matching and a date-only fallback.
2. Recreate the source cohort, reviewed vital/laboratory features and patient split. See legacy modules `08_rebuild.py`, `09_observation_review.py` and `10_prepare_reviewed_features.py`, then `prepare_rebuilt_core14_training.py`. The branch was split by patient before eligibility filtering. Data are not included here.
3. `run_revised_phenotypes_20261027.py --check-only` checks prepared training inputs. The full run uses `--workers 3` and is computationally intensive. It fits ten MICE-PMM datasets, observed-data models and resampling over k = 2–6.
4. `freeze_revised_inputs_20261027.py` records the original frozen inputs. It also requires historical acceptance/QA files, original frozen-input hash checks and provenance archives that are not shipped. This is a known portability prerequisite, not a check to delete or bypass. Review `accept_rebuilt_cohort_20261027.py` when rebuilding those checks in an authorized workspace.
5. `run_missingness_depth_20261028.py`, `cross_query_anchor_depth_20261028.py` and `verify_missingness_depth_20261028.py` supply the perturbation diagnostics.
6. `improve_phenotypes_20261029.py` and `verify_improvement_20261029.py` cover optimisation, complete-medoid candidates and held-out assignments. Preserve the validation/test separation and the original train-only limits.

## 3. Shared-definition branch

The original staged workflow is represented by `start_common_definition_20261029.py`, `audit_common_source_semantics_20261029.py`, `common_sofa_lab_bounds_20261030.py` and `run_common_pipeline_20261030.py`. The latter orchestrates extraction, chart compaction/verification, landmark selection, features, verification, model fitting and reporting. Its upstream source tables and laboratory bounds must already exist.

The eICU source adapters additionally require credentialed, culture-linked extraction caches produced in the original mapping workflow; `extract_eicu_culture_linked_sources.py` is included as source. All intermediate Parquet files, remote result caches and QA sentinels remain absent. Reproduction therefore requires reconstructing the source adapters, not merely renaming columns or running the final model script.

`build_common_identified_20261030.py` identifies the subset whose first landmark is unchanged across admissible SOFA bounds. `common_features_20261030.py` extracts the retrospective 6-hour/24-hour features. `fit_common_models_20261030.py` fits only MIMIC and freezes observed-value models for each k. `external_common_20261104.py` verifies and assigns the baseline and route-expanded eICU cohorts. It writes sensitive patient-level intermediates and must only run in the authorized private workspace. Route expansions remain post hoc sensitivity analyses, not certified hospital codebook equivalence.

## 4. Standalone prepared-data assignment

```shell
python scripts/assign_prepared.py --features /private/eligible_features.csv --bundle /private/common_frozen_bundle --out-dir /private/new_assignment
```

This wrapper requires a nonmissing boolean `feature_eligible` flag set upstream and all 14 named feature columns. It reads one original frozen bundle and verifies its manifest before use. It never computes infection events, SOFA or t0, and never substitutes an alternative cohort rule. The output uses input row position rather than transferring patient identifiers. All medoid distances must be computable; labels are not assigned from only the available subset. Ties follow the original smallest-label rule. No refitting, imputation or k selection is done externally.

## 5. Verification limits

Synthetic tests exercise clipping, missing pairs, complete-distance requirements, deterministic ties and a separately coded scalar distance calculation. They verify the packaged kernel, not clinical event semantics or the published patient counts. Full clinical replay requires the restricted inputs and provenance files described above. Reusing cached fits is not a fresh model run.
