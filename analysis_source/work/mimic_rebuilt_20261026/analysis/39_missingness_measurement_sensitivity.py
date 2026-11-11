"""Outcome-blinded sensitivity analysis for missingness and measurement process.

This script does not modify the frozen primary clustering. It quantifies whether
cluster assignments are aligned with feature availability and reruns PAM under
two prespecified diagnostic representations:

1. indicator-balanced distance: each variable contributes a value component and
   an availability component with equal within-variable weight;
2. availability-only distance: clusters are formed from missingness indicators
   alone to quantify measurement-process structure.

All fits use the existing training-only core14 eligible population and never
load outcomes, validation/test rows, or outcome-derived fields.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_samples

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "analysis"))
v = importlib.import_module("26_domain_balanced_v7")
m = importlib.import_module("18_physiology_v5")


def feature_limits(frame: pd.DataFrame, features: list[str]) -> dict[str, tuple[float, float]]:
    return {f: (float(frame[f].quantile(0.01)), float(frame[f].quantile(0.99))) for f in features}


def indicator_balanced_distance(
    frame: pd.DataFrame,
    features: list[str],
    limits: dict[str, tuple[float, float]],
    block: int = 256,
) -> np.ndarray:
    """Equal-variable distance with value and availability components.

    For each variable, the contribution is 0.5*availability disagreement plus
    0.5*the normalized absolute value difference when both values are observed.
    Missing values therefore remain visible without receiving twice the weight
    of an observed value.
    """
    x = frame[features].to_numpy(dtype=float)
    valid = np.isfinite(x)
    scaled = np.zeros_like(x, dtype=float)
    for j, f in enumerate(features):
        lo, hi = limits[f]
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"invalid training limit for {f}: {(lo, hi)}")
        if hi > lo:
            scaled[:, j] = np.clip(np.nan_to_num(x[:, j], nan=lo), lo, hi)
            scaled[:, j] = (scaled[:, j] - lo) / (hi - lo)
    n = len(frame)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        a_valid = valid[start:stop, None, :]
        b_valid = valid[None, :, :]
        both = a_valid & b_valid
        availability = np.logical_xor(a_valid, b_valid).astype(np.float32)
        value = np.abs(scaled[start:stop, None, :] - scaled[None, :, :])
        score = 0.5 * availability + 0.5 * value * both
        out[start:stop] = score.mean(axis=2, dtype=np.float32)
    np.fill_diagonal(out, 0.0)
    return out


def availability_distance(frame: pd.DataFrame, features: list[str], block: int = 256) -> np.ndarray:
    """Hamming distance on feature-availability indicators."""
    valid = frame[features].notna().to_numpy(dtype=np.uint8)
    n = len(frame)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        out[start:stop] = np.mean(valid[start:stop, None, :] != valid[None, :, :], axis=2, dtype=np.float32)
    np.fill_diagonal(out, 0.0)
    return out


def fit_grid(
    distance: np.ndarray,
    raw: pd.DataFrame,
    features: list[str],
    tag: str,
    base_labels: dict[int, np.ndarray],
    out_data: Path,
    out_tables: Path,
    seed: int,
    candidate_k: list[int],
    pam_restarts: int,
    max_iter: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in candidate_k:
        fit, restarts = m.c.fit_pam(distance, k, pam_restarts, seed + k, max_iter)
        sil = silhouette_samples(distance, fit["labels"], metric="precomputed")
        sizes = np.bincount(fit["labels"], minlength=k)
        base = base_labels[k]
        agreement, mapped, _ = m.c.aligned(base, fit["labels"], k)
        rows.append({
            "analysis": tag,
            "k": k,
            "n": len(raw),
            "silhouette_mean": float(sil.mean()),
            "silhouette_negative_fraction": float((sil < 0).mean()),
            "min_cluster_fraction": float(sizes.min() / len(raw)),
            "ari_vs_primary_missing_aware": float(adjusted_rand_score(base, fit["labels"])),
            "aligned_agreement_vs_primary": float(agreement),
            "loss": float(fit["loss"]),
        })
        pd.DataFrame({**{c: raw[c] for c in m.IDS}, "cluster": fit["labels"] + 1}).to_parquet(
            out_data / f"{tag}_k{k}_labels.parquet", index=False
        )
        pd.DataFrame(restarts).to_csv(out_tables / f"{tag}_k{k}_restarts.csv", index=False)
        del fit, restarts, sil
    return rows


def main() -> None:
    started = time.time()
    pointer = json.loads((ROOT / "logs/clustering_v7_latest.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "config/phenotype_primary_v1.json").read_text(encoding="utf-8"))
    data_root = Path(pointer["data_path"])
    tables = Path(pointer["tables_path"])
    reports = Path(pointer["report_path"])
    raw = pd.read_parquet(data_root / "core14_sensitivity" / "eligible_train_unimputed.parquet")
    features = list(cfg["features_primary"])
    if len(raw) != 14138:
        raise ValueError(f"unexpected core14 population size: {len(raw)}")
    forbidden = {"hosp_mortality", "event_type", "endpoint", "follow_up_days", "deathtime", "dischtime"}
    loaded_forbidden = forbidden.intersection(raw.columns)
    if loaded_forbidden:
        raise ValueError(f"outcome-like columns loaded into sensitivity input: {sorted(loaded_forbidden)}")

    out_data = data_root / "measurement_sensitivity"
    out_data.mkdir(exist_ok=True)
    out_tables = tables
    # The primary phenotype configuration is equal-feature Gower. The older
    # core14_sensitivity labels are domain-balanced and are retained only as a
    # separate sensitivity family; they must not be used as the comparator here.
    base_labels = {
        k: pd.read_parquet(data_root / f"equal_feature_k{k}_labels.parquet")["cluster"].to_numpy() - 1
        for k in cfg["clustering"]["candidate_k"]
    }

    availability = raw[features].notna()
    summary = pd.DataFrame({
        "feature": features,
        "n_observed": availability.sum(axis=0).to_numpy(),
        "missing_n": (~availability).sum(axis=0).to_numpy(),
        "missing_pct": (~availability).mean(axis=0).to_numpy(),
    })
    summary.to_csv(out_tables / "measurement_feature_missingness.csv", index=False)
    raw["observed_feature_n"] = availability.sum(axis=1).to_numpy()
    raw["observed_domain_n"] = sum(raw[fs].notna().any(axis=1) for fs in cfg["domains"].values()).to_numpy()
    raw["missing_pattern"] = availability.astype(int).astype(str).agg("".join, axis=1)
    raw[["observed_feature_n", "observed_domain_n", "missing_pattern"]].to_csv(
        out_tables / "measurement_patient_missingness.csv", index=False
    )
    pattern_counts = raw["missing_pattern"].value_counts().rename_axis("missing_pattern").reset_index(name="n")
    pattern_counts["pct"] = pattern_counts["n"] / len(raw)
    pattern_counts.head(50).to_csv(out_tables / "measurement_missing_patterns_top50.csv", index=False)

    cluster_rows = []
    for k, labels in base_labels.items():
        for cluster in sorted(np.unique(labels)):
            mask = labels == cluster
            cluster_rows.append({
                "analysis": "primary_equal_feature_missing_aware",
                "k": k,
                "cluster": cluster + 1,
                "n": int(mask.sum()),
                "observed_feature_n_mean": float(raw.loc[mask, "observed_feature_n"].mean()),
                "observed_feature_n_median": float(raw.loc[mask, "observed_feature_n"].median()),
                "observed_domain_n_mean": float(raw.loc[mask, "observed_domain_n"].mean()),
                "missing_pattern_count": int(raw.loc[mask, "missing_pattern"].nunique()),
            })
    pd.DataFrame(cluster_rows).to_csv(out_tables / "measurement_cluster_missingness_profiles.csv", index=False)

    limits = feature_limits(raw, features)
    indicator_d = indicator_balanced_distance(raw, features, limits)
    availability_d = availability_distance(raw, features)
    all_rows = []
    all_rows.extend(fit_grid(
        indicator_d, raw, features, "indicator_balanced", base_labels, out_data, out_tables,
        cfg["clustering"]["seed"] + 61000, cfg["clustering"]["candidate_k"],
        cfg["clustering"]["pam_restarts"], 200,
    ))
    del indicator_d
    all_rows.extend(fit_grid(
        availability_d, raw, features, "availability_only", base_labels, out_data, out_tables,
        cfg["clustering"]["seed"] + 62000, cfg["clustering"]["candidate_k"],
        cfg["clustering"]["pam_restarts"], 200,
    ))
    del availability_d
    pd.DataFrame(all_rows).to_csv(out_tables / "measurement_sensitivity_metrics.csv", index=False)

    manifest = {
        "status": "complete",
        "run_id": pointer["run_id"],
        "population": "core14 equal-feature eligible training rows",
        "n": len(raw),
        "features": features,
        "representations": [
            "indicator_balanced: equal within-variable value and availability components",
            "availability_only: Hamming distance on missingness indicators",
        ],
        "candidate_k": cfg["clustering"]["candidate_k"],
        "outcomes_loaded": False,
        "validation_test_used": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
    }
    (reports / "measurement_sensitivity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
