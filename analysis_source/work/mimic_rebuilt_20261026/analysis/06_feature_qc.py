r"""
06_feature_qc.py
特征表质控 —— 读取 03 特征表 + 01 队列，输出缺失率 / 分布 / 异常值 / 血压一致性 / 队列对账 / 结局核对。

前置：`python analysis/04_download_features.py` 已生成 `data/features.csv`。
运行：
    python analysis/06_feature_qc.py

输出：
    outputs/tables/feature_missingness.csv   每特征缺失率
    outputs/tables/feature_summary.csv       每特征分布摘要
    outputs/tables/feature_qc.csv            每特征超生理范围计数
    outputs/tables/cohort_reconciliation.csv 队列对账（01 vs 03）
    控制台打印完整质控摘要

关键提醒（方法学）：
    - 本脚本只「计数并报告」异常，不自动删除/改写任何值；是否剔除由后续冻结方案决定。
    - 生理范围是初步阈值，正式分析前需在 analysis_plan 中冻结。
    - 结局（hosp_mortality/deathtime/dischtime）来自 01 队列，特征表本身不含结局。
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT_TABLES = os.path.join(ROOT, "outputs", "tables")
OUT_REPORTS = os.path.join(ROOT, "outputs", "reports")

FEATURES_CSV = os.path.join(DATA, "features.csv")
COHORT_CSV = os.path.join(DATA, "sepsis3_cohort.csv")

# 生理合理范围（初步阈值；超范围 -> 计入异常，不自动剔除）
RANGES = {
    "heart_rate": (20, 300),
    "sbp": (40, 260),
    "map": (30, 200),
    "resp_rate": (4, 60),
    "temp_c": (30, 43),
    "spo2": (50, 100),
    "gcs_min": (3, 15),
    "lactate": (0.1, 30),
    "wbc": (0.1, 100),
    "platelet": (1, 1500),
    "hemoglobin": (3, 22),
    "creatinine": (0.1, 30),
    "bun": (1, 200),
    "bilirubin": (0.1, 50),
    "sodium": (100, 180),
    "potassium": (1.5, 9),
    "bicarbonate": (5, 45),
    "glucose": (20, 1200),
    "pao2": (30, 500),
    "fio2": (21, 100),
    "pao2_fio2_ratio": (20, 800),
}

BINARY_COLS = ["vent_pre_t0", "vaso_pre_t0", "crrt_pre_t0"]

# 连续特征（用于分布摘要）
CONT_COLS = [
    "age", "charlson_score", "heart_rate", "sbp", "map", "resp_rate",
    "temp_c", "spo2", "gcs_min", "lactate", "wbc", "platelet", "hemoglobin",
    "creatinine", "bun", "bilirubin", "sodium", "potassium", "bicarbonate",
    "glucose", "pao2", "fio2", "pao2_fio2_ratio",
]


def load_features():
    if not os.path.exists(FEATURES_CSV):
        raise SystemExit(f"[阻塞] 找不到 {FEATURES_CSV}。请先运行 `python analysis/04_download_features.py`。")
    df = pd.read_csv(FEATURES_CSV, low_memory=False)
    return df


def load_cohort():
    if not os.path.exists(COHORT_CSV):
        raise SystemExit(f"[阻塞] 找不到 {COHORT_CSV}。")
    df = pd.read_csv(COHORT_CSV, low_memory=False)
    return df


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def qc_missingness(df, features):
    """每特征缺失率。"""
    rows = []
    for col in features:
        n = len(df)
        n_miss = int(df[col].isna().sum())
        rows.append({"feature": col, "n": n, "n_missing": n_miss,
                     "pct_missing": round(100 * n_miss / n, 2)})
    out = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    print(out.to_string(index=False))
    return out


def qc_summary(df, cont_cols):
    """连续特征分布摘要。"""
    rows = []
    for col in cont_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "feature": col,
            "n_nonmissing": int(s.notna().sum()),
            "mean": round(s.mean(), 3),
            "sd": round(s.std(), 3),
            "min": round(s.min(), 3),
            "q1": round(s.quantile(0.25), 3),
            "median": round(s.median(), 3),
            "q3": round(s.quantile(0.75), 3),
            "max": round(s.max(), 3),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def qc_outliers(df, ranges):
    """超生理范围计数（仅报告，不剔除）。"""
    rows = []
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"feature": col, "plausible_range": f"[{lo}, {hi}]",
                     "n_out_of_range": n_out,
                     "pct_out_of_range": round(100 * n_out / max(len(s), 1), 2)})
    out = pd.DataFrame(rows).sort_values("n_out_of_range", ascending=False)
    print(out.to_string(index=False))
    return out


def qc_bp_consistency(df):
    """有创/无创血压混取中位数 -> SBP/MAP 生理矛盾检查。"""
    section("血压一致性（有创/无创混取中位数的矛盾）")
    d = df[["sbp", "map"]].copy()
    for c in ("sbp", "map"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    both = d.dropna()
    n_map_ge_sbp = int((both["map"] >= both["sbp"]).sum())
    n_map_gt_sbp = int((both["map"] > both["sbp"]).sum())
    print(f"sbp/map 均非空的记录数：{len(both)}")
    print(f"  map >= sbp（生理不可能）：{n_map_ge_sbp}  ({100*n_map_ge_sbp/max(len(both),1):.2f}%)")
    print(f"  map >  sbp：{n_map_gt_sbp}")
    print(f"  sbp 与 map 的 Pearson r = {both['sbp'].corr(both['map']):.3f}  (正常应明显 >0)")
    if n_map_ge_sbp > 0:
        print("  => 提示：混取有创+无创中位数会造成 SBP 与 MAP 来自不同来源、生理不自洽。")
        print("     建议后续冻结方案采用「有创优先」，或分开有创/无创分别聚合。")
    return n_map_ge_sbp


def qc_binary(df):
    section("二元变量取值检查（应只含 0/1）")
    for col in BINARY_COLS:
        if col not in df.columns:
            print(f"  {col}: 列不存在")
            continue
        vc = df[col].value_counts(dropna=False).sort_index()
        print(f"  {col}: {dict(vc)}")


def qc_reconciliation(features, cohort):
    """01 队列 vs 03 特征表对账（去重差异 + 结局核对）。"""
    section("队列对账（01 未去重 vs 03 已去重）")
    n_stays = len(cohort)
    n_patients_cohort = cohort["subject_id"].nunique()
    multi = int((cohort.groupby("subject_id").size() > 1).sum())
    print(f"01 队列（sepsis3 stay，未去重）：{n_stays} 行")
    print(f"  其中唯一患者：{n_patients_cohort}")
    print(f"  有 >1 次脓毒症 stay 的患者数：{multi}  （这 {multi} 例在 03 会被压成 1 行）")

    n_feat = len(features)
    n_patients_feat = features["subject_id"].nunique()
    print(f"\n03 特征表（去重后，每患者首次事件）：{n_feat} 行")
    print(f"  唯一患者数：{n_patients_feat}")
    print(f"  subject_id 是否唯一（应=True）：{features['subject_id'].is_unique}")

    # 覆盖：特征表 stay_id 是否都在队列里
    in_cohort = int(features["stay_id"].isin(cohort["stay_id"]).sum())
    print(f"\n特征表 stay_id 能匹配到 01 队列：{in_cohort} / {n_feat}")

    # 结局合并核对
    keep = ["stay_id", "sofa_time", "age", "gender", "hosp_mortality", "deathtime", "dischtime"]
    m = features.merge(cohort[keep], on="stay_id", how="left", suffixes=("", "_cohort"))

    # t0 与 sofa_time 一致性
    t0_match = 0
    if "t0" in m.columns and "sofa_time" in m.columns:
        t0_match = int((m["t0"].astype(str) == m["sofa_time"].astype(str)).sum())
    print(f"\nt0 == cohort.sofa_time 一致：{t0_match} / {len(m)}")

    # age / gender 一致性
    if "age_cohort" in m.columns:
        age_match = int((pd.to_numeric(m["age"], errors="coerce")
                         == pd.to_numeric(m["age_cohort"], errors="coerce")).sum())
        print(f"age 与队列一致：{age_match} / {len(m)}")
    if "gender_cohort" in m.columns:
        g_match = int((m["gender"].astype(str) == m["gender_cohort"].astype(str)).sum())
        print(f"gender 与队列一致：{g_match} / {len(m)}")

    # 结局
    n_event = int(m["hosp_mortality"].fillna(0).sum())
    n_known = int(m["hosp_mortality"].notna().sum())
    print(f"\n结局（hosp_mortality）：")
    print(f"  已知结局：{n_known} / {len(m)}；缺失结局：{len(m) - n_known}")
    print(f"  院内死亡事件数：{n_event}")
    print(f"  院内死亡率（按已知结局）：{100*n_event/max(n_known,1):.2f}%")

    # 生存时间非负检查
    t0 = pd.to_datetime(m["t0"], errors="coerce")
    dt = pd.to_datetime(m["deathtime"], errors="coerce")
    dis = pd.to_datetime(m["dischtime"], errors="coerce")
    ev = m["hosp_mortality"].fillna(0).astype(int) == 1
    end = dt.where(ev, dis)  # 事件者用 deathtime，否则 dischtime
    fu_days = (end - t0).dt.total_seconds() / 86400.0
    n_neg = int((fu_days < 0).sum())
    print(f"  follow_up < 0（t0 晚于结局时间，异常）：{n_neg}")
    print(f"  中位随访：{fu_days.median():.2f} 天  (IQR {fu_days.quantile(0.25):.2f}–{fu_days.quantile(0.75):.2f})")

    return pd.DataFrame({
        "metric": ["cohort_rows", "cohort_unique_patients", "multi_stay_patients",
                   "features_rows", "features_unique_patients",
                   "stay_id_in_cohort", "t0_matches_sofa_time",
                   "n_known_outcome", "n_event", "mortality_pct",
                   "followup_negative"],
        "value": [n_stays, n_patients_cohort, multi, n_feat, n_patients_feat,
                  in_cohort, t0_match, n_known, n_event,
                  round(100 * n_event / max(n_known, 1), 2), n_neg],
    })


def main():
    os.makedirs(OUT_TABLES, exist_ok=True)
    os.makedirs(OUT_REPORTS, exist_ok=True)

    features = load_features()
    cohort = load_cohort()

    print("特征表列：", list(features.columns))

    section("1. 基本形状")
    print(f"特征表：{features.shape[0]} 行 × {features.shape[1]} 列")
    print(f"stay_id 重复数：{int(features['stay_id'].duplicated().sum())}")
    print(f"subject_id 重复数：{int(features['subject_id'].duplicated().sum())}")
    print(f"hadm_id 重复数：{int(features['hadm_id'].duplicated().sum())}")

    section("2. 特征缺失率（%），按缺失率降序")
    miss = qc_missingness(features, [c for c in features.columns
                                     if c not in ("subject_id", "stay_id", "hadm_id", "t0", "gender")])
    miss.to_csv(os.path.join(OUT_TABLES, "feature_missingness.csv"), index=False)

    section("3. 连续特征分布摘要")
    summ = qc_summary(features, CONT_COLS)
    summ.to_csv(os.path.join(OUT_TABLES, "feature_summary.csv"), index=False)

    section("4. 超生理范围计数（仅报告，不剔除）")
    qc = qc_outliers(features, RANGES)
    qc.to_csv(os.path.join(OUT_TABLES, "feature_qc.csv"), index=False)

    qc_bp_consistency(features)
    qc_binary(features)

    section("5. 队列对账与结局核对")
    recon = qc_reconciliation(features, cohort)
    recon.to_csv(os.path.join(OUT_TABLES, "cohort_reconciliation.csv"), index=False)

    section("质控结论要点")
    print("- 检查上面 5 节，确认：特征表应「一行一患者、无重复」；")
    print("- 特征表行数 < 队列行数属正常（01 未去重、03 已去重，差异=多次住院患者数）；")
    print("- 若有 follow_up<0 或 t0 与 sofa_time 不一致，需回查 SQL 后再进入切分；")
    print("- 缺失率、异常值、血压矛盾记入结果，供 D 阶段冻结方案时决策。")

    print(f"\n已写入：{OUT_TABLES}")


if __name__ == "__main__":
    main()
