r"""
04_download_features.py
运行 03_feature_extraction.sql，把特征宽表拉到本地 CSV/parquet。

前置：gcloud 已认证（同 02）。运行：
  python analysis/04_download_features.py

说明：禁用 BigQuery Storage API（走 REST），并自动打印各特征原始缺失率，
      直接对应论文“报告原始缺失率”的要求。
"""
import os
import sys

import pandas as pd
from google.cloud import bigquery

HERE = os.path.dirname(os.path.abspath(__file__))
SQL_FILE = os.path.join(HERE, "03_feature_extraction.sql")
OUT_DIR = os.path.join(HERE, "..", "data")
PROJECT = os.environ.get("GCP_PROJECT")


def main() -> int:
    with open(SQL_FILE, encoding="utf-8") as f:
        sql = f.read()

    client = bigquery.Client(project=PROJECT) if PROJECT else bigquery.Client()
    print("运行 03_feature_extraction.sql（特征提取，可能耗时几分钟）…")
    df = client.query(sql).to_dataframe(create_bqstorage_client=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "features.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n完成：{len(df)} 行 × {len(df.columns)} 列")
    print(f"已保存 CSV：{csv_path}")

    try:
        df.to_parquet(os.path.join(OUT_DIR, "features.parquet"), index=False)
        print("已保存 parquet：features.parquet")
    except Exception as e:  # noqa: BLE001
        print(f"（跳过 parquet：{e}）")

    print("\n各特征原始缺失率（%），只列有缺失的：")
    miss = df.isna().mean().mul(100).round(1).sort_values(ascending=False)
    for col, rate in miss.items():
        if rate > 0:
            print(f"  {col:<16} {rate:>5}%")

    print("\n数值特征摘要：")
    print(df.describe().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
