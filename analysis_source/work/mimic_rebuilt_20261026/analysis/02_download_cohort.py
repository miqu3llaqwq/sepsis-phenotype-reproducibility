r"""
02_download_cohort.py
从 BigQuery 拉取 Sepsis-3 队列（01_sepsis3_cohort.sql 的结果）到本地 CSV/parquet。

前置：首次运行前完成一次 Google 认证（二选一）：
  A) gcloud CLI（推荐，复用你现在的 Google 账号）：
       1) winget install Google.CloudSDK   （装完重开终端）
       2) gcloud auth application-default login
  B) 服务账号（如果有 key.json）：
       set GOOGLE_APPLICATION_CREDENTIALS=C:\路径\key.json

运行：
  python analysis/02_download_cohort.py

说明：
  - 如果报 "project" 相关错误，说明 ADC 无法推断计费项目，
    设置环境变量后重跑：  set GCP_PROJECT=你的项目ID
  - 依赖：google-cloud-bigquery、db-dtypes、pandas（均已装）
"""
import os
import sys

import pandas as pd
from google.cloud import bigquery

HERE = os.path.dirname(os.path.abspath(__file__))
SQL_FILE = os.path.join(HERE, "01_sepsis3_cohort.sql")
OUT_DIR = os.path.join(HERE, "..", "data")
PROJECT = os.environ.get("GCP_PROJECT")  # 可选；ADC 推断不出项目时设置


def main() -> int:
    with open(SQL_FILE, encoding="utf-8") as f:
        sql = f.read()

    client = bigquery.Client(project=PROJECT) if PROJECT else bigquery.Client()
    print("开始查询 BigQuery（MIMIC-IV v3.1 sepsis3 队列）…")
    # 禁用 BigQuery Storage API（默认走 gRPC，容易被网络/代理拦截，报 503 UNAVAILABLE）。
    # 改用 REST 下载结果：对本队列（约 2 万行）足够快，且走的是查询本身已成功的 HTTPS 通道。
    df = client.query(sql).to_dataframe(create_bqstorage_client=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "sepsis3_cohort.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n完成：{len(df)} 行 × {len(df.columns)} 列")
    print(f"已保存 CSV：{csv_path}")

    # parquet 可选（更快、保留类型），缺 pyarrow 就跳过
    try:
        parquet_path = os.path.join(OUT_DIR, "sepsis3_cohort.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"已保存 parquet：{parquet_path}")
    except Exception as e:  # noqa: BLE001
        print(f"（跳过 parquet：{e}）")

    print("\n列与类型：")
    print(df.dtypes.to_string())
    print("\n结局变量分布：")
    for col in ["hosp_mortality", "mort_28d", "mort_90d"]:
        if col in df.columns:
            print(f"  {col}: {df[col].value_counts(dropna=False).to_dict()}")
    print("\n前 3 行预览：")
    print(df.head(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
