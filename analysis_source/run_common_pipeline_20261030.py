"""Reproducible sequential entry point; preserves original analysis directories."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STAGES=['extract_common_core_20261030.py','compact_common_chart_20261030.py','verify_compact_chart_20261030.py','build_common_identified_20261030.py','common_features_20261030.py','verify_common_core_20261030.py','fit_common_models_20261030.py','report_common_1to3_20261030.py']
if __name__=='__main__':
    for script in STAGES:
        print('STAGE',script,flush=True)
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/script)],cwd=ROOT,check=True)
