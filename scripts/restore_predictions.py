"""Copies the saved test and validation predictions from the repository (results/predictions) to data/interim/predictions, where the analysis, figure and audit scripts read them.

The predictions are the record of the single-use test runs: the test scripts refuse to run a second time, so only the four main horizons can be regenerated (run_test_evaluation.py --export,
which reproduces the first run exactly); all other predictions (72/168 h, quantile and conformal inputs, day-ahead models, LightGBM, GRU) come from this folder.
Usage: py -I scripts/restore_predictions.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "results" / "predictions", ROOT / "data" / "interim" / "predictions"
DST.mkdir(parents=True, exist_ok=True)
files = sorted(SRC.glob("*.npz"))
for f in files:
    shutil.copy2(f, DST / f.name)
print(f"copied {len(files)} prediction files from {SRC.relative_to(ROOT)} to {DST.relative_to(ROOT)}")
