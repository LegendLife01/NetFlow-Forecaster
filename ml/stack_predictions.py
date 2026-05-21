"""Validation-selected stacking for benchmark attempts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_utils import FEATURES, weighted_mae


def load_pair(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    val_p = run_dir / "results" / "val_predictions.csv"
    val_a = run_dir / "results" / "val_actuals.csv"
    test_p = run_dir / "results" / "predictions.csv"
    if not (val_p.exists() and val_a.exists() and test_p.exists()):
        return None
    return (
        pd.read_csv(val_p)[FEATURES].to_numpy(dtype=float),
        pd.read_csv(val_a)[FEATURES].to_numpy(dtype=float),
        pd.read_csv(test_p)[FEATURES].to_numpy(dtype=float),
    )


def stack_attempts(run_dirs: list[Path], output_dir: Path) -> Path | None:
    pairs = [(run_dir, load_pair(run_dir)) for run_dir in run_dirs]
    pairs = [(run_dir, pair) for run_dir, pair in pairs if pair is not None]
    if len(pairs) < 2:
        return None
    scored = []
    for run_dir, (val_pred, val_actual, _) in pairs:
        scored.append((weighted_mae(val_actual, val_pred), run_dir, val_pred, val_actual))
    scored.sort(key=lambda item: item[0])
    _, run_a, pred_a, actual = scored[0]
    _, run_b, pred_b, _ = scored[1]
    best_w = 0.5
    best_score = float("inf")
    for weight in np.linspace(0.0, 1.0, 21):
        score = weighted_mae(actual, weight * pred_a + (1.0 - weight) * pred_b)
        if score < best_score:
            best_score = score
            best_w = float(weight)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_a, output_dir, dirs_exist_ok=True)
    test_a = pd.read_csv(run_a / "results" / "predictions.csv")[FEATURES].to_numpy(dtype=float)
    test_b = pd.read_csv(run_b / "results" / "predictions.csv")[FEATURES].to_numpy(dtype=float)
    final = best_w * test_a + (1.0 - best_w) * test_b
    pd.DataFrame(final, columns=FEATURES).to_csv(output_dir / "results" / "predictions.csv", index=False)
    (output_dir / "json" / "stacking.json").write_text(
        f'{{"run_a":"{run_a}","run_b":"{run_b}","weight_a":{best_w:.3f}}}',
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_model.py")), "--run-dir", str(output_dir)], check=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    result = stack_attempts([Path(item) for item in args.runs], Path(args.output_dir))
    print(result or "no_stack")


if __name__ == "__main__":
    main()
