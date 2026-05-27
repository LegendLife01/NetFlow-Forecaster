"""Validation-selected stacking for benchmark attempts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_utils import FEATURES, weighted_mae


def load_pair(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    val_p = run_dir / "results" / "val_predictions.csv"
    val_a = run_dir / "results" / "val_actuals.csv"
    test_p = run_dir / "results" / "predictions.csv"
    test_a = run_dir / "results" / "actuals.csv"
    if not (val_p.exists() and val_a.exists() and test_p.exists() and test_a.exists()):
        return None
    return (
        pd.read_csv(val_p)[FEATURES].to_numpy(dtype=float),
        pd.read_csv(val_a)[FEATURES].to_numpy(dtype=float),
        pd.read_csv(test_p)[FEATURES].to_numpy(dtype=float),
        pd.read_csv(test_a)[FEATURES].to_numpy(dtype=float),
    )


def align_tail(*arrays: np.ndarray) -> list[np.ndarray]:
    """Align forecast arrays by their most recent samples."""
    min_len = min(len(array) for array in arrays)
    return [array[-min_len:] for array in arrays]


def stack_attempts(run_dirs: list[Path], output_dir: Path) -> Path | None:
    pairs = [(run_dir, load_pair(run_dir)) for run_dir in run_dirs]
    pairs = [(run_dir, pair) for run_dir, pair in pairs if pair is not None]
    if len(pairs) < 2:
        return None

    # load_pair returns (val_pred, val_actual, test_pred, test_actual)
    val_preds = [pair[0] for _, pair in pairs]
    val_actuals_list = [pair[1] for _, pair in pairs]
    test_preds = [pair[2] for _, pair in pairs]

    # Different candidates can yield slightly different validation/test lengths
    # depending on their split logic. Align everything on the most recent tail.
    val_min = min(*(len(arr) for arr in val_preds), *(len(arr) for arr in val_actuals_list))
    val_preds = [arr[-val_min:] for arr in val_preds]
    val_actuals_list = [arr[-val_min:] for arr in val_actuals_list]
    # Use a shared aligned actuals array for weight search (they should match closely).
    val_actuals = val_actuals_list[0]
    test_min = min(len(arr) for arr in test_preds)
    test_preds = [arr[-test_min:] for arr in test_preds]
    n_runs = len(pairs)
    n_features = val_actuals.shape[1]
    best_weights = np.zeros((n_features, n_runs), dtype=float)

    for feat_idx in range(n_features):
        best_score = float("inf")
        best_w = np.zeros(n_runs, dtype=float)
        for _ in range(3000):
            weights = np.random.dirichlet(np.ones(n_runs))
            candidate = sum(weights[i] * val_preds[i][:, feat_idx] for i in range(n_runs))
            score = float(np.mean(np.abs(candidate - val_actuals[:, feat_idx])))
            if score < best_score:
                best_score = score
                best_w = weights
        best_weights[feat_idx] = best_w

    final = np.zeros_like(test_preds[0])
    for feat_idx in range(n_features):
        for run_idx in range(n_runs):
            final[:, feat_idx] += best_weights[feat_idx, run_idx] * test_preds[run_idx][:, feat_idx]

    run_scores = [weighted_mae(val_actuals_list[i], val_preds[i]) for i in range(n_runs)]
    best_run_idx = int(np.argmin(run_scores))
    best_run_dir = pairs[best_run_idx][0]

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(best_run_dir, output_dir, dirs_exist_ok=True)
    pd.DataFrame(final, columns=FEATURES).to_csv(output_dir / "results" / "predictions.csv", index=False)
    actual_a = pd.read_csv(best_run_dir / "results" / "actuals.csv")[FEATURES].to_numpy(dtype=float)
    if len(actual_a) != len(final):
        actual_a = actual_a[-len(final) :]
    pd.DataFrame(actual_a, columns=FEATURES).to_csv(output_dir / "results" / "actuals.csv", index=False)

    stacking_meta = {
        "n_runs_stacked": n_runs,
        "best_run": str(best_run_dir),
        "weights": {
            feature: best_weights[idx].tolist() for idx, feature in enumerate(FEATURES)
        },
        "val_scores": {str(pairs[i][0].name): float(run_scores[i]) for i in range(n_runs)},
    }
    (output_dir / "json" / "stacking.json").write_text(
        json.dumps(stacking_meta, indent=2), encoding="utf-8"
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
