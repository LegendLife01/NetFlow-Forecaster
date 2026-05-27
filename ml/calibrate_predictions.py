"""Validation-only prediction calibration for benchmark attempts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_utils import FEATURES, feature_quality


@dataclass
class CalibrationParams:
    scale: list[float]
    bias: list[float]
    persistence_weight: list[float]
    thresholds: dict[str, float]


def persistence_baseline(actuals: np.ndarray) -> np.ndarray:
    base = np.empty_like(actuals)
    base[0] = actuals[0]
    base[1:] = actuals[:-1]
    return base


def calibrate(val_actuals: np.ndarray, val_predictions: np.ndarray, thresholds: dict[str, float]) -> CalibrationParams:
    persistence = persistence_baseline(val_actuals)
    scales: list[float] = []
    biases: list[float] = []
    weights: list[float] = []
    for idx, feature in enumerate(FEATURES):
        std = max(float(np.std(val_actuals[:, idx], ddof=0)), 1e-9)
        base_mae = float(np.mean(np.abs(val_actuals[:, idx] - persistence[:, idx])))
        actual_spikes = val_actuals[:, idx] > thresholds[feature]
        best = (-1e9, 1.0, 0.0, 0.0)
        for scale in np.linspace(0.85, 1.15, 7):
            for bias in np.linspace(-0.1 * std, 0.1 * std, 5):
                for weight in np.linspace(0.0, 0.5, 6):
                    pred = scale * val_predictions[:, idx] + bias + weight * persistence[:, idx]
                    pred = pred / (1.0 + weight)
                    mae = float(np.mean(np.abs(val_actuals[:, idx] - pred)))
                    pred_spikes = pred > thresholds[feature]
                    tp = float(np.sum(actual_spikes & pred_spikes))
                    fp = float(np.sum(~actual_spikes & pred_spikes))
                    fn = float(np.sum(actual_spikes & ~pred_spikes))
                    precision = tp / max(tp + fp, 1.0)
                    recall = tp / max(tp + fn, 1.0)
                    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
                    score = feature_quality(
                        mae,
                        base_mae,
                        0.0,
                        f1,
                        int(actual_spikes.sum()),
                        int(pred_spikes.sum()),
                    )
                    if feature == "traffic_mbps":
                        if int(pred_spikes.sum()) < 5:
                            score -= 10.0
                        actual_count = int(actual_spikes.sum())
                        predicted_count = int(pred_spikes.sum())
                        if actual_count > 0 and predicted_count > 1.3 * actual_count:
                            score -= 15.0 * (predicted_count / max(actual_count, 1) - 1.3)
                    if score > best[0]:
                        best = (score, float(scale), float(bias), float(weight))
        _, scale, bias, weight = best
        scales.append(scale)
        biases.append(bias)
        weights.append(weight)
    return CalibrationParams(scales, biases, weights, thresholds)


def apply_calibration(predictions: np.ndarray, params: CalibrationParams, persistence: np.ndarray) -> np.ndarray:
    calibrated = predictions.copy()
    for idx, feature in enumerate(FEATURES):
        weight = params.persistence_weight[idx]
        calibrated[:, idx] = params.scale[idx] * calibrated[:, idx] + params.bias[idx] + weight * persistence[:, idx]
        calibrated[:, idx] = calibrated[:, idx] / (1.0 + weight)
    return np.clip(calibrated, 0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actuals", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    actuals = pd.read_csv(args.actuals)[FEATURES].to_numpy(dtype=float)
    preds = pd.read_csv(args.predictions)[FEATURES].to_numpy(dtype=float)
    thresholds = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8"))["spike_thresholds"]
    params = calibrate(actuals, preds, thresholds)
    Path(args.output).write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
