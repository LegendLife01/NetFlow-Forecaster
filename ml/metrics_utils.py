"""Shared scoring helpers for telemetry forecasting."""

from __future__ import annotations

import numpy as np


FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]
FEATURE_WEIGHTS = {"traffic_mbps": 0.50, "latency_ms": 0.25, "packet_loss_pct": 0.25}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def spike_thresholds_from_train(y_train: np.ndarray, multiplier: float = 1.0) -> dict[str, float]:
    return {
        feature: float(y_train[:, idx].mean() + multiplier * y_train[:, idx].std(ddof=0))
        for idx, feature in enumerate(FEATURES)
    }


def spike_thresholds_from_quantile(y_train: np.ndarray, quantile: float = 0.90) -> dict[str, float]:
    return {
        feature: float(np.quantile(y_train[:, idx], quantile))
        for idx, feature in enumerate(FEATURES)
    }


def compute_spike_scores(actuals: np.ndarray, predictions: np.ndarray, thresholds: dict[str, float]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for idx, feature in enumerate(FEATURES):
        threshold = float(thresholds[feature])
        actual_spikes = actuals[:, idx] > threshold
        predicted_spikes = predictions[:, idx] > threshold
        true_positive = int(np.logical_and(actual_spikes, predicted_spikes).sum())
        false_positive = int(np.logical_and(~actual_spikes, predicted_spikes).sum())
        false_negative = int(np.logical_and(actual_spikes, ~predicted_spikes).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        scores[feature] = {
            "threshold": threshold,
            "actual_spikes": int(actual_spikes.sum()),
            "predicted_spikes": int(predicted_spikes.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return scores


def weighted_mae(actuals: np.ndarray, predictions: np.ndarray, weights: dict[str, float] = FEATURE_WEIGHTS) -> float:
    values = []
    for idx, feature in enumerate(FEATURES):
        values.append(weights[feature] * float(np.mean(np.abs(actuals[:, idx] - predictions[:, idx]))))
    return float(sum(values))


def feature_quality(
    model_mae: float,
    baseline_mae: float,
    r2: float,
    spike_f1: float,
    actual_spikes: int,
    predicted_spikes: int = 0,
) -> float:
    """Score a single feature prediction 0–100.

    Scoring breakdown:
      - 55%  Log-ratio MAE improvement: log2(baseline_mae / model_mae) clamped [0,1].
              This reaches 1.0 when the model is 2x better than baseline (not 100x),
              making the score sensitive to realistic improvements rather than requiring
              extreme MAE reduction to score well.
      - 45%  R² clamped to [0,1]. Negative R² contributes 0 (no penalty added to spike
              score, but the error_score is dragged down). This directly penalises runs
              where the model is worse than predicting the mean.
      - error_score feeds 60% of the total; spike_score feeds 40%.

    When actual_spikes == 0 the spike component is 1.0 provided the model is not
    hallucinating spikes (predicted_spikes < 5). This prevents the old bug where a
    model that predicts nothing scored 75% because there happened to be no spikes.
    """
    import math

    ratio = max(baseline_mae, 1e-9) / max(model_mae, 1e-9)
    mae_score = clamp(math.log2(ratio))
    r2_score_val = clamp(r2)
    error_score = 0.55 * mae_score + 0.45 * r2_score_val

    if actual_spikes == 0:
        spike_score = 1.0 if predicted_spikes < 5 else clamp(1.0 - predicted_spikes / 20.0)
    else:
        spike_score = clamp(spike_f1)

    return 100.0 * (0.60 * error_score + 0.40 * spike_score)


def quality_score_v2(per_feature: dict[str, dict[str, float]], weights: dict[str, float] = FEATURE_WEIGHTS) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    score = 0.0
    for feature in FEATURES:
        row = per_feature[feature]
        score += weights[feature] * feature_quality(
            float(row["model_mae"]),
            float(row["baseline_mae"]),
            float(row.get("r2", 0.0)),
            float(row.get("spike_f1", 0.0)),
            int(row.get("actual_spikes", 0)),
            int(row.get("predicted_spikes", 0)),
        )
    return float(score / total_weight)


def summarize_gates(
    overall_quality: float,
    avg_mae_improvement: float,
    per_feature_rows: list[dict[str, float]],
    traffic_spike_f1: float,
    traffic_predicted_spikes: int,
    model_quality: float,
    persistence_quality: float,
) -> dict[str, bool]:
    gates = {
        "quality_ge_90": overall_quality >= 90.0,
        "mae_improvement_ge_15": avg_mae_improvement >= 15.0,
        "beats_persistence_each_feature_mae": all(float(row["mae_improvement_pct"]) > 0.0 for row in per_feature_rows),
        "traffic_spike_f1_ge_0_50": traffic_spike_f1 >= 0.50,
        "traffic_predicted_spikes_ge_5": traffic_predicted_spikes >= 5,
        "model_quality_gt_persistence": model_quality > persistence_quality,
    }
    r2_values = [float(row.get("model_r2", 0.0)) for row in per_feature_rows]
    gates["all_features_r2_ge_neg_0_1"] = all(v >= -0.1 for v in r2_values)
    return gates


def diagnose_quality_shortfall(summary: dict) -> dict[str, str | float]:
    per_feature = summary.get("per_feature", [])
    if not per_feature:
        return {"bottleneck_feature": "unknown", "bottleneck_reason": "missing_per_feature_metrics", "suggested_candidate": "hybrid_default"}
    worst = min(per_feature, key=lambda row: float(row.get("quality_pct", 0.0)))
    feature = str(worst.get("metric", "unknown"))
    reason = "low_quality"
    if float(worst.get("mae_improvement_pct", 0.0)) <= 0.0:
        reason = "does_not_beat_persistence_mae"
    elif float(worst.get("model_r2", 0.0)) < 0.1:
        reason = "low_r2"
    gates = summary.get("gates_passed", {})
    if not gates.get("traffic_spike_f1_ge_0_50", True):
        reason = "low_traffic_spike_f1"
        feature = "traffic_mbps"
    suggestion = "gb_spike" if reason in {"does_not_beat_persistence_mae", "low_traffic_spike_f1"} else "hybrid_aggressive"
    return {
        "bottleneck_feature": feature,
        "bottleneck_reason": reason,
        "suggested_candidate": suggestion,
        "quality": float(summary.get("overall", {}).get("normalized_quality_pct", 0.0)),
    }
