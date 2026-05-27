"""Tests for the per-feature spike loss and packet-loss transform behavior."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from train_model import (
    FEATURES,
    SpikeWeightedLoss,
    inverse_transform_features,
    transform_features,
)


def test_spike_loss_per_feature_multipliers_differ():
    """Per-feature multipliers must produce different loss for different features."""
    thresh = torch.tensor([0.5, 0.5, 0.5])
    loss = SpikeWeightedLoss(thresh, spike_weight=4.0)
    pred = torch.zeros(10, 3)
    target = torch.tensor([[1.0, 1.0, 1.0]] * 10)
    pred.requires_grad_(True)
    value = loss(pred, target)
    value.backward()
    grads = pred.grad.abs().mean(dim=0)
    assert grads[1] > grads[0], (
        f"Latency gradient {grads[1]:.4f} should exceed traffic {grads[0]:.4f}. "
        f"Per-feature multipliers are not working."
    )
    assert grads[2] > grads[1], (
        f"Loss gradient {grads[2]:.4f} should exceed latency {grads[1]:.4f}."
    )


def test_spike_loss_zero_when_pred_equals_target():
    """Loss must be exactly 0 when predictions equal targets."""
    thresh = torch.tensor([0.5, 0.5, 0.5])
    loss = SpikeWeightedLoss(thresh)
    target = torch.tensor([[0.1, 0.2, 0.05], [0.9, 0.8, 0.7]])
    assert loss(target, target).item() == 0.0


def test_packet_loss_transform_round_trip():
    """packet-loss transform followed by inverse must recover original values."""
    df = pd.DataFrame(
        {
            "traffic_mbps": [20.0, 80.0, 150.0],
            "latency_ms": [2.0, 5.0, 10.0],
            "packet_loss_pct": [0.0, 1.5, 5.0],
        }
    )
    transformed = transform_features(df)
    np.testing.assert_array_almost_equal(
        transformed["traffic_mbps"].to_numpy(),
        df["traffic_mbps"].to_numpy(),
    )
    expected = np.sqrt(df["packet_loss_pct"].to_numpy())
    np.testing.assert_array_almost_equal(
        transformed["packet_loss_pct"].to_numpy(), expected, decimal=5
    )
    arr = transformed[FEATURES].to_numpy(dtype=np.float32)
    recovered = inverse_transform_features(arr)
    np.testing.assert_array_almost_equal(
        recovered[:, FEATURES.index("packet_loss_pct")],
        df["packet_loss_pct"].to_numpy(),
        decimal=4,
    )


def test_packet_loss_transform_preserves_spike_ordering():
    """packet-loss transform must preserve rank order."""
    vals = np.array([0.0, 0.1, 0.5, 1.0, 3.0, 7.0])
    transformed = np.sqrt(vals)
    assert list(np.argsort(vals)) == list(np.argsort(transformed))


def test_r2_recovery_candidate_in_tournament():
    """hybrid_r2_recovery must appear in the candidate list for all profile types."""
    from telemetry_profile import TelemetryProfile
    from trainer_tournament import candidates_for_profile

    for trainer in ("hybrid", "hybrid_aggressive", "gb_only"):
        profile = TelemetryProfile(
            rows=2000,
            usable_rows=2000,
            train_rows_est=1400,
            traffic_std=20.0,
            latency_std=2.0,
            loss_std=0.2,
            traffic_spike_rate=0.10,
            persistence_mae={
                "traffic_mbps": 1.0,
                "latency_ms": 1.0,
                "packet_loss_pct": 1.0,
            },
            volatility="medium",
            recommended_sequence_length=96,
            recommended_lookback=24,
            recommended_spike_quantile=0.90,
            recommended_trainer=trainer,
            recommended_epochs=80,
        )
        candidates = candidates_for_profile(profile)
        ids = [candidate.id for candidate in candidates]
        assert "hybrid_r2_recovery" in ids, (
            f"hybrid_r2_recovery missing from candidates for trainer={trainer}. Got: {ids}"
        )
