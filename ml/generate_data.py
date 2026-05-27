"""Generate synthetic network telemetry for the ML pipeline.

The generated series mimics a spine-leaf fabric under business-hour load:
traffic follows daily/weekly seasonality, latency rises with utilization,
packet loss appears under congestion, and occasional incidents create
correlated spikes across all metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]


def generate_traffic_data(
    hours: int = 720,
    output: str | Path = "ml/telemetry_generated.csv",
    seed: int | None = None,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=hours, freq="h")
    hour = np.arange(hours) % 24
    day = (np.arange(hours) // 24) % 7

    business_window = (hour >= 7) & (hour <= 20)
    base_traffic = np.where(
        business_window,
        45 + 38 * np.sin((hour - 7) * np.pi / 13),
        12 + 7 * np.sin(hour * np.pi / 12),
    )
    weekly_factor = np.where(day >= 5, 0.70, 1.0)
    trend = np.linspace(0.0, 7.0, hours)
    seasonal_amplitude = 1.0 + 0.10 * np.cos(np.linspace(0, 4 * np.pi, hours))
    traffic = base_traffic * weekly_factor * seasonal_amplitude + trend

    regime_state = np.empty(hours, dtype=object)
    regime_multipliers = np.ones(hours, dtype=float)
    regime_latency_shift = np.zeros(hours, dtype=float)
    regime_loss_shift = np.zeros(hours, dtype=float)
    time_index = 0
    while time_index < hours:
        remaining = hours - time_index
        if remaining < 8:
            duration = remaining
        else:
            duration = int(rng.integers(8, min(40, remaining) + 1))
        regime = rng.choice(
            ["normal", "busy", "degraded", "incident"],
            p=[0.62, 0.20, 0.13, 0.05],
        )
        for offset in range(duration):
            idx = time_index + offset
            regime_state[idx] = regime
            if regime == "normal":
                regime_multipliers[idx] = 1.0
                regime_latency_shift[idx] = 0.0
                regime_loss_shift[idx] = 0.0
            elif regime == "busy":
                regime_multipliers[idx] = 1.18
                regime_latency_shift[idx] = 1.6
                regime_loss_shift[idx] = 0.05
            elif regime == "degraded":
                regime_multipliers[idx] = 0.90
                regime_latency_shift[idx] = 3.8
                regime_loss_shift[idx] = 0.24
            else:
                regime_multipliers[idx] = 1.35
                regime_latency_shift[idx] = 6.8
                regime_loss_shift[idx] = 0.48
        if regime in {"busy", "incident"}:
            spike_start = time_index + int(rng.integers(0, duration))
            spike_width = min(hours - spike_start, int(rng.integers(4, max(5, duration))))
            spike_ramp = np.linspace(0.0, 1.0, spike_width)
            traffic[spike_start : spike_start + spike_width] += spike_ramp * rng.uniform(20, 90)
        time_index += duration

    noise = rng.normal(0, 3.4, hours)
    traffic = np.clip(traffic * regime_multipliers + noise + rng.uniform(-3.0, 3.0, hours), 0.0, None)
    load = np.clip(traffic / 140.0, 0.0, 1.9)

    packet_loss = np.clip(
        0.03
        + load * 0.18
        + regime_loss_shift
        + rng.normal(0, 0.05, hours)
        + rng.uniform(0.0, 0.12, hours),
        0.0,
        100.0,
    )

    latency = np.clip(
        1.8
        + load * 5.0
        + packet_loss * 8.2
        + np.maximum(load - 0.57, 0.0) ** 2 * 18.5
        + regime_latency_shift
        + rng.normal(0, 0.7, hours),
        0.5,
        None,
    )

    traffic = np.clip(traffic - packet_loss * 0.7 + rng.normal(0, 1.8, hours), 0.0, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "traffic_mbps": np.round(traffic, 3),
            "latency_ms": np.round(latency, 3),
            "packet_loss_pct": np.round(packet_loss, 3),
            "source": "synthetic",
        }
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} rows -> {output}")
    print(df[FEATURES].describe().round(3).to_string())
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=720, help="Number of hourly rows to generate.")
    parser.add_argument("--output", default="ml/telemetry_generated.csv", help="CSV path to write.")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible random seed.")
    parser.add_argument("--start", default="2024-01-01", help="Start timestamp accepted by pandas.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_traffic_data(args.hours, args.output, args.seed, args.start)
