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
    output: str | Path = "telemetry.csv",
    seed: int | None = None,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=hours, freq="h")
    hour = np.arange(hours) % 24
    day = (np.arange(hours) // 24) % 7

    business_window = (hour >= 7) & (hour <= 20)
    daily_wave = np.where(
        business_window,
        55 + 35 * np.sin((hour - 7) * np.pi / 13),
        14 + 6 * np.sin(hour * np.pi / 12),
    )
    weekly_factor = np.where(day >= 5, 0.55, 1.0)
    trend = np.linspace(0, 8, hours)
    noise = rng.normal(0, 4.5, hours)
    traffic = np.clip((daily_wave + trend) * weekly_factor + noise, 0, None)

    incident_count = max(4, hours // 36)
    incident_idx = rng.choice(hours, size=min(incident_count, hours), replace=False)
    traffic[incident_idx] += rng.uniform(55, 125, len(incident_idx))

    load = traffic / 150.0
    latency = 2.0 + load * 8.5 + rng.normal(0, 0.35, hours)
    latency[incident_idx] += rng.uniform(5, 22, len(incident_idx))
    latency = np.clip(latency, 0.5, None)

    packet_loss = np.clip(rng.normal(0.08, 0.04, hours), 0, None)
    high_load = traffic > 90
    packet_loss[high_load] += rng.uniform(0.25, 2.4, high_load.sum())
    packet_loss[incident_idx] += rng.uniform(0.9, 7.5, len(incident_idx))
    packet_loss = np.clip(packet_loss, 0.0, 100.0)

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
    parser.add_argument("--output", default="telemetry.csv", help="CSV path to write.")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible random seed.")
    parser.add_argument("--start", default="2024-01-01", help="Start timestamp accepted by pandas.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_traffic_data(args.hours, args.output, args.seed, args.start)
