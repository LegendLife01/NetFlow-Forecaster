"""Profile telemetry CSVs before automated benchmarking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from metrics_utils import FEATURES, weighted_mae
from train_model import load_dataset


@dataclass
class TelemetryProfile:
    rows: int
    usable_rows: int
    train_rows_est: int
    traffic_std: float
    latency_std: float
    loss_std: float
    traffic_spike_rate: float
    persistence_mae: dict[str, float]
    volatility: str
    recommended_sequence_length: int
    recommended_lookback: int
    recommended_spike_quantile: float
    recommended_trainer: str
    recommended_epochs: int


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def profile_telemetry(path: Path, train_ratio: float = 0.70, test_ratio: float = 0.82) -> TelemetryProfile:
    df = load_dataset(path)
    usable_rows = len(df)
    if usable_rows < 120:
        raise ValueError("Need at least 120 rows after feature engineering")
    values = df[FEATURES].to_numpy(dtype=float)
    train_rows = max(10, min(usable_rows - 2, int(usable_rows * train_ratio)))
    val_start = train_rows
    val_end = max(val_start + 2, min(usable_rows, int(usable_rows * test_ratio)))
    train = values[:train_rows]
    val = values[val_start:val_end]
    persistence = np.empty_like(val)
    persistence[0] = values[val_start - 1]
    persistence[1:] = val[:-1]
    persistence_mae = {feature: float(np.mean(np.abs(val[:, idx] - persistence[:, idx]))) for idx, feature in enumerate(FEATURES)}
    traffic_threshold = float(np.quantile(train[:, 0], 0.90))
    spike_rate = float(np.mean(train[:, 0] > traffic_threshold))
    traffic_mean = max(float(np.mean(train[:, 0])), 1e-9)
    cv = float(np.std(train[:, 0], ddof=0) / traffic_mean)
    if cv > 0.6:
        volatility = "high"
    elif cv > 0.25:
        volatility = "medium"
    else:
        volatility = "low"
    seq_len = clamp_int(int(usable_rows * 0.05), 24, 96)
    lookback = min(24, max(4, seq_len // 2))
    spike_quantile = 0.85 if spike_rate < 0.05 else 0.90
    persist_weighted = weighted_mae(val, persistence)
    if usable_rows < 400 or persist_weighted < 0.5:
        trainer = "gb_only"
    elif cv > 0.6:
        trainer = "hybrid_aggressive"
    else:
        trainer = "hybrid"
    source = " ".join(str(value) for value in df.get("source", []))[:500].lower() if "source" in df.columns else ""
    traffic_min = float(np.min(values[:, 0]))
    traffic_span = float(np.max(values[:, 0]) - traffic_min)
    kaggle_like = "kaggle" in path.name.lower() or "kaggle" in source or (8.0 <= traffic_min <= 12.5 and 100.0 <= traffic_span <= 260.0)
    if kaggle_like:
        trainer = "gb_only"
        spike_quantile = 0.88
    epochs = 40 if usable_rows < 1000 else 60
    return TelemetryProfile(
        rows=int(len(df)),
        usable_rows=int(usable_rows),
        train_rows_est=int(train_rows),
        traffic_std=float(np.std(train[:, 0], ddof=0)),
        latency_std=float(np.std(train[:, 1], ddof=0)),
        loss_std=float(np.std(train[:, 2], ddof=0)),
        traffic_spike_rate=spike_rate,
        persistence_mae=persistence_mae,
        volatility=volatility,
        recommended_sequence_length=seq_len,
        recommended_lookback=lookback,
        recommended_spike_quantile=spike_quantile,
        recommended_trainer=trainer,
        recommended_epochs=epochs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    profile = profile_telemetry(Path(args.data))
    payload = asdict(profile)
    text = json.dumps(payload, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
