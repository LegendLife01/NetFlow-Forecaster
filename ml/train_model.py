"""Train a multivariate LSTM for network telemetry prediction."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]
DEFAULT_MODEL = "lstm_model.pth"


class MultivariateTrafficLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        dropout = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, max(32, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_size // 2), output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def create_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for idx in range(len(data) - seq_len):
        x.append(data[idx : idx + seq_len])
        y.append(data[idx + seq_len])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run generate_data.py or collect_telemetry.py first.")
    df = pd.read_csv(path)
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    df = df.dropna(subset=FEATURES).copy()
    if len(df) < 30:
        raise ValueError(f"{path} has only {len(df)} usable rows; collect or generate more telemetry.")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="telemetry.csv", help="Input telemetry CSV.")
    parser.add_argument("--sequence-length", type=int, default=48, help="Lookback window size.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--hidden-size", type=int, default=256, help="LSTM hidden units.")
    parser.add_argument("--layers", type=int, default=2, help="LSTM layer count.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Chronological train split.")
    parser.add_argument("--seed", type=int, default=7, help="Torch and NumPy seed.")
    parser.add_argument("--output", default=DEFAULT_MODEL, help="Model weights path.")
    parser.add_argument("--output-dir", default=".", help="Directory for all training artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.output)
    if not model_path.is_absolute():
        model_path = output_dir / model_path

    df = load_dataset(data_path)
    raw = df[FEATURES].to_numpy(dtype=np.float32)
    print(f"Loaded {len(raw)} rows from {data_path}")
    for idx, feature in enumerate(FEATURES):
        print(f"  {feature:<16} {raw[:, idx].min():8.3f} to {raw[:, idx].max():8.3f}")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw)
    x, y = create_sequences(scaled, args.sequence_length)
    if len(x) < 10:
        raise ValueError("Not enough sequences. Reduce --sequence-length or collect more rows.")

    split = max(1, min(len(x) - 1, int(len(x) * args.train_split)))
    x_train = torch.tensor(x[:split])
    y_train = torch.tensor(y[:split])
    x_test = torch.tensor(x[split:])
    y_test = torch.tensor(y[split:])
    print(f"Training samples: {len(x_train)} | Test samples: {len(x_test)}")

    model = MultivariateTrafficLSTM(len(FEATURES), args.hidden_size, args.layers, len(FEATURES))
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    train_losses: list[float] = []

    print("Training multivariate LSTM...")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(float(loss.item()))
        train_losses.append(float(loss.item()))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch + 1:3d}/{args.epochs} | loss={loss.item():.6f} | lr={lr:.5f}")

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_test).numpy()
        actual_scaled = y_test.numpy()

    predictions = scaler.inverse_transform(pred_scaled)
    actuals = scaler.inverse_transform(actual_scaled)

    metrics = {}
    print("\nTest performance:")
    print(f"{'feature':<18} {'MAE':>10} {'RMSE':>10}")
    for idx, feature in enumerate(FEATURES):
        mae = mean_absolute_error(actuals[:, idx], predictions[:, idx])
        rmse = float(np.sqrt(mean_squared_error(actuals[:, idx], predictions[:, idx])))
        metrics[feature] = {"mae": float(mae), "rmse": rmse}
        print(f"{feature:<18} {mae:10.3f} {rmse:10.3f}")

    torch.save(model.state_dict(), model_path)
    scaler_params = {
        "features": FEATURES,
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "data_range": scaler.data_range_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
    }
    (output_dir / "scaler_params.json").write_text(json.dumps(scaler_params, indent=2), encoding="utf-8")
    pd.DataFrame(predictions, columns=FEATURES).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(actuals, columns=FEATURES).to_csv(output_dir / "actuals.csv", index=False)
    pd.DataFrame({"epoch": np.arange(1, len(train_losses) + 1), "mse_loss": train_losses}).to_csv(
        output_dir / "train_losses.csv",
        index=False,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if data_path.resolve() != (output_dir / data_path.name).resolve():
        shutil.copy2(data_path, output_dir / data_path.name)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved human-readable data artifacts -> {output_dir}")


if __name__ == "__main__":
    main()
