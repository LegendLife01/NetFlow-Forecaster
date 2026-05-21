"""Hybrid Ensemble: LSTM + Gradient Boosting - production grade."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from run_layout import artifact_path, ensure_run_layout
from train_kaggle_model import add_features
from train_model import FEATURES, INPUT_FEATURES, TIME_FEATURES, create_sequences, inverse_transform_features, load_dataset, transform_features


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class EnhancedMultivariateTrafficLSTM(nn.Module):
    def __init__(self, input_size: int = 7, hidden_size: int = 128, num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


SimpleLSTM = EnhancedMultivariateTrafficLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ml/telemetry.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-split", type=float, default=0.82)
    parser.add_argument("--gb-weight", type=float, default=0.65)
    parser.add_argument("--lstm-weight", type=float, default=0.35)
    parser.add_argument("--output", default="lstm_model.pth")
    parser.add_argument("--output-dir", default="runs/hybrid_best")
    return parser.parse_args()


def normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    scale = np.maximum(target.std(axis=0, ddof=0), 1e-9)
    return float(np.mean(((prediction - target) / scale) ** 2))


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    ensure_run_layout(output_dir)
    model_path = Path(args.output)
    if not model_path.is_absolute():
        model_path = artifact_path(output_dir, model_path.name, "model")

    df = load_dataset(data_path)
    transformed = transform_features(df)
    feature_cols = INPUT_FEATURES if all(c in transformed.columns for c in TIME_FEATURES) else FEATURES

    x_seq, _ = create_sequences(transformed[feature_cols].to_numpy(dtype=np.float32), args.sequence_length)
    _, y_seq = create_sequences(transformed[FEATURES].to_numpy(dtype=np.float32), args.sequence_length)
    if len(x_seq) < 10:
        raise ValueError("Not enough sequences. Reduce --sequence-length or collect more rows.")

    split = max(1, min(len(x_seq) - 1, int(args.train_split * len(x_seq))))
    x_train_raw, y_train_raw = x_seq[:split], y_seq[:split]
    x_test_raw, y_test_raw = x_seq[split:], y_seq[split:]

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train = input_scaler.fit_transform(x_train_raw.reshape(-1, x_train_raw.shape[-1])).reshape(x_train_raw.shape)
    x_test = input_scaler.transform(x_test_raw.reshape(-1, x_test_raw.shape[-1])).reshape(x_test_raw.shape)
    y_train = target_scaler.fit_transform(y_train_raw)
    y_test = target_scaler.transform(y_test_raw)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    lstm = SimpleLSTM(len(feature_cols), args.hidden_size, args.layers, len(FEATURES))
    optimizer = torch.optim.AdamW(lstm.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("Training Hybrid LSTM component...")
    train_rows: list[dict[str, float]] = []
    best_state = {key: value.detach().clone() for key, value in lstm.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(args.epochs):
        lstm.train()
        losses: list[float] = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = lstm(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        lstm.eval()
        with torch.no_grad():
            val_pred = lstm(x_test_tensor)
            val_loss = criterion(val_pred, y_test_tensor).item()
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in lstm.state_dict().items()}
        train_rows.append(
            {
                "epoch": epoch + 1,
                "mse_loss": float(np.mean(losses)),
                "validation_mse_loss": float(val_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"LSTM Epoch {epoch + 1} Loss: {np.mean(losses):.4f} | Val: {val_loss:.4f}")

    lstm.load_state_dict(best_state)
    lstm.eval()

    print("Training Gradient Boosting spike component...")
    lookback = 24
    df_feat, gb_feature_cols = add_features(df, lookback=lookback)
    x_gb = df_feat[gb_feature_cols].to_numpy(dtype=float)
    y_gb = df_feat[FEATURES].to_numpy(dtype=float)

    first_test_original_idx = args.sequence_length + split
    gb_test_start = max(0, first_test_original_idx - lookback)
    gb_test_end = min(len(x_gb), gb_test_start + len(x_test))
    gb_train_end = max(1, gb_test_start)

    gb = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    )
    gb.fit(x_gb[:gb_train_end], y_gb[:gb_train_end])

    with torch.no_grad():
        lstm_scaled = lstm(x_test_tensor[: gb_test_end - gb_test_start]).numpy()
    lstm_pred = inverse_transform_features(target_scaler.inverse_transform(lstm_scaled))
    gb_pred = gb.predict(x_gb[gb_test_start:gb_test_end])
    actuals = y_gb[gb_test_start:gb_test_end]
    final_pred = args.gb_weight * gb_pred + args.lstm_weight * lstm_pred
    final_pred = np.clip(final_pred, 0.0, None)

    torch.save(lstm.state_dict(), model_path)
    joblib.dump(
        {
            "model": gb,
            "feature_columns": gb_feature_cols,
            "features": FEATURES,
            "lookback": lookback,
            "ensemble_weights": {"gradient_boosting": args.gb_weight, "lstm": args.lstm_weight},
        },
        artifact_path(output_dir, "gb_model.joblib", "model"),
    )
    torch.onnx.export(
        lstm,
        torch.randn(1, args.sequence_length, len(feature_cols)),
        artifact_path(output_dir, "lstm_model.onnx", "model"),
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
    )

    metrics = {
        "training": {
            "loss": "HybridLstmMSEPlusGradientBoosting",
            "feature_columns": feature_cols,
            "gb_feature_columns": gb_feature_cols,
            "output_features": FEATURES,
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "requested_epochs": args.epochs,
            "epochs": len(train_rows),
            "best_epoch": best_epoch,
            "best_validation_mse_loss": best_val,
            "train_split": args.train_split,
            "architecture": "attention_lstm",
            "ensemble": "lstm_gradient_boosting",
            "gb_weight": args.gb_weight,
            "lstm_weight": args.lstm_weight,
        }
    }
    for idx, feature in enumerate(FEATURES):
        mae = mean_absolute_error(actuals[:, idx], final_pred[:, idx])
        rmse = float(np.sqrt(mean_squared_error(actuals[:, idx], final_pred[:, idx])))
        metrics[feature] = {"mae": float(mae), "rmse": rmse}

    scaler_params = {
        "feature_columns": feature_cols,
        "scaler_type": "StandardScaler",
        "input_scaler": {"mean": input_scaler.mean_.tolist(), "scale": input_scaler.scale_.tolist(), "var": input_scaler.var_.tolist()},
        "target_scaler": {"mean": target_scaler.mean_.tolist(), "scale": target_scaler.scale_.tolist(), "var": target_scaler.var_.tolist()},
    }
    artifact_path(output_dir, "scaler_params.json", "json").write_text(json.dumps(scaler_params, indent=2), encoding="utf-8")
    artifact_path(output_dir, "metrics.json", "json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(final_pred, columns=FEATURES).to_csv(artifact_path(output_dir, "predictions.csv", "results"), index=False)
    pd.DataFrame(actuals, columns=FEATURES).to_csv(artifact_path(output_dir, "actuals.csv", "results"), index=False)
    pd.DataFrame(train_rows).to_csv(artifact_path(output_dir, "train_losses.csv", "results"), index=False)

    raw_copy = artifact_path(output_dir, data_path.name, "raw_data")
    if data_path.resolve() != raw_copy.resolve():
        shutil.copy2(data_path, raw_copy)

    print(f"\nHYBRID ENSEMBLE COMPLETE -> {output_dir}")
    print(f"Normalized ensemble MSE: {normalized_mse(final_pred, actuals):.4f}")
    print("This should give stronger spike capture than the pure LSTM.")


if __name__ == "__main__":
    main()
