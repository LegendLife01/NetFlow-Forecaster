"""Create a prediction dashboard and spike summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]
LABELS = ["Traffic (Mbps)", "Latency (ms)", "Packet Loss (%)"]
UNITS = ["Mbps", "ms", "%"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="telemetry.csv", help="Input telemetry CSV.")
    parser.add_argument("--predictions", default="predictions.csv", help="Predictions CSV artifact.")
    parser.add_argument("--actuals", default="actuals.csv", help="Actuals CSV artifact.")
    parser.add_argument("--losses", default="train_losses.csv", help="Training loss CSV artifact.")
    parser.add_argument("--sensitivity", type=float, default=1.5, help="Std-dev multiplier for spike thresholds.")
    parser.add_argument("--output", default="traffic_prediction_dashboard.png", help="PNG dashboard path.")
    parser.add_argument("--output-dir", default=".", help="Directory containing artifacts and dashboard output.")
    return parser.parse_args()


def style_axis(ax, title: str) -> None:
    ax.set_facecolor("#161a22")
    ax.set_title(title, color="#e6edf3", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors="#9aa4b2", labelsize=8)
    ax.xaxis.label.set_color("#9aa4b2")
    ax.yaxis.label.set_color("#9aa4b2")
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#586170")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#263040", alpha=0.45, linewidth=0.6)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def in_output_dir(path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else output_dir / path

    predictions = pd.read_csv(in_output_dir(args.predictions))[FEATURES].to_numpy()
    actuals = pd.read_csv(in_output_dir(args.actuals))[FEATURES].to_numpy()
    losses_df = pd.read_csv(in_output_dir(args.losses))
    losses = losses_df["mse_loss"].to_numpy()
    df = pd.read_csv(args.data).dropna(subset=FEATURES)

    thresholds = {
        feature: float(df[feature].mean() + args.sensitivity * df[feature].std(ddof=0))
        for feature in FEATURES
    }
    spikes = {
        feature: np.where(predictions[:, idx] > thresholds[feature])[0]
        for idx, feature in enumerate(FEATURES)
    }

    print("Spike thresholds:")
    for feature in FEATURES:
        print(f"  {feature:<16} {thresholds[feature]:8.3f} | predicted spikes={len(spikes[feature])}")

    bg = "#0d1117"
    panel = "#161a22"
    colors = ["#58a6ff", "#3fb950", "#f2cc60"]
    alert = "#ff6b6b"
    text = "#e6edf3"
    muted = "#9aa4b2"
    x = np.arange(len(actuals))

    fig = plt.figure(figsize=(18, 14), facecolor=bg)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)
    fig.suptitle("AI-Driven Network Telemetry Forecast", color=text, fontsize=18, fontweight="bold", y=0.985)

    for idx, (feature, label, color) in enumerate(zip(FEATURES, LABELS, colors)):
        ax = fig.add_subplot(gs[0, idx])
        ax.plot(x, actuals[:, idx], color=color, linewidth=1.2, label="Actual")
        ax.plot(x, predictions[:, idx], color="#f0f6fc", linewidth=1.0, linestyle="--", alpha=0.75, label="Predicted")
        ax.axhline(thresholds[feature], color=alert, linestyle=":", linewidth=1.1, label="Spike threshold")
        spike_idx = spikes[feature]
        if len(spike_idx):
            ax.scatter(spike_idx, predictions[spike_idx, idx], color=alert, s=24, zorder=5)
        ax.set_xlabel("Test sample")
        ax.set_ylabel(label)
        ax.legend(facecolor=panel, labelcolor=text, edgecolor="#586170", fontsize=7)
        style_axis(ax, f"Actual vs Predicted - {label}")

    metrics = {}
    for idx, (feature, label, color) in enumerate(zip(FEATURES, LABELS, colors)):
        ax = fig.add_subplot(gs[1, idx])
        errors = predictions[:, idx] - actuals[:, idx]
        metrics[feature] = {
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "spikes": int(len(spikes[feature])),
            "threshold": thresholds[feature],
        }
        ax.hist(errors, bins=30, color=color, edgecolor=panel, alpha=0.9)
        ax.axvline(0, color="#f0f6fc", linestyle="--", linewidth=1.1)
        ax.set_xlabel(f"Prediction error ({UNITS[idx]})")
        ax.set_ylabel("Frequency")
        style_axis(ax, f"Error Distribution - {label}")

    ax_loss = fig.add_subplot(gs[2, 0])
    ax_loss.plot(losses, color="#f2cc60", linewidth=1.5)
    ax_loss.fill_between(np.arange(len(losses)), losses, color="#f2cc60", alpha=0.18)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE")
    style_axis(ax_loss, "Training Loss")

    ax_corr = fig.add_subplot(gs[2, 1])
    corr = np.corrcoef(df[FEATURES].tail(len(actuals)).to_numpy().T)
    im = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    names = ["Traffic", "Latency", "Loss"]
    ax_corr.set_xticks(range(3), names, color=muted)
    ax_corr.set_yticks(range(3), names, color=muted)
    for row in range(3):
        for col in range(3):
            ax_corr.text(col, row, f"{corr[row, col]:.2f}", ha="center", va="center", color=text, fontsize=9)
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    style_axis(ax_corr, "Telemetry Correlation")

    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.set_facecolor(panel)
    ax_stats.axis("off")
    rows = [
        ("Rows", f"{len(df):,}"),
        ("Test samples", f"{len(actuals):,}"),
        ("Sensitivity", f"{args.sensitivity:.1f}x std"),
    ]
    for idx, feature in enumerate(FEATURES):
        rows.append((f"{feature} MAE", f"{metrics[feature]['mae']:.3f} {UNITS[idx]}"))
    for feature in FEATURES:
        rows.append((f"{feature} spikes", str(metrics[feature]["spikes"])))
    for idx, (label, value) in enumerate(rows):
        ypos = 0.94 - idx * 0.09
        ax_stats.text(0.05, ypos, f"{label}:", color=muted, fontsize=9, transform=ax_stats.transAxes)
        ax_stats.text(0.55, ypos, value, color=text, fontsize=9, fontweight="bold", transform=ax_stats.transAxes)
    ax_stats.set_title("Model Summary", color=text, fontsize=11, fontweight="bold", pad=8)

    ax_timeline = fig.add_subplot(gs[3, :])
    ax_timeline.plot(x, actuals[:, 0], color=colors[0], linewidth=1.2, alpha=0.75, label="Actual traffic")
    ax_timeline.plot(x, predictions[:, 0], color="#f0f6fc", linewidth=1.0, linestyle="--", alpha=0.7, label="Predicted traffic")
    ax_timeline.axhline(thresholds["traffic_mbps"], color=alert, linestyle=":", linewidth=1.1)
    traffic_spikes = spikes["traffic_mbps"]
    if len(traffic_spikes):
        ax_timeline.scatter(traffic_spikes, predictions[traffic_spikes, 0], color=alert, s=36, zorder=5)
    ax_latency = ax_timeline.twinx()
    ax_latency.plot(x, actuals[:, 1], color=colors[1], linewidth=1.0, alpha=0.45, label="Actual latency")
    ax_latency.set_ylabel("Latency (ms)", color=muted)
    ax_latency.tick_params(colors=muted, labelsize=8)
    ax_latency.spines["right"].set_color("#586170")
    ax_latency.spines["top"].set_visible(False)
    ax_timeline.set_xlabel("Test sample")
    ax_timeline.set_ylabel("Traffic (Mbps)")
    ax_timeline.legend(loc="upper left", facecolor=panel, labelcolor=text, edgecolor="#586170", fontsize=8)
    ax_latency.legend(loc="upper right", facecolor=panel, labelcolor=text, edgecolor="#586170", fontsize=8)
    style_axis(ax_timeline, "Traffic Forecast with Latency Overlay")

    dashboard_path = in_output_dir(args.output)
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor=bg)
    (output_dir / "spike_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Dashboard saved -> {dashboard_path}")
    print(f"Spike summary saved -> {output_dir / 'spike_summary.json'}")


if __name__ == "__main__":
    main()
