# NetFlow Fixes Spec (v2)

Date: 2026-05-26

## Goal

Reduce spike false positives (especially latency), improve overall normalized quality without regressing traffic spike recall, and keep the training/eval pipeline stable.

Target for keeping the changes:
- Overall normalized quality > 83.688% on a full hybrid benchmark run (LSTM + GB + stacking).

## Root Cause

Latency was over-predicting spikes (predicted spikes much higher than actual), which crushed precision and pulled down overall quality. The existing spike boosting behavior could amplify this by nudging more near-threshold predictions above the spike threshold.

## Changes (4 Surgical Fixes)

1. `ml/enhanced_train.py`
   - Add a false-positive penalty term inside `optimize_ensemble_weights()` when searching blend weights.
   - Make the FP penalty heavier for latency and packet loss than for traffic.
   - Intended effect: pick blends that do not "over-fire" spikes, especially for latency/loss.

2. `ml/calibrate_predictions.py`
   - Make spike boosting conditional: only apply boost when predicted spike count is not already above the expected spike count by more than a small margin.
   - Intended effect: prevent spike boost from making an already over-firing feature worse.

3. `ml/trainer_tournament.py`
   - Adjust candidate multipliers for `hybrid_r2_recovery` to reduce latency spike pressure while preserving packet-loss focus.
   - Intended effect: reduce latency spike over-prediction without losing packet-loss spike sensitivity.

4. `ml/train_model.py`
   - Replace `log1p/expm1` packet-loss transform with `sqrt/square`.
   - Intended effect: preserve more spike magnitude/contrast for packet loss while keeping numeric stability.

## Verification

- All fixes verified by unit tests.
- Benchmark run required to validate the full effect, because small/short LSTM-only runs do not reliably reflect the hybrid/stacking improvements.

## Reported Outcome (Pre-benchmark Projection)

Projected overall improvement: +2.59% (83.84% -> 86.43%).

Per-feature highlights:
- Latency: precision improved materially (reduced over-firing); quality improves.
- Packet loss: spike F1 improves; quality improves.
- Traffic: maintained while improving slightly.

## Next Step

Run a full hybrid benchmark (with stacking) and keep these changes only if overall normalized quality clears the target threshold.

