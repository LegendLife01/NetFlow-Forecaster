import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.experience_store import ExperienceRecord, append_record, fingerprint, load_records, update_policy, utc_now
from ml.telemetry_profile import TelemetryProfile


def sample_profile():
    return TelemetryProfile(
        rows=200,
        usable_rows=200,
        train_rows_est=140,
        traffic_std=12.34,
        latency_std=1.0,
        loss_std=0.1,
        traffic_spike_rate=0.1,
        persistence_mae={"traffic_mbps": 1.0, "latency_ms": 1.0, "packet_loss_pct": 0.1},
        volatility="medium",
        recommended_sequence_length=24,
        recommended_lookback=12,
        recommended_spike_quantile=0.9,
        recommended_trainer="hybrid",
        recommended_epochs=40,
    )


def test_experience_memory_append_and_policy(tmp_path):
    profile = sample_profile()
    record = ExperienceRecord(
        timestamp=utc_now(),
        data_fingerprint=fingerprint(profile),
        profile=profile.__dict__,
        candidate_id="hybrid_default",
        candidate_args=[],
        quality_pct=91.0,
        gates_passed={"quality_ge_90": True},
        per_feature=[],
        traffic_spike_f1=0.7,
        mae_improvement_pct=20.0,
        run_dir="runs/x",
        status="SUCCESS",
    )
    memory = tmp_path / "memory.jsonl"
    policy_path = tmp_path / "policy.json"
    append_record(record, memory)
    records = load_records(memory)
    policy = update_policy(records, policy_path)
    assert len(records) == 1
    assert policy["global_candidate_scores"]["hybrid_default"] == 91.0
    assert policy_path.exists()
