"""Append-only experience memory for telemetry benchmark attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from telemetry_profile import TelemetryProfile


# Resolve paths relative to this source file so they work regardless of
# which directory the script is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH: Path = _REPO_ROOT / "runs" / ".experience" / "memory.jsonl"
POLICY_PATH: Path = _REPO_ROOT / "runs" / ".experience" / "policy.json"


@dataclass
class ExperienceRecord:
    timestamp: str
    data_fingerprint: str
    profile: dict
    candidate_id: str
    candidate_args: list[str]
    quality_pct: float
    gates_passed: dict
    per_feature: list[dict]
    traffic_spike_f1: float
    mae_improvement_pct: float
    run_dir: str
    status: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(profile: TelemetryProfile) -> str:
    payload = (
        profile.usable_rows,
        profile.volatility,
        round(profile.traffic_spike_rate, 3),
        round(profile.traffic_std, 2),
    )
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def append_record(record: ExperienceRecord, memory_path: Path = MEMORY_PATH) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def load_records(memory_path: Path = MEMORY_PATH) -> list[ExperienceRecord]:
    if not memory_path.exists():
        return []
    records = []
    for line in memory_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(ExperienceRecord(**json.loads(line)))
    return records


def similar_records(fp: str, limit: int = 50, memory_path: Path = MEMORY_PATH) -> list[ExperienceRecord]:
    records = load_records(memory_path)
    exact = [record for record in records if record.data_fingerprint == fp]
    if exact:
        return exact[-limit:]
    return records[-limit:]


def update_policy(records: list[ExperienceRecord], policy_path: Path = POLICY_PATH) -> dict:
    policy = {"version": 1, "global_candidate_scores": {}, "candidate_attempts": {}, "by_fingerprint": {}, "by_volatility": {}}
    grouped: dict[str, list[ExperienceRecord]] = {}
    for record in records:
        grouped.setdefault(record.candidate_id, []).append(record)
    for candidate, items in grouped.items():
        policy["global_candidate_scores"][candidate] = sum(item.quality_pct for item in items) / len(items)
        policy["candidate_attempts"][candidate] = len(items)
    by_fp: dict[str, list[ExperienceRecord]] = {}
    by_vol: dict[str, list[ExperienceRecord]] = {}
    for record in records:
        by_fp.setdefault(record.data_fingerprint, []).append(record)
        volatility = str(record.profile.get("volatility", "unknown"))
        by_vol.setdefault(volatility, []).append(record)
    for fp, items in by_fp.items():
        best = max(items, key=lambda item: item.quality_pct)
        policy["by_fingerprint"][fp] = {
            "best_candidate": best.candidate_id,
            "mean_quality": sum(item.quality_pct for item in items) / len(items),
            "count": len(items),
        }
    for volatility, items in by_vol.items():
        best = max(items, key=lambda item: item.quality_pct)
        policy["by_volatility"][volatility] = {
            "best_candidate": best.candidate_id,
            "mean_quality": sum(item.quality_pct for item in items) / len(items),
            "count": len(items),
        }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return policy


def load_policy(policy_path: Path = POLICY_PATH) -> dict:
    if not policy_path.exists():
        return {}
    return json.loads(policy_path.read_text(encoding="utf-8"))


def update_policy_incremental(memory_path: Path = MEMORY_PATH, policy_path: Path = POLICY_PATH) -> dict:
    return update_policy(load_records(memory_path), policy_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-policy", action="store_true")
    args = parser.parse_args()
    if args.rebuild_policy:
        policy = update_policy_incremental()
        print(json.dumps(policy, indent=2))
    else:
        print(json.dumps({"records": len(load_records()), "policy": str(POLICY_PATH)}, indent=2))


if __name__ == "__main__":
    main()
