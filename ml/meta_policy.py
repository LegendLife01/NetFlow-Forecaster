"""Rank benchmark candidates using persisted experience."""

from __future__ import annotations

import math

from experience_store import fingerprint
from telemetry_profile import TelemetryProfile
from trainer_tournament import Candidate, candidates_for_profile


def rank_candidates(profile: TelemetryProfile, policy: dict, attempts: list) -> list[Candidate]:
    candidates = candidates_for_profile(profile)
    tried = {attempt["candidate"]["id"] for attempt in attempts if "candidate" in attempt}
    if not policy:
        return [candidate for candidate in candidates if candidate.id not in tried]
    fp = fingerprint(profile)
    global_scores = policy.get("global_candidate_scores", {})
    fp_policy = policy.get("by_fingerprint", {}).get(fp, {})
    vol_policy = policy.get("by_volatility", {}).get(profile.volatility, {})
    total = max(1, sum(int(row.get("count", 1)) for row in policy.get("by_fingerprint", {}).values()))
    base_order = {candidate.id: idx for idx, candidate in enumerate(candidates)}

    ranked = []
    for candidate in candidates:
        if candidate.id in tried:
            continue
        historical = float(global_scores.get(candidate.id, 50.0))
        if fp_policy.get("best_candidate") == candidate.id:
            historical = max(historical, float(fp_policy.get("mean_quality", historical)))
        if vol_policy.get("best_candidate") == candidate.id:
            historical = max(historical, float(vol_policy.get("mean_quality", historical)))
        global_score = float(global_scores.get(candidate.id, 50.0))
        rule_priority = 100.0 - base_order[candidate.id] * 10.0
        count = max(1, sum(1 for item in global_scores if item == candidate.id))
        exploration = math.sqrt(2.0 * math.log(total + 1.0) / count) * 5.0
        score = 0.45 * historical + 0.25 * global_score + 0.20 * rule_priority + 0.10 * exploration
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked]
