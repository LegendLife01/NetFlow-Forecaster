import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.meta_policy import rank_candidates
from tests.test_experience_store import sample_profile


def test_meta_policy_prioritizes_known_good_candidate():
    profile = sample_profile()
    policy = {
        "global_candidate_scores": {"gb_spike": 95.0, "hybrid_default": 50.0},
        "by_fingerprint": {},
        "by_volatility": {"medium": {"best_candidate": "gb_spike", "mean_quality": 94.0, "count": 3}},
    }
    ranked = rank_candidates(profile, policy, [])
    assert ranked[0].id == "gb_spike"
