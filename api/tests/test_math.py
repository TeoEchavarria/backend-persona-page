import pytest

from app.config import Settings
from app.db import SimilarityStats
from app.search import confidence, confidence_band, reciprocal_rank_fusion, z_score


# ── Search ────────────────────────────────────────────────────────────


def test_rrf_rewards_items_ranked_well_in_both_lists():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert max(scores, key=scores.get) == "b"
    assert scores["a"] > scores["c"]


def test_confidence_bands():
    settings = Settings(z_high=3.0, z_medium=2.0, gap_high=0.05)
    assert confidence_band(3.5, 0.0, settings) == "high"
    assert confidence_band(2.5, 0.10, settings) == "high"
    assert confidence_band(2.5, 0.01, settings) == "medium"
    assert confidence_band(1.0, 0.50, settings) == "far"


def test_confidence_from_similarity_stats():
    result = confidence(SimilarityStats(mean=0.80, std=0.02, top=[0.88, 0.84]), Settings(z_high=3.0, z_medium=2.0))
    assert result.z == pytest.approx(4.0)
    assert result.gap == pytest.approx(0.04)
    assert result.band == "high"
    assert confidence(SimilarityStats(0, 0, []), Settings()).band == "far"
    assert z_score(1.0, 1.0, 0.0) == 0.0
