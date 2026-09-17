import numpy as np
import pytest

from app.config import Settings
from app.db import SimilarityStats
from app.embeddings import normalize
from app.search import confidence, confidence_band, reciprocal_rank_fusion, z_score
from app.session import effective_query, update_context


def unit(*values: float) -> np.ndarray:
    return normalize(np.array(values, dtype=float))


# ── Search ────────────────────────────────────────────────────────────


def test_rrf_rewards_items_ranked_well_in_both_lists():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert max(scores, key=scores.get) == "b"
    assert scores["a"] > scores["c"]


def test_confidence_bands():
    settings = Settings(margin_high=0.08, margin_medium=0.05, gap_high=0.04)
    assert confidence_band(0.09, 0.0, settings) == "high"
    assert confidence_band(0.06, 0.05, settings) == "high"
    assert confidence_band(0.06, 0.01, settings) == "medium"
    assert confidence_band(0.03, 0.50, settings) == "far"


def test_confidence_from_similarity_stats():
    settings = Settings(margin_high=0.07, margin_medium=0.05)
    result = confidence(SimilarityStats(mean=0.80, std=0.02, top=[0.88, 0.84]), settings)
    assert result.margin == pytest.approx(0.08)
    assert result.z == pytest.approx(4.0)
    assert result.gap == pytest.approx(0.04)
    assert result.band == "high"
    assert confidence(SimilarityStats(0, 0, []), Settings()).band == "far"
    assert z_score(1.0, 1.0, 0.0) == 0.0


# ── Rocchio ───────────────────────────────────────────────────────────


def test_effective_query_blends_and_stays_unit_length():
    q, c = unit(1, 0), unit(0, 1)
    assert effective_query(q, None) is q
    blended = effective_query(q, c, alpha=0.7)
    assert np.linalg.norm(blended) == pytest.approx(1.0)
    assert blended[0] > blended[1] > 0


def test_update_context_drifts_towards_what_is_read():
    first = update_context(None, unit(1, 0))
    assert np.allclose(first, unit(1, 0))
    context = first
    for _ in range(10):
        context = update_context(context, unit(0, 1), beta=0.8)
    assert context[1] > context[0]
    assert np.linalg.norm(context) == pytest.approx(1.0)
