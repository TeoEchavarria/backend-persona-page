import numpy as np
import pytest

from app import intent, markov
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
    # A short query naming a technology has little margin but literal evidence: never "far".
    assert confidence_band(0.03, 0.0, settings, lexical=True) == "medium"


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


# ── Markov ────────────────────────────────────────────────────────────

CENTROIDS = np.stack([unit(1, 0, 0), unit(0.9, 0.1, 0), unit(0, 1, 0), unit(0, 0.1, 1)])


def test_prior_is_a_stochastic_matrix_without_self_loops():
    prior = markov.semantic_prior(CENTROIDS, temperature=0.1)
    assert np.allclose(prior.sum(axis=1), 1)
    assert np.allclose(np.diag(prior), 0)
    assert prior[0].argmax() == 1


def test_without_events_transitions_equal_the_prior():
    prior = markov.semantic_prior(CENTROIDS, 0.1)
    assert np.allclose(markov.transition_matrix(np.zeros((4, 4)), prior, strength=2.0), prior)


def test_observed_transitions_shift_the_matrix():
    prior = markov.semantic_prior(CENTROIDS, 0.1)
    index = {10: 0, 11: 1, 12: 2, 13: 3}
    visits = [("s", "read", 10), ("s", "read", 10), ("s", "finish", 10), ("s", "read", 12)] * 1
    visits += [("t", "read", 10), ("t", "read", 12)]
    counts = markov.transition_counts(visits, index, finished_weight=2.0)
    assert counts[0, 2] == pytest.approx(3.0)
    assert counts.sum() == pytest.approx(3.0)

    transitions = markov.transition_matrix(counts, prior, strength=2.0)
    assert np.allclose(transitions.sum(axis=1), 1)
    assert transitions[0, 2] > prior[0, 2]
    assert transitions[0].argmax() == 2


def test_random_walk_reaches_two_steps_away():
    # A chain 0 -> 1 -> 2: from 0, node 2 is only reachable through 1.
    chain = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    scores = markov.random_walk_with_restart(chain, np.array([1.0, 0, 0]), restart_probability=0.3)
    assert scores.sum() == pytest.approx(1.0)
    assert scores[0] > scores[1] > scores[2] > 0


def test_pagerank_prefers_the_hub():
    star = np.array([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]], dtype=float)
    ranks = markov.pagerank(star, damping=0.85)
    assert ranks.sum() == pytest.approx(1.0)
    assert ranks.argmax() == 1


def test_mmr_skips_near_duplicates():
    embeddings = np.stack([unit(1, 0), unit(1, 0.01), unit(0, 1)])
    relevance = np.array([1.0, 0.95, 0.6])
    assert markov.maximal_marginal_relevance(relevance, embeddings, [0, 1, 2], k=2, balance=0.5) == [0, 2]
    assert markov.maximal_marginal_relevance(relevance, embeddings, [0, 1, 2], k=2, balance=1.0) == [0, 1]


# ── HMM ───────────────────────────────────────────────────────────────


def test_forward_moves_belief_towards_the_observed_intent_and_is_sticky():
    centroids = np.stack([unit(1, 0), unit(0, 1)])
    transitions = intent.sticky_transitions(2, stickiness=0.8)
    assert np.allclose(transitions.sum(axis=1), 1)

    belief = intent.forward_step(None, unit(1, 0.2), centroids, transitions, temperature=0.1)
    assert belief.sum() == pytest.approx(1.0)
    assert belief[0] > 0.9

    # One ambiguous observation does not erase what came before.
    belief = intent.forward_step(belief, unit(1, 1), centroids, transitions, temperature=0.1)
    assert belief[0] > 0.5

    for _ in range(3):
        belief = intent.forward_step(belief, unit(0, 1), centroids, transitions, temperature=0.1)
    assert belief[1] > 0.9
