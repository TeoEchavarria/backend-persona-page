"""Note-to-note transitions: a semantic prior updated with what visitors actually do.

    P(j | i) = (c_ij + λ s_ij) / (Σ_k c_ik + λ)

s is a row-softmax of note similarities (the Dirichlet prior), c the observed transition counts.
"""

from collections.abc import Iterable

import numpy as np


def softmax_rows(similarities: np.ndarray, temperature: float) -> np.ndarray:
    logits = similarities / temperature
    np.fill_diagonal(logits, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def semantic_prior(centroids: np.ndarray, temperature: float) -> np.ndarray:
    return softmax_rows(centroids @ centroids.T, temperature)


def transition_counts(
    visits: Iterable[tuple[object, str, object]], index: dict, finished_weight: float
) -> np.ndarray:
    """Count note-to-note moves per session; leaving a note read to the end counts `finished_weight`."""
    counts = np.zeros((len(index), len(index)))
    for path in _session_paths(visits, index):
        for (source, finished), (target, _) in zip(path, path[1:]):
            counts[source, target] += finished_weight if finished else 1.0
    return counts


def transition_matrix(counts: np.ndarray, prior: np.ndarray, strength: float) -> np.ndarray:
    return (counts + strength * prior) / (counts.sum(axis=1, keepdims=True) + strength)


def random_walk_with_restart(
    transitions: np.ndarray, restart: np.ndarray, restart_probability: float, tol: float = 1e-10, max_iter: int = 500
) -> np.ndarray:
    """Power iteration of r = (1 - c) Pᵀ r + c e. With c = 1 - damping and uniform e it is PageRank."""
    scores = restart.copy()
    for _ in range(max_iter):
        updated = (1 - restart_probability) * transitions.T @ scores + restart_probability * restart
        if np.abs(updated - scores).sum() < tol:
            return updated
        scores = updated
    return scores


def pagerank(transitions: np.ndarray, damping: float) -> np.ndarray:
    uniform = np.full(len(transitions), 1 / len(transitions))
    return random_walk_with_restart(transitions, uniform, 1 - damping)


def maximal_marginal_relevance(
    relevance: np.ndarray, embeddings: np.ndarray, candidates: list[int], k: int, balance: float
) -> list[int]:
    """Pick k candidates trading relevance against similarity to what is already picked."""
    top = max((relevance[i] for i in candidates), default=0.0)
    relevance = relevance / top if top > 0 else relevance
    chosen: list[int] = []
    remaining = list(candidates)
    while remaining and len(chosen) < k:
        def marginal(i: int) -> float:
            redundancy = max((float(embeddings[i] @ embeddings[j]) for j in chosen), default=0.0)
            return balance * relevance[i] - (1 - balance) * redundancy

        best = max(remaining, key=marginal)
        chosen.append(best)
        remaining.remove(best)
    return chosen


def _session_paths(visits: Iterable[tuple[object, str, object]], index: dict) -> list[list[tuple[int, bool]]]:
    """Collapse each session's events into [(note, finished)], one entry per consecutive visit."""
    paths: dict[object, list[list]] = {}
    for session, kind, note in visits:
        if note not in index:
            continue
        path = paths.setdefault(session, [])
        node = index[note]
        if not path or path[-1][0] != node:
            path.append([node, False])
        if kind == "finish":
            path[-1][1] = True
    return [[(node, finished) for node, finished in path] for path in paths.values()]
