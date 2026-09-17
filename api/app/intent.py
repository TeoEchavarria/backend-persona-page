"""What is the visitor after? A small HMM over hand-written intents, updated online with the forward algorithm.

    α_t(z) ∝ p(o_t | z) · Σ_z' α_{t-1}(z') A(z', z)

Emissions are von Mises–Fisher-like: p(o | z) ∝ exp(cos(o, μ_z) / τ). With a shared τ the
normalizing constant is the same for every intent, so it cancels when α is normalized.
"""

import numpy as np


def sticky_transitions(n: int, stickiness: float) -> np.ndarray:
    if n == 1:
        return np.ones((1, 1))
    matrix = np.full((n, n), (1 - stickiness) / (n - 1))
    np.fill_diagonal(matrix, stickiness)
    return matrix


def emission_likelihoods(observation: np.ndarray, centroids: np.ndarray, temperature: float) -> np.ndarray:
    logits = centroids @ observation / temperature
    return np.exp(logits - logits.max())


def forward_step(
    belief: np.ndarray | None, observation: np.ndarray, centroids: np.ndarray, transitions: np.ndarray, temperature: float
) -> np.ndarray:
    prior = np.full(len(centroids), 1 / len(centroids)) if belief is None else belief @ transitions
    posterior = prior * emission_likelihoods(observation, centroids, temperature)
    return posterior / posterior.sum()
