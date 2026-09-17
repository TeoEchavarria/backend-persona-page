import numpy as np

from app.embeddings import normalize


def effective_query(query: np.ndarray, context: np.ndarray | None, alpha: float = 0.7) -> np.ndarray:
    if context is None:
        return query
    return normalize(alpha * query + (1 - alpha) * context)


def update_context(context: np.ndarray | None, read: np.ndarray, beta: float = 0.8) -> np.ndarray:
    if context is None:
        return read
    return normalize(beta * context + (1 - beta) * read)
