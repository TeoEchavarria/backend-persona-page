from dataclasses import dataclass

import numpy as np
import psycopg

from app import db
from app.config import Settings

MESSAGES = {
    "high": "Esto responde directamente a lo que buscas.",
    "medium": "Hay coincidencias parciales; puede que no sea exactamente lo que buscas.",
    "far": "No encontré nada cercano. Estos son los fragmentos menos lejanos.",
}


@dataclass(frozen=True)
class Confidence:
    band: str
    margin: float
    z: float
    gap: float
    top_similarity: float
    message: str


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    vector_rank: int | None
    text_rank: int | None


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return scores


def z_score(value: float, mean: float, std: float) -> float:
    return 0.0 if std == 0 else (value - mean) / std


def confidence_band(margin: float, gap: float, settings: Settings) -> str:
    """Bands use the raw margin over the mean, not z: off-topic queries score uniformly low,
    which shrinks the spread and inflates their z-score."""
    if margin >= settings.margin_high or (margin >= settings.margin_medium and gap >= settings.gap_high):
        return "high"
    if margin >= settings.margin_medium:
        return "medium"
    return "far"


def confidence(stats: db.SimilarityStats, settings: Settings) -> Confidence:
    if not stats.top:
        return Confidence("far", 0.0, 0.0, 0.0, 0.0, MESSAGES["far"])
    top = stats.top[0]
    gap = top - stats.top[1] if len(stats.top) > 1 else 0.0
    margin = top - stats.mean
    band = confidence_band(margin, gap, settings)
    return Confidence(band, margin, z_score(top, stats.mean, stats.std), gap, top, MESSAGES[band])


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    query_vector: np.ndarray,
    kind: str | None,
    settings: Settings,
) -> list[Hit]:
    by_vector = db.vector_ranking(conn, query_vector, kind, settings.candidate_pool)
    by_text = db.text_ranking(conn, query, kind, settings.candidate_pool)
    vector_ids = [chunk_id for chunk_id, _ in by_vector]
    text_ids = [chunk_id for chunk_id, _ in by_text]
    fused = reciprocal_rank_fusion([vector_ids, text_ids], settings.rrf_k)
    best = sorted(fused, key=fused.get, reverse=True)[: settings.results_limit]
    return [
        Hit(
            chunk_id=chunk_id,
            score=fused[chunk_id],
            vector_rank=_rank_of(chunk_id, vector_ids),
            text_rank=_rank_of(chunk_id, text_ids),
        )
        for chunk_id in best
    ]


def _rank_of(item: str, ranking: list[str]) -> int | None:
    return ranking.index(item) + 1 if item in ranking else None
