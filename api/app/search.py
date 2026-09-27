from dataclasses import dataclass

import numpy as np
import psycopg

from app import db
from app.config import Settings

MESSAGES = {
    "es": {
        "high": "Esto responde directamente a lo que buscas.",
        "medium": "Hay coincidencias parciales; puede que no sea exactamente lo que buscas.",
        "far": "No encontré nada cercano. Estos son los fragmentos menos lejanos.",
    },
    "en": {
        "high": "This answers what you are looking for.",
        "medium": "These are partial matches; they may not be exactly what you are looking for.",
        "far": "Nothing close came up. These are the least distant paragraphs.",
    },
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


def confidence(stats: db.SimilarityStats, settings: Settings, lang: str = "es") -> Confidence:
    messages = MESSAGES.get(lang, MESSAGES["es"])
    if not stats.top:
        return Confidence("far", 0.0, 0.0, 0.0, 0.0, messages["far"])
    top = stats.top[0]
    gap = top - stats.top[1] if len(stats.top) > 1 else 0.0
    margin = top - stats.mean
    band = confidence_band(margin, gap, settings)
    return Confidence(band, margin, z_score(top, stats.mean, stats.std), gap, top, messages[band])


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    query_vector: np.ndarray,
    kind: str | None,
    settings: Settings,
    since: int | None = None,
    lang: str = "es",
) -> tuple[list[Hit], int]:
    """The best hits, and how many distinct candidates the two rankings produced."""
    by_vector = db.vector_ranking(conn, query_vector, kind, settings.candidate_pool, since, lang)
    by_text = db.text_ranking(conn, query, kind, settings.candidate_pool, since, lang)
    vector_ids = [chunk_id for chunk_id, _ in by_vector]
    text_ids = [chunk_id for chunk_id, _ in by_text]
    fused = reciprocal_rank_fusion([vector_ids, text_ids], settings.rrf_k)
    best = sorted(fused, key=fused.get, reverse=True)[: settings.results_limit]
    hits = [
        Hit(
            chunk_id=chunk_id,
            score=fused[chunk_id],
            vector_rank=_rank_of(chunk_id, vector_ids),
            text_rank=_rank_of(chunk_id, text_ids),
        )
        for chunk_id in best
    ]
    return hits, len(fused)


def _rank_of(item: str, ranking: list[str]) -> int | None:
    return ranking.index(item) + 1 if item in ranking else None
