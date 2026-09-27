"""Note graph and intents, rebuilt from the database at most once a minute per process."""

import time
from dataclasses import dataclass

import numpy as np
import psycopg

from app import db, intent, markov
from app.config import Settings
from app.embeddings import normalize

CACHE_SECONDS = 60


@dataclass
class NoteGraph:
    notes: list[dict]
    index: dict[str, int]
    centroids: np.ndarray
    prior: np.ndarray
    counts: np.ndarray
    transitions: np.ndarray
    pagerank: np.ndarray

    def position(self, slug: str) -> int | None:
        return next((i for i, note in enumerate(self.notes) if note["slug"] == slug), None)


@dataclass
class Intents:
    ids: list[str]
    labels: list[str]
    centroids: np.ndarray
    transitions: np.ndarray

    def distribution(self, belief: list[float] | np.ndarray | None) -> list[dict]:
        probabilities = np.full(len(self.ids), 1 / len(self.ids)) if belief is None else np.asarray(belief)
        return [
            {"id": i, "label": label, "probability": float(p)}
            for i, label, p in zip(self.ids, self.labels, probabilities)
        ]


_cache: dict[str, tuple[float, object]] = {}


def invalidate() -> None:
    _cache.clear()


def _cached(key: str, build):
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    value = build()
    _cache[key] = (time.monotonic(), value)
    return value


def load_graph(conn: psycopg.Connection, settings: Settings, lang: str = "es") -> NoteGraph | None:
    return _cached(f"graph:{lang}", lambda: build_graph(conn, settings, lang))


def load_intents(conn: psycopg.Connection, settings: Settings, lang: str = "es") -> Intents | None:
    return _cached(f"intents:{lang}", lambda: build_intents(conn, settings, lang))


def build_graph(conn: psycopg.Connection, settings: Settings, lang: str = "es") -> NoteGraph | None:
    """One graph per language; transitions are counted by slug, so both languages feed both graphs."""
    rows = db.note_centroids(conn, lang)
    if len(rows) < 2:
        return None
    notes = [{key: row[key] for key in ("id", "slug", "title", "kind", "published")} for row in rows]
    index = {note["slug"]: i for i, note in enumerate(notes)}
    centroids = normalize(np.stack([np.asarray(row["centroid"], dtype=np.float64) for row in rows]))
    prior = markov.semantic_prior(centroids, settings.markov_temperature)
    visits = [(row["session_id"], row["kind"], row["slug"]) for row in db.note_visits(conn)]
    counts = markov.transition_counts(visits, index, settings.finished_weight)
    transitions = markov.transition_matrix(counts, prior, settings.markov_lambda)
    return NoteGraph(
        notes=notes,
        index=index,
        centroids=centroids,
        prior=prior,
        counts=counts,
        transitions=transitions,
        pagerank=markov.pagerank(transitions, settings.pagerank_damping),
    )


def build_intents(conn: psycopg.Connection, settings: Settings, lang: str = "es") -> Intents | None:
    centroid_by_slug = {
        row["slug"]: np.asarray(row["centroid"], dtype=np.float64) for row in db.note_centroids(conn, lang)
    }
    ids, labels, centroids = [], [], []
    for row in db.list_intents(conn, lang):
        members = [centroid_by_slug[slug] for slug in row["note_slugs"] if slug in centroid_by_slug]
        if members:
            ids.append(row["id"])
            labels.append(row["label"])
            centroids.append(np.mean(members, axis=0))
    if not ids:
        return None
    return Intents(
        ids=ids,
        labels=labels,
        centroids=normalize(np.stack(centroids)),
        transitions=intent.sticky_transitions(len(ids), settings.intent_stickiness),
    )


def related(graph: NoteGraph, slug: str, k: int, settings: Settings) -> list[tuple[dict, float]]:
    source = graph.position(slug)
    if source is None:
        return []
    restart = np.zeros(len(graph.notes))
    restart[source] = 1.0
    scores = markov.random_walk_with_restart(graph.transitions, restart, settings.restart_probability)
    candidates = [i for i in range(len(graph.notes)) if i != source]
    chosen = markov.maximal_marginal_relevance(scores, graph.centroids, candidates, k, settings.mmr_lambda)
    return [(graph.notes[i], float(scores[i])) for i in chosen]
