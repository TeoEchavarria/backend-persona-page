"""A visitor's session: the Rocchio context vector and the belief over intents."""

from dataclasses import dataclass
from uuid import UUID

import numpy as np
import psycopg

from app import db, graph, intent
from app.config import Settings
from app.schemas import ContextOut, IntentOut, NoteRef, SessionState
from app.session import update_context


@dataclass
class Visitor:
    id: UUID
    context: np.ndarray | None
    belief: np.ndarray | None
    lang: str = "es"  # the language the visitor is reading in: picks the note graph and intent labels


def load(conn: psycopg.Connection, session_id: UUID, lang: str = "es") -> Visitor:
    row = db.get_or_create_session(conn, session_id)
    context = None if row["context"] is None else np.asarray(row["context"], dtype=np.float32)
    belief = None if row["intent_belief"] is None else np.asarray(row["intent_belief"])
    return Visitor(session_id, context, belief, lang)


def save(conn: psycopg.Connection, visitor: Visitor) -> None:
    belief = None if visitor.belief is None else [float(p) for p in visitor.belief]
    db.save_session(conn, visitor.id, visitor.context, belief)


def observe(conn: psycopg.Connection, settings: Settings, visitor: Visitor, vector: np.ndarray, read: bool) -> None:
    """Every observation updates the intent belief; only reading moves the context."""
    if read:
        visitor.context = update_context(visitor.context, vector, settings.rocchio_beta)
    intents = graph.load_intents(conn, settings, visitor.lang)
    if intents is not None:
        belief = visitor.belief if visitor.belief is not None and len(visitor.belief) == len(intents.ids) else None
        visitor.belief = intent.forward_step(
            belief, vector.astype(np.float64), intents.centroids, intents.transitions, settings.intent_temperature
        )


def context_out(
    conn: psycopg.Connection,
    settings: Settings,
    visitor: Visitor,
    applied: bool = False,
    query: np.ndarray | None = None,
) -> ContextOut:
    note = None
    notes = graph.load_graph(conn, settings, visitor.lang)
    if visitor.context is not None and notes is not None:
        closest = notes.notes[int(np.argmax(notes.centroids @ visitor.context))]
        note = NoteRef(**{key: closest[key] for key in ("slug", "title", "kind", "published")})
    alignment = None
    if visitor.context is not None and query is not None:
        alignment = float(query @ visitor.context)
    return ContextOut(
        available=visitor.context is not None,
        applied=applied,
        note=note,
        alpha=settings.rocchio_alpha,
        beta=settings.rocchio_beta,
        query_alignment=alignment,
    )


def intents_out(conn: psycopg.Connection, settings: Settings, visitor: Visitor) -> list[IntentOut]:
    intents = graph.load_intents(conn, settings, visitor.lang)
    if intents is None:
        return []
    belief = visitor.belief if visitor.belief is not None and len(visitor.belief) == len(intents.ids) else None
    return [IntentOut(**item) for item in intents.distribution(belief)]


def state(conn: psycopg.Connection, settings: Settings, visitor: Visitor) -> SessionState:
    return SessionState(
        session_id=visitor.id,
        context=context_out(conn, settings, visitor),
        intents=intents_out(conn, settings, visitor),
    )
