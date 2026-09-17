"""A visitor's session: the Rocchio context vector built from what they read."""

from dataclasses import dataclass
from uuid import UUID

import numpy as np
import psycopg

from app import db
from app.config import Settings
from app.embeddings import normalize
from app.schemas import ContextOut, NoteRef, SessionState
from app.session import update_context


@dataclass
class Visitor:
    id: UUID
    context: np.ndarray | None


def load(conn: psycopg.Connection, session_id: UUID) -> Visitor:
    row = db.get_or_create_session(conn, session_id)
    context = None if row["context"] is None else np.asarray(row["context"], dtype=np.float32)
    return Visitor(session_id, context)


def save(conn: psycopg.Connection, visitor: Visitor) -> None:
    db.save_session(conn, visitor.id, visitor.context)


def observe(conn: psycopg.Connection, settings: Settings, visitor: Visitor, vector: np.ndarray, read: bool) -> None:
    """Only reading moves the context; a query is logged but leaves it alone."""
    if read:
        visitor.context = update_context(visitor.context, vector, settings.rocchio_beta)


def context_out(
    conn: psycopg.Connection,
    settings: Settings,
    visitor: Visitor,
    applied: bool = False,
    query: np.ndarray | None = None,
) -> ContextOut:
    note = None
    rows = db.note_centroids(conn) if visitor.context is not None else []
    if rows:
        centroids = normalize(np.stack([row["centroid"] for row in rows]))
        closest = rows[int(np.argmax(centroids @ visitor.context))]
        note = NoteRef(slug=closest["slug"], title=closest["title"], kind=closest["kind"])
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


def state(conn: psycopg.Connection, settings: Settings, visitor: Visitor) -> SessionState:
    return SessionState(session_id=visitor.id, context=context_out(conn, settings, visitor))
