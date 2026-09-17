from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from app import db, visitor
from app.config import Settings, get_settings
from app.schemas import EventRequest, SessionState

router = APIRouter()


@router.post("/events", response_model=SessionState)
def record_event(
    event: EventRequest,
    conn: psycopg.Connection = Depends(db.get_connection),
    settings: Settings = Depends(get_settings),
) -> SessionState:
    session = visitor.load(conn, event.session_id)
    note_id = db.note_id_for_slug(conn, event.note_slug) if event.note_slug else None

    if event.chunk_id:
        found = db.chunk_embedding(conn, event.chunk_id)
        if found is None:
            raise HTTPException(404, "Fragmento desconocido")
        embedding, note_id = found
        if event.kind in ("read", "select"):
            visitor.observe(conn, settings, session, embedding, read=True)
    elif note_id is None:
        raise HTTPException(422, "El evento necesita chunk_id o note_slug")

    db.insert_event(conn, session.id, event.kind, chunk_id=event.chunk_id, note_id=note_id)
    visitor.save(conn, session)
    return visitor.state(conn, settings, session)


@router.get("/sessions/{session_id}", response_model=SessionState)
def session_state(
    session_id: UUID,
    conn: psycopg.Connection = Depends(db.get_connection),
    settings: Settings = Depends(get_settings),
) -> SessionState:
    return visitor.state(conn, settings, visitor.load(conn, session_id))


@router.delete("/sessions/{session_id}/context", response_model=SessionState)
def clear_context(
    session_id: UUID,
    conn: psycopg.Connection = Depends(db.get_connection),
    settings: Settings = Depends(get_settings),
) -> SessionState:
    session = visitor.load(conn, session_id)
    session.context = None
    visitor.save(conn, session)
    return visitor.state(conn, settings, session)
