import psycopg
from fastapi import APIRouter, Depends, HTTPException

from app import db
from app.schemas import ChunkOut, NoteOut, NoteSummary

router = APIRouter()


@router.get("/notes", response_model=list[NoteSummary])
def list_notes(conn: psycopg.Connection = Depends(db.get_connection)) -> list[NoteSummary]:
    return [NoteSummary(**row) for row in db.list_notes(conn)]


@router.get("/notes/{slug}", response_model=NoteOut)
def get_note(slug: str, conn: psycopg.Connection = Depends(db.get_connection)) -> NoteOut:
    note = db.get_note(conn, slug)
    if note is None:
        raise HTTPException(404, "Nota no encontrada")
    chunks = [ChunkOut(**row) for row in db.note_chunks(conn, note["id"])]
    return NoteOut(**note, chunks=chunks)
