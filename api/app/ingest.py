"""Replace the indexed content with a new set of notes, embedding only what changed."""

from dataclasses import dataclass

import psycopg

from app import db, graph
from app.chunking import Note
from app.embeddings import Encoder


@dataclass(frozen=True)
class SyncReport:
    notes: int
    chunks: int
    embedded: int
    removed_chunks: int
    removed_notes: int


def sync(conn: psycopg.Connection, encoder: Encoder, notes: list[Note], intents: list[dict], force: bool = False) -> SyncReport:
    """Notes missing from `notes` are deleted: the caller always sends the whole content."""
    known = set() if force else db.chunk_ids(conn)
    pending = [(note, chunk) for note in notes for chunk in note.chunks if chunk.id not in known]
    embeddings = encoder.encode_passages([chunk.text_with_context for _, chunk in pending]) if pending else []

    with conn.transaction():
        note_ids = {note.slug: db.upsert_note(conn, note) for note in notes}
        for note in notes:
            for chunk in note.chunks:
                if chunk.id in known:
                    db.update_position(conn, note_ids[note.slug], chunk)
        for (note, chunk), embedding in zip(pending, embeddings):
            db.insert_chunk(conn, note_ids[note.slug], chunk, embedding)
        removed_chunks = db.delete_chunks_except(conn, {chunk.id for note in notes for chunk in note.chunks})
        removed_notes = db.delete_notes_except(conn, set(note_ids))
        db.replace_intents(conn, intents)

    graph.invalidate()
    return SyncReport(
        notes=len(notes),
        chunks=sum(len(note.chunks) for note in notes),
        embedded=len(pending),
        removed_chunks=removed_chunks,
        removed_notes=removed_notes,
    )
