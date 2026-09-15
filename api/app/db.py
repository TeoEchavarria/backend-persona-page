from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.chunking import Chunk, Note
from app.config import get_settings

SCHEMA = Path(__file__).with_name("schema.sql")


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    # prepare_threshold=None keeps it working behind transaction poolers (Neon, pgbouncer).
    with psycopg.connect(
        get_settings().database_url, row_factory=dict_row, autocommit=True, prepare_threshold=None
    ) as conn:
        register_vector(conn)
        yield conn


def get_connection() -> Iterator[psycopg.Connection]:
    with connect() as conn:
        yield conn


def init_schema() -> None:
    settings = get_settings()
    sql = (
        SCHEMA.read_text()
        .replace("{dim}", str(settings.embedding_dim))
        .replace("{text_config}", settings.text_search_config)
    )
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(sql)


def ping(conn: psycopg.Connection) -> bool:
    return conn.execute("select 1 as ok").fetchone()["ok"] == 1


# ── Indexing ──────────────────────────────────────────────────────────


def upsert_note(conn: psycopg.Connection, note: Note) -> int:
    row = conn.execute(
        """
        insert into notes (slug, kind, title, tags, summary, link)
        values (%s, %s, %s, %s, %s, %s)
        on conflict (slug) do update
        set kind = excluded.kind, title = excluded.title, tags = excluded.tags,
            summary = excluded.summary, link = excluded.link
        returning id
        """,
        (note.slug, note.kind, note.title, note.tags, note.summary, note.link),
    ).fetchone()
    return row["id"]


def chunk_ids(conn: psycopg.Connection) -> set[str]:
    return {row["id"] for row in conn.execute("select id from chunks")}


def insert_chunk(conn: psycopg.Connection, note_id: int, chunk: Chunk, embedding: np.ndarray) -> None:
    conn.execute(
        """
        insert into chunks (id, note_id, position, headings, text, text_with_context, embedding)
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict (id) do update
        set note_id = excluded.note_id, position = excluded.position, headings = excluded.headings,
            text = excluded.text, text_with_context = excluded.text_with_context, embedding = excluded.embedding
        """,
        (chunk.id, note_id, chunk.position, list(chunk.headings), chunk.text, chunk.text_with_context, embedding),
    )


def update_position(conn: psycopg.Connection, note_id: int, chunk: Chunk) -> None:
    conn.execute(
        "update chunks set note_id = %s, position = %s where id = %s and (note_id, position) <> (%s, %s)",
        (note_id, chunk.position, chunk.id, note_id, chunk.position),
    )


def delete_chunks_except(conn: psycopg.Connection, keep: set[str]) -> int:
    return conn.execute("delete from chunks where not (id = any(%s))", (list(keep),)).rowcount


def delete_notes_except(conn: psycopg.Connection, slugs: set[str]) -> int:
    return conn.execute("delete from notes where not (slug = any(%s))", (list(slugs),)).rowcount
