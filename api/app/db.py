from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

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


def as_array(value) -> np.ndarray | None:
    return None if value is None else np.asarray(value.to_numpy(), dtype=np.float32)


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


def replace_intents(conn: psycopg.Connection, intents: list[dict]) -> None:
    with conn.transaction():
        conn.execute("delete from intents")
        for position, intent in enumerate(intents):
            conn.execute(
                "insert into intents (id, position, label, description, note_slugs) values (%s, %s, %s, %s, %s)",
                (intent["id"], position, intent["label"], intent.get("description", ""), intent.get("notes", [])),
            )


def purge_old_sessions(conn: psycopg.Connection, days: int) -> int:
    return conn.execute(
        "delete from sessions where updated_at < now() - make_interval(days => %s)", (days,)
    ).rowcount


# ── Search ────────────────────────────────────────────────────────────

_KIND_FILTER = "(%(kind)s::text is null or n.kind = %(kind)s)"


@dataclass(frozen=True)
class SimilarityStats:
    mean: float
    std: float
    top: list[float]


def vector_ranking(conn: psycopg.Connection, vector: np.ndarray, kind: str | None, limit: int) -> list[tuple[str, float]]:
    # <#> is the negative inner product; with unit vectors, -(a <#> b) is the cosine.
    rows = conn.execute(
        f"""
        select c.id, -(c.embedding <#> %(v)s) as similarity
        from chunks c join notes n on n.id = c.note_id
        where {_KIND_FILTER}
        order by c.embedding <#> %(v)s
        limit %(limit)s
        """,
        {"v": vector, "kind": kind, "limit": limit},
    )
    return [(row["id"], row["similarity"]) for row in rows]


def similarity_stats(conn: psycopg.Connection, vector: np.ndarray, kind: str | None) -> SimilarityStats:
    row = conn.execute(
        f"""
        select avg(s) as mean, coalesce(stddev_pop(s), 0) as std,
               (array_agg(s order by s desc))[1:2] as top
        from (
            select -(c.embedding <#> %(v)s) as s
            from chunks c join notes n on n.id = c.note_id
            where {_KIND_FILTER}
        ) scored
        """,
        {"v": vector, "kind": kind},
    ).fetchone()
    return SimilarityStats(row["mean"] or 0.0, row["std"], list(row["top"] or []))


def text_ranking(conn: psycopg.Connection, query: str, kind: str | None, limit: int) -> list[tuple[str, float]]:
    # plainto_tsquery ANDs every word; OR-ing them suits natural-language questions better.
    config = get_settings().text_search_config
    rows = conn.execute(
        f"""
        with q as (
            select nullif(replace(plainto_tsquery('{config}', %(q)s)::text, '&', '|'), '')::tsquery as tsq
        )
        select c.id, ts_rank(c.tsv, q.tsq) as rank
        from chunks c join notes n on n.id = c.note_id, q
        where q.tsq is not null and c.tsv @@ q.tsq and {_KIND_FILTER}
        order by rank desc
        limit %(limit)s
        """,
        {"q": query, "kind": kind, "limit": limit},
    )
    return [(row["id"], row["rank"]) for row in rows]


def fetch_chunks(conn: psycopg.Connection, ids: list[str], vector: np.ndarray) -> dict[str, dict]:
    rows = conn.execute(
        """
        select c.id, c.text, c.headings, c.position, -(c.embedding <#> %s) as similarity,
               n.id as note_id, n.slug, n.title, n.kind
        from chunks c join notes n on n.id = c.note_id
        where c.id = any(%s)
        """,
        (vector, ids),
    )
    return {row["id"]: row for row in rows}


def chunk_embedding(conn: psycopg.Connection, chunk_id: str) -> tuple[np.ndarray, int] | None:
    row = conn.execute("select embedding, note_id from chunks where id = %s", (chunk_id,)).fetchone()
    return (as_array(row["embedding"]), row["note_id"]) if row else None


# ── Notes ─────────────────────────────────────────────────────────────


def list_notes(conn: psycopg.Connection) -> list[dict]:
    return conn.execute("select id, slug, kind, title, tags, summary, link from notes order by id").fetchall()


def get_note(conn: psycopg.Connection, slug: str) -> dict | None:
    return conn.execute(
        "select id, slug, kind, title, tags, summary, link from notes where slug = %s", (slug,)
    ).fetchone()


def note_chunks(conn: psycopg.Connection, note_id: int) -> list[dict]:
    return conn.execute(
        "select id, position, headings, text from chunks where note_id = %s order by position", (note_id,)
    ).fetchall()


def note_centroids(conn: psycopg.Connection) -> list[dict]:
    rows = conn.execute(
        """
        select n.id, n.slug, n.kind, n.title, avg(c.embedding) as centroid
        from notes n join chunks c on c.note_id = n.id
        group by n.id order by n.id
        """
    ).fetchall()
    return [{**row, "centroid": as_array(row["centroid"])} for row in rows]


def note_id_for_slug(conn: psycopg.Connection, slug: str) -> int | None:
    row = conn.execute("select id from notes where slug = %s", (slug,)).fetchone()
    return row["id"] if row else None


def list_intents(conn: psycopg.Connection) -> list[dict]:
    return conn.execute("select id, label, description, note_slugs from intents order by position").fetchall()


# ── Sessions and events ───────────────────────────────────────────────


def get_or_create_session(conn: psycopg.Connection, session_id: UUID) -> dict:
    conn.execute("insert into sessions (id) values (%s) on conflict (id) do nothing", (session_id,))
    row = conn.execute("select id, context, intent_belief from sessions where id = %s", (session_id,)).fetchone()
    return {**row, "context": as_array(row["context"])}


def save_session(
    conn: psycopg.Connection, session_id: UUID, context: np.ndarray | None, belief: list[float] | None
) -> None:
    conn.execute(
        "update sessions set context = %s, intent_belief = %s, updated_at = now() where id = %s",
        (context, belief, session_id),
    )


def insert_event(
    conn: psycopg.Connection,
    session_id: UUID,
    kind: str,
    chunk_id: str | None = None,
    note_id: int | None = None,
    query: str | None = None,
) -> None:
    conn.execute(
        "insert into events (session_id, kind, chunk_id, note_id, query) values (%s, %s, %s, %s, %s)",
        (session_id, kind, chunk_id, note_id, query),
    )


def note_visits(conn: psycopg.Connection) -> list[dict]:
    """Every note-level event, grouped by session in time order: the raw material for transition counts."""
    return conn.execute(
        """
        select session_id, kind, note_id from events
        where note_id is not null and kind in ('read', 'select', 'finish')
        order by session_id, created_at, id
        """
    ).fetchall()
