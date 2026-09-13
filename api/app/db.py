from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    # prepare_threshold=None keeps it working behind transaction poolers (Neon, pgbouncer).
    with psycopg.connect(
        get_settings().database_url, row_factory=dict_row, autocommit=True, prepare_threshold=None
    ) as conn:
        yield conn


def get_connection() -> Iterator[psycopg.Connection]:
    with connect() as conn:
        yield conn


def ping(conn: psycopg.Connection) -> bool:
    return conn.execute("select 1 as ok").fetchone()["ok"] == 1
