import logging
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import db
from app.config import get_settings
from app.embeddings import get_encoder
from app.routes import admin, events, notes, search

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # A local model loads in seconds; do it before the first request, not inside one.
    # A failure here must not take /health down with it; searches report it instead.
    if get_settings().embedding_backend == "local":
        try:
            get_encoder()
        except RuntimeError as error:
            logger.error("Could not load the local encoder: %s", error)
    yield


app = FastAPI(title="Portfolio search", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

app.include_router(search.router)
app.include_router(events.router)
app.include_router(notes.router)
app.include_router(admin.router)


@app.get("/")
def root():
    return {"name": "portfolio-search", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health():
    """Tells an unreachable database apart from one that is reachable but not synced yet."""
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=5) as conn:
            has_vector = conn.execute("select 1 from pg_extension where extname = 'vector'").fetchone()
            has_notes = conn.execute("select to_regclass('notes')").fetchone()[0] is not None
            notes = conn.execute("select count(*) from notes").fetchone()[0] if has_notes else 0
    except psycopg.Error as error:
        logger.error("Health check failed: %s", error)
        return JSONResponse({"status": "error", "database": "unreachable"}, status_code=503)
    if not (has_vector and has_notes):
        return JSONResponse(
            {"status": "error", "database": "ok", "schema": "missing — run the first content sync"},
            status_code=503,
        )
    return {"status": "ok", "database": "ok", "notes": notes}
