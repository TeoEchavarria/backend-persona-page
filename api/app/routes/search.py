import datetime as dt
import time
from uuid import uuid4

import psycopg
from fastapi import APIRouter, Depends

from app import db, search, visitor
from app.config import Settings, get_settings
from app.embeddings import Encoder, get_encoder
from app.routes.errors import embedding_errors
from app.schemas import ConfidenceOut, NoteRef, SearchRequest, SearchResponse, SearchResult
from app.session import effective_query

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def run_search(
    request: SearchRequest,
    conn: psycopg.Connection = Depends(db.get_connection),
    encoder: Encoder = Depends(get_encoder),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    started = time.perf_counter()
    query = request.query.strip()
    session = visitor.load(conn, request.session_id or uuid4(), request.lang)
    with embedding_errors():
        query_vector = encoder.encode_queries([query])[0]

    use_context = request.use_context and session.context is not None
    ranked_by = effective_query(query_vector, session.context if use_context else None, settings.rocchio_alpha)
    hits, total = search.hybrid_search(conn, query, ranked_by, request.kind, settings, request.since, request.lang)
    # Confidence is judged on the literal query: context may reorder results, never make them look closer.
    stats = db.similarity_stats(conn, query_vector, request.kind, request.since, request.lang)
    # Judged on the literal query's ranking: the context may reorder, never make a match look closer.
    literal_hits, _ = (
        (hits, total) if ranked_by is query_vector
        else search.hybrid_search(conn, query, query_vector, request.kind, settings, request.since, request.lang)
    )
    confidence = search.confidence(stats, settings, request.lang, search.lexical_agreement(literal_hits))
    rows = db.fetch_chunks(conn, [hit.chunk_id for hit in hits], query_vector, query, request.lang)
    results = [_result(hit, rows[hit.chunk_id]) for hit in hits if hit.chunk_id in rows]
    if request.order == "date":
        results.sort(key=lambda result: result.note.published or dt.date.min, reverse=True)

    db.insert_event(conn, session.id, "query", query=query)
    visitor.observe(conn, settings, session, query_vector, read=False)
    visitor.save(conn, session)

    return SearchResponse(
        session_id=session.id,
        query=query,
        results=results,
        confidence=ConfidenceOut(
            band=confidence.band,
            message=confidence.message,
            margin=confidence.margin,
            z=confidence.z,
            gap=confidence.gap,
            top_similarity=confidence.top_similarity,
            mean_similarity=stats.mean,
            lexical=confidence.lexical,
            margin_high=settings.margin_high,
            margin_medium=settings.margin_medium,
            gap_high=settings.gap_high,
        ),
        context=visitor.context_out(conn, settings, session, applied=use_context, query=query_vector),
        intents=visitor.intents_out(conn, settings, session),
        rrf_k=settings.rrf_k,
        total=total,
        took_ms=round((time.perf_counter() - started) * 1000),
    )


def _result(hit: search.Hit, row: dict) -> SearchResult:
    return SearchResult(
        chunk_id=hit.chunk_id,
        text=row["text"],
        highlighted=row["highlighted"],
        headings=row["headings"],
        note=NoteRef(slug=row["slug"], title=row["title"], kind=row["kind"], published=row["published"]),
        score=hit.score,
        similarity=row["similarity"],
        vector_rank=hit.vector_rank,
        text_rank=hit.text_rank,
    )
