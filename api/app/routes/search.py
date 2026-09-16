import psycopg
from fastapi import APIRouter, Depends

from app import db, search
from app.config import Settings, get_settings
from app.embeddings import Encoder, get_encoder
from app.routes.errors import embedding_errors
from app.schemas import ConfidenceOut, NoteRef, SearchRequest, SearchResponse, SearchResult

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def run_search(
    request: SearchRequest,
    conn: psycopg.Connection = Depends(db.get_connection),
    encoder: Encoder = Depends(get_encoder),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    query = request.query.strip()
    with embedding_errors():
        query_vector = encoder.encode_queries([query])[0]

    hits = search.hybrid_search(conn, query, query_vector, request.kind, settings)
    stats = db.similarity_stats(conn, query_vector, request.kind)
    confidence = search.confidence(stats, settings)
    rows = db.fetch_chunks(conn, [hit.chunk_id for hit in hits], query_vector)

    return SearchResponse(
        query=query,
        results=[_result(hit, rows[hit.chunk_id]) for hit in hits if hit.chunk_id in rows],
        confidence=ConfidenceOut(
            band=confidence.band,
            message=confidence.message,
            margin=confidence.margin,
            z=confidence.z,
            gap=confidence.gap,
            top_similarity=confidence.top_similarity,
            mean_similarity=stats.mean,
            margin_high=settings.margin_high,
            margin_medium=settings.margin_medium,
            gap_high=settings.gap_high,
        ),
        rrf_k=settings.rrf_k,
    )


def _result(hit: search.Hit, row: dict) -> SearchResult:
    return SearchResult(
        chunk_id=hit.chunk_id,
        text=row["text"],
        headings=row["headings"],
        note=NoteRef(slug=row["slug"], title=row["title"], kind=row["kind"]),
        score=hit.score,
        similarity=row["similarity"],
        vector_rank=hit.vector_rank,
        text_rank=hit.text_rank,
    )
