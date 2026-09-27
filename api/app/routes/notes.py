import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query

from app import db, graph
from app.config import Settings, get_settings
from app.schemas import ChunkOut, GraphEdge, GraphNode, GraphOut, Lang, NoteOut, NoteSummary, RelatedNote

router = APIRouter()


@router.get("/notes", response_model=list[NoteSummary])
def list_notes(
    lang: Lang = "es",
    conn: psycopg.Connection = Depends(db.get_connection),
    settings: Settings = Depends(get_settings),
) -> list[NoteSummary]:
    """Every note, most central first (global PageRank over the transition graph)."""
    notes = graph.load_graph(conn, settings, lang)
    centrality = {} if notes is None else {n["slug"]: float(p) for n, p in zip(notes.notes, notes.pagerank)}
    summaries = [NoteSummary(**row, centrality=centrality.get(row["slug"])) for row in db.list_notes(conn, lang)]
    return sorted(summaries, key=lambda note: note.centrality or 0.0, reverse=True)


@router.get("/notes/{slug}", response_model=NoteOut)
def get_note(slug: str, lang: Lang = "es", conn: psycopg.Connection = Depends(db.get_connection)) -> NoteOut:
    note = db.get_note(conn, slug, lang)
    if note is None:
        raise HTTPException(404, "Nota no encontrada")
    chunks = [ChunkOut(**row) for row in db.note_chunks(conn, note["id"])]
    return NoteOut(**note, chunks=chunks)


@router.get("/notes/{slug}/related", response_model=list[RelatedNote])
def related_notes(
    slug: str,
    k: int = Query(3, ge=1, le=10),
    lang: Lang = "es",
    conn: psycopg.Connection = Depends(db.get_connection),
    settings: Settings = Depends(get_settings),
) -> list[RelatedNote]:
    notes = graph.load_graph(conn, settings, lang)
    if notes is None or notes.position(slug) is None:
        raise HTTPException(404, "Nota no encontrada")
    return [
        RelatedNote(**note, score=score)
        for note, score in graph.related(notes, slug, k, settings)
    ]


@router.get("/graph", response_model=GraphOut)
def note_graph(
    lang: Lang = "es",
    conn: psycopg.Connection = Depends(db.get_connection),
    settings: Settings = Depends(get_settings),
) -> GraphOut:
    notes = graph.load_graph(conn, settings, lang)
    nodes, edges = [], []
    if notes is not None:
        nodes = [GraphNode(**note, pagerank=float(p)) for note, p in zip(notes.notes, notes.pagerank)]
        for i, source in enumerate(notes.notes):
            for j, target in enumerate(notes.notes):
                if i != j:
                    edges.append(
                        GraphEdge(
                            source=source["slug"],
                            target=target["slug"],
                            probability=float(notes.transitions[i, j]),
                            prior=float(notes.prior[i, j]),
                            observed=float(notes.counts[i, j]),
                        )
                    )
    return GraphOut(
        nodes=nodes,
        edges=edges,
        strength=settings.markov_lambda,
        temperature=settings.markov_temperature,
        restart_probability=settings.restart_probability,
        damping=settings.pagerank_damping,
        finished_weight=settings.finished_weight,
        intent_stickiness=settings.intent_stickiness,
        intent_temperature=settings.intent_temperature,
    )
