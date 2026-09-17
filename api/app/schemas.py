from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

Kind = Literal["project", "skill"]
Band = Literal["high", "medium", "far"]


class NoteRef(BaseModel):
    slug: str
    title: str
    kind: Kind


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    session_id: UUID | None = None
    kind: Kind | None = None
    use_context: bool = True


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    headings: list[str]
    note: NoteRef
    score: float
    similarity: float
    vector_rank: int | None
    text_rank: int | None


class ConfidenceOut(BaseModel):
    band: Band
    message: str
    margin: float
    z: float
    gap: float
    top_similarity: float
    mean_similarity: float
    margin_high: float
    margin_medium: float
    gap_high: float


class ContextOut(BaseModel):
    available: bool
    applied: bool
    note: NoteRef | None = None
    alpha: float
    beta: float
    query_alignment: float | None = None


class SearchResponse(BaseModel):
    session_id: UUID
    query: str
    results: list[SearchResult]
    confidence: ConfidenceOut
    context: ContextOut
    rrf_k: int


class EventRequest(BaseModel):
    session_id: UUID
    kind: Literal["read", "select", "finish"]
    chunk_id: str | None = Field(default=None, max_length=64)
    note_slug: str | None = Field(default=None, max_length=120)


class SessionState(BaseModel):
    session_id: UUID
    context: ContextOut


class ChunkOut(BaseModel):
    id: str
    position: int
    headings: list[str]
    text: str


class NoteSummary(NoteRef):
    tags: list[str]
    summary: str | None
    link: str | None


class NoteOut(NoteSummary):
    chunks: list[ChunkOut]


class NoteSource(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9-]+$", max_length=120)
    kind: Kind
    source: str = Field(max_length=200_000)


class ContentSync(BaseModel):
    notes: list[NoteSource] = Field(max_length=2_000)
    force: bool = False


class SyncResult(BaseModel):
    notes: int
    chunks: int
    embedded: int
    removed_chunks: int
    removed_notes: int
