from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal["project", "skill"]
Band = Literal["high", "medium", "far"]


class NoteRef(BaseModel):
    slug: str
    title: str
    kind: Kind


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    kind: Kind | None = None


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
    z: float
    gap: float
    top_similarity: float
    z_high: float
    z_medium: float
    gap_high: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    confidence: ConfidenceOut
    rrf_k: int


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
