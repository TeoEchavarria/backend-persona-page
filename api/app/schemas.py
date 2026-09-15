from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal["project", "skill"]


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
