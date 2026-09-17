import secrets

from fastapi import APIRouter, Depends, Header, HTTPException

from app import db, ingest
from app.chunking import parse_source
from app.config import Settings, get_settings
from app.embeddings import Encoder, get_encoder
from app.routes.errors import embedding_errors
from app.schemas import ContentSync, SyncResult

router = APIRouter(prefix="/admin")


def require_admin(authorization: str = Header(default=""), settings: Settings = Depends(get_settings)) -> None:
    if not settings.admin_token:
        raise HTTPException(403, "La sincronización está deshabilitada: falta ADMIN_TOKEN")
    if not secrets.compare_digest(authorization.encode(), f"Bearer {settings.admin_token}".encode()):
        raise HTTPException(401, "Token inválido")


@router.put("/content", response_model=SyncResult, dependencies=[Depends(require_admin)])
def sync_content(content: ContentSync, encoder: Encoder = Depends(get_encoder)) -> SyncResult:
    """Replace the whole content with what the frontend sends. Only new or edited paragraphs are embedded."""
    notes = [parse_source(note.slug, note.kind, note.source) for note in content.notes]
    slugs = [note.slug for note in notes]
    if len(slugs) != len(set(slugs)):
        raise HTTPException(422, "Hay notas con el mismo slug")

    # The first sync against an empty database creates the schema (and the vector extension).
    db.init_schema()
    with db.connect() as conn, embedding_errors():
        report = ingest.sync(conn, encoder, notes, content.force)
        db.purge_old_sessions(conn, get_settings().event_retention_days)
    return SyncResult(**report.__dict__)
