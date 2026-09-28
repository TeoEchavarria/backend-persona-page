import secrets

import yaml
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
    notes = []
    for note in content.notes:
        try:
            notes.append(parse_source(note.slug, note.kind, note.source, note.lang))
        except yaml.YAMLError as error:
            raise HTTPException(422, f"Frontmatter inválido en {note.lang}/{note.slug}: {error}") from error
    keys = [(note.slug, note.lang) for note in notes]
    if len(keys) != len(set(keys)):
        raise HTTPException(422, "Hay notas con el mismo slug en el mismo idioma")
    intents = [intent.model_dump() for intent in content.intents]

    # The first sync against an empty database creates the schema (and the vector extension).
    db.init_schema()
    with db.connect() as conn, embedding_errors():
        report = ingest.sync(conn, encoder, notes, intents, content.force)
        db.purge_old_sessions(conn, get_settings().event_retention_days)
    return SyncResult(**report.__dict__)
