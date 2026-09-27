import logging
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


@contextmanager
def embedding_errors() -> Iterator[None]:
    try:
        yield
    except httpx.HTTPError as error:
        detail = error.response.text[:300] if isinstance(error, httpx.HTTPStatusError) else ""
        logger.error("Embeddings request failed: %s %s", error, detail)
        raise HTTPException(503, "El servicio de embeddings no respondió. Intenta de nuevo en unos segundos.") from error
