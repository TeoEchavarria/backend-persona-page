from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from fastapi import HTTPException


@contextmanager
def embedding_errors() -> Iterator[None]:
    try:
        yield
    except httpx.HTTPError as error:
        raise HTTPException(503, "El servicio de embeddings no respondió. Intenta de nuevo en unos segundos.") from error
