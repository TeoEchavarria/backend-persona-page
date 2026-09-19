import os
import threading
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings, get_settings


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class Encoder(Protocol):
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def encode_passages(self, texts: list[str]) -> np.ndarray: ...


class _PrefixedEncoder:
    """e5-style models expect 'query: ' / 'passage: ' prefixes; both backends apply them the same way."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.settings.query_prefix + t for t in texts])

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.settings.passage_prefix + t for t in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class LocalEncoder(_PrefixedEncoder):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class HttpEncoder(_PrefixedEncoder):
    """Hugging Face feature-extraction API (serverless or a dedicated endpoint)."""

    def _encode(self, texts: list[str]) -> np.ndarray:
        url = self.settings.embedding_api_url.format(model=self.settings.embedding_model)
        headers = {"Authorization": f"Bearer {self.settings.embedding_api_token}"}
        response = httpx.post(url, json={"inputs": texts}, headers=headers, timeout=20)
        response.raise_for_status()
        vectors = np.asarray(response.json(), dtype=np.float32)
        if vectors.ndim == 3:  # token embeddings: mean-pool them
            vectors = vectors.mean(axis=1)
        return normalize(vectors)


_encoder: Encoder | None = None
_encoder_lock = threading.Lock()


def get_encoder() -> Encoder:
    """One encoder per process; requests run in a thread pool, so creation is locked."""
    global _encoder
    with _encoder_lock:
        if _encoder is None:
            settings = get_settings()
            if settings.embedding_backend == "http":
                _encoder = HttpEncoder(settings)
            else:
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                _encoder = LocalEncoder(settings)
        return _encoder
