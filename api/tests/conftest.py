import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://portfolio:portfolio@localhost:5433/portfolio_test"
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["EMBEDDING_BACKEND"] = "local"
os.environ["ADMIN_TOKEN"] = ADMIN_TOKEN = "test-token"

FIXTURES = Path(__file__).parent / "fixtures" / "content"

TOPICS = [
    {"plantas", "tomate", "sol", "riego", "compost", "suelo", "jardín", "raíces", "verano"},
    {"pan", "harina", "horno", "hornear", "sopa", "verduras", "cebolla", "levadura", "zanahoria", "cocina"},
    {"api", "fastapi", "postgres", "vectores", "arquitectura", "proyecto", "ejemplo"},
]


class TopicEncoder:
    """Deterministic stand-in for the model: one dimension per topic plus a little per-word noise.
    Text with no topic words lands equally close to every topic, like an ambiguous query."""

    calls = 0

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        TopicEncoder.calls += len(texts)
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in zip(vectors, texts):
            for word in re.findall(r"\w+", text.lower()):
                for topic, words in enumerate(TOPICS):
                    row[topic] += word in words
                row[3 + int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 0.05
            if not row[:3].any():
                row[:3] = 1.0
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def content_payload(**extra) -> dict:
    """The fixture notes in the shape the frontend sends them."""
    kinds = {"proyectos": "project", "skills": "skill"}
    notes = [
        {"slug": path.stem, "kind": kinds[path.parent.name], "source": path.read_text()}
        for folder in kinds
        for path in sorted((FIXTURES / folder).glob("*.md"))
    ]
    return {"notes": notes, "intents": json.loads((FIXTURES / "intents.json").read_text()), **extra}


def sync(client, **extra):
    return client.put(
        "/admin/content", json=content_payload(**extra), headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
    )


@pytest.fixture(scope="session")
def client():
    import psycopg
    from fastapi.testclient import TestClient

    try:
        psycopg.connect(TEST_DATABASE_URL, connect_timeout=2).close()
    except psycopg.OperationalError:
        pytest.skip(f"Postgres de prueba no disponible en {TEST_DATABASE_URL}")

    from app.embeddings import get_encoder
    from app.main import app

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("drop table if exists events, sessions, intents, chunks, notes cascade")

    app.dependency_overrides[get_encoder] = TopicEncoder
    with TestClient(app) as test_client:
        response = sync(test_client)
        assert response.status_code == 200, response.text
        yield test_client
    app.dependency_overrides.clear()
