from tests.conftest import TopicEncoder, content_payload, sync


def search(client, query: str, **body) -> dict:
    response = client.post("/search", json={"query": query, **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "database": "ok"}


def test_syncing_the_same_content_embeds_nothing(client):
    before = TopicEncoder.calls
    report = sync(client).json()
    assert report == {"notes": 3, "chunks": 8, "embedded": 0, "removed_chunks": 0, "removed_notes": 0}
    assert TopicEncoder.calls == before


def test_sync_requires_the_admin_token(client):
    assert client.put("/admin/content", json=content_payload()).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.put("/admin/content", json=content_payload(), headers=wrong).status_code == 401


def test_sync_updates_edits_and_removals(client):
    payload = content_payload()
    cocina = next(note for note in payload["notes"] if note["slug"] == "cocina")
    cocina["source"] = cocina["source"].replace("fuego lento", "fuego muy lento")
    payload["notes"] = [note for note in payload["notes"] if note["slug"] != "jardin"]
    headers = {"Authorization": "Bearer test-token"}

    report = client.put("/admin/content", json=payload, headers=headers).json()
    assert report["embedded"] == 1
    assert report["removed_notes"] == 1

    assert sync(client).json()["embedded"] == 3  # jardín's two paragraphs and the original soup


def test_search_returns_paragraph_note_and_score(client):
    body = search(client, "¿cuánto sol necesita el tomate?")

    top = body["results"][0]
    assert top["note"] == {"slug": "jardin", "title": "Jardín", "kind": "project"}
    assert "tomate" in top["text"]
    assert top["headings"] == ["Plantas"]
    assert top["score"] > 0 and top["vector_rank"] == 1
    assert body["confidence"]["band"] in {"high", "medium", "far"}
    assert len(body["results"]) <= 5


def test_kind_filter(client):
    body = search(client, "horno pan", kind="project")
    assert body["results"]
    assert all(result["note"]["kind"] == "project" for result in body["results"])
