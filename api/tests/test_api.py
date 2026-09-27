from uuid import uuid4

import pytest

from tests.conftest import TopicEncoder, content_payload, sync


def search(client, query: str, **body) -> dict:
    response = client.post("/search", json={"query": query, **body})
    assert response.status_code == 200, response.text
    return response.json()


def read(client, session_id: str, chunk_id: str, kind: str = "read") -> dict:
    response = client.post("/events", json={"session_id": session_id, "kind": kind, "chunk_id": chunk_id})
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "database": "ok", "notes": 3}


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
    assert client.get("/notes/jardin").status_code == 404

    assert sync(client).json()["embedded"] == 3  # jardín's two paragraphs and the original soup


def test_search_returns_paragraph_note_and_score(client):
    body = search(client, "¿cuánto sol necesita el tomate?")

    top = body["results"][0]
    assert top["note"] == {"slug": "jardin", "title": "Jardín", "kind": "project"}
    assert "tomate" in top["text"]
    assert top["headings"] == ["Plantas"]
    assert top["score"] > 0 and top["vector_rank"] == 1
    assert body["confidence"]["band"] in {"high", "medium", "far"}
    assert body["context"]["applied"] is False
    assert len(body["results"]) <= 5


def test_kind_filter(client):
    body = search(client, "horno pan", kind="project")
    assert body["results"]
    assert all(result["note"]["kind"] == "project" for result in body["results"])


def test_reading_builds_context_that_tilts_ambiguous_queries(client):
    ambiguous = "¿qué me recomiendas?"
    session_id = search(client, ambiguous)["session_id"]
    note = client.get("/notes/cocina").json()

    for chunk in note["chunks"]:
        state = read(client, session_id, chunk["id"], kind="select")
    assert state["context"]["available"] is True
    assert state["context"]["note"]["slug"] == "cocina"

    with_context = search(client, ambiguous, session_id=session_id)
    assert with_context["context"]["applied"] is True
    assert with_context["results"][0]["note"]["slug"] == "cocina"

    without_context = search(client, ambiguous, session_id=session_id, use_context=False)
    assert without_context["context"]["applied"] is False
    assert without_context["context"]["available"] is True

    cleared = client.delete(f"/sessions/{session_id}/context").json()
    assert cleared["context"]["available"] is False


def test_intent_belief_follows_what_is_read(client):
    session_id = str(uuid4())
    jardin = client.get("/notes/jardin").json()
    for chunk in jardin["chunks"]:
        state = read(client, session_id, chunk["id"])

    intents = {intent["id"]: intent["probability"] for intent in state["intents"]}
    assert sum(intents.values()) == pytest.approx(1.0)
    assert intents["casa"] > intents["tecnica"]


def test_note_view_and_unknown_note(client):
    note = client.get("/notes/ejemplo").json()
    assert note["title"] == "Proyecto de ejemplo"
    assert [chunk["position"] for chunk in note["chunks"]] == [0, 1, 2, 3]
    assert client.get("/notes/no-existe").status_code == 404


def test_events_validate_their_target(client):
    session_id = str(uuid4())
    assert client.post("/events", json={"session_id": session_id, "kind": "read"}).status_code == 422
    assert (
        client.post("/events", json={"session_id": session_id, "kind": "read", "chunk_id": "nope"}).status_code == 404
    )
    finish = client.post("/events", json={"session_id": session_id, "kind": "finish", "note_slug": "jardin"})
    assert finish.status_code == 200


def test_related_notes_and_graph(client):
    related = client.get("/notes/jardin/related", params={"k": 2}).json()
    assert len(related) == 2
    assert "jardin" not in {note["slug"] for note in related}

    graph = client.get("/graph").json()
    assert {node["slug"] for node in graph["nodes"]} == {"ejemplo", "jardin", "cocina"}
    outgoing = sum(edge["probability"] for edge in graph["edges"] if edge["source"] == "jardin")
    assert outgoing == pytest.approx(1.0)
    assert sum(node["pagerank"] for node in graph["nodes"]) == pytest.approx(1.0)


def test_notes_are_listed_most_central_first(client):
    notes = client.get("/notes").json()
    centralities = [note["centrality"] for note in notes]
    assert len(notes) == 3
    assert centralities == sorted(centralities, reverse=True)
