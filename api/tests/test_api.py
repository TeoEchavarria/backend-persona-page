from tests.conftest import TopicEncoder, content_payload, sync


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
