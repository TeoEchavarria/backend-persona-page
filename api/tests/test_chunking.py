import datetime as dt
from pathlib import Path

from app.chunking import chunk_id, load_notes, parse_note, with_context

FIXTURES = Path(__file__).parent / "fixtures" / "content"


def test_note_produces_expected_chunks():
    note = parse_note(FIXTURES / "proyectos" / "ejemplo.md")

    assert note.slug == "ejemplo"
    assert note.kind == "project"
    assert note.tags == ["prueba"]
    assert note.published == dt.date(2026, 1, 10)
    assert [(c.headings, c.text) for c in note.chunks] == [
        ((), "Párrafo inicial sin sección."),
        (
            ("Arquitectura",),
            "La API usa FastAPI y guarda vectores en Postgres. Esta línea sigue el mismo párrafo.",
        ),
        (("Arquitectura", "Detalles"), "- primer punto\n- segundo punto con un enlace"),
        (("Resultados",), "1. uno\n2. dos"),
    ]
    assert [c.position for c in note.chunks] == [0, 1, 2, 3]


def test_context_prefix_keeps_title_and_nearest_headings():
    assert with_context("Nota", ("A", "B", "C"), "texto") == "Nota > B > C > texto"
    assert with_context("Nota", (), "texto") == "Nota > texto"

    note = parse_note(FIXTURES / "proyectos" / "ejemplo.md")
    assert note.chunks[2].text_with_context.startswith("Proyecto de ejemplo > Arquitectura > Detalles > - primer")


def test_ids_are_stable_hashes_of_the_contextual_text():
    first = parse_note(FIXTURES / "proyectos" / "ejemplo.md")
    second = parse_note(FIXTURES / "proyectos" / "ejemplo.md")

    assert [c.id for c in first.chunks] == [c.id for c in second.chunks]
    assert first.chunks[1].id == chunk_id(first.chunks[1].text_with_context)
    assert len({c.id for c in first.chunks}) == len(first.chunks)


def test_kind_comes_from_the_folder():
    kinds = {note.slug: note.kind for note in load_notes(FIXTURES)}
    assert kinds == {"ejemplo": "project", "jardin": "project", "cocina": "skill"}
