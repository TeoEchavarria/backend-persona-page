import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
from markdown_it import MarkdownIt
from markdown_it.token import Token

KIND_BY_FOLDER = {"proyectos": "project", "skills": "skill", "anotaciones": "note", "experiencia": "experience"}
CONTEXT_DEPTH = 2

_markdown = MarkdownIt("commonmark")


@dataclass(frozen=True)
class Chunk:
    id: str
    position: int
    headings: tuple[str, ...]
    text: str
    text_with_context: str


@dataclass
class Note:
    slug: str
    kind: str
    title: str
    tags: list[str]
    summary: str | None
    link: str | None
    published: dt.date | None = None
    lang: str = "es"
    chunks: list[Chunk] = field(default_factory=list)


def load_notes(content_dir: Path) -> list[Note]:
    return [
        parse_note(path)
        for folder in KIND_BY_FOLDER
        for path in sorted((content_dir / folder).glob("*.md"))
    ]


def parse_note(path: Path) -> Note:
    return parse_source(path.stem, KIND_BY_FOLDER[path.parent.name], path.read_text())


def parse_source(slug: str, kind: str, source: str, lang: str = "es") -> Note:
    """A markdown file with frontmatter (title, tags, summary, link, date); frontmatter may override slug and kind."""
    post = frontmatter.loads(source)
    title = post.get("title") or slug
    note = Note(
        slug=post.get("slug") or slug,
        kind=post.get("kind") or kind,
        title=title,
        tags=list(post.get("tags") or []),
        summary=post.get("summary"),
        link=post.get("link"),
        published=_as_date(post.get("date")),
        lang=lang,
    )
    note.chunks = chunk_markdown(post.content, title, lang)
    return note


def _as_date(value) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)) if value else None


def chunk_markdown(markdown: str, title: str, lang: str = "es") -> list[Chunk]:
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []
    tokens = _markdown.parse(markdown)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "heading_open":
            level = int(token.tag[1])
            headings = [h for h in headings if h[0] < level] + [(level, tokens[i + 1].content.strip())]
            i = _skip_block(tokens, i)
            continue
        if (token.level == 0 and token.nesting == 1) or token.type in {"fence", "code_block"}:
            end = _skip_block(tokens, i)
            text = _plain_text(tokens[i:end]).strip()
            if text:
                path = tuple(h[1] for h in headings)
                chunks.append(_make_chunk(len(chunks), title, path, text, lang))
            i = end
            continue
        i += 1
    return chunks


def with_context(title: str, headings: tuple[str, ...], text: str) -> str:
    """'Proyecto > Sección > párrafo', keeping only the nearest headings so the prefix stays short."""
    return " > ".join([title, *headings[-CONTEXT_DEPTH:], text])


def chunk_id(text_with_context: str, lang: str = "es") -> str:
    """Stable id: a hash of the contextual text and its language (a paragraph can read the same in both)."""
    return hashlib.sha256(f"{lang}:{text_with_context}".encode()).hexdigest()[:20]


def _make_chunk(position: int, title: str, headings: tuple[str, ...], text: str, lang: str) -> Chunk:
    contextual = with_context(title, headings, text)
    return Chunk(chunk_id(contextual, lang), position, headings, text, contextual)


def _skip_block(tokens: list[Token], start: int) -> int:
    """Index right after the block that opens at `start`."""
    depth = 0
    for i in range(start, len(tokens)):
        depth += tokens[i].nesting
        if depth <= 0:
            return i + 1
    return len(tokens)


def _plain_text(block: list[Token]) -> str:
    lines: list[str] = []
    bullet = ""
    for token in block:
        if token.type == "list_item_open":
            bullet = f"{token.info}. " if token.info else "- "
        elif token.type == "inline":
            lines.append(bullet + "".join(_inline_text(child) for child in token.children or []))
            bullet = ""
        elif token.type in {"fence", "code_block"}:
            lines.append(token.content.rstrip("\n"))
    return "\n".join(lines)


def _inline_text(token: Token) -> str:
    if token.type in {"softbreak", "hardbreak"}:
        return " " if token.type == "softbreak" else "\n"
    if token.type in {"text", "code_inline", "image"}:
        return token.content
    return ""
