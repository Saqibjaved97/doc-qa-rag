"""Split markdown documents into retrievable chunks, one per section (long sections are split further)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    doc: str        # file name
    title: str      # document title (first # heading)
    section: str    # section heading
    text: str       # section body

    @property
    def source(self) -> str:
        return f"{self.doc} > {self.section}"

    @property
    def indexed_text(self) -> str:
        """What the retrievers see: title and heading carry a lot of meaning, so they are included."""
        return f"{self.title}. {self.section}. {self.text}"


def split_markdown(doc: str, markdown: str, max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    title, section, body, chunks = doc, doc, [], []

    def flush():
        words = " ".join(body).split()
        if not words:
            return
        step = max_words - overlap
        for start in range(0, max(len(words) - overlap, 1), step):
            chunks.append(Chunk(doc, title, section, " ".join(words[start:start + max_words])))

    for line in markdown.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            body = []
            if len(m.group(1)) == 1:
                title = m.group(2).strip()
            section = m.group(2).strip()
        elif line.strip():
            body.append(line.strip())
    flush()
    return chunks


def load_folder(folder: str | Path, **kw) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        chunks.extend(split_markdown(path.name, path.read_text(encoding="utf-8"), **kw))
    if not chunks:
        raise ValueError(f"no markdown files in {folder}")
    return chunks
