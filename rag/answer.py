"""Turn retrieved chunks into an answer with sources.

Without an API key the answer is extractive: the sentences from the retrieved chunks that best
match the question, each with its source. With OPENAI_API_KEY set (and `pip install openai`), the
chunks are sent to an LLM with instructions to answer only from them and to cite the sources.
"""
from __future__ import annotations

import os
import re

from .chunk import Chunk
from .text import tokenize

SYSTEM_PROMPT = (
    "You answer questions about the company IT helpdesk using ONLY the numbered sources. "
    "Cite sources like [1]. If the sources do not contain the answer, say you don't know and suggest "
    "contacting the service desk. Answer in 1-3 sentences."
)


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    sources = "\n\n".join(f"[{i}] ({c.source})\n{c.text}" for i, (c, _) in enumerate(hits, 1))
    return f"Sources:\n{sources}\n\nQuestion: {question}"


def extractive_answer(question: str, hits: list[tuple[Chunk, float]], max_sentences: int = 2) -> dict:
    q = set(tokenize(question))
    candidates = []
    for rank, (chunk, _) in enumerate(hits):
        for sent in re.split(r"(?<=[.!?])\s+", chunk.text):
            overlap = len(q & set(tokenize(sent)))
            if overlap:
                candidates.append((overlap - 0.3 * rank, sent.strip(), chunk.source))
    candidates.sort(key=lambda x: -x[0])
    picked = candidates[:max_sentences]
    if not picked:
        return {"answer": "I could not find this in the knowledge base. Please contact the service desk.", "sources": []}
    return {"answer": " ".join(s for _, s, _ in picked), "sources": sorted({src for *_, src in picked})}


def llm_answer(question: str, hits: list[tuple[Chunk, float]]) -> dict:
    from openai import OpenAI   # optional dependency
    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_prompt(question, hits)}],
    )
    return {"answer": resp.choices[0].message.content.strip(), "sources": [c.source for c, _ in hits]}


def answer(question: str, hits: list[tuple[Chunk, float]], use_llm: bool | None = None) -> dict:
    if use_llm is None:
        use_llm = bool(os.getenv("OPENAI_API_KEY"))
    return llm_answer(question, hits) if use_llm else extractive_answer(question, hits)
