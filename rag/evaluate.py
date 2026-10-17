"""Measure how often each retriever puts the right section near the top.

    python -m rag.evaluate
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .chunk import load_folder
from .retrieve import RETRIEVERS

ROOT = Path(__file__).resolve().parent.parent


def load_questions(path=ROOT / "eval" / "questions.csv") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(kb=ROOT / "kb", questions=None, ks=(1, 3, 5)) -> dict:
    chunks = load_folder(kb)
    questions = questions or load_questions()
    results = {}
    for key, cls in RETRIEVERS.items():
        retriever = cls(chunks)
        ranks = []          # 1-based rank of the first correct section, or None
        for q in questions:
            hits = retriever.search(q["question"], k=len(chunks))
            rank = next((i for i, (c, _) in enumerate(hits, 1) if c.doc == q["doc"] and c.section == q["section"]), None)
            ranks.append(rank)
        row = {f"recall@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in ks}
        row["mrr"] = float(np.mean([1 / r if r else 0 for r in ranks]))
        row["misses"] = [q["question"] for q, r in zip(questions, ranks) if not r or r > 3]
        results[cls.name] = row
    return {"n_chunks": len(chunks), "n_questions": len(questions), "retrievers": results}


def main():
    r = evaluate()
    print(f"{r['n_questions']} questions over {r['n_chunks']} chunks")
    print(f"{'retriever':<26}{'recall@1':>10}{'recall@3':>10}{'recall@5':>10}{'MRR':>8}")
    for name, row in r["retrievers"].items():
        print(f"{name:<26}{row['recall@1']:>10.3f}{row['recall@3']:>10.3f}{row['recall@5']:>10.3f}{row['mrr']:>8.3f}")
    best = max(r["retrievers"], key=lambda n: r["retrievers"][n]["mrr"])
    print(f"\nQuestions where {best} has the right section outside the top 3:")
    for q in r["retrievers"][best]["misses"]:
        print("  -", q)


if __name__ == "__main__":
    main()
