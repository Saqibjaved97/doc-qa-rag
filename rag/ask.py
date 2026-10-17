"""Ask a question about the knowledge base.

    python -m rag.ask "How do I reset my password?"
    python -m rag.ask --retriever bm25 --k 3 "Can I print from home?"
    python -m rag.ask --kb path/to/your/markdown/folder "your question"
"""
from __future__ import annotations

import argparse

from .answer import answer
from .chunk import load_folder
from .evaluate import ROOT
from .retrieve import RETRIEVERS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--kb", default=str(ROOT / "kb"), help="folder with markdown files")
    p.add_argument("--retriever", choices=list(RETRIEVERS), default="hybrid")
    p.add_argument("--k", type=int, default=3, help="number of chunks to use")
    p.add_argument("--no-llm", action="store_true", help="always use the extractive answer")
    a = p.parse_args()

    retriever = RETRIEVERS[a.retriever](load_folder(a.kb))
    hits = retriever.search(a.question, k=a.k)
    result = answer(a.question, hits, use_llm=False if a.no_llm else None)
    print(result["answer"])
    print("\nSources:")
    for c, score in hits:
        print(f"  {c.source}  (score {score:.3f})")


if __name__ == "__main__":
    main()
