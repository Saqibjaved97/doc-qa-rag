# Document Q&A with Retrieval (RAG)

Ask questions in plain language about a set of markdown documents and get a short answer with its
sources. The project builds and evaluates the **retrieval** half of a retrieval-augmented
generation (RAG) system: chunking, three retrievers (TF-IDF, a from-scratch BM25, and a hybrid),
and a labelled question set to measure them. Answers are extractive by default; set an OpenAI API
key to have an LLM write the answer from the retrieved sources, with citations.

```
$ python -m rag.ask --no-llm "Can I print from home?"
Printing to office printers from home is not possible. Save the document and print it at the office.

Sources:
  printer.md > Printing from home  (score 0.033)
  printer.md > Printing  (score 0.032)
  home-office.md > Home office equipment  (score 0.031)
```

## Why

An LLM can only answer from the context it is given, so in RAG the retriever decides most of the
answer quality. If the right passage is not in the top results, no prompt will fix it. This project
measures that step instead of assuming it works.

## How it works

- **Knowledge base** (`kb/`): 16 short IT-helpdesk articles for a fictional company (VPN,
  passwords, MFA, printing, onboarding, phishing, ...). Written for this project.
- **Chunking** (`rag/chunk.py`): one chunk per markdown section; long sections are split into
  120-word windows with 20 words overlap. The document title and section heading are indexed with
  the text. 64 chunks in total.
- **Retrievers** (`rag/retrieve.py`):
  - TF-IDF with word 1–2-grams and cosine similarity (scikit-learn)
  - Okapi BM25 (k1 = 1.5, b = 0.75), implemented in NumPy
  - Hybrid: reciprocal rank fusion of the two
- **Evaluation** (`rag/evaluate.py`, `eval/questions.csv`): 48 questions, each labelled with the
  section that answers it. Many are phrased in different words than the article
  ("I am getting a new mobile..." for a section titled "New phone").
- **Answering** (`rag/answer.py`): extractive (best-matching sentences plus sources), or an LLM
  prompted to answer only from the numbered sources and cite them.

## Results

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| TF-IDF | 0.73 | 0.88 | 0.90 | 0.81 |
| **BM25** | **0.79** | **0.90** | **0.92** | **0.85** |
| Hybrid (TF-IDF + BM25) | 0.75 | 0.90 | 0.92 | 0.83 |

- **BM25 is the best single retriever here**: the right section is first for 79% of questions and
  in the top 3 for 90%. With three chunks passed to the LLM, about 1 question in 10 cannot be
  answered correctly no matter how good the model is.
- **The hybrid does not help on this data**, because both retrievers are keyword-based and fail on
  the same questions. Fusion pays off when the methods make different mistakes.
- **The misses are vocabulary mismatches**: "mobile" vs "phone", "wireless" vs "Wi-Fi", "remote
  connection" vs "VPN", "departing employee" vs "leaver". This is the known limit of lexical
  search and the reason production RAG systems add dense embeddings.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m rag.evaluate                                  # table above + list of misses
python -m rag.ask "How long are backups kept?"
python -m rag.ask --kb path/to/your/docs "your question"    # any folder of .md files
pytest -q                                               # 10 tests
```

LLM answers (optional):

```bash
pip install openai
export OPENAI_API_KEY=...          # Windows: set OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini    # any chat model you have access to
python -m rag.ask "I clicked a link in a suspicious email. What now?"
```

## Project layout

```
kb/                   knowledge base (markdown)
eval/questions.csv    48 questions with the correct document and section
rag/
  text.py             tokeniser, stop words, light stemming
  chunk.py            markdown chunking
  retrieve.py         TF-IDF, BM25, hybrid
  evaluate.py         recall@k and MRR per retriever
  answer.py           extractive answer or LLM answer with citations
  ask.py              command line
tests/                10 tests, including the LLM path with a fake client
```

## Ideas for next steps

- Add a dense retriever (e.g. `sentence-transformers` embeddings) and check whether the hybrid
  then closes the vocabulary gap.
- Add a cross-encoder re-ranker on the top 20 chunks.
- Evaluate answer quality, not only retrieval: does the LLM answer stay faithful to the sources?

## License

MIT
