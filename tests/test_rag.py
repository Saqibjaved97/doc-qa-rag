import sys
import types
import unittest

from rag.answer import answer, build_prompt, extractive_answer
from rag.chunk import Chunk, load_folder, split_markdown
from rag.evaluate import ROOT, evaluate
from rag.retrieve import BM25Retriever, HybridRetriever, TfidfRetriever
from rag.text import stem, tokenize

DOC = """# Printing

Intro text about printers.

## Paper jam
Open the tray and remove the paper.

## Toner
Toner is in the cabinet.
"""


class TextTests(unittest.TestCase):
    def test_tokenize_removes_stopwords_and_stems(self):
        self.assertEqual(tokenize("How do I reset the printers?"), ["reset", "printer"])
        self.assertEqual(stem("policies"), "policy")


class ChunkTests(unittest.TestCase):
    def test_one_chunk_per_section(self):
        chunks = split_markdown("printer.md", DOC)
        self.assertEqual([c.section for c in chunks], ["Printing", "Paper jam", "Toner"])
        self.assertTrue(all(c.title == "Printing" for c in chunks))

    def test_long_sections_are_split_with_overlap(self):
        body = " ".join(f"w{i}" for i in range(300))
        chunks = split_markdown("long.md", f"# Long\n{body}", max_words=120, overlap=20)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[1].text.split()[0], "w100")

    def test_knowledge_base_loads(self):
        self.assertGreater(len(load_folder(ROOT / "kb")), 40)


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_folder(ROOT / "kb")

    def test_each_retriever_finds_exact_match(self):
        for cls_ in (TfidfRetriever, BM25Retriever, HybridRetriever):
            top = cls_(self.chunks).search("What does VPN error 812 mean?", k=1)[0][0]
            self.assertEqual((top.doc, top.section), ("vpn-setup.md", "Common problems"), cls_.name)

    def test_evaluation_quality(self):
        r = evaluate()
        self.assertEqual(r["n_questions"], 48)
        self.assertGreaterEqual(r["retrievers"]["BM25"]["recall@5"], 0.85)


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.hits = [(Chunk("printer.md", "Printing", "Toner", "Toner is in the cabinet. Paper is too."), 1.0)]

    def test_extractive_answer_has_source(self):
        a = extractive_answer("Where is the toner?", self.hits)
        self.assertIn("Toner is in the cabinet.", a["answer"])
        self.assertEqual(a["sources"], ["printer.md > Toner"])

    def test_no_match_says_so(self):
        a = extractive_answer("capital of France", self.hits)
        self.assertEqual(a["sources"], [])

    def test_prompt_numbers_sources(self):
        self.assertIn("[1] (printer.md > Toner)", build_prompt("toner?", self.hits))

    def test_llm_path_with_fake_client(self):
        calls = {}

        class FakeOpenAI:
            def __init__(self):
                self.chat = types.SimpleNamespace(completions=self)

            def create(self, **kw):
                calls.update(kw)
                msg = types.SimpleNamespace(content=" In the cabinet [1]. ")
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

        sys.modules["openai"] = types.SimpleNamespace(OpenAI=FakeOpenAI)
        try:
            a = answer("Where is the toner?", self.hits, use_llm=True)
        finally:
            del sys.modules["openai"]
        self.assertEqual(a["answer"], "In the cabinet [1].")
        self.assertEqual(calls["messages"][0]["role"], "system")
        self.assertIn("Where is the toner?", calls["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
