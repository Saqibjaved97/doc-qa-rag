import tempfile
import unittest
from pathlib import Path

from tickets.model import TicketClassifier
from tickets.train import train


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "model.joblib"
        cls.results = train(n=1500, seed=1, model_path=cls.path, report_dir=cls.tmp.name, plot=False)
        cls.model = TicketClassifier.load(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_beats_baseline_on_unseen_wording(self):
        self.assertGreater(self.results["category"]["accuracy"], 0.8)
        cv = self.results["cv_macro_f1"]["category"]
        self.assertGreater(cv["Logistic Regression"], cv["Majority class (baseline)"] + 0.5)

    def test_prediction_format(self):
        p = self.model.predict("I was charged twice for my subscription.")
        self.assertEqual(set(p), {"category", "category_confidence", "priority", "priority_confidence", "needs_review"})
        self.assertTrue(0 <= p["category_confidence"] <= 1)

    def test_english_and_german_examples(self):
        self.assertEqual(self.model.predict("Hi, my laptop does not turn on anymore. Thanks")["category"], "hardware")
        self.assertEqual(self.model.predict("Hallo, die Rechnung hat den falschen Betrag. Danke")["category"], "billing")
        self.assertEqual(self.model.predict("Das VPN trennt sich ständig, alle Nutzer sind betroffen.")["category"], "network")


if __name__ == "__main__":
    unittest.main()
