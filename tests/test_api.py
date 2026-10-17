import importlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

HAVE_FASTAPI = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None


@unittest.skipUnless(HAVE_FASTAPI, "fastapi and httpx are not installed")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from tickets.train import train

        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "model.joblib"
        train(n=1200, seed=2, model_path=path, report_dir=cls.tmp.name, plot=False)
        os.environ["MODEL_PATH"] = str(path)
        import app.main as main
        main = importlib.reload(main)          # pick up MODEL_PATH
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_predict(self):
        r = self.client.post("/predict", json={"text": "My laptop does not turn on. Urgent, the whole team is blocked."})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["category"], "hardware")
        self.assertIn(body["priority"], {"low", "medium", "high"})

    def test_rejects_empty_text(self):
        self.assertEqual(self.client.post("/predict", json={"text": ""}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
