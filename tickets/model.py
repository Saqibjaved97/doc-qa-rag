"""Text classifiers for ticket category and priority.

Each target gets its own pipeline: word n-grams plus character n-grams (robust to typos and German
compound words) feeding a linear model. Logistic regression gives calibrated-enough probabilities
for a confidence score; the API uses them to route low-confidence tickets to a human.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def features() -> FeatureUnion:
    return FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])


def candidates(seed: int = 42) -> dict:
    """Models compared in training. The last one is the production choice."""
    return {
        "Majority class (baseline)": Pipeline([("f", features()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "Naive Bayes": Pipeline([("f", features()), ("clf", ComplementNB())]),
        "Linear SVM": Pipeline([("f", features()), ("clf", LinearSVC(C=0.5, random_state=seed))]),
        "Logistic Regression": Pipeline([("f", features()), ("clf", LogisticRegression(C=8, max_iter=2000))]),
    }


@dataclass
class TicketClassifier:
    category: Pipeline
    priority: Pipeline
    threshold: float = 0.5   # below this confidence the ticket should go to a human

    def predict(self, text: str) -> dict:
        out = {}
        for name, model in (("category", self.category), ("priority", self.priority)):
            proba = model.predict_proba([text])[0]
            best = proba.argmax()
            out[name] = str(model.classes_[best])
            out[f"{name}_confidence"] = round(float(proba[best]), 3)
        out["needs_review"] = out["category_confidence"] < self.threshold
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TicketClassifier":
        return joblib.load(path)
