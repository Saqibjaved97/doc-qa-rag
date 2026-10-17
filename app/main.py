"""REST API for the ticket classifier.

    uvicorn app.main:app --reload
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"text": "Mein Laptop geht nicht mehr an, bitte dringend helfen"}'
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from tickets.model import TicketClassifier

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/ticket_classifier.joblib"))

app = FastAPI(title="Support Ticket Classifier", version="1.0.0",
              description="Predicts category and priority of English and German support tickets.")
_model: TicketClassifier | None = None


def get_model() -> TicketClassifier:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():          # first start without a trained model: train one
            from tickets.train import train
            train(model_path=MODEL_PATH, plot=False)
        _model = TicketClassifier.load(MODEL_PATH)
    return _model


class Ticket(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, examples=["I was charged twice for my subscription. Urgent!"])


class Prediction(BaseModel):
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    needs_review: bool = Field(description="True when the category confidence is low; route to a human")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket) -> Prediction:
    return Prediction(**get_model().predict(ticket.text))
