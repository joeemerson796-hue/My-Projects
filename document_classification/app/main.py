"""FastAPI application for document classification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import DocumentClassifier

app = FastAPI(
    title="Document Classification API",
    description="Classify technical documents into categories using ML models",
    version="1.0.0",
)

classifier: DocumentClassifier | None = None


class DocumentInput(BaseModel):
    text: str = Field(..., min_length=10, example="The GPU renders 3D graphics using ray tracing.")
    backend: str = Field(default="tfidf", example="tfidf")


class ClassificationResponse(BaseModel):
    category: str
    label: int
    probabilities: dict[str, float]


@app.on_event("startup")
def load_model():
    global classifier
    classifier = DocumentClassifier(backend="tfidf")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": classifier is not None}


@app.post("/classify", response_model=ClassificationResponse)
def classify_document(doc: DocumentInput):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = classifier.predict(doc.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
