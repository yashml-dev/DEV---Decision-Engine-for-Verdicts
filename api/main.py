"""Run:  uvicorn api.main:app --reload"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dev.schema import load_schema
from dev.predict import Predictor

app = FastAPI(title="DEV - Decision Engine for Verdicts", version="0.1.0")
schema = load_schema()
predictor = None


class DecideRequest(BaseModel):
    state: dict | str
    questions: list[str] | None = None


@app.on_event("startup")
def _load():
    global predictor
    path = os.getenv("DEV_CHECKPOINT", "checkpoints/dev-v0.1.pt")
    if os.path.exists(path):
        predictor = Predictor(schema, path, float(os.getenv("DEV_THRESHOLD", "0.85")))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None, "questions": list(schema)}


@app.post("/decide")
def decide(req: DecideRequest):
    if predictor is None:
        raise HTTPException(503, "No checkpoint found. Train first: python -m dev.train")
    unknown = [q for q in (req.questions or []) if q not in schema]
    if unknown:
        raise HTTPException(422, f"Unknown questions: {unknown}")
    text = req.state if isinstance(req.state, str) else " ".join(str(v) for v in req.state.values())
    return predictor.decide(text, req.questions)
