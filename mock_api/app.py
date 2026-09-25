"""
Mock ForgetMeNot API for local contract testing.

Two endpoints matter here:
  POST /api/v1/events                     -- what the worker sends (docs/EVENT_API.md)
  GET  /api/v1/patients/{id}/visitors     -- what the worker needs back
                                             (docs/VISITOR_GALLERY_API.md)

No auth. Events kept in memory + dumped to data/mock_events/events.json.
The visitors endpoint serves a locally enrolled gallery.pkl if one exists, so
the full backend-sync path can be exercised before the real API exists.

  uvicorn mock_api.app:app --host 127.0.0.1 --port 5155
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DUMP_DIR = ROOT / "data" / "mock_events"
GALLERY_PKL = ROOT / "data" / "faces" / "gallery.pkl"

EMBEDDING_MODEL = "keras-facenet-0.3.2"
EMBEDDING_DIM = 512
DEFAULT_THRESHOLD = 0.675

EventType = Literal["fall_detected", "visitor_recognized", "visitor_unknown"]
SourceType = Literal["room_cam", "door_cam"]


class EventCreate(BaseModel):
    patient_id: int = Field(gt=0)
    type: EventType
    label: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    source: SourceType


class EventOut(EventCreate):
    id: str
    created_at: str


app = FastAPI(title="FMN mock events + visitors", version="0.2.0")
_store: list[EventOut] = []


def _dump() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    path = DUMP_DIR / "events.json"
    path.write_text(
        json.dumps([e.model_dump(mode="json") for e in _store], indent=2),
        encoding="utf-8",
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    body = await request.body()
    if not body:
        return JSONResponse(status_code=400, content={"detail": "empty body"})
    try:
        json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"detail": "malformed JSON"})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/events", status_code=201, response_model=EventOut)
def create_event(body: EventCreate) -> EventOut:
    event = EventOut(
        id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        patient_id=body.patient_id,
        type=body.type,
        label=body.label,
        confidence=body.confidence,
        source=body.source,
    )
    _store.append(event)
    _dump()
    return event


@app.get("/api/v1/events", response_model=list[EventOut])
def list_events() -> list[EventOut]:
    """Dev-only list. Not required of production FMN."""
    return list(_store)


def _fake_visitors() -> list[dict]:
    """Seeded unit vectors. Exercises transport only -- these match nobody."""
    rng = np.random.default_rng(7)
    out = []
    for name in ("dr_khan", "ammi"):
        vectors = []
        for _ in range(3):
            v = rng.normal(size=EMBEDDING_DIM)
            vectors.append((v / np.linalg.norm(v)).round(8).tolist())
        out.append({"name": name, "relation": "demo", "embeddings": vectors})
    return out


def _visitors_from_gallery() -> list[dict] | None:
    """Serve a real locally enrolled gallery so recognition actually works."""
    if not GALLERY_PKL.is_file():
        return None
    with open(GALLERY_PKL, "rb") as f:
        blob = pickle.load(f)
    people = blob.get("people", blob) if isinstance(blob, dict) else blob
    return [
        {
            "name": name,
            "relation": "enrolled_locally",
            "embeddings": [np.asarray(e, dtype=float).round(8).tolist() for e in embs],
        }
        for name, embs in people.items()
    ]


@app.get("/api/v1/patients/{patient_id}/visitors")
def list_visitors(patient_id: int) -> dict:
    """Stand-in for the endpoint the backend must implement."""
    if patient_id <= 0:
        raise HTTPException(status_code=422, detail="patient_id must be positive")

    visitors = _visitors_from_gallery()
    served_from = "gallery.pkl"
    if visitors is None:
        visitors = _fake_visitors()
        served_from = "seeded_fake_vectors"

    return {
        "patient_id": patient_id,
        "embedding_model": EMBEDDING_MODEL,
        "threshold": DEFAULT_THRESHOLD,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "served_from": served_from,
        "visitors": visitors,
    }
