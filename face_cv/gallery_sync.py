"""Fetch the enrolled visitor gallery from the FMN backend.

This is the other half of the seam in pipeline.load_visitor_gallery. The
backend owns enrollment (caregivers upload photos in the app); the worker only
needs the resulting vectors. Contract: docs/VISITOR_GALLERY_API.md.

Vectors are cached to disk after a successful fetch so a restart during a
network outage still recognizes people.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .pipeline import EMBEDDING_DIM, EMBEDDING_MODEL, Gallery

VISITORS_PATH = "/api/v1/patients/{patient_id}/visitors"


class GallerySyncError(Exception):
    """Raised when the visitor gallery cannot be fetched or is unusable."""


def cache_path(cache_dir: str | Path, patient_id: int) -> Path:
    return Path(cache_dir) / f"visitors_patient_{patient_id}.pkl"


def _parse_response(body: dict[str, Any]) -> tuple[Gallery, dict[str, Any]]:
    visitors = body.get("visitors")
    if not isinstance(visitors, list):
        raise GallerySyncError("response has no 'visitors' list")

    gallery: Gallery = {}
    for entry in visitors:
        name = entry.get("name")
        vectors = entry.get("embeddings")
        if not name or not isinstance(vectors, list):
            raise GallerySyncError(f"visitor entry missing name/embeddings: {entry!r}")
        parsed: list[np.ndarray] = []
        for vec in vectors:
            arr = np.asarray(vec, dtype=np.float64)
            if arr.shape != (EMBEDDING_DIM,):
                raise GallerySyncError(
                    f"{name}: expected {EMBEDDING_DIM}-d vector, got shape {arr.shape}"
                )
            norm = float(np.linalg.norm(arr))
            if not 0.99 <= norm <= 1.01:
                raise GallerySyncError(
                    f"{name}: vector is not L2-normalized (norm={norm:.4f}). "
                    "See docs/EMBEDDING_CONTRACT.md"
                )
            parsed.append(arr)
        if parsed:
            gallery[name] = parsed

    meta = {
        "embedding_model": body.get("embedding_model"),
        "threshold": body.get("threshold"),
        "updated_at": body.get("updated_at"),
        "source": "backend",
    }
    return gallery, meta


def fetch_visitor_gallery(
    base_url: str,
    *,
    patient_id: int,
    cache_dir: str | Path | None = None,
    timeout: float = 10.0,
    allow_model_mismatch: bool = False,
) -> tuple[Gallery, dict[str, Any]]:
    """GET this patient's enrolled visitors. Writes a local cache on success."""
    url = base_url.rstrip("/") + VISITORS_PATH.format(patient_id=int(patient_id))
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
    except httpx.HTTPError as e:
        raise GallerySyncError(f"cannot reach {url}: {e}") from e

    if resp.status_code >= 400:
        raise GallerySyncError(f"GET {url} -> {resp.status_code}: {resp.text[:300]}")

    try:
        body = resp.json()
    except ValueError as e:
        raise GallerySyncError(f"GET {url} returned non-JSON body") from e

    gallery, meta = _parse_response(body)

    served = meta.get("embedding_model")
    if served != EMBEDDING_MODEL:
        message = (
            f"embedding model mismatch: backend sent {served!r}, this worker "
            f"matches {EMBEDDING_MODEL!r}. Scores from different models are not "
            "comparable and recognition will fail silently. "
            "See docs/EMBEDDING_CONTRACT.md"
        )
        if not allow_model_mismatch:
            raise GallerySyncError(message)

    if cache_dir is not None and gallery:
        path = cache_path(cache_dir, patient_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"people": gallery, **meta}, f)
        meta["cached_to"] = str(path)

    return gallery, meta
