"""Load enrolled visitor embeddings from Qdrant.

The backend upserts one 512-d FaceNet vector per photo into the
``visitor_faces`` collection. This module scrolls that collection for one
patient and returns the same {name: [vectors]} gallery the live matcher uses.

Credentials come from the environment or a ``.env`` file at the repo root
(``URL`` / ``QDRANT_URL``, ``API_KEY`` / ``QDRANT_API_KEY``). A successful
fetch is cached to disk so a later restart still recognizes people if Qdrant
is unreachable.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .gallery_sync import GallerySyncError, cache_path
from .pipeline import EMBEDDING_DIM, EMBEDDING_MODEL, DEFAULT_DETECTOR, Gallery

DEFAULT_COLLECTION = "visitor_faces"
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env_file(path: Path = _ENV_FILE) -> None:
    """Set missing environment variables from ``.env``. Does not override."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def qdrant_settings(collection: str | None = None) -> tuple[str, str, str]:
    """Return (url, api_key, collection) from the environment."""
    load_env_file()
    url = (os.environ.get("QDRANT_URL") or os.environ.get("URL") or "").strip()
    api_key = (os.environ.get("QDRANT_API_KEY") or os.environ.get("API_KEY") or "").strip()
    name = (
        collection
        or os.environ.get("QDRANT_COLLECTION")
        or DEFAULT_COLLECTION
    ).strip()
    if not url or not api_key:
        raise GallerySyncError(
            "Qdrant credentials missing. Set URL and API_KEY in .env "
            "(or QDRANT_URL and QDRANT_API_KEY)."
        )
    if not url.endswith(":6333"):
        url = url.rstrip("/") + ":6333"
    return url, api_key, name


def _as_vector(point: Any) -> np.ndarray:
    raw = point.vector
    if isinstance(raw, dict):
        if not raw:
            raise GallerySyncError(f"point {point.id} has no vector")
        raw = next(iter(raw.values()))
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (EMBEDDING_DIM,):
        raise GallerySyncError(
            f"point {point.id}: expected {EMBEDDING_DIM}-d vector, got shape {arr.shape}"
        )
    norm = float(np.linalg.norm(arr))
    if not 0.99 <= norm <= 1.01:
        raise GallerySyncError(
            f"point {point.id}: vector is not L2-normalized (norm={norm:.4f})"
        )
    return arr


def _gallery_from_points(
    points: list[Any],
    *,
    allow_model_mismatch: bool,
) -> tuple[Gallery, dict[str, Any]]:
    gallery: Gallery = {}
    models: set[str] = set()
    for point in points:
        payload = point.payload or {}
        name = payload.get("name")
        if not name:
            raise GallerySyncError(f"point {point.id} is missing payload name")
        model = payload.get("embedding_model")
        if model:
            models.add(str(model))
        gallery.setdefault(str(name), []).append(_as_vector(point))

    if not models:
        served = None
    elif len(models) == 1:
        served = next(iter(models))
    else:
        raise GallerySyncError(
            f"collection mixed embedding models: {sorted(models)}. "
            "Re-enroll with a single model."
        )

    if served != EMBEDDING_MODEL:
        message = (
            f"embedding model mismatch: Qdrant has {served!r}, this worker "
            f"matches {EMBEDDING_MODEL!r}. See docs/EMBEDDING_CONTRACT.md"
        )
        if not allow_model_mismatch:
            raise GallerySyncError(message)

    meta = {
        "embedding_model": served or EMBEDDING_MODEL,
        "detector": DEFAULT_DETECTOR,
        "source": "qdrant",
    }
    return gallery, meta


def fetch_qdrant_gallery(
    *,
    patient_id: int,
    collection: str | None = None,
    cache_dir: str | Path | None = None,
    timeout: float = 30.0,
    allow_model_mismatch: bool = False,
) -> tuple[Gallery, dict[str, Any]]:
    """Scroll this patient's points. Writes a local cache on success."""
    url, api_key, collection = qdrant_settings(collection)
    query = Filter(
        must=[
            FieldCondition(key="patient_id", match=MatchValue(value=int(patient_id)))
        ]
    )
    try:
        client = QdrantClient(
            url=url, api_key=api_key, timeout=timeout, prefer_grpc=False
        )
        points: list[Any] = []
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=collection,
                scroll_filter=query,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points.extend(batch)
            if offset is None or not batch:
                break
    except GallerySyncError:
        raise
    except Exception as e:
        raise GallerySyncError(f"cannot read {collection} from Qdrant: {e}") from e

    gallery, meta = _gallery_from_points(
        points, allow_model_mismatch=allow_model_mismatch
    )
    meta["collection"] = collection

    if cache_dir is not None and gallery:
        path = cache_path(cache_dir, patient_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"people": gallery, **meta}, f)
        meta["cached_to"] = str(path)

    return gallery, meta
