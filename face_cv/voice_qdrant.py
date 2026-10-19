"""Match a live speech embedding against Qdrant collection ``visitor_voices``.

The care PC embeds the clip. Qdrant holds the enrolled vectors and returns the
nearest one for this patient. Nothing is read from a local gallery file.
The worker does not create the collection; the backend team does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType

from .gallery_sync import GallerySyncError
from .qdrant_gallery import qdrant_settings
from .speakers import SPEAKER_DIM, VOICE_COLLECTION, SpeakerMatch, match_from_hit


def _client(collection: str, timeout: float) -> tuple[QdrantClient, str]:
    url, api_key, name = qdrant_settings(collection)
    return (
        QdrantClient(url=url, api_key=api_key, timeout=timeout, prefer_grpc=False),
        name,
    )


def _patient_filter(patient_id: int) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="patient_id", match=MatchValue(value=int(patient_id)))
        ]
    )


def _ensure_patient_index(client: QdrantClient, collection: str) -> None:
    """Qdrant Cloud rejects a patient_id filter until this integer index exists."""
    client.create_payload_index(
        collection_name=collection,
        field_name="patient_id",
        field_schema=PayloadSchemaType.INTEGER,
        wait=True,
    )


def _nearest(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    patient_id: int,
) -> list[Any]:
    query_filter = _patient_filter(patient_id)
    if hasattr(client, "query_points"):
        found = client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=query_filter,
            limit=1,
            with_payload=True,
        )
        return list(found.points)
    return list(
        client.search(  # type: ignore[attr-defined]
            collection_name=collection,
            query_vector=vector,
            query_filter=query_filter,
            limit=1,
            with_payload=True,
        )
    )


def count_voices(
    *,
    patient_id: int,
    collection: str = VOICE_COLLECTION,
    timeout: float = 15.0,
) -> int:
    """How many enrolled voice points this patient has. Raises GallerySyncError."""
    client, name = _client(collection, timeout)
    try:
        _ensure_patient_index(client, name)
        counted = client.count(
            collection_name=name,
            count_filter=_patient_filter(patient_id),
            exact=True,
        )
    except Exception as e:
        raise GallerySyncError(f"cannot read {name} from Qdrant: {e}") from e
    return int(getattr(counted, "count", 0))


def search_voice(
    *,
    patient_id: int,
    embedding: np.ndarray,
    threshold: float,
    collection: str = VOICE_COLLECTION,
    timeout: float = 15.0,
) -> SpeakerMatch:
    """Nearest enrolled voice for this patient. Below threshold is no match."""
    probe = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if probe.shape != (SPEAKER_DIM,):
        raise GallerySyncError(
            f"voice embedding is {probe.shape[0]}-d, {VOICE_COLLECTION} is {SPEAKER_DIM}-d"
        )
    client, name = _client(collection, timeout)
    try:
        _ensure_patient_index(client, name)
        hits = _nearest(client, name, probe.tolist(), patient_id)
    except Exception as e:
        raise GallerySyncError(f"cannot search {name}: {e}") from e
    if not hits:
        return SpeakerMatch(False, "", 0.0)
    top = hits[0]
    payload = top.payload or {}
    return match_from_hit(
        name=payload.get("name") or payload.get("visitor_name"),
        score=float(top.score or 0.0),
        threshold=threshold,
        embedding_model=payload.get("embedding_model"),
        relationship=payload.get("relationship") or payload.get("relation"),
    )
