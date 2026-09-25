"""HTTP client for ForgetMeNot event API (mock or real)."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:5155"
EVENTS_PATH = "/api/v1/events"


class FmnClientError(Exception):
    """Raised when the event POST fails (network or HTTP error)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def post_event(
    base_url: str,
    *,
    patient_id: int,
    event_type: str = "visitor_unknown",
    label: str = "unknown",
    confidence: float = 0.0,
    source: str = "door_cam",
    timeout: float = 5.0,
) -> dict[str, Any]:
    """
    POST one activity event. Returns the JSON body on 2xx.

    Raises FmnClientError on timeout, connection failure, or non-2xx.
    """
    url = base_url.rstrip("/") + EVENTS_PATH
    payload = {
        "patient_id": int(patient_id),
        "type": event_type,
        "label": label,
        "confidence": float(confidence),
        "source": source,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
    except httpx.TimeoutException as e:
        raise FmnClientError(f"timeout posting to {url}") from e
    except httpx.HTTPError as e:
        raise FmnClientError(f"HTTP error posting to {url}: {e}") from e

    if resp.status_code >= 400:
        detail = resp.text[:300]
        raise FmnClientError(
            f"POST {url} -> {resp.status_code}: {detail}",
            status_code=resp.status_code,
        )
    return resp.json()
