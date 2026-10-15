# FMN Event API — Backend Contract

**Caller:** care-PC face worker (this repo)
**Auth:** not required for this endpoint
**Scope:** accept one event JSON, persist it, return the created row. Camera
frames never leave the care PC.

This is the **same endpoint** the fall worker uses. If you have already
implemented it for `fall_detected`, you only need to accept two more `type`
values and `source: door_cam` — see the enum rows below. No new endpoint.

---

## Context — what runs on the care PC (not on your servers)

ForgetMeNot caregivers use UUID; patients use integer `id`. On a home **care
PC** this worker watches a door camera, decides who is at the door, then calls
**only** `POST /api/v1/events`. Face images and model inference stay on that
machine.

### Visitor recognition (`source: door_cam`)

1. Door camera frames are read on the care PC.
2. The largest face in the frame is detected and cropped.
3. The crop becomes a 512-d FaceNet embedding.
4. That embedding is compared by cosine similarity against the patient's
   enrolled visitors, loaded at startup from Qdrant (`--from-qdrant`) or from
   the HTTP gallery (see `VISITOR_GALLERY_API.md`).
5. Best score **≥ threshold** → that visitor. Below → unknown.
6. We require **persistence** (several consecutive frames agreeing) plus a
   per-identity **cooldown**, so one visit produces one event and not fifty.
7. On event we POST:
   - `type`: `visitor_recognized` or `visitor_unknown`
   - `label`: the visitor's display name, or `unknown`
   - `confidence`: cosine score clamped to `[0, 1]`
   - `source`: `door_cam`
   - `patient_id`: that patient's integer PK
8. Optionally a local JPEG snapshot is saved for demo — **never uploaded** via
   this API.

### What this means for backend

| You receive | You do **not** receive / run |
| --- | --- |
| Small JSON event about a patient | Camera frames or visitor snapshots |
| `patient_id` as **int** | Caregiver UUID in this payload (derive via patient → caregiver) |
| Activity types for `activity_logs` | FaceNet / MTCNN inference for live frames |

`care PC (recognize) → POST /api/v1/events → insert activity_logs → (your FCM / UI path)`

---

## Endpoint

| | |
| --- | --- |
| Method | `POST` |
| Path | `/api/v1/events` |

---

## Request body

```json
{
  "patient_id": 1,
  "type": "visitor_recognized",
  "label": "dr_khan",
  "confidence": 0.82,
  "source": "door_cam"
}
```

An unknown visitor looks like this:

```json
{
  "patient_id": 1,
  "type": "visitor_unknown",
  "label": "unknown",
  "confidence": 0.41,
  "source": "door_cam"
}
```

| Field | Type | Rules |
| --- | --- | --- |
| `patient_id` | integer | Existing patient primary key (`patients.id`) |
| `type` | string enum | `fall_detected` \| `visitor_recognized` \| `visitor_unknown` |
| `label` | string | Visitor display name, or `unknown`. Fall worker sends `floor_down` |
| `confidence` | float | `0.0`–`1.0` |
| `source` | string enum | `room_cam` \| `door_cam` |

Note on `confidence` for visitor events: it is a cosine similarity, not a
probability. `0.82` means "quite similar to the enrolled photos," not "82%
certain." Treat it as a score to display or log, not a percentage.

---

## Success — `201 Created`

```json
{
  "id": "<activity_log id>",
  "created_at": "2026-09-25T09:00:00+00:00",
  "patient_id": 1,
  "type": "visitor_recognized",
  "label": "dr_khan",
  "confidence": 0.82,
  "source": "door_cam"
}
```

---

## Errors

| Status | When |
| --- | --- |
| `400` | Empty body or malformed JSON |
| `422` | Invalid / missing fields, unknown enums, bad confidence, unknown `patient_id` |

---

## Server-side processing

1. Validate the body.
2. Resolve `patient_id` → patient (and linked caregiver). Reject if missing.
3. Insert into `activity_logs`.
4. Return `201` with the created row.

**Must land in `activity_logs`:** `fall_detected`, `visitor_recognized`,
`visitor_unknown`.

FCM / UI reactions are out of scope for this contract — use your existing
`activity_logs` path.

---

## Out of scope for this endpoint

- Uploading face images or snapshots
- Running recognition models on the server for live frames
- Specifying FCM payload details
- `GET /api/v1/events` (mock-only helper in this repo)
