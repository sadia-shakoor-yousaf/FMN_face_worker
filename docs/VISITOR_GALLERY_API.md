# Visitor Gallery API — what the face worker needs from the backend

**Caller:** care-PC face worker (this repo)
**Direction:** worker reads. This is the only endpoint the worker needs *besides*
`POST /api/v1/events`.
**Status:** not implemented yet. A working stand-in ships in `mock_api/app.py`.

---

## Why this exists

Caregivers enrol visitors **in the app**, not on the care PC. Nobody SSHes into
a patient's living room to drop JPEGs in a folder. So the enrolled faces start
their life on your side and have to reach the worker.

The worker fetches this list at startup, holds it in memory, caches it to disk
(so a restart during an outage still works), and re-fetches periodically — so a
visitor enrolled at 3:00pm is recognized by 3:05pm without anyone restarting
anything.

The worker does **not** query this endpoint per video frame. Once per sync.

---

## Endpoint

| | |
| --- | --- |
| Method | `GET` |
| Path | `/api/v1/patients/{patient_id}/visitors` |
| Auth | same as the rest of your API; the worker can send a bearer token |

`patient_id` is the integer `patients.id`, same as in the event API.

---

## Response — `200 OK`

```json
{
  "patient_id": 1,
  "embedding_model": "keras-facenet-0.3.2",
  "threshold": 0.675,
  "updated_at": "2026-09-25T14:03:11+00:00",
  "visitors": [
    {
      "name": "dr_khan",
      "relation": "doctor",
      "embeddings": [
        [0.0123, -0.0456, "... 512 floats total ..."],
        [0.0118, -0.0461, "... 512 floats total ..."]
      ]
    },
    {
      "name": "ammi",
      "relation": "daughter",
      "embeddings": [
        [0.0301, 0.0077, "... 512 floats total ..."]
      ]
    }
  ]
}
```

| Field | Type | Rules |
| --- | --- | --- |
| `patient_id` | integer | Echo of the path parameter |
| `embedding_model` | string | Must be `keras-facenet-0.3.2`. The worker **refuses to match** on an unknown value — see `EMBEDDING_CONTRACT.md` |
| `threshold` | float | Optional. Omit and the worker uses its own default (0.675) |
| `updated_at` | ISO 8601 | Timestamp of the most recent enrollment change, so the worker can skip unchanged syncs |
| `visitors[].name` | string | **This is what the caregiver sees.** It goes straight into the event `label` |
| `visitors[].relation` | string | Optional, display only |
| `visitors[].embeddings` | array of arrays | One 512-float vector per enrolled photo, **L2-normalized** (`‖v‖ = 1`) |

### Rules the worker enforces on arrival

- Every vector is exactly **512** floats. Wrong length is a hard error.
- Every vector has L2 norm within `1.0 ± 0.01`. Un-normalized vectors are
  rejected, because the 0.675 threshold assumes unit length.
- `embedding_model` must match. On a mismatch the worker stops rather than
  guessing, because mismatched vectors accept strangers instead of erroring.

### Empty is valid

A patient with nobody enrolled returns `"visitors": []`. The worker logs it and
exits with a clear message. Not an error on your side.

---

## Errors

| Status | When |
| --- | --- |
| `404` | No such patient |
| `422` | `patient_id` not a positive integer |

---

## Keep 5–6 vectors per person

Each enrolled photo becomes one vector, and the worker scores a live face
against **all** of a person's vectors, taking the best. That is what makes
recognition survive a change of glasses or lighting. Don't average them into a
single vector server-side — it measurably hurts, and the worker already handles
the list.

---

## If you'd rather not run TensorFlow

Return photo URLs instead of vectors and the care PC will embed them. Same
endpoint, different payload:

```json
{
  "patient_id": 1,
  "updated_at": "2026-09-25T14:03:11+00:00",
  "visitors": [
    {
      "name": "dr_khan",
      "relation": "doctor",
      "photos": [
        "https://your-storage/visitors/12/a.jpg",
        "https://your-storage/visitors/12/b.jpg"
      ]
    }
  ]
}
```

This version needs **no ML libraries on your server** and removes the whole
embedding-mismatch risk class. Tell us which shape you're building and the
worker adapts — it's one function on our side either way.

---

## Test it without writing anything

The mock in this repo already serves this contract:

```bash
.venv-mock\Scripts\python.exe -m uvicorn mock_api.app:app --host 127.0.0.1 --port 5155
curl http://127.0.0.1:5155/api/v1/patients/1/visitors
```

Compare your response shape against that.
