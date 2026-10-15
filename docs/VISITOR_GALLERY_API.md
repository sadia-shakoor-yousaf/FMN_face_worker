# Visitor gallery — where enrolled embeddings live

**Caller:** care-PC face worker (this repo)
**Current path:** Qdrant Cloud. The backend generates FaceNet vectors at enrollment and upserts them. The worker reads them with `--from-qdrant`.
**Older path:** `GET /api/v1/patients/{patient_id}/visitors`, still supported by `--gallery-url`. A stand-in ships in `mock_api/app.py`.

Events are unchanged: the worker still only POSTs `POST /api/v1/events`. It never uploads photos.

---

## Qdrant (what the backend is building)

The collection already exists. Do not create a second one.

| | |
| --- | --- |
| Collection | `visitor_faces` |
| Vector size | 512 |
| Distance | Cosine |
| Credentials | `URL` + `API_KEY` in `.env` (aliases: `QDRANT_URL`, `QDRANT_API_KEY`). Optional `QDRANT_COLLECTION`. |

One enrolled photo is one point. Generate the vector with the path in [`EMBEDDING_CONTRACT.md`](EMBEDDING_CONTRACT.md). Do not use Qdrant's built-in embedding models.

```python
client.upsert(
    collection_name="visitor_faces",
    points=[{
        "id": "<stable id per photo, so a re-upload replaces it>",
        "vector": ["...512 floats, L2-normalized..."],
        "payload": {
            "patient_id": 39,
            "name": "ammi",
            "relation": "daughter",
            "embedding_model": "keras-facenet-0.3.2",
            "source_file": "10.jpg",
        },
    }],
)
```

| Payload field | Rules |
| --- | --- |
| `patient_id` | Integer `patients.id`. Indexed. The worker filters on this. |
| `name` | The label that goes into the event. Required. |
| `relation` | Optional, display only |
| `embedding_model` | Must be `keras-facenet-0.3.2`. A mismatch aborts the sync. |
| `source_file` | Optional photo name |

The worker scrolls this patient's points once at startup, groups them by `name`, and matches in memory. It does not query Qdrant per frame. A successful fetch is cached to `data/faces/visitors_patient_<id>.pkl`.

```bash
python workers/face_worker.py --cam 0 --patient-id 39 --from-qdrant --api-url https://YOUR_API_HOST
```

---

## HTTP gallery (older path)

**Direction:** worker reads, if started with `--gallery-url`.
**Status:** not the path enrollment is using. Kept so a backend that serves vectors over HTTP still works.

### Why this exists

Caregivers enrol visitors **in the app**, not on the care PC. This endpoint was the original way to hand those vectors to the worker. The worker fetches the list at startup, holds it in memory, and caches it to disk. It does not query this endpoint per video frame. A visitor enrolled after the worker has started is picked up on the next launch.

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
