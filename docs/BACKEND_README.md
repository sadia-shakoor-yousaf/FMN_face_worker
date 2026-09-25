# Backend guide — ForgetMeNot Face / Visitor Worker

This guide is for the **FMN backend team**.
You do **not** need to understand machine learning. You need to:

1. Accept two new event types on an endpoint you already have.
2. Serve one new endpoint that lists a patient's enrolled visitors.
3. Confirm a visitor event shows up in `activity_logs`.

If you have never cloned a GitHub repo or used Python before, follow this page
top to bottom.

---

## What is this project? (in plain words)

ForgetMeNot has:

- A **caregiver app** (Flutter)
- A **backend API** (your FastAPI / Supabase stack)
- A **care-PC worker** (this repo) that watches a door camera in the patient's home

When someone appears at the door, the worker decides whether it is a person the
family enrolled, then sends a tiny JSON message to your server:

```http
POST /api/v1/events
```

with `type: visitor_recognized` (and the person's name) or
`type: visitor_unknown`.

**Important:** the worker does **not** upload camera frames or face photos. Only
the event JSON.

This is the **same endpoint** the fall worker already uses. If `fall_detected`
works, you mostly need to accept two more `type` values plus
`source: door_cam`. Full field list: [`EVENT_API.md`](EVENT_API.md).

---

## The part that is genuinely new

Falls need no data from you. Visitors do — the worker has to know *who* is
enrolled, and families enrol people **in the app**, not on the care PC.

So there is one new endpoint to build:

```http
GET /api/v1/patients/{patient_id}/visitors
```

Contract: [`VISITOR_GALLERY_API.md`](VISITOR_GALLERY_API.md).

You have a choice about what it returns, and it changes how much work you do:

| Option | You return | You install | Risk |
| --- | --- | --- | --- |
| **A** | Photo URLs + names | Nothing | None |
| **B** | 512-number vectors + names | TensorFlow 2.13, keras-facenet, mtcnn (~2GB) | Must match our processing exactly |

Option B is fine — but if you generate the vectors, your processing has to match
the care PC's *exactly*, or recognition silently stops working while every
endpoint still returns `200`. That is not hypothetical: in our testing, vectors
from a mismatched path let through **100% of strangers**.

So if you pick B, two things are mandatory:

1. Read [`EMBEDDING_CONTRACT.md`](EMBEDDING_CONTRACT.md).
2. Run `python scripts/verify_embedding.py` on your server and get `PASS`.

If TensorFlow on the API server sounds unappealing (it is CPU-only on Windows at
that version), pick **A** and we will do the processing here.

---

## What you care about vs what you can ignore

| Relevant to backend | You can ignore |
| --- | --- |
| [`EVENT_API.md`](EVENT_API.md) — request/response contract | How FaceNet works inside |
| [`VISITOR_GALLERY_API.md`](VISITOR_GALLERY_API.md) — the endpoint we need | Threshold tuning research |
| `scripts/smoke_post.py` — test your endpoint with no camera | `data/live_logs/` snapshots (local only) |
| `patient_id` = **integer** patient PK | Caregiver UUID (you resolve it from the patient row) |
| Inserting into `activity_logs` | Running recognition on live frames |

---

## What you need installed first

Do this once on your laptop.

### 1) Git

Git downloads the project from GitHub: <https://git-scm.com/downloads>

Check it works:

```bash
git --version
```

### 2) Python 3.11 — not 3.12, not 3.13

This worker pins TensorFlow 2.13, which only supports Python up to **3.11**.
Download from <https://www.python.org/downloads/release/python-3119/> and tick
"Add Python to PATH" during install.

Check:

```bash
python --version
```

If that prints 3.12 or later, you have the wrong one. On Windows try
`py -3.11 --version`, and use `py -3.11` in place of `python` below.

---

## Get the project

```bash
git clone <this-repo-url> FMN_face_worker
cd FMN_face_worker
```

Everything below assumes you are **inside** the `FMN_face_worker` folder (the
one containing `requirements.txt`).

---

## Create a virtual environment

A venv keeps these packages out of your system Python.

```bash
py -3.11 -m venv .venv
```

Activate it:

**Windows (PowerShell):**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. If it doesn't, the commands below
will install into the wrong place.

Then install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-mock.txt   # optional, for the local fake server
```

This downloads TensorFlow, so expect a few hundred MB and a few minutes.

> **Do not "upgrade to latest."** The pins in `requirements.txt` are
> deliberate: TensorFlow 2.16+ ships Keras 3, which refuses to load the face
> model this worker uses.

---

## Path 1: test YOUR endpoint with no camera

This is the fastest way to check your `POST /api/v1/events` accepts visitor
events.

```bash
python scripts/smoke_post.py --api-url https://YOUR_API_HOST --patient-id 1
python scripts/smoke_post.py --api-url https://YOUR_API_HOST --patient-id 1 --type visitor_unknown
```

Examples of `--api-url`:

- `http://127.0.0.1:8000` (API on your own machine)
- `https://api.yourdomain.com` (deployed API)

Do **not** put `/api/v1/events` in `--api-url`. The script adds that path for
you. So if your full URL is `https://x.com/api/v1/events`, pass only
`https://x.com`.

### Success looks like

- Terminal prints JSON with an `id` and `created_at`
- A new `activity_logs` row exists:
  - `type` = `visitor_recognized`
  - `label` = `dr_khan`
  - `source` = `door_cam`
  - `patient_id` = the int you passed

### If it fails

| Symptom | Likely cause |
| --- | --- |
| `connection` / timeout | Wrong host, server not running, firewall |
| `404` | Path is not `/api/v1/events` |
| `422` | Body validation — check the `type` enum accepts `visitor_recognized` and `visitor_unknown`, and `source` accepts `door_cam` |
| `400` | Empty / bad JSON handling on your side |

Same test with curl (optional):

```bash
curl -X POST https://YOUR_API_HOST/api/v1/events ^
  -H "Content-Type: application/json" ^
  -d "{\"patient_id\":1,\"type\":\"visitor_recognized\",\"label\":\"dr_khan\",\"confidence\":0.82,\"source\":\"door_cam\"}"
```

(On macOS/Linux use `\` at end of lines instead of `^`, and a single-quoted
`-d '...'` JSON string.)

---

## Path 2: check your visitors endpoint shape

Run the mock server and compare its response to yours.

**Terminal 1:**

```bash
uvicorn mock_api.app:app --host 127.0.0.1 --port 5155
```

**Terminal 2:**

```bash
curl http://127.0.0.1:5155/api/v1/patients/1/visitors
```

Your endpoint should return the same field names and types. Then point the
worker at your server instead:

```bash
python workers/face_worker.py --cam 0 --patient-id 1 --gallery-url https://YOUR_API_HOST --api-url https://YOUR_API_HOST
```

The worker logs `synced N visitor(s) from ...` on success, or tells you exactly
which rule your payload broke (vector length, normalization, or model
identifier).

---

## Path 3: run the real worker with a camera

You need enrolled faces first. Locally that means a few photos per person:

```text
data/faces/gallery/
    your_name/      1.jpg 2.jpg 3.jpg 4.jpg 5.jpg
    a_colleague/    1.jpg 2.jpg 3.jpg 4.jpg 5.jpg
```

Then:

```bash
python scripts/enroll_faces.py --detector haar
python workers/face_worker.py --cam 0 --patient-id 1 --detector haar --api-url https://YOUR_API_HOST
```

### What you should see

1. `VISITOR visitor_recognized name=your_name score=0.7xx`
2. Then `POST visitor_recognized ok id=...`
3. A matching row in `activity_logs`

Stand in front of the camera as someone **not** enrolled and you should get
`visitor_unknown` instead.

If POST fails, the worker **keeps running** and only prints a warning. Fix the
API and try again.

Local snapshots (not uploaded) appear under `data/live_logs/`.

---

## Your backend checklist

- [ ] `POST /api/v1/events` accepts `visitor_recognized` and `visitor_unknown`
- [ ] `source` enum accepts `door_cam`
- [ ] `patient_id` accepted as **integer**
- [ ] Valid event → insert `activity_logs` → `201` + created row
- [ ] `smoke_post.py` succeeds against your server for both visitor types
- [ ] `GET /api/v1/patients/{id}/visitors` implemented per [`VISITOR_GALLERY_API.md`](VISITOR_GALLERY_API.md)
- [ ] Decided Option A (photo URLs) or Option B (vectors)
- [ ] **If Option B:** `scripts/verify_embedding.py` prints `PASS` on your server
- [ ] Caregiver notification path wired the same way as your other activity logs

You do **not** need to run recognition on live camera frames.

---

## Common beginner mistakes

1. **Installed Python 3.12+** — TensorFlow 2.13 won't install. Use 3.11.
2. **Forgot to activate `.venv`** — `pip` then installs somewhere else. Activate first (prompt shows `(.venv)`).
3. **Put the full path in `--api-url`** — use the base URL only, not `.../api/v1/events`.
4. **Used a UUID for `patient_id`** — must be the patient's **int** primary key.
5. **Upgraded TensorFlow** — breaks model loading. Keep the pins.
6. **Ran commands from the wrong folder** — you must be inside `FMN_face_worker`.
7. **Enrolled with one detector, ran with another** — scores shift. Use the same `--detector` in both commands.

---

## Need more technical detail?

| Doc | For |
| --- | --- |
| [`EVENT_API.md`](EVENT_API.md) | Exact JSON contract for events |
| [`VISITOR_GALLERY_API.md`](VISITOR_GALLERY_API.md) | The endpoint we need from you |
| [`EMBEDDING_CONTRACT.md`](EMBEDDING_CONTRACT.md) | Required if you generate vectors |
| [`PIPELINE.md`](PIPELINE.md) | How the worker decides who is at the door |
| [`../README.md`](../README.md) | Shorter overview + all flags |

If something here is unclear, ask the vision teammate which `--api-url` and
which patient id to use in your environment.
