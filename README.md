# ForgetMeNot — Care-PC Face / Visitor Worker

Local **visitor recognition worker** for the ForgetMeNot (FMN) FYP.
Runs on a care-PC / laptop: watches a door camera (or a demo video), decides who
is at the door, and posts a small JSON event to the FMN backend.

Sibling of [`FMN_fall_worker`](https://github.com/sadia-shakoor-yousaf/FMN_fall_worker) —
same event endpoint, same packaging, different sensor. This repo is meant for
**backend (and frontend) integration testing**. Research notebooks and
evaluation live in a separate thesis repo.

> **Backend teammates (start here):**
> Step-by-step clone, Python setup, and how to hit your real API — written for beginners:
> [`docs/BACKEND_README.md`](docs/BACKEND_README.md)

---

## What this does

```text
Door camera / video
  → face detector (largest face)
  → FaceNet 512-d embedding
  → cosine match vs enrolled visitors
  → streak + cooldown
  → POST /api/v1/events  { visitor_recognized | visitor_unknown, patient_id, ... }
```

- **Camera frames stay on the care PC.** Only the event JSON is sent.
- `patient_id` **is an integer** (your `patients.id` PK).
- Push notifications (FCM) are **your** concern after you write `activity_logs`.

More detail:

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — recognition pipeline and design decisions
- [`docs/EVENT_API.md`](docs/EVENT_API.md) — the event contract (shared with the fall worker)
- [`docs/VISITOR_GALLERY_API.md`](docs/VISITOR_GALLERY_API.md) — **the one endpoint we need from the backend**
- [`docs/EMBEDDING_CONTRACT.md`](docs/EMBEDDING_CONTRACT.md) — required only if the backend generates vectors

---

## Requirements

| Item | Notes |
| --- | --- |
| OS | Windows / macOS / Linux |
| Python | **3.11 only** (see `.python-version`) — TensorFlow 2.13 does not support 3.12+ |
| Hardware | CPU. TensorFlow 2.13 on Windows is CPU-only |
| Camera | USB webcam **or** an `.mp4` clip for offline demo |

---

## Project layout

```text
FMN_face_worker/
├── README.md
├── requirements.txt              # worker runtime (pinned — see note below)
├── requirements-mock.txt         # + FastAPI mock server
├── face_cv/                      # detectors, FaceNet, matching, sync, HTTP client
├── workers/face_worker.py        # main entrypoint
├── scripts/enroll_faces.py       # photos → local gallery.pkl
├── scripts/verify_embedding.py   # prove two machines share an embedding space
├── scripts/smoke_post.py         # POST a visitor event without a camera
├── mock_api/                     # local stand-in for both endpoints
├── docs/                         # pipeline + contracts
└── data/                         # gallery, logs, snapshots (all gitignored)
```

No model weights are committed. FaceNet downloads on first run; Haar cascades
ship inside OpenCV.

---

## Setup

```bash
git clone <this-repo-url> FMN_face_worker
cd FMN_face_worker

py -3.11 -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: mock API for offline contract testing
pip install -r requirements-mock.txt
```

> **The pins are load-bearing.** `keras-facenet` loads a Keras-2 model that
> TensorFlow 2.16+ (Keras 3) refuses, and this package generation predates
> numpy 2.x. Do not upgrade TensorFlow or numpy.

---

## Quick start

### 1) Enrol a few faces

Folder name = the label the caregiver sees.

```text
data/faces/gallery/
    dr_khan/   1.jpg 2.jpg 3.jpg 4.jpg 5.jpg
    ammi/      1.jpg 2.jpg 3.jpg 4.jpg 5.jpg
```

Five or six varied photos per person — different lighting, glasses on and off,
slight angles. Six frames from one selfie burst teach it one lighting condition
and nothing else.

```bash
python scripts/enroll_faces.py               # accurate, slow (MTCNN)
python scripts/enroll_faces.py --detector haar   # much faster on CPU
```

Nothing under `data/faces/` is committed. It is biometric data of real people;
keep it that way.

### 2) Mock API (no real backend yet)

**Terminal 1 — mock server**

```bash
uvicorn mock_api.app:app --host 127.0.0.1 --port 5155
```

**Terminal 2 — smoke POST (no camera)**

```bash
python scripts/smoke_post.py --api-url http://127.0.0.1:5155 --patient-id 1
python scripts/smoke_post.py --api-url http://127.0.0.1:5155 --patient-id 1 --type visitor_unknown
```

### 3) Run the worker

```bash
# live webcam
python workers/face_worker.py --cam 0 --api-url http://127.0.0.1:5155 --patient-id 1

# video file, headless
python workers/face_worker.py --source path/to/clip.mp4 --no-preview --api-url http://127.0.0.1:5155 --patient-id 1

# local only, no POST
python workers/face_worker.py --cam 0 --api-url ""
```

On an event you should see `POST visitor_recognized ok id=...` in the log. Mock
events are also written to `data/mock_events/events.json`.

---

## Test against the **real** FMN backend

1. Implement `POST /api/v1/events` per [`docs/EVENT_API.md`](docs/EVENT_API.md).
2. Implement `GET /api/v1/patients/{id}/visitors` per [`docs/VISITOR_GALLERY_API.md`](docs/VISITOR_GALLERY_API.md).
3. Use a real patient integer id from your DB.
4. Point the worker at your API (base URL only — the worker appends the paths):

```bash
python workers/face_worker.py ^
  --cam 0 ^
  --api-url https://YOUR_API_HOST ^
  --gallery-url https://YOUR_API_HOST ^
  --patient-id 1 ^
  --no-preview
```

(macOS/Linux: use `\` line continuations instead of `^`.)

Confirm a new `activity_logs` row: `type = visitor_recognized`,
`source = door_cam`, `label = <visitor name>`.

Omit `--gallery-url` to keep using the local gallery while their endpoint is
still being built.

---

## Where enrolled faces come from

The worker reads vectors through one seam, so the source can change without
touching the frame loop:

| Source | Flag | Use |
| --- | --- | --- |
| Local `gallery.pkl` | `--gallery` (default) | Development, demos, offline |
| Photo folder | `--gallery data/faces/gallery` | Re-embed on the spot |
| FMN backend | `--gallery-url https://host` | Production; caregivers enrol in the app |

The backend path caches vectors under `data/faces/`, so a restart during a
network outage still recognizes people. If the backend reports a different
`embedding_model` than this worker matches, the sync **fails loudly** instead of
silently accepting strangers — see [`docs/EMBEDDING_CONTRACT.md`](docs/EMBEDDING_CONTRACT.md).

---

## Useful flags

| Flag | Example | Meaning |
| --- | --- | --- |
| `--api-url` | `https://api...` | Base URL; `""` disables POST |
| `--patient-id` | `1` | Required for POST |
| `--gallery` | `data/faces/gallery.pkl` | Local enrolled visitors |
| `--gallery-url` | `https://api...` | Fetch enrolled vectors from the backend |
| `--detector` | `mtcnn` / `haar` | **Must match enrollment** |
| `--threshold` | `0.675` | Match cutoff (default comes from the gallery) |
| `--need` | `5` | Consecutive frames on one identity before firing |
| `--cooldown` | `120` | Seconds before the same identity fires again |
| `--source` | `clip.mp4` | Offline demo |
| `--cam` | `0` | Webcam index |
| `--skip` | `3` | Every Nth frame (help slow CPUs) |
| `--no-preview` | | Headless |
| `--no-snapshot` | | Don't write visitor JPEGs |

Logs and snapshots land in `data/live_logs/`.

---

## Honest limitations

- **The 0.675 threshold was tuned on photos**, not on a 480p door camera with
  backlight and motion blur. Expect to re-check it with `--threshold` on real
  hardware. A "wrong" unknown is usually a threshold issue, not a broken API.
- **MTCNN is slow on CPU** — use `--detector haar` for anything live, and enrol
  with the same detector.
- **Largest face only.** Two people at the door produce one event. Matches the
  fall worker's largest-person rule and the solo-elderly deployment this product
  targets.
- **`confidence` is a cosine score, not a probability.** `0.82` means "similar to
  the enrolled photos," not "82% sure."

---

## Environment notes

- First run downloads FaceNet weights and feels slow; later runs are faster.
- Preview window: press `q` or `Esc` to quit.
- POST failures are warnings only — the camera loop keeps running.
- Nothing under `data/` is committed: enrollment photos, embeddings, and visitor
  snapshots are all personal data.

---

## Out of scope (this repo)

- Training / evaluation notebooks (thesis repo)
- Fall / room-cam worker ([`FMN_fall_worker`](https://github.com/sadia-shakoor-yousaf/FMN_fall_worker), same event contract)
- Uploading images or video to the cloud
- Caregiver-app enrollment UI
