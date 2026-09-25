# Face worker pipeline

How the care-PC process turns door-camera frames into a visitor event.

---

## High-level flow

```text
Door camera / video file
        │
        ▼
   Face detector (MTCNN or Haar) → largest face box
        │
        ▼
   Crop → 160×160 RGB
        │
        ▼
   FaceNet → 512-d embedding, L2-normalized
        │
        ▼
   Cosine similarity vs every enrolled photo of every visitor
   (best score per person wins)
        │
        ▼
   best ≥ threshold (0.675) ? visitor name : "unknown"
        │
        ▼
   NEED consecutive frames on the same identity (default 5)
        │
        ▼
   Per-identity cooldown (default 120s) ──► skip duplicate events
        │
        ▼
   POST /api/v1/events   +   optional local JPEG snapshot
```

Enrolled vectors come from either a local `gallery.pkl`
(`scripts/enroll_faces.py`) or the backend (`--gallery-url`). The frame loop is
identical either way.

---

## Modules in this repo

| Path | Role |
| --- | --- |
| `workers/face_worker.py` | Entrypoint: capture, overlay, logging, POST |
| `face_cv/pipeline.py` | Detectors, FaceNet embedding, gallery matching |
| `face_cv/live.py` | Streak + per-identity cooldown policy |
| `face_cv/gallery_sync.py` | Fetch + cache enrolled vectors from the backend |
| `face_cv/fmn_client.py` | HTTP POST client for the event API |
| `scripts/enroll_faces.py` | Photos folder → local `gallery.pkl` |
| `scripts/verify_embedding.py` | Prove two machines share an embedding space |
| `scripts/smoke_post.py` | POST a visitor event with no camera |
| `mock_api/app.py` | Local stand-in for both endpoints |

No model weights are committed. FaceNet weights download on first run;
Haar cascades ship inside OpenCV.

---

## Decisions a reviewer will ask about

**Why cosine and a threshold instead of training a classifier?**
A classifier over enrolled people has no way to say "nobody." Visitors who were
never enrolled are the common case at a front door, so open-set rejection is the
requirement, and a similarity threshold gives it directly.

**Where does 0.675 come from?**
A sweep on the research test set: the smallest cut that still rejected ≥95% of
imposters. Safety-first, because accepting a stranger is worse than asking a
known visitor to step closer. It is tuned for FaceNet on **photos** — a live
480p camera is a different domain, so expect to re-check it (`--threshold`).

**Why only the largest face?**
Mirrors the fall worker's largest-person rule. ForgetMeNot targets elderly
patients living alone, so one-face-per-frame matches the deployment. Two people
at the door produce one event, for the larger face.

**Why the streak and cooldown?**
Without them, a visitor standing in frame for ten seconds generates one event
per processed frame. The caregiver would get dozens of notifications for one
visit.

---

## Performance notes

MTCNN is accurate and slow; Haar is roughly an order of magnitude faster on CPU
with slightly weaker unknown rejection. Whichever you pick, **enrollment and
live matching must use the same detector** — the crops differ systematically, so
mixing them shifts every score. The worker warns if the gallery metadata
disagrees with `--detector`.

On a slow CPU, raise `--skip` before lowering accuracy anywhere else.

---

## CLI knobs you'll use most

| Flag | Meaning |
| --- | --- |
| `--cam 0` | USB camera index |
| `--source clip.mp4` | File instead of camera |
| `--api-url URL` | FMN base URL (empty `""` = no POST) |
| `--patient-id N` | Integer `patients.id` |
| `--gallery PATH` | Local `.pkl` or photo folder |
| `--gallery-url URL` | Fetch enrolled vectors from the backend instead |
| `--detector mtcnn\|haar` | Must match enrollment |
| `--threshold 0.675` | Override the match cutoff |
| `--need 5` | Consecutive frames on one identity before firing |
| `--cooldown 120` | Seconds before the same identity fires again |
| `--skip N` | Process every Nth frame if CPU is slow |
| `--no-preview` | Headless |
| `--no-snapshot` | Don't write visitor JPEGs |
