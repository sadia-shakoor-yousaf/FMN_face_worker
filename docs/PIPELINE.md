# Face worker pipeline

How the care-PC process turns door-camera frames into a visitor event.

---

## High-level flow

```text
Door camera / video file
        │
        ▼
   Face detector (Haar by default, MTCNN offline) → largest face box
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
        ├─ known: NEED consecutive frames (default 5) ──► visitor_recognized
        │
        └─ unknown for 10s (a gap under 1s does not reset the clock)
                │
                ▼
           Mic opens. Wait up to 30s for speech to *start*.
           The care PC embeds those few seconds and asks Qdrant
           collection visitor_voices for the nearest enrolled voice.
           Audio is not uploaded. The clip is saved under data/live_logs/
           next to the snapshot (voice_*.wav) so the match can be replayed.
                │
                ├─ voice names them ──► visitor_recognized
                ├─ speech, no match ──► visitor_unknown
                └─ nobody speaks ────► visitor_unknown

   If the face becomes known during the wait, the face wins and the mic closes.
   One event, then cooldown (default 120s) until that face has left.
   POST /api/v1/events   +   optional local JPEG snapshot
```

Enrolled vectors come from Qdrant (`--from-qdrant`, collection `visitor_faces`), a local `gallery.pkl` (`scripts/enroll_faces.py`), or the backend HTTP gallery (`--gallery-url`). The frame loop is identical either way. Qdrant is read once at startup, filtered by `patient_id`, then matched in memory.

---

## Modules in this repo

| Path | Role |
| --- | --- |
| `workers/face_worker.py` | Entrypoint: capture, overlay, logging, POST |
| `face_cv/pipeline.py` | Detectors, FaceNet embedding, gallery matching |
| `face_cv/visit.py` | Face-first visit decision, including the voice wait |
| `face_cv/speech.py` | Energy gate: wait for speech, then a short clip |
| `face_cv/mic.py` | Microphone session, opened only for that wait |
| `face_cv/speakers.py` | Turn one Qdrant voice hit into a name or a miss |
| `face_cv/speaker_embed.py` | ECAPA embedding of the live clip, care PC only |
| `face_cv/voice_qdrant.py` | Search Qdrant collection `visitor_voices` |
| `face_cv/live.py` | Older streak helper. The worker decides in `visit.py` |
| `face_cv/qdrant_gallery.py` | Scroll enrolled vectors from Qdrant and cache them |
| `face_cv/gallery_sync.py` | Fetch + cache enrolled vectors from the HTTP gallery |
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

**Why does an unknown face wait, then listen?**
A known face still announces on the short streak. Voice runs only after the
face has been unknown for about 10 seconds, and only long enough for someone
to start talking (about 30 seconds). The model scores a few seconds of that
speech, not a half-minute recording. Silence at the end of the wait is still
an unknown visitor: the face already failed. One event is posted either way.
The live clip is embedded on the care PC. The enrolled vectors are in Qdrant
`visitor_voices`, and the nearest-neighbor score comes from that search.
The backend receives the visitor event and nothing else.

---

## Performance notes

Measured on the development machine, 640×480 frames:

| Stage | Cost |
| --- | --- |
| Haar detection | ~190 ms/frame |
| MTCNN detection | ~2000 ms/frame (plus ~10 s one-off init) |
| FaceNet embedding | ~50 ms per detected face |
| End-to-end, Haar | ~5–8 FPS |

MTCNN is therefore **offline-only** in practice: at 0.5 FPS a `--need 5` streak
would take ten seconds of standing still. Haar is the default on both sides.

Whichever you pick, **enrollment and live matching must use the same detector**
— the crops differ systematically, so mixing them shifts every score. The worker
warns if the gallery metadata disagrees with `--detector`.

What has *not* been measured: whether Haar's looser crops cost accuracy versus
MTCNN's on real door-camera footage. The research comparison was done on clean
photos, where both reached the same Rank-1.

On a slow CPU, raise `--skip` before lowering accuracy anywhere else.

---

## CLI knobs you'll use most

| Flag | Meaning |
| --- | --- |
| `--cam 0` | USB camera index |
| `--source clip.mp4` | File instead of camera |
| `--api-url URL` | FMN base URL for the event POST |
| `--no-post` | Recognize and log only, no API call |
| `--patient-id N` | Integer `patients.id` |
| `--from-qdrant` | Load this patient's vectors from Qdrant (`URL` + `API_KEY` in `.env`) |
| `--qdrant-collection NAME` | Collection override (default `visitor_faces`) |
| `--gallery PATH` | Local `.pkl` or photo folder |
| `--gallery-url URL` | Fetch enrolled vectors from the HTTP gallery instead |
| `--detector mtcnn\|haar` | Must match enrollment |
| `--threshold 0.675` | Override the match cutoff |
| `--need 5` | Consecutive frames before a *known* face fires |
| `--unknown-hold 10` | Seconds of unknown face before the mic opens |
| `--voice-wait 30` | Seconds to wait for speech to start |
| `--dropout 1` | Face gap that does not reset the unknown clock |
| `--no-voice` | Post unknown when the face hold ends, no microphone |
| `--voice-collection visitor_voices` | Qdrant collection of enrolled voice vectors |
| `--voice-threshold 0.25` | Cosine cutoff on the Qdrant search score |
| `--speech-rms 0.02` | Mic energy that counts as speech starting |
| `--cooldown 120` | Seconds before the same identity fires again |
| `--skip N` | Process every Nth frame if CPU is slow |
| `--no-preview` | Headless |
| `--no-snapshot` | Don't write visitor JPEGs |
