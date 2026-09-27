# Artifacts

FaceNet weights are **not** stored in this repo. `keras-facenet` downloads them on first `embed_face()` / `enroll_faces.py` run (~90 MB).

| File | Purpose |
| --- | --- |
| `embedding_reference.json` | Optional output of `scripts/verify_embedding.py --write-reference`. Used to prove two FaceNet installs produce the same 512-d vectors. |

No LSTM / YOLO files live here. This worker is FaceNet + a detector only.
