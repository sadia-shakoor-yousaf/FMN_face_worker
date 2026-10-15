# Embedding contract — read this before generating vectors

**Who needs this:** whoever turns an uploaded enrollment photo into a face vector.

If the care PC and the backend both produce embeddings, they must land in the
**same space**. This is not a nice-to-have. A mismatch raises no error anywhere:
the numbers stay valid, the API stays happy, and recognition silently degrades
to nonsense.

We measured this during the research phase. Vectors from a different model,
scored against a threshold tuned for FaceNet, gave **0% unknown rejection** —
every stranger was accepted as a known visitor. That is the failure mode.

---

## The pinned path

Both sides must do exactly this, in this order:

| Step | Requirement |
| --- | --- |
| 1. Read image | Any loader, but the array handed to step 2 must be **RGB** |
| 2. Detect face | OpenCV Haar frontal-face cascade, largest box by area (`w * h`) |
| 3. Crop | `img[y : y+h, x : x+w]`, clamping negative `x`/`y` to 0 |
| 4. Resize | `cv2.resize(crop, (160, 160))` |
| 5. Cast | `float32`, values kept in `0..255` (**no** /255 scaling) |
| 6. Embed | `keras_facenet.FaceNet().embeddings(batch)` → 512-d |
| 7. Normalize | Divide by the L2 norm, so `‖v‖ = 1` |

Pins: `tensorflow==2.13.0`, `keras-facenet==0.3.2`, `mtcnn==0.1.1`,
`scipy<1.14`, `numpy<2`. (`keras-facenet` imports scipy without declaring it,
so pip will not install it for you.)

Model identifier used on the wire: **`keras-facenet-0.3.2`**.

### Why Haar and not MTCNN in step 2

Steps 5–7 define the embedding space; step 2 only has to **match the live
worker**, because detector crops differ systematically and shift every score.
The live worker is stuck with Haar — MTCNN costs ~2s per frame on CPU, which no
door camera can absorb — so enrollment uses Haar too. Nothing stops the project
switching both sides to MTCNN later; they just have to move together.

Reference implementation is `face_cv/pipeline.py` in this repo
(`largest_face` + `embed_face`). Copying those two functions is the safest
route.

---

## The four things that break it silently

1. **BGR instead of RGB.** OpenCV's `imread` returns BGR; PIL returns RGB. Both
   produce a valid 512-d vector. Only one is correct. This is the most common
   failure, by a wide margin.
2. **A different FaceNet.** DeepFace's `Facenet512`, InsightFace, ArcFace, and
   `keras-facenet` are *different models*. Similar names, unrelated spaces.
3. **Scaling to 0..1.** `keras-facenet` does its own prewhitening internally.
   Dividing by 255 first shifts every vector.
4. **Forgetting the L2 normalization.** Then cosine similarity is no longer a
   plain dot product and the 0.675 threshold is meaningless.

---

## Prove it matches — one command

This repo ships a conformance check. It embeds a fixed, seeded 160×160 array
(deterministic, no image file, nobody's actual face) and compares against a
committed reference vector.

```bash
python scripts/verify_embedding.py
```

- `PASS` → same embedding space, enrollment vectors are comparable.
- `FAIL` → the printed checklist tells you which of the four causes to check.

The reference in `docs/conformance.json` is produced once, on the care PC, with
`python scripts/verify_embedding.py --emit`.

Run this **before** wiring up enrollment, not after the demo fails.

---

## Versioning

Store the model identifier on every Qdrant point as `embedding_model`
(`keras-facenet-0.3.2`). The worker refuses to match vectors from an identifier
it does not recognize, rather than producing confident nonsense. Collection,
payload, and the upsert shape are in [`VISITOR_GALLERY_API.md`](VISITOR_GALLERY_API.md).

If the model ever changes, every enrolled visitor must be re-embedded and the
threshold re-tuned. Thresholds do not transfer between models.

---

## Where the vector goes

The backend upserts each vector into Qdrant collection `visitor_faces`. The care PC does not embed enrollment photos on that path. FaceNet therefore runs in two places (enrollment service and care PC), which is why the conformance check above is mandatory.

Storing the JPEG and letting the care PC embed it would keep FaceNet in one place, but that is not what the worker does today. It only reads vectors.
