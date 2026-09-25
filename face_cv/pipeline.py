"""Face detect -> FaceNet embed -> gallery cosine match.

Same policy as the semester face-recognition project: largest face only,
512-d L2-normalized FaceNet embedding, cosine similarity against every
enrolled photo of a person (max wins), reject below threshold.

Threshold 0.675 comes from the custom-set sweep (smallest cut with
unknown-rejection >= 95%). It is valid only for FaceNet embeddings --
swapping the embedder means re-tuning it.

Anything that produces embeddings for this worker (here, or on the backend)
must match EMBEDDING_MODEL exactly. See docs/EMBEDDING_CONTRACT.md.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_THRESHOLD = 0.675
FACE_SIZE = (160, 160)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
UNKNOWN_LABEL = "unknown"
GALLERY_VERSION = 1

# Identity of the embedding space. Vectors from a different model or
# preprocessing path are not comparable, and mixing them fails silently.
EMBEDDING_MODEL = "keras-facenet-0.3.2"
EMBEDDING_DIM = 512

Gallery = dict[str, list[np.ndarray]]
Box = tuple[int, int, int, int]

_embedder = None
_detectors: dict[str, Any] = {}


def get_embedder():
    """Lazy-init FaceNet (keras-facenet, 512-d, 160x160 input)."""
    global _embedder
    if _embedder is None:
        try:
            from keras_facenet import FaceNet
        except ImportError as e:
            raise RuntimeError(
                "keras-facenet is not installed in this environment. "
                "This worker needs its own Python 3.11 venv: "
                "pip install -r requirements.txt"
            ) from e
        _embedder = FaceNet()
    return _embedder


class _MtcnnDetector:
    """Accurate, slow. Use for enrollment."""

    name = "mtcnn"

    def __init__(self) -> None:
        try:
            from mtcnn.mtcnn import MTCNN
        except ImportError as e:
            raise RuntimeError("mtcnn is not installed in this environment") from e
        self._impl = MTCNN()

    def boxes(self, img_rgb: np.ndarray) -> list[Box]:
        found = []
        for face in self._impl.detect_faces(img_rgb):
            x, y, w, h = face["box"]
            found.append((max(0, int(x)), max(0, int(y)), int(w), int(h)))
        return sorted(found, key=lambda b: b[2] * b[3], reverse=True)


class _HaarDetector:
    """~20x faster than MTCNN on CPU, slightly worse unknown rejection."""

    name = "haar"

    def __init__(self) -> None:
        cascade_path = (
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        self._impl = cv2.CascadeClassifier(str(cascade_path))
        if self._impl.empty():
            raise RuntimeError(f"cannot load Haar cascade: {cascade_path}")

    def boxes(self, img_rgb: np.ndarray) -> list[Box]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        found = self._impl.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in found]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def get_detector(name: str = "mtcnn"):
    """Lazy-init a face detector by name ('mtcnn' or 'haar')."""
    key = name.lower()
    if key not in _detectors:
        if key == "mtcnn":
            _detectors[key] = _MtcnnDetector()
        elif key == "haar":
            _detectors[key] = _HaarDetector()
        else:
            raise ValueError(f"unknown detector: {name!r} (use 'mtcnn' or 'haar')")
    return _detectors[key]


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_face(img_rgb: np.ndarray, box: Box) -> np.ndarray | None:
    x, y, w, h = box
    crop = img_rgb[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, FACE_SIZE)


def largest_face(
    img_rgb: np.ndarray,
    detector_name: str = "mtcnn",
) -> tuple[np.ndarray, Box] | None:
    """Return (160x160 RGB crop, box) for the biggest face, or None."""
    boxes = get_detector(detector_name).boxes(img_rgb)
    if not boxes:
        return None
    face = crop_face(img_rgb, boxes[0])
    if face is None:
        return None
    return face, boxes[0]


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


def embed_face(face_rgb: np.ndarray) -> np.ndarray:
    """160x160 RGB uint8 crop -> L2-normalized 512-d float vector."""
    face = face_rgb.astype("float32")
    if face.ndim == 3:
        face = np.expand_dims(face, axis=0)
    emb = get_embedder().embeddings(face)[0]
    return _l2_normalize(emb)


def score_against_gallery(query_emb: np.ndarray, gallery: Gallery) -> dict[str, float]:
    """Per-person similarity, max over that person's enrollment photos."""
    return {
        name: max(float(np.dot(query_emb, e)) for e in embs)
        for name, embs in gallery.items()
        if embs
    }


def identify(
    query_emb: np.ndarray,
    gallery: Gallery,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str, float]:
    """Return (person name, score) or (UNKNOWN_LABEL, best score)."""
    scores = score_against_gallery(query_emb, gallery)
    if not scores:
        return UNKNOWN_LABEL, 0.0
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    if best_score < threshold:
        return UNKNOWN_LABEL, best_score
    return best_name, best_score


def to_confidence(score: float) -> float:
    """Cosine is [-1, 1]; the event API only accepts [0, 1]."""
    return float(min(max(score, 0.0), 1.0))


def build_gallery(
    photos_dir: str | Path,
    *,
    detector_name: str = "mtcnn",
    verbose: bool = True,
) -> Gallery:
    """Embed every photo under ``photos_dir/<person_name>/<image>``."""
    photos_dir = Path(photos_dir)
    if not photos_dir.is_dir():
        raise FileNotFoundError(
            f"photo folder not found: {photos_dir}\n"
            f"expected layout: {photos_dir}/<person_name>/1.jpg"
        )

    gallery: Gallery = {}
    for person_dir in sorted(p for p in photos_dir.iterdir() if p.is_dir()):
        embeddings: list[np.ndarray] = []
        images = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        for img_path in images:
            try:
                got = largest_face(read_rgb(img_path), detector_name)
            except Exception as e:
                if verbose:
                    print(f"  [skip] {person_dir.name}/{img_path.name}: {e}")
                continue
            if got is None:
                if verbose:
                    print(f"  [skip] {person_dir.name}/{img_path.name}: no face found")
                continue
            embeddings.append(embed_face(got[0]))
        if embeddings:
            gallery[person_dir.name] = embeddings
            if verbose:
                print(f"  enrolled {person_dir.name}: {len(embeddings)} photo(s)")
        elif verbose:
            print(f"  [skip] {person_dir.name}: no usable photos")
    return gallery


def save_gallery(
    gallery: Gallery,
    path: str | Path,
    *,
    detector_name: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "version": GALLERY_VERSION,
        "detector": detector_name,
        "threshold": float(threshold),
        "embedding_model": EMBEDDING_MODEL,
        "people": gallery,
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def load_visitor_gallery(
    source: str | Path,
    *,
    patient_id: int | None = None,
) -> tuple[Gallery, dict[str, Any]]:
    """Load enrolled visitors from disk. Returns ({name: [embeddings]}, meta).

    This is the local half of the seam between the worker and wherever
    enrollments live: a .pkl written by scripts/enroll_faces.py, a folder of
    photos embedded on the spot, or a cache of a backend sync.

    For the backend path see face_cv.gallery_sync.fetch_visitor_gallery.
    The worker's frame loop is identical either way.
    """
    source = Path(source)
    if source.is_dir():
        gallery = build_gallery(source, verbose=False)
        return gallery, {
            "detector": "mtcnn",
            "threshold": DEFAULT_THRESHOLD,
            "embedding_model": EMBEDDING_MODEL,
        }

    if not source.is_file():
        raise FileNotFoundError(
            f"visitor gallery not found: {source}\n"
            "run: python scripts/enroll_faces.py"
        )

    with open(source, "rb") as f:
        blob = pickle.load(f)

    if isinstance(blob, dict) and "people" in blob:
        meta = {k: v for k, v in blob.items() if k != "people"}
        return blob["people"], meta
    return blob, {}
