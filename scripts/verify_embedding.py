"""
Prove two machines produce the same face embeddings.

If the backend generates enrollment vectors and the care PC generates live
vectors, both must land in the *same* embedding space. A mismatch does not
raise an error anywhere -- cosine scores just quietly stop meaning anything,
and every stranger scores like a match. This script is the guard.

The reference input is a fixed, seeded 160x160 array rather than a photo, so
the check is deterministic, needs no image files, and contains nobody's face.
It isolates the model + preprocessing + normalization. Pass --image to also
check the detector/crop half of the path.

  python scripts/verify_embedding.py --emit      # we do this once
  python scripts/verify_embedding.py             # backend runs this and compares
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_cv.pipeline import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FACE_SIZE,
    embed_face,
    largest_face,
    read_rgb,
)

REFERENCE_PATH = ROOT / "docs" / "conformance.json"
REFERENCE_SEED = 20260925
# Cosine below this between two runs means the embedding paths differ.
MIN_COSINE = 0.9999


def reference_input() -> np.ndarray:
    """Deterministic stand-in for a 160x160 RGB uint8 face crop."""
    rng = np.random.default_rng(REFERENCE_SEED)
    return rng.integers(0, 256, (*FACE_SIZE, 3), dtype=np.uint8)


def describe(vec: np.ndarray) -> dict:
    return {
        "dim": int(vec.shape[0]),
        "l2_norm": round(float(np.linalg.norm(vec)), 6),
        "first8": [round(float(v), 6) for v in vec[:8]],
        "sha256": hashlib.sha256(
            np.round(vec, 5).astype(np.float64).tobytes()
        ).hexdigest()[:16],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Embedding conformance check")
    p.add_argument(
        "--emit",
        action="store_true",
        help="Write this machine's vector as the reference (run once, by us)",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Also embed a real photo to check the detect+crop path",
    )
    p.add_argument("--detector", choices=["mtcnn", "haar"], default="mtcnn")
    args = p.parse_args()

    print(f"expected model: {EMBEDDING_MODEL}")
    print("embedding the fixed reference input ...")
    vec = embed_face(reference_input())
    info = describe(vec)

    if vec.shape != (EMBEDDING_DIM,):
        print(
            f"FAIL: got {vec.shape[0]}-d vector, expected {EMBEDDING_DIM}-d. "
            "Wrong model.",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(info, indent=2))

    if args.image is not None:
        got = largest_face(read_rgb(args.image), args.detector)
        if got is None:
            print(f"WARN: no face found in {args.image}")
        else:
            print(f"\n{args.image.name} ({args.detector}):")
            print(json.dumps(describe(embed_face(got[0])), indent=2))

    if args.emit:
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.write_text(
            json.dumps(
                {
                    "embedding_model": EMBEDDING_MODEL,
                    "seed": REFERENCE_SEED,
                    "input_shape": [*FACE_SIZE, 3],
                    "reference": info,
                    "vector": [round(float(v), 8) for v in vec],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nreference written -> {REFERENCE_PATH}")
        print("commit this file; the backend compares against it")
        return 0

    if not REFERENCE_PATH.is_file():
        print(
            f"\nno reference at {REFERENCE_PATH}\n"
            "run with --emit on the machine whose embeddings are authoritative",
            file=sys.stderr,
        )
        return 1

    ref = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    ref_vec = np.asarray(ref["vector"], dtype=np.float64)
    cosine = float(np.dot(vec, ref_vec))
    max_diff = float(np.max(np.abs(vec - ref_vec)))

    print(f"\nreference model: {ref['embedding_model']}")
    print(f"cosine vs reference : {cosine:.8f}")
    print(f"max abs difference  : {max_diff:.8f}")

    if cosine < MIN_COSINE:
        print(
            f"\nFAIL: cosine {cosine:.6f} < {MIN_COSINE}. This machine is NOT in "
            "the same embedding space.\nCheck, in order: model version, RGB vs "
            "BGR channel order, 160x160 resize, float32 cast, L2 normalization.\n"
            "See docs/EMBEDDING_CONTRACT.md",
            file=sys.stderr,
        )
        return 1

    print("\nPASS: same embedding space. Enrollment vectors are comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
