"""
Enroll visitor faces from photos into a local gallery the face worker can load.

This is the local / offline path. In production the caregiver app uploads the
photos and the backend serves the vectors (docs/VISITOR_GALLERY_API.md).

Expected layout (folder name = the label the caregiver sees):

  data/faces/gallery/dr_khan/1.jpg 2.jpg ...
  data/faces/gallery/ammi/1.jpg 2.jpg ...

  python scripts/enroll_faces.py
  python scripts/enroll_faces.py --detector haar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_cv.pipeline import DEFAULT_THRESHOLD, build_gallery, save_gallery

DEFAULT_PHOTOS_DIR = ROOT / "data" / "faces" / "gallery"
DEFAULT_OUT = ROOT / "data" / "faces" / "gallery.pkl"
MIN_PHOTOS = 3


def main() -> int:
    p = argparse.ArgumentParser(description="Build the visitor face gallery")
    p.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--detector",
        choices=["mtcnn", "haar"],
        default="mtcnn",
        help="Enrollment is offline, so the slow accurate one is fine (default mtcnn)",
    )
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = p.parse_args()

    print(f"photos : {args.photos_dir}")
    print(f"detector: {args.detector}")
    try:
        gallery = build_gallery(args.photos_dir, detector_name=args.detector)
    except FileNotFoundError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    if not gallery:
        print(
            f"FAIL: no faces enrolled from {args.photos_dir}",
            file=sys.stderr,
        )
        return 1

    thin = {n: len(e) for n, e in gallery.items() if len(e) < MIN_PHOTOS}
    for name, count in thin.items():
        print(f"WARN: {name} has only {count} photo(s); add more for stable matching")

    save_gallery(
        gallery,
        args.out,
        detector_name=args.detector,
        threshold=args.threshold,
    )
    total = sum(len(e) for e in gallery.values())
    print(f"\nenrolled {len(gallery)} people, {total} embeddings")
    print(f"gallery -> {args.out}")
    print(f"threshold = {args.threshold}")
    print(
        f"\nnext: python workers/face_worker.py --cam 0 --patient-id 1 "
        f"--detector {args.detector}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
