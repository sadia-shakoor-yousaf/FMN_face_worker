"""
Capture enrollment photos straight from the webcam.

Enrolling from the *same* camera the worker will watch removes the domain gap
between clean photographs and live 480p video, which is the usual reason a
known person comes back as "unknown".

Only frames where a face is actually detected get saved, so you cannot end up
with an empty crop or a picture of the ceiling.

  python scripts/capture_enrollment.py --name sadia
  python scripts/capture_enrollment.py --name ammi --count 8

Move a little between shots: turn your head slightly, change expression, take
glasses off. Six near-identical frames teach it one pose and nothing else.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_cv.camera import open_camera, resolve_camera_index
from face_cv.pipeline import DEFAULT_DETECTOR, get_detector

DEFAULT_OUT_DIR = ROOT / "data" / "faces" / "gallery"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture enrollment photos")
    p.add_argument(
        "--name",
        required=True,
        help="Folder name = the label the caregiver sees (e.g. dr_khan)",
    )
    p.add_argument("--count", type=int, default=6, help="Photos to capture")
    p.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Minimum seconds between saved photos, so they differ",
    )
    p.add_argument(
        "--cam",
        type=int,
        default=None,
        help="Camera index. Default: USB/JLC webcam (not the laptop camera).",
    )
    p.add_argument(
        "--cam-name",
        type=str,
        default=None,
        help='Substring of the device name, e.g. "USB 2.0" or "JLC"',
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--detector",
        choices=["mtcnn", "haar"],
        default=DEFAULT_DETECTOR,
        help=f"Only used to confirm a face is present (default {DEFAULT_DETECTOR})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    person_dir = args.out_dir / args.name
    person_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(person_dir.glob("*.jpg"))
    if existing:
        print(f"note: {person_dir} already has {len(existing)} photo(s); adding more")

    detector = get_detector(args.detector)
    try:
        cam_index, cam_label = resolve_camera_index(args.cam, args.cam_name)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"FAIL: cannot open camera {cam_index} ({cam_label})", file=sys.stderr)
        return 1
    print(f"camera {cam_index}: {cam_label}")

    print(f"saving to {person_dir}")
    print(f"target: {args.count} photos. Press q or Esc to stop early.\n")

    saved = len(existing)
    target = saved + args.count
    last_save = 0.0

    try:
        while saved < target:
            ok, frame = cap.read()
            if not ok:
                print("FAIL: camera stopped returning frames", file=sys.stderr)
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = detector.boxes(rgb)
            now = time.perf_counter()
            ready = now - last_save >= args.interval

            vis = frame.copy()
            if boxes:
                x, y, w, h = boxes[0]
                colour = (0, 180, 0) if ready else (0, 165, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)

            status = (
                "no face - move into frame"
                if not boxes
                else ("capturing..." if ready else "hold on, changing pose helps")
            )
            cv2.putText(
                vis,
                f"{saved - len(existing)}/{args.count}  {status}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 0) if boxes and ready else (0, 0, 255),
                2,
            )
            cv2.imshow("enrollment capture - q to quit", vis)

            if boxes and ready:
                saved += 1
                out = person_dir / f"{saved}.jpg"
                cv2.imwrite(str(out), frame)
                last_save = now
                print(f"  saved {out.name}  (face at {boxes[0]})")

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("stopped by user")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    total = len(sorted(person_dir.glob("*.jpg")))
    print(f"\n{person_dir} now has {total} photo(s)")
    if total < 3:
        print("WARN: fewer than 3 photos; matching will be unstable")
        return 1

    print("\nnext:")
    print("  python scripts/enroll_faces.py")
    print("  python workers/face_worker.py --cam 0 --no-post")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
