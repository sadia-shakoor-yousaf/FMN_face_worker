"""
Care-PC face worker (door camera).

Camera -> detect largest face -> FaceNet embedding -> cosine match against the
enrolled visitor gallery -> streak + cooldown -> POST to FMN event API.

  python workers/face_worker.py --cam 0 --patient-id 1
  python workers/face_worker.py --source path/to/clip.mp4 --no-preview
  python workers/face_worker.py --cam 0 --no-post   # local log only

Gallery comes from a local .pkl (scripts/enroll_faces.py) or, with
--gallery-url, from the FMN backend. See docs/VISITOR_GALLERY_API.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_cv.camera import enumerate_named_cameras, open_camera, resolve_camera_index
from face_cv.fmn_client import DEFAULT_BASE_URL, FmnClientError, post_event
from face_cv.gallery_sync import GallerySyncError, cache_path, fetch_visitor_gallery
from face_cv.live import DEFAULT_COOLDOWN_S, DEFAULT_NEED, LiveFaceRecognizer
from face_cv.pipeline import (
    DEFAULT_DETECTOR,
    DEFAULT_THRESHOLD,
    UNKNOWN_LABEL,
    embed_face,
    identify,
    largest_face,
    load_visitor_gallery,
    to_confidence,
)

DEFAULT_GALLERY = ROOT / "data" / "faces" / "gallery.pkl"
DEFAULT_CACHE_DIR = ROOT / "data" / "faces"
DEFAULT_LOG_DIR = ROOT / "data" / "live_logs"


def _setup_log(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"face_worker_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    log = logging.getLogger("face_worker")
    log.info("log file: %s", log_path)
    return log


def _load_gallery(args: argparse.Namespace, log: logging.Logger):
    """Backend sync if --gallery-url, else local file/folder. Cache is fallback."""
    if not args.gallery_url:
        return load_visitor_gallery(args.gallery, patient_id=args.patient_id)

    if not args.patient_id:
        raise RuntimeError("--gallery-url requires --patient-id")

    try:
        gallery, meta = fetch_visitor_gallery(
            args.gallery_url,
            patient_id=args.patient_id,
            cache_dir=args.cache_dir,
            allow_model_mismatch=args.allow_model_mismatch,
        )
        log.info(
            "synced %d visitor(s) from %s (updated_at=%s)",
            len(gallery),
            args.gallery_url,
            meta.get("updated_at", "?"),
        )
        return gallery, meta
    except GallerySyncError as e:
        cached = cache_path(args.cache_dir, args.patient_id)
        if not cached.is_file():
            raise RuntimeError(f"{e}\nno local cache at {cached}") from e
        log.warning("sync failed (%s); falling back to cache %s", e, cached)
        return load_visitor_gallery(cached, patient_id=args.patient_id)


def _draw_overlay(
    frame: np.ndarray,
    *,
    box: tuple[int, int, int, int] | None,
    name: str | None,
    score: float,
    run: int,
    need: int,
    fps: float,
    event: bool,
) -> np.ndarray:
    out = frame.copy()
    known = name is not None and name != UNKNOWN_LABEL
    color = (0, 180, 0) if known else (0, 0, 255)

    if box is not None:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if name is not None:
            cv2.putText(
                out,
                f"{name} {score:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

    status = name if name is not None else "no face"
    lines = [f"face: {status}  streak {run}/{need}", f"FPS: {fps:.1f}"]
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28

    if event:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, 8)
        cv2.putText(
            out,
            "VISITOR EVENT",
            (12, out.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ForgetMeNot live face worker")
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
    p.add_argument(
        "--list-cams",
        action="store_true",
        help="Print camera indexes and names, then exit",
    )
    p.add_argument("--width", type=int, default=1920, help="Capture width (JLC 1080p)")
    p.add_argument("--height", type=int, default=1080, help="Capture height")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional video file instead of camera (demo / offline test)",
    )
    p.add_argument(
        "--patient-id",
        type=int,
        default=None,
        help="Patient integer id (required when posting to API)",
    )
    p.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"FMN base URL for POST /api/v1/events (default {DEFAULT_BASE_URL}). "
        "Use --no-post to disable posting.",
    )
    p.add_argument(
        "--no-post",
        action="store_true",
        help="Recognize and log locally without calling the API. Prefer this "
        'over --api-url "", which PowerShell silently drops.',
    )
    p.add_argument(
        "--gallery",
        type=Path,
        default=DEFAULT_GALLERY,
        help="Enrolled visitors (.pkl from scripts/enroll_faces.py, or a photo folder)",
    )
    p.add_argument(
        "--gallery-url",
        type=str,
        default="",
        help="Fetch visitors from the backend instead of --gallery "
        "(GET /api/v1/patients/<id>/visitors)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Where synced vectors are cached for offline restarts",
    )
    p.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="Match even if the backend used a different embedding model (unsafe)",
    )
    p.add_argument(
        "--detector",
        choices=["mtcnn", "haar"],
        default=DEFAULT_DETECTOR,
        help="Must match whatever you enrolled with. haar is far faster on CPU",
    )
    p.add_argument("--threshold", type=float, default=None, help="default from gallery")
    p.add_argument(
        "--need",
        type=int,
        default=DEFAULT_NEED,
        help=f"Consecutive frames on the same identity (default {DEFAULT_NEED})",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_S,
        help="Seconds before the same identity can fire again",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=3,
        help="Process every Nth frame (raise if CPU FPS is low)",
    )
    p.add_argument("--no-preview", action="store_true", help="Headless (log only)")
    p.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Do not save a local JPEG per visitor event",
    )
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames (tests)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = _setup_log(args.log_dir)

    if args.list_cams:
        cams = enumerate_named_cameras()
        if not cams:
            log.error("no cameras found")
            return 1
        for idx, name in cams:
            log.info("cam %s: %s", idx, name)
        return 0

    api_url = "" if args.no_post else (args.api_url or "").strip()
    post_enabled = bool(api_url)
    if post_enabled and not args.patient_id:
        log.warning(
            "POST enabled (%s) but --patient-id missing; events will not be posted",
            api_url,
        )

    try:
        gallery, meta = _load_gallery(args, log)
    except (FileNotFoundError, RuntimeError) as e:
        log.error("%s", e)
        return 1
    if not gallery:
        log.error("gallery is empty (no enrolled visitors)")
        return 1

    threshold = args.threshold
    if threshold is None:
        threshold = float(meta.get("threshold") or DEFAULT_THRESHOLD)

    enrolled_with = meta.get("detector")
    if enrolled_with and enrolled_with != args.detector:
        log.warning(
            "gallery was enrolled with %s but running %s; crops differ, so scores "
            "shift. Re-enroll with --detector %s or run with --detector %s",
            enrolled_with,
            args.detector,
            args.detector,
            enrolled_with,
        )

    recognizer = LiveFaceRecognizer(need=args.need, cooldown_s=args.cooldown)

    cam_label = None
    if args.source is not None:
        source = args.source
        cap = cv2.VideoCapture(source)
    else:
        try:
            cam_index, cam_label = resolve_camera_index(args.cam, args.cam_name)
        except RuntimeError as e:
            log.error("%s", e)
            return 1
        source = cam_index
        cap = open_camera(cam_index, args.width, args.height)
    if not cap.isOpened():
        log.error("cannot open source: %s", source)
        return 1
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if cam_label:
        log.info("camera %s (%s) %sx%s", source, cam_label, got_w, got_h)

    skip = max(int(args.skip), 1)
    log.info(
        "start source=%s patient_id=%s people=%d threshold=%.3f need=%s "
        "cooldown=%.0fs detector=%s skip=%s preview=%s api_url=%s",
        source,
        args.patient_id or "(none)",
        len(gallery),
        threshold,
        args.need,
        args.cooldown,
        args.detector,
        skip,
        not args.no_preview,
        api_url or "(disabled)",
    )
    log.info("enrolled: %s", ", ".join(sorted(gallery)))

    t0 = time.perf_counter()
    n_proc = 0
    n_read = 0
    fps_ema = 0.0
    flash_until = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.info("end of stream")
                break
            n_read += 1
            if (n_read - 1) % skip != 0:
                continue

            t_frame = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            got = largest_face(rgb, args.detector)

            if got is None:
                name: str | None = None
                score = 0.0
                box = None
            else:
                face, box = got
                name, score = identify(embed_face(face), gallery, threshold)

            now = time.perf_counter()
            event = recognizer.push(name, score, now_s=now)

            n_proc += 1
            inst = 1.0 / max(time.perf_counter() - t_frame, 1e-6)
            fps_ema = inst if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * inst)

            if event is not None:
                flash_until = now + 2.0
                event_type = (
                    "visitor_recognized" if event.known else "visitor_unknown"
                )
                confidence = to_confidence(event.confidence)
                log.warning(
                    "VISITOR %s name=%s score=%.3f patient_id=%s",
                    event_type,
                    event.name,
                    event.confidence,
                    args.patient_id or "(none)",
                )
                if not args.no_snapshot:
                    shot = args.log_dir / (
                        f"visitor_{time.strftime('%Y%m%d_%H%M%S')}_{event.name}.jpg"
                    )
                    cv2.imwrite(str(shot), frame)
                    log.info("snapshot saved: %s", shot)
                if post_enabled and args.patient_id:
                    try:
                        posted = post_event(
                            api_url,
                            patient_id=args.patient_id,
                            event_type=event_type,
                            label=event.name,
                            confidence=confidence,
                            source="door_cam",
                        )
                        log.info("POST %s ok id=%s", event_type, posted.get("id", "?"))
                    except FmnClientError as e:
                        log.warning("POST %s failed: %s", event_type, e)

            if not args.no_preview:
                vis = _draw_overlay(
                    frame,
                    box=box,
                    name=name,
                    score=score,
                    run=recognizer.consecutive_hits,
                    need=args.need,
                    fps=fps_ema,
                    event=now < flash_until,
                )
                cv2.imshow("ForgetMeNot face worker", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    log.info("quit by user")
                    break

            if args.max_frames is not None and n_proc >= args.max_frames:
                log.info("hit --max-frames=%s", args.max_frames)
                break
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    avg = n_proc / elapsed if elapsed > 0 else 0.0
    log.info(
        "done processed=%d read=%d elapsed=%.1fs avg_FPS=%.2f ema_FPS=%.2f",
        n_proc,
        n_read,
        elapsed,
        avg,
        fps_ema,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
