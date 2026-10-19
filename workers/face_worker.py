"""
Care-PC face worker (door camera).

Camera -> detect largest face -> FaceNet embedding -> cosine match against the
enrolled visitor gallery. A known face posts immediately. An unknown face is
held, then the mic gets one chance to name them, and one visitor event is posted.

  python workers/face_worker.py --cam 0 --patient-id 1
  python workers/face_worker.py --source path/to/clip.mp4 --no-preview
  python workers/face_worker.py --cam 0 --no-post   # local log only

Gallery comes from Qdrant (--from-qdrant), a local .pkl
(scripts/enroll_faces.py), or, with --gallery-url, the FMN backend.
See docs/VISITOR_GALLERY_API.md.
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
from face_cv.mic import Ear, SpeechMatched, SpeechStarted
from face_cv.speech import write_wav
from face_cv.gallery_sync import GallerySyncError, cache_path, fetch_visitor_gallery
from face_cv.speakers import DEFAULT_SPEAKER_THRESHOLD, VOICE_COLLECTION
from face_cv.voice_qdrant import count_voices
from face_cv.qdrant_gallery import fetch_qdrant_gallery
from face_cv.visit import (
    DEFAULT_COOLDOWN_S,
    DEFAULT_DROPOUT_S,
    DEFAULT_NEED,
    DEFAULT_UNKNOWN_HOLD_S,
    DEFAULT_VOICE_WAIT_S,
    VisitDecider,
    VisitStep,
)
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
    """Qdrant, backend sync, or local file. Remote loads fall back to cache."""
    if args.from_qdrant and args.gallery_url:
        raise RuntimeError("use either --from-qdrant or --gallery-url, not both")

    if args.from_qdrant:
        if not args.patient_id:
            raise RuntimeError("--from-qdrant requires --patient-id")
        try:
            gallery, meta = fetch_qdrant_gallery(
                patient_id=args.patient_id,
                collection=args.qdrant_collection,
                cache_dir=args.cache_dir,
                allow_model_mismatch=args.allow_model_mismatch,
            )
            log.info(
                "synced %d visitor(s) from Qdrant collection %s",
                len(gallery),
                meta.get("collection", args.qdrant_collection),
            )
            return gallery, meta
        except GallerySyncError as e:
            return _gallery_from_cache(args, log, e)

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
        return _gallery_from_cache(args, log, e)


def _log_voices(args: argparse.Namespace, log: logging.Logger) -> None:
    """Confirm Qdrant has this patient's voice points. Matching happens later."""
    if args.no_voice:
        return
    if not args.patient_id:
        log.warning("voice match needs --patient-id; speech will not name anyone")
        return
    try:
        n = count_voices(patient_id=args.patient_id, collection=args.voice_collection)
    except GallerySyncError as e:
        log.warning(
            "voice collection %s is not readable (%s); speech will not name anyone",
            args.voice_collection,
            e,
        )
        return
    log.info(
        "voices: %d point(s) in Qdrant %s for patient %s threshold=%.3f",
        n,
        args.voice_collection,
        args.patient_id,
        args.voice_threshold,
    )


def _gallery_from_cache(args: argparse.Namespace, log: logging.Logger, error: Exception):
    cached = cache_path(args.cache_dir, args.patient_id)
    if not cached.is_file():
        raise RuntimeError(f"{error}\nno local cache at {cached}") from error
    log.warning("sync failed (%s); falling back to cache %s", error, cached)
    return load_visitor_gallery(cached, patient_id=args.patient_id)


def _draw_overlay(
    frame: np.ndarray,
    *,
    box: tuple[int, int, int, int] | None,
    name: str | None,
    score: float,
    status: str,
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

    lines = [status, f"FPS: {fps:.1f}"]
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


def _status_line(decider: VisitDecider, *, name: str | None, now_s: float) -> str:
    face = name if name is not None else "no face"
    if decider.phase == "listen":
        waited = decider.listen_elapsed(now_s)
        return f"listening for speech {waited:.0f}/{decider.voice_wait_s:.0f}s  face: {face}"
    unknown_for = decider.unknown_elapsed(now_s)
    if unknown_for > 0:
        return (
            f"face unknown {unknown_for:.0f}/{decider.unknown_hold_s:.0f}s  "
            f"streak {decider.consecutive_hits}/{decider.need}"
        )
    return f"face: {face}  streak {decider.consecutive_hits}/{decider.need}"


def _publish_visit(
    step: VisitStep,
    *,
    frame: np.ndarray,
    now_s: float,
    args: argparse.Namespace,
    api_url: str,
    post_enabled: bool,
    ear: Ear,
    log: logging.Logger,
    flash_until: float,
) -> float:
    if step.start_listen:
        log.info("face unknown long enough; opening microphone")
        ear.start()
    if step.event is not None:
        event = step.event
        event_type = "visitor_recognized" if event.known else "visitor_unknown"
        confidence = to_confidence(event.confidence)
        if event.via == "silence":
            log.info("voice wait ended with no speech")
        log.warning(
            "VISITOR %s name=%s score=%.3f via=%s patient_id=%s",
            event_type,
            event.name,
            event.confidence,
            event.via,
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
        flash_until = now_s + 2.0
    if step.stop_listen:
        ear.stop()
    return flash_until


def _clip_label(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return safe or "nomatch"


def _write_voice_clip(note: SpeechMatched, log_dir: Path, log: logging.Logger) -> Path | None:
    audio = note.audio
    if audio is None or int(audio.size) == 0 or note.sample_rate <= 0:
        return None
    label = _clip_label(note.match.name) if note.match.matched else "nomatch"
    path = log_dir / f"voice_{time.strftime('%Y%m%d_%H%M%S')}_{label}.wav"
    try:
        write_wav(path, audio, note.sample_rate)
    except OSError as e:
        log.warning("voice clip not saved: %s", e)
        return None
    return path


def _poll_ear(
    ear: Ear,
    decider: VisitDecider,
    log: logging.Logger,
    args: argparse.Namespace,
) -> VisitStep:
    step = VisitStep()
    now = time.perf_counter()
    for note in ear.poll():
        if isinstance(note, SpeechStarted):
            decider.note_speech(now)
            log.info("speech started; matching a short clip")
        elif isinstance(note, SpeechMatched):
            found = note.match
            clip = None if args.no_snapshot else _write_voice_clip(note, args.log_dir, log)
            samples = 0 if note.audio is None else int(note.audio.size)
            log.warning(
                "VOICE matched=%s name=%s score=%.3f relationship=%s threshold=%.3f "
                "samples=%d clip=%s",
                found.matched,
                found.name or "(none)",
                found.confidence,
                found.relationship or "(none)",
                args.voice_threshold,
                samples,
                clip or "(not saved)",
            )
            step = decider.resolve_voice(
                matched=found.matched,
                name=found.name,
                confidence=found.confidence,
                now_s=now,
            )
    return step


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
        "--from-qdrant",
        action="store_true",
        help="Load enrolled embeddings from Qdrant (URL and API_KEY in .env) "
        "instead of --gallery. Requires --patient-id.",
    )
    p.add_argument(
        "--qdrant-collection",
        type=str,
        default=None,
        help="Qdrant collection name (default visitor_faces, or QDRANT_COLLECTION)",
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
        "--unknown-hold",
        type=float,
        default=DEFAULT_UNKNOWN_HOLD_S,
        help="Seconds an unknown face must stay unknown before the mic opens",
    )
    p.add_argument(
        "--voice-wait",
        type=float,
        default=DEFAULT_VOICE_WAIT_S,
        help="Seconds to wait for speech to start after the face gives up",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT_S,
        help="Face gap, in seconds, that does not reset the unknown clock",
    )
    p.add_argument(
        "--no-voice",
        action="store_true",
        help="Post the unknown visitor when the face hold ends, without the mic",
    )
    p.add_argument(
        "--speech-rms",
        type=float,
        default=0.02,
        help="Mic energy that counts as speech starting",
    )
    p.add_argument(
        "--voice-collection",
        type=str,
        default=VOICE_COLLECTION,
        help=f"Qdrant collection of enrolled voice vectors (default {VOICE_COLLECTION})",
    )
    p.add_argument(
        "--voice-threshold",
        type=float,
        default=DEFAULT_SPEAKER_THRESHOLD,
        help=f"Speaker cosine cutoff returned by Qdrant (default {DEFAULT_SPEAKER_THRESHOLD})",
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

    decider = VisitDecider(
        need=args.need,
        cooldown_s=args.cooldown,
        unknown_hold_s=args.unknown_hold,
        voice_wait_s=args.voice_wait,
        dropout_s=args.dropout,
        voice=not args.no_voice,
    )
    _log_voices(args, log)
    ear = Ear(
        patient_id=args.patient_id,
        threshold=args.voice_threshold,
        collection=args.voice_collection,
        log=log,
        rms_threshold=args.speech_rms,
    )

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
        "cooldown=%.0fs unknown_hold=%.0fs voice_wait=%.0fs voice=%s "
        "detector=%s skip=%s preview=%s api_url=%s",
        source,
        args.patient_id or "(none)",
        len(gallery),
        threshold,
        args.need,
        args.cooldown,
        args.unknown_hold,
        args.voice_wait,
        not args.no_voice,
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
            n_proc += 1
            inst = 1.0 / max(time.perf_counter() - t_frame, 1e-6)
            fps_ema = inst if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * inst)

            publish = dict(
                frame=frame,
                now_s=now,
                args=args,
                api_url=api_url,
                post_enabled=post_enabled,
                ear=ear,
                log=log,
            )
            flash_until = _publish_visit(decider.push(name, score, now), flash_until=flash_until, **publish)
            flash_until = _publish_visit(
                _poll_ear(ear, decider, log, args),
                flash_until=flash_until,
                **publish,
            )
            flash_until = _publish_visit(decider.tick(now), flash_until=flash_until, **publish)

            if not args.no_preview:
                vis = _draw_overlay(
                    frame,
                    box=box,
                    name=name,
                    score=score,
                    status=_status_line(decider, name=name, now_s=now),
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
        ear.stop()
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
