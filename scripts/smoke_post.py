"""
POST one visitor event with no camera (API smoke test).

  python scripts/smoke_post.py
  python scripts/smoke_post.py --api-url https://YOUR_HOST --patient-id 1
  python scripts/smoke_post.py --type visitor_unknown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_cv.fmn_client import DEFAULT_BASE_URL, FmnClientError, post_event


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-test POST /api/v1/events")
    p.add_argument("--api-url", default=DEFAULT_BASE_URL)
    p.add_argument("--patient-id", type=int, default=1)
    p.add_argument(
        "--type",
        dest="event_type",
        choices=["visitor_recognized", "visitor_unknown"],
        default="visitor_recognized",
    )
    p.add_argument("--label", default=None, help="default: dr_khan / unknown")
    p.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="default: 0.82 recognized / 0.41 unknown",
    )
    p.add_argument("--source", default="door_cam")
    args = p.parse_args()

    recognized = args.event_type == "visitor_recognized"
    label = args.label if args.label is not None else (
        "dr_khan" if recognized else "unknown"
    )
    # An unknown visitor scored *below* the match threshold, so a high
    # confidence next to visitor_unknown would be a contradiction.
    confidence = args.confidence if args.confidence is not None else (
        0.82 if recognized else 0.41
    )

    try:
        out = post_event(
            args.api_url,
            patient_id=args.patient_id,
            event_type=args.event_type,
            label=label,
            confidence=confidence,
            source=args.source,
        )
    except FmnClientError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
