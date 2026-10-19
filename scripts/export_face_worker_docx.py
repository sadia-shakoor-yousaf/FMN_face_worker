"""Build a Word document of the face-worker docs with real table formatting."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

OUT = Path(__file__).resolve().parents[1] / "FMN_Face_Worker_Documentation.docx"

NAVY = "1F4E79"
NAVY_RGB = RGBColor(0x1F, 0x4E, 0x79)
ALT = "D6E3F0"
WHITE = "FFFFFF"


def shade(cell, hex_color: str) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    el = OxmlElement("w:shd")
    el.set(qn("w:fill"), hex_color)
    el.set(qn("w:val"), "clear")
    tcPr.append(el)


def set_cell_borders(cell) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "8")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "8FAADC")
        borders.append(el)
    tcPr.append(borders)


def set_run(run, *, size=11, bold=False, color=None, name="Calibri") -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), name)


def add_para(doc, text, *, style=None, size=11, bold=False, space_after=8):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    run = p.add_run(text)
    set_run(run, size=size, bold=bold)
    return p


def heading(doc, text, level=1) -> None:
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = NAVY_RGB
        run.font.name = "Calibri"


def add_table(doc, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(h)
        set_run(run, size=11, bold=True, color=RGBColor(255, 255, 255))
        shade(cell, NAVY)
        set_cell_borders(cell)
    for r_i, row in enumerate(rows):
        fill = ALT if r_i % 2 == 0 else WHITE
        for c_i, val in enumerate(row):
            cell = table.rows[r_i + 1].cells[c_i]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(val)
            set_run(run, size=10.5)
            shade(cell, fill)
            set_cell_borders(cell)
    doc.add_paragraph()


def bullet(doc, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.clear()
    run = p.add_run(text)
    set_run(run, size=11)


def numbered(doc, text: str) -> None:
    p = doc.add_paragraph(style="List Number")
    p.clear()
    run = p.add_run(text)
    set_run(run, size=11)


def code_block(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.left_indent = Cm(0.4)
    run = p.add_run(text)
    set_run(run, size=9.5, name="Consolas")
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F2F2F2")
    shd.set(qn("w:val"), "clear")
    pPr.append(shd)


def build() -> Path:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2)
    section.right_margin = Cm(2)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("ForgetMeNot")
    set_run(r, size=26, bold=True, color=NAVY_RGB)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Care-PC Face / Visitor Worker")
    set_run(r, size=18, bold=True, color=NAVY_RGB)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run("Technical documentation  ·  FYP  ·  enrollment, pipeline, and API contracts")
    set_run(r, size=11, color=RGBColor(0x5B, 0x5B, 0x5B))

    heading(doc, "1. Purpose", 1)
    add_para(
        doc,
        "Local visitor-recognition worker for the ForgetMeNot (FMN) FYP. It runs on a "
        "care-PC / laptop, watches a door camera (or a demo video), decides who is at "
        "the door, and posts a small JSON event to the FMN backend. Camera frames stay "
        "on the care PC. Only the event JSON is sent.",
    )
    add_para(
        doc,
        "Sibling of FMN_fall_worker: same event endpoint and packaging, different sensor. "
        "patient_id is an integer (patients.id). Push notifications (FCM) are handled "
        "after activity_logs is written.",
    )

    heading(doc, "2. End-to-end pipeline", 1)
    add_table(
        doc,
        ["Step", "What happens"],
        [
            ["1. Capture", "Door camera / video file (USB/JLC 1080p by default, not the laptop cam)"],
            ["2. Detect", "Haar (default) or MTCNN (offline) finds face boxes; largest face only"],
            ["3. Crop", "Face crop resized to 160×160 RGB"],
            ["4. Embed", "FaceNet → 512-d embedding, L2-normalized"],
            ["5. Match", "Cosine similarity vs every enrolled photo; best score per person wins"],
            ["6. Decide", "Score ≥ 0.675 → visitor name; below → unknown"],
            ["7. Persist", "NEED consecutive frames (default 5) on the same identity"],
            ["8. Cooldown", "Per-identity cooldown (default 120 s) skips duplicate visits"],
            ["9. Notify", "POST /api/v1/events + optional local JPEG snapshot (never uploaded)"],
        ],
    )

    heading(doc, "3. Requirements", 1)
    add_table(
        doc,
        ["Item", "Notes"],
        [
            ["OS", "Windows / macOS / Linux"],
            ["Python", "3.11 only (.python-version). TensorFlow 2.13 does not support 3.12+"],
            ["Hardware", "CPU. TensorFlow 2.13 on Windows is CPU-only"],
            ["Camera", "USB webcam (JLC / USB 2.0 Camera) or an .mp4 clip for offline demo"],
        ],
    )

    heading(doc, "4. Project layout", 1)
    add_table(
        doc,
        ["Path", "Role"],
        [
            ["workers/face_worker.py", "Entrypoint: capture, overlay, logging, POST"],
            ["face_cv/pipeline.py", "Detectors, FaceNet embedding, gallery matching"],
            ["face_cv/live.py", "Streak + per-identity cooldown policy"],
            ["face_cv/gallery_sync.py", "Fetch + cache enrolled vectors from the backend"],
            ["face_cv/fmn_client.py", "HTTP POST client for the event API"],
            ["face_cv/camera.py", "USB/JLC camera auto-select (skip laptop webcam)"],
            ["scripts/capture_enrollment.py", "Webcam → enrollment photos"],
            ["scripts/enroll_faces.py", "Photos folder → local gallery.pkl"],
            ["scripts/verify_embedding.py", "Prove two machines share an embedding space"],
            ["scripts/smoke_post.py", "POST a visitor event with no camera"],
            ["mock_api/app.py", "Local stand-in for both endpoints"],
            ["docs/", "Pipeline + API contracts"],
            ["data/", "Gallery, logs, snapshots (gitignored biometric data)"],
        ],
    )
    add_para(
        doc,
        "No model weights are committed. FaceNet weights download on first run; Haar cascades ship inside OpenCV.",
    )

    heading(doc, "5. Setup", 1)
    code_block(
        doc,
        "py -3.11 -m venv .venv\n"
        ".venv\\Scripts\\activate\n"
        "python -m pip install --upgrade pip\n"
        "pip install -r requirements.txt",
    )
    add_para(
        doc,
        "The pins are load-bearing. keras-facenet loads a Keras-2 model that TensorFlow "
        "2.16+ (Keras 3) refuses, and this package generation predates numpy 2.x. Do not "
        "upgrade TensorFlow or numpy.",
    )
    add_para(
        doc,
        "The mock API needs a separate venv (.venv-mock). TensorFlow 2.13 pins "
        "typing-extensions<4.6 while FastAPI’s pydantic v2 needs >=4.6.1.",
        bold=False,
    )

    heading(doc, "6. Enrollment (enroll_faces.py)", 1)
    add_para(
        doc,
        "This is the local / offline path. In production the caregiver app uploads photos "
        "and the backend serves vectors (Visitor Gallery API). Folder name = the label "
        "the caregiver sees.",
    )
    add_table(
        doc,
        ["Step", "Command", "Result"],
        [
            [
                "1. Capture",
                "python scripts/capture_enrollment.py --name dr_khan",
                "Saves only frames with a detected face from the JLC webcam",
            ],
            [
                "2. Repeat",
                "python scripts/capture_enrollment.py --name ammi",
                "One folder per visitor under data/faces/gallery/",
            ],
            [
                "3. Embed",
                "python scripts/enroll_faces.py",
                "Haar detector (default) builds data/faces/gallery.pkl",
            ],
            [
                "4. Optional",
                "python scripts/enroll_faces.py --detector mtcnn",
                "Slower; worker must use --detector mtcnn as well",
            ],
        ],
    )
    add_table(
        doc,
        ["enroll_faces.py flag", "Default", "Meaning"],
        [
            ["--photos-dir", "data/faces/gallery", "Root folder of visitor subfolders"],
            ["--out", "data/faces/gallery.pkl", "Serialized gallery the worker loads"],
            ["--detector", "haar", "Must match the live worker detector"],
            ["--threshold", "0.675", "Stored with the gallery; worker uses it unless overridden"],
        ],
    )
    add_para(
        doc,
        "Expected layout: data/faces/gallery/dr_khan/*.jpg and data/faces/gallery/ammi/*.jpg. "
        "Five or six varied photos per person. Enrollment and live matching must use the same detector.",
    )

    heading(doc, "7. Where enrolled faces come from", 1)
    add_table(
        doc,
        ["Source", "Flag", "Use"],
        [
            ["Local gallery.pkl", "--gallery (default)", "Development, demos, offline"],
            ["Photo folder", "--gallery data/faces/gallery", "Re-embed on the spot"],
            ["FMN backend", "--gallery-url https://host", "Production; caregivers enrol in the app"],
        ],
    )
    add_para(
        doc,
        "The backend path caches vectors under data/faces/, so a restart during a network "
        "outage still recognizes people. If the backend reports a different embedding_model, "
        "sync fails loudly instead of silently accepting strangers.",
    )

    heading(doc, "8. Run the worker", 1)
    add_table(
        doc,
        ["Mode", "Command"],
        [
            [
                "Live JLC camera + API",
                "python workers/face_worker.py --api-url http://127.0.0.1:5155 --patient-id 39",
            ],
            ["List cameras", "python workers/face_worker.py --list-cams"],
            [
                "Force USB index 1",
                "python workers/face_worker.py --cam 1 --api-url http://127.0.0.1:5155 --patient-id 39",
            ],
            [
                "Video file, headless",
                "python workers/face_worker.py --source clip.mp4 --no-preview --api-url http://127.0.0.1:5155 --patient-id 1",
            ],
            ["Local only (no POST)", "python workers/face_worker.py --no-post"],
        ],
    )
    add_para(
        doc,
        "On this PC, cam 0 is HP Wide Vision HD Camera (laptop) and cam 1 is USB 2.0 Camera "
        "(JLC 1080p). With no --cam flag the worker auto-picks the JLC. Use --no-post, not "
        "--api-url \"\": PowerShell silently discards an empty string.",
    )

    heading(doc, "9. Useful flags", 1)
    add_table(
        doc,
        ["Flag", "Example", "Meaning"],
        [
            ["--api-url", "https://api...", "Base URL for the event POST"],
            ["--no-post", "—", "Recognize and log only, no API call"],
            ["--patient-id", "1", "Required for POST"],
            ["--gallery", "data/faces/gallery.pkl", "Local enrolled visitors"],
            ["--gallery-url", "https://api...", "Fetch enrolled vectors from the backend"],
            ["--detector", "mtcnn / haar", "Must match enrollment"],
            ["--threshold", "0.675", "Match cutoff (default comes from the gallery)"],
            ["--need", "5", "Consecutive frames on one identity before firing"],
            ["--cooldown", "120", "Seconds before the same identity fires again"],
            ["--source", "clip.mp4", "Offline demo"],
            ["--cam", "auto USB/JLC", "Webcam index (1 = USB 2.0 Camera on this PC)"],
            ["--cam-name", "USB 2.0", "Pick camera by name substring"],
            ["--list-cams", "—", "Print camera indexes and names, then exit"],
            ["--skip", "3", "Every Nth frame (help slow CPUs)"],
            ["--no-preview", "—", "Headless"],
            ["--no-snapshot", "—", "Do not write visitor JPEGs"],
        ],
    )

    heading(doc, "10. Performance (640×480, development machine)", 1)
    add_table(
        doc,
        ["Stage", "Cost"],
        [
            ["Haar detection", "~190 ms/frame"],
            ["MTCNN detection", "~2000 ms/frame (plus ~10 s one-off init)"],
            ["FaceNet embedding", "~50 ms per detected face"],
            ["End-to-end, Haar", "~5–8 FPS"],
        ],
    )
    add_para(
        doc,
        "MTCNN is therefore offline-only in practice: at 0.5 FPS a --need 5 streak would "
        "take ten seconds of standing still. Haar is the default on both sides.",
    )

    heading(doc, "11. Event API — POST /api/v1/events", 1)
    add_para(
        doc,
        "Same endpoint as the fall worker. If fall_detected is already implemented, accept "
        "two more type values and source: door_cam. Auth is not required for this endpoint. "
        "Frames never leave the care PC.",
    )
    add_table(
        doc,
        ["Property", "Value"],
        [
            ["Method", "POST"],
            ["Path", "/api/v1/events"],
        ],
    )
    add_table(
        doc,
        ["You receive", "You do not receive / run"],
        [
            ["Small JSON event about a patient", "Camera frames or visitor snapshots"],
            ["patient_id as int", "Caregiver UUID in this payload"],
            ["Activity types for activity_logs", "FaceNet / MTCNN inference for live frames"],
        ],
    )
    heading(doc, "Request body fields", 2)
    add_table(
        doc,
        ["Field", "Type", "Rules"],
        [
            ["patient_id", "integer", "Existing patient primary key (patients.id)"],
            ["type", "string enum", "fall_detected | visitor_recognized | visitor_unknown"],
            ["label", "string", "Visitor display name, or unknown. Fall worker sends floor_down"],
            ["confidence", "float", "0.0–1.0 (cosine score, not a probability)"],
            ["source", "string enum", "room_cam | door_cam"],
        ],
    )
    heading(doc, "Example — recognized visitor", 2)
    code_block(
        doc,
        "{\n"
        '  "patient_id": 1,\n'
        '  "type": "visitor_recognized",\n'
        '  "label": "dr_khan",\n'
        '  "confidence": 0.82,\n'
        '  "source": "door_cam"\n'
        "}",
    )
    heading(doc, "Example — unknown visitor", 2)
    code_block(
        doc,
        "{\n"
        '  "patient_id": 1,\n'
        '  "type": "visitor_unknown",\n'
        '  "label": "unknown",\n'
        '  "confidence": 0.41,\n'
        '  "source": "door_cam"\n'
        "}",
    )
    heading(doc, "HTTP errors", 2)
    add_table(
        doc,
        ["Status", "When"],
        [
            ["201 Created", "Event persisted; body echoes id, created_at, and the posted fields"],
            ["400", "Empty body or malformed JSON"],
            ["422", "Invalid / missing fields, unknown enums, bad confidence, unknown patient_id"],
        ],
    )

    heading(doc, "12. Visitor Gallery API — GET /api/v1/patients/{id}/visitors", 1)
    add_para(
        doc,
        "The only extra endpoint besides POST /api/v1/events. The worker fetches this list "
        "at startup, caches it, and re-fetches periodically. It does not query per video frame.",
    )
    add_table(
        doc,
        ["Property", "Value"],
        [
            ["Method", "GET"],
            ["Path", "/api/v1/patients/{patient_id}/visitors"],
            ["Auth", "Same as the rest of the API; worker can send a bearer token"],
        ],
    )
    add_table(
        doc,
        ["Field", "Type", "Rules"],
        [
            ["patient_id", "integer", "Echo of the path parameter"],
            [
                "embedding_model",
                "string",
                "Must be keras-facenet-0.3.2. Worker refuses unknown values",
            ],
            ["threshold", "float", "Optional. Omit and the worker uses 0.675"],
            ["updated_at", "ISO 8601", "Last enrollment change; worker can skip unchanged syncs"],
            ["visitors[].name", "string", "What the caregiver sees; becomes event label"],
            ["visitors[].relation", "string", "Optional, display only"],
            [
                "visitors[].embeddings",
                "array of arrays",
                "One 512-float L2-normalized vector per enrolled photo",
            ],
        ],
    )
    add_table(
        doc,
        ["Status", "When"],
        [
            ["200 OK", "Gallery payload (empty visitors[] is valid)"],
            ["404", "No such patient"],
            ["422", "patient_id not a positive integer"],
        ],
    )
    bullet(doc, "Every vector is exactly 512 floats.")
    bullet(doc, "Every vector has L2 norm within 1.0 ± 0.01.")
    bullet(doc, "Keep 5–6 vectors per person.")

    heading(doc, "13. Design decisions", 1)
    add_table(
        doc,
        ["Question", "Decision"],
        [
            [
                "Why cosine + threshold, not a classifier?",
                "A classifier cannot say “nobody.” Open-set rejection is required at a front door.",
            ],
            [
                "Where does 0.675 come from?",
                "Smallest cut with ≥95% unknown rejection on the research photo set. Re-check on live camera.",
            ],
            [
                "Why only the largest face?",
                "Matches the fall worker’s largest-person rule and solo-elderly deployment.",
            ],
            [
                "Why streak and cooldown?",
                "Without them, ten seconds in frame produces dozens of caregiver notifications.",
            ],
        ],
    )

    heading(doc, "14. Limitations", 1)
    bullet(
        doc,
        "The 0.675 threshold was tuned on photos, not a 480p door camera with backlight and motion blur.",
    )
    bullet(
        doc,
        "MTCNN is unusable live on CPU (~2 s/frame vs ~190 ms Haar). Haar’s accuracy on a real door camera has not been measured against MTCNN.",
    )
    bullet(doc, "Largest face only: two people at the door produce one event.")
    bullet(
        doc,
        "confidence is a cosine score, not a probability. 0.82 means similar to enrolled photos, not 82% sure.",
    )

    heading(doc, "15. Environment notes", 1)
    bullet(doc, "First run downloads FaceNet weights and feels slow; later runs are faster.")
    bullet(doc, "Preview window: press q or Esc to quit.")
    bullet(doc, "POST failures are warnings only — the camera loop keeps running.")
    bullet(doc, "Nothing under data/ is committed: photos, embeddings, and snapshots are personal data.")
    bullet(doc, "Stop the fall worker before starting the face worker if both would use the same USB camera.")

    heading(doc, "16. Out of scope (this repo)", 1)
    add_table(
        doc,
        ["Item", "Where it lives"],
        [
            ["Training / evaluation notebooks", "Thesis / face-reco repo"],
            ["Fall / room-cam worker", "FMN_fall_worker (same event contract)"],
            ["Uploading images or video to the cloud", "Not done; frames stay on the care PC"],
            ["Caregiver-app enrollment UI", "FMN frontend"],
        ],
    )

    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer.add_run("ForgetMeNot FYP  ·  Face / Visitor Worker documentation  ·  tables formatted for Word")
    set_run(fr, size=9, color=RGBColor(0x66, 0x66, 0x66))

    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(path)
