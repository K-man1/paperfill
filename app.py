"""
Flask backend for the AI PDF editor.

Endpoints:
  POST /api/upload    Receive PDF, run preprocessor, return structure JSON
                       (with bboxes stripped) so the frontend can show a preview.
  POST /api/fill      Receive structure-id, call OpenAI to fill in slots,
                       render the filled PDF, return download URL.
  GET  /api/download/<job_id>   Stream the filled PDF.
  GET  /api/preview/<job_id>/<page>   Stream a page image for preview.
"""

import base64
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

import fitz
from flask import (Flask, jsonify, request, send_file, render_template,
                   abort, redirect, url_for, session)
from openai import OpenAI

# Load .env file if present (no external dependency)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from preprocess import preprocess_pdf
from render import render_overlays_pdf, build_overlays_from_structure
from handwriting.client import handwriting_enabled, generate_handwriting
from context_sources import (extract_file_text, fetch_youtube_transcript,
                             assemble_context)


# ---- Setup ---------------------------------------------------------------

BASE_DIR = Path(__file__).parent
UPLOADS = BASE_DIR / "uploads"
OUTPUTS = BASE_DIR / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 10
ALLOWED_EXT = {".pdf"}

# Curated handwriting style presets. These are real in-distribution IAM word
# images (the data One-DM was trained on), which produce far more consistent
# output than arbitrary user photos. Keys are the ids the UI sends to /api/style.
STYLES_DIR = BASE_DIR / "handwriting" / "styles"
STYLE_PRESETS = {
    "neat":  {"label": "Neat",  "file": "neat.png",  "hint": "clean print"},
    "light": {"label": "Light", "file": "light.png", "hint": "light print"},
    "bold":  {"label": "Bold",  "file": "bold.png",  "hint": "bold print"},
}

def _preset_b64(preset_id: str) -> str | None:
    info = STYLE_PRESETS.get(preset_id)
    if not info:
        return None
    path = STYLES_DIR / info["file"]
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

PASSWORD_ADMIN = "alien"

# The user access code is changeable from the admin panel and persisted to
# disk so it survives restarts and is shared across all gunicorn workers. It
# is read fresh from disk on every login attempt, so a change takes effect
# immediately for every worker process.
DEFAULT_USER_PASSWORD = "spurs"
USER_PASSWORD_PATH = BASE_DIR / "user_password.txt"

def get_user_password() -> str:
    try:
        pw = USER_PASSWORD_PATH.read_text().strip()
        if pw:
            return pw
    except OSError:
        pass
    return DEFAULT_USER_PASSWORD

def set_user_password(pw: str) -> None:
    USER_PASSWORD_PATH.write_text(pw.strip())

# In-memory sign-in log persisted to disk on each write.
SIGNIN_LOG_PATH = BASE_DIR / "signin_log.json"
SIGNIN_LOG: list[dict] = []

def _load_signin_log():
    if SIGNIN_LOG_PATH.exists():
        try:
            SIGNIN_LOG.extend(json.loads(SIGNIN_LOG_PATH.read_text()))
        except Exception:
            pass

def _save_signin_log():
    SIGNIN_LOG_PATH.write_text(json.dumps(SIGNIN_LOG, indent=2))

def _record_signin(result: str):
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ip": request.remote_addr or "unknown",
        "ua": (request.user_agent.string or "")[:200],
        "result": result,  # 'user', 'admin', 'failed'
    }
    SIGNIN_LOG.append(entry)
    _save_signin_log()

_load_signin_log()

# OpenAI-compatible client. Uses the Hack Club AI proxy by default
# (free, no credit card). Reads HCAI_API_KEY from environment / .env.
_openai_client = None
def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("HCAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set HCAI_API_KEY in .env or environment."
            )
        _openai_client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get(
                "OPENAI_BASE_URL",
                "https://ai.hackclub.com/proxy/v1",
            ),
        )
    return _openai_client

# Job store. Kept in-memory for speed but mirrored to disk so that every
# gunicorn worker can find a job it didn't create (uploads and fills can land
# on different worker processes).
JOBS: dict[str, dict] = {}


def _job_meta_path(job_id: str) -> Path:
    return OUTPUTS / job_id / "job.json"


def _hw_dir(job_id: str) -> Path:
    return OUTPUTS / job_id / "hw"


def _load_hw_images(job_id: str) -> dict[str, bytes]:
    """Read previously generated handwriting PNGs keyed by overlay id."""
    d = _hw_dir(job_id)
    if not d.exists():
        return {}
    return {p.stem: p.read_bytes() for p in d.glob("*.png")}


def _generate_hw_for_job(job_id: str) -> None:
    """If the job has a handwriting style + the Modal service is configured,
    generate a PNG per overlay and cache it on disk for rendering."""
    job = JOBS[job_id]
    style_b64 = job.get("style_b64")
    if not (style_b64 and handwriting_enabled()):
        return
    items = {ov["id"]: ov.get("text", "") for ov in job.get("overlays", [])}
    try:
        images = generate_handwriting(style_b64, items)
    except Exception as e:
        print(f"[handwriting] generation failed, falling back to text: {e}")
        return
    d = _hw_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):           # clear stale words from a prior fill
        old.unlink()
    for ov_id, png in images.items():
        (d / f"{ov_id}.png").write_bytes(png)


def save_job(job_id: str) -> None:
    """Mirror a job's metadata to disk so other workers can load it."""
    job = JOBS.get(job_id)
    if job is None:
        return
    (OUTPUTS / job_id).mkdir(exist_ok=True)
    _job_meta_path(job_id).write_text(json.dumps(job))


def load_job(job_id: str) -> dict | None:
    """Return a job from memory, falling back to its on-disk copy."""
    if not job_id:
        return None
    job = JOBS.get(job_id)
    if job is not None:
        return job
    path = _job_meta_path(job_id)
    if not path.exists():
        return None
    try:
        job = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    JOBS[job_id] = job
    return job


# ---- Helpers -------------------------------------------------------------

def new_job_id() -> str:
    return secrets.token_urlsafe(12)


def strip_bboxes_for_llm(structure: dict) -> dict:
    """
    Return a copy of the structure with bboxes and other rendering-only
    fields removed. This is what we send to the LLM — keeps the prompt
    short and prevents the model from trying to reason about coordinates.
    """
    units_for_llm = []
    for u in structure["units"]:
        clean = {
            "unit_id": u["unit_id"],
            "type": u["type"],
            "prompt": u["prompt_text"],
        }
        if u["type"] == "inline_blanks":
            clean["slots"] = [s["slot_id"] for s in u["slots"]]
        elif u["type"] == "table":
            # Flatten all slot ids in the table so the LLM sees the
            # complete list it needs to fill.
            ids = []
            for row in u["table_cells"]:
                for cell in row:
                    if cell is None:
                        continue
                    ids.extend(s["slot_id"] for s in cell["slots"])
            clean["slots"] = ids
        elif u["type"] == "open_response":
            # The open-response answer is keyed by unit_id.
            clean["answer_key"] = u["unit_id"]
        units_for_llm.append(clean)
    return {"units": units_for_llm}


def call_openai_to_fill(structure_for_llm: dict, instructions: str = "") -> dict[str, str]:
    """
    Single API call that returns a JSON object mapping slot_id / unit_id
    to the answer string. Uses Structured Outputs / JSON mode so we don't
    have to babysit the format.

    `instructions` is optional free-text from the user — e.g. an answer key
    they already have, or guidance like "answer in Spanish". When present it
    should take priority over the model's own knowledge.
    """
    system = (
        "You are filling in a worksheet PDF. You receive a list of units. "
        "For each unit:\n"
        "  - 'inline_blanks' or 'table': the prompt contains {{slot_id}} "
        "    placeholders. Return the answer for each slot_id.\n"
        "  - 'open_response': the prompt is a question. Return one answer "
        "    keyed by the unit's answer_key, kept to a few sentences.\n"
        "Use the context in each prompt to figure out what kind of "
        "answer fits (a single word, a phrase, a conjugated verb form, "
        "a name, a date, etc.). Be accurate. If you genuinely don't know "
        "something factual (e.g. the user's name), pick a reasonable "
        "placeholder like 'Student'.\n"
        "If the user provides instructions or an answer key, treat those as "
        "authoritative and prefer them over your own knowledge.\n"
        "Return ONLY a JSON object: {\"<slot_or_unit_id>\": \"<answer>\", ...}. "
        "No prose, no markdown, no <think> tags, no explanations — JSON only. "
        "/no_think"
    )

    structure_json = json.dumps(structure_for_llm, ensure_ascii=False)
    instructions = (instructions or "").strip()

    if instructions:
        user = (
            "User-provided answer key / instructions (use these as the "
            "authoritative source — prefer them over your own knowledge):\n"
            f"{instructions}\n\n"
            f"Worksheet to fill:\n{structure_json}"
        )
    else:
        user = structure_json

    response = get_openai_client().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "qwen/qwen3-32b"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = _extract_json_object(content)
    flat = _flatten_answers(parsed)
    if not flat:
        print(f"[fill] WARNING: LLM returned no usable answers. Raw (first 800 chars):\n{content[:800]}")
    return flat


_KEY_SUFFIX_RE = re.compile(r"(s\d+|u\d+)$")


def _normalize_key(k: str) -> str:
    """
    Models sometimes return composite keys like 'u1-s2', 'u3_s1', 'unit1.s4',
    'slot_s5'. Pull out the trailing 'sN' (slot) or 'uN' (unit) id we use
    in the structure. If no match, return the original.
    """
    m = _KEY_SUFFIX_RE.search(k)
    return m.group(1) if m else k


def _flatten_answers(obj: dict) -> dict[str, str]:
    """
    Normalize LLM answer responses to {slot_or_unit_id: answer_string}.
    Handles:
      - flat {"s1": "x"}
      - nested {"u1": {"s1": "x", "s2": "y"}}
      - composite keys {"u1-s1": "x"}
      - mixed: {"u3": "open response", "u1-s1": "answer"}
    """
    out: dict[str, str] = {}
    for k, v in (obj or {}).items():
        if isinstance(v, str):
            out[_normalize_key(k)] = v
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, str):
                    out[_normalize_key(sk)] = sv
        # ignore lists/numbers — model went off-spec
    return out


def _extract_json_object(text: str) -> dict:
    """
    Robustly pull a JSON object out of a model response. Handles:
      - <think>...</think> blocks (qwen3 reasoning models)
      - ```json fenced code blocks
      - leading/trailing prose
    Falls back to the first {...} balanced span if direct parse fails.
    """
    if not text:
        return {}
    # Strip reasoning blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Last resort: find the first balanced { ... } and try that
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start:i+1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    break
    return {}


# ---- Routes --------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        pw = request.form.get("password", "")
        if pw == PASSWORD_ADMIN:
            _record_signin("admin")
            session["role"] = "admin"
            return redirect(url_for("admin"))
        elif pw == get_user_password():
            _record_signin("user")
            session["role"] = "user"
            return redirect(url_for("index"))
        else:
            _record_signin("failed")
            return render_template("login.html", error="Incorrect access code. Please try again.")
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    user_count = sum(1 for e in SIGNIN_LOG if e["result"] == "user")
    admin_count = sum(1 for e in SIGNIN_LOG if e["result"] == "admin")
    fail_count = sum(1 for e in SIGNIN_LOG if e["result"] == "failed")
    return render_template(
        "admin.html",
        logs=SIGNIN_LOG,
        total=len(SIGNIN_LOG),
        user_count=user_count,
        admin_count=admin_count,
        fail_count=fail_count,
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        current_user_password=get_user_password(),
        pw_status=request.args.get("pw_status"),
    )


@app.post("/admin/user-password")
def change_user_password():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    new_pw = (request.form.get("new_password") or "").strip()
    confirm = (request.form.get("confirm_password") or "").strip()
    if len(new_pw) < 3:
        return redirect(url_for("admin", pw_status="short"))
    if new_pw != confirm:
        return redirect(url_for("admin", pw_status="mismatch"))
    if new_pw == PASSWORD_ADMIN:
        return redirect(url_for("admin", pw_status="conflict"))
    set_user_password(new_pw)
    return redirect(url_for("admin", pw_status="ok"))


@app.route("/")
def index():
    # Gated: require a sign-in (user or admin) before the filler is shown.
    if not session.get("role"):
        return redirect(url_for("login"))
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "only PDF files allowed"}), 400

    job_id = new_job_id()
    pdf_path = UPLOADS / f"{job_id}.pdf"
    f.save(pdf_path)

    # Optional: force every page through the vision model (for text-layer PDFs
    # whose layout the heuristics miss, e.g. bullet-list study guides).
    force_vision = (request.form.get("force_vision") or "").lower() in ("1", "true", "on", "yes")

    # Quick sanity check + preprocess
    try:
        structure = preprocess_pdf(str(pdf_path), force_vision=force_vision)
    except Exception as e:
        pdf_path.unlink(missing_ok=True)
        return jsonify({"error": f"could not parse PDF: {e}"}), 400

    # Render preview images of each page so the frontend can show
    # what was uploaded.
    doc = fitz.open(str(pdf_path))
    preview_dir = OUTPUTS / job_id
    preview_dir.mkdir(exist_ok=True)
    page_sizes = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=110)
        pix.save(str(preview_dir / f"page-{i}.png"))
        rect = page.rect
        page_sizes.append({"width": rect.width, "height": rect.height})
    page_count = len(doc)
    doc.close()

    JOBS[job_id] = {
        "pdf_path": str(pdf_path),
        "original_name": Path(f.filename).name,  # for the download filename
        "structure": structure,
        "page_count": page_count,
        "page_sizes": page_sizes,
        "overlays": None,
        "filled_path": None,
    }
    save_job(job_id)

    # Build a frontend-safe summary (no bboxes; they're huge and useless
    # to the UI).
    summary = {
        "job_id": job_id,
        "page_count": page_count,
        "unit_count": structure["unit_count"],
        "slot_count": structure["slot_count"],
        "units": [
            {
                "unit_id": u["unit_id"],
                "type": u["type"],
                "page": u["page"],
                "prompt": u["prompt_text"],
            }
            for u in structure["units"]
        ],
    }
    return jsonify(summary)


@app.post("/api/context")
def context():
    """
    Extract reference material the AI should use when filling the sheet.

    Multipart body:
      files       -> zero or more reference files (PDF / text / image)
      youtube_urls-> JSON array of YouTube URLs (string)

    Returns {context: "<combined labelled text>", sources: [{name, chars}]}.
    The frontend passes `context` back into /api/fill.
    """
    sources: list[tuple[str, str]] = []
    summary = []

    for f in request.files.getlist("files"):
        if not f.filename:
            continue
        text = extract_file_text(f.filename, f.read())
        sources.append((f"Reference file: {f.filename}", text))
        summary.append({"name": f.filename, "kind": "file", "chars": len(text)})

    raw_urls = request.form.get("youtube_urls", "[]")
    try:
        urls = json.loads(raw_urls)
    except (TypeError, json.JSONDecodeError):
        urls = []
    for url in urls if isinstance(urls, list) else []:
        url = str(url).strip()
        if not url:
            continue
        text = fetch_youtube_transcript(url)
        sources.append((f"YouTube transcript: {url}", text))
        summary.append({"name": url, "kind": "youtube", "chars": len(text)})

    return jsonify({"context": assemble_context(sources), "sources": summary})


@app.post("/api/fill")
def fill():
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    job = load_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404

    instructions = str(data.get("instructions", ""))[:8000]
    context_text = str(data.get("context", ""))[:30000]
    if context_text.strip():
        instructions = (
            f"{instructions}\n\nReference material the user attached "
            f"(use it as authoritative source material):\n{context_text}"
        ).strip()
    structure_for_llm = strip_bboxes_for_llm(job["structure"])
    try:
        answers = call_openai_to_fill(structure_for_llm, instructions)
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {e}"}), 502

    overlays = build_overlays_from_structure(job["structure"], answers)
    job["answers"] = answers
    job["overlays"] = overlays

    _generate_hw_for_job(job_id)          # no-op unless a style is attached

    try:
        _rerender_job(job_id)
    except Exception as e:
        return jsonify({"error": f"render failed: {e}"}), 500
    save_job(job_id)

    return jsonify({
        "job_id": job_id,
        "answers": answers,
        "overlays": overlays,
        "page_count": job["page_count"],
        "page_sizes": job["page_sizes"],
    })


@app.get("/api/styles")
def list_styles():
    """List the curated handwriting presets for the UI picker."""
    return jsonify({
        "enabled": handwriting_enabled(),
        "presets": [{"id": k, "label": v["label"], "hint": v["hint"]}
                    for k, v in STYLE_PRESETS.items()],
    })


@app.get("/api/styles/<preset_id>.png")
def style_preview(preset_id: str):
    """Serve a preset's reference image so the picker can show what it looks like."""
    info = STYLE_PRESETS.get(preset_id)
    if not info:
        abort(404)
    return send_file(STYLES_DIR / info["file"], mimetype="image/png")


@app.post("/api/style")
def upload_style():
    """Attach a handwriting style to a job. Accepts, in priority order:
      - a preset id  (form 'preset' or JSON {preset})  ← recommended
      - a multipart file field 'style'
      - JSON {style_b64}
    """
    data = request.get_json(silent=True) or {}
    job_id = request.form.get("job_id") or data.get("job_id")
    job = load_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404

    preset = request.form.get("preset") or data.get("preset")
    file = request.files.get("style")
    if preset:
        style_b64 = _preset_b64(preset)
        if not style_b64:
            return jsonify({"error": f"unknown preset '{preset}'"}), 400
    elif file is not None:
        style_b64 = base64.b64encode(file.read()).decode()
    else:
        style_b64 = data.get("style_b64", "")
    if not style_b64:
        return jsonify({"error": "no style provided"}), 400

    job["style_b64"] = style_b64
    save_job(job_id)
    return jsonify({"ok": True, "handwriting_service": handwriting_enabled()})


def _rerender_job(job_id: str) -> None:
    """Re-render the filled PDF + page PNG previews from the job's current overlays."""
    job = JOBS[job_id]
    filled_path = OUTPUTS / f"{job_id}-filled.pdf"
    render_overlays_pdf(job["pdf_path"], job["overlays"], str(filled_path),
                        images=_load_hw_images(job_id))
    doc = fitz.open(str(filled_path))
    preview_dir = OUTPUTS / job_id
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=110)
        pix.save(str(preview_dir / f"filled-{i}.png"))
    doc.close()
    job["filled_path"] = str(filled_path)


@app.post("/api/update")
def update():
    """
    Replace the job's overlays with the client-provided list and re-render.
    Body: {job_id, overlays: [{id, page, bbox:[x0,y0,x1,y1], text, mode?}, ...]}
    """
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    job = load_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    overlays = data.get("overlays")
    if not isinstance(overlays, list):
        return jsonify({"error": "overlays must be a list"}), 400

    cleaned = []
    max_page = job["page_count"] - 1
    for ov in overlays:
        try:
            bbox = [float(x) for x in ov["bbox"]]
            if len(bbox) != 4:
                continue
            page = int(ov.get("page", 0))
            if page < 0 or page > max_page:
                continue
            font = ov.get("font", "sans")
            if font not in ("sans", "serif", "mono"):
                font = "sans"
            try:
                size = float(ov.get("size", 11))
            except (TypeError, ValueError):
                size = 11
            size = max(6.0, min(48.0, size))
            cleaned.append({
                "id": str(ov.get("id", "")),
                "page": page,
                "bbox": bbox,
                "text": str(ov.get("text", "")),
                "mode": ov.get("mode", "region"),
                "font": font,
                "size": size,
                "bold": bool(ov.get("bold", False)),
                "italic": bool(ov.get("italic", False)),
                "underline": bool(ov.get("underline", False)),
            })
        except (KeyError, TypeError, ValueError):
            continue

    job["overlays"] = cleaned
    try:
        _rerender_job(job_id)
    except Exception as e:
        return jsonify({"error": f"render failed: {e}"}), 500
    save_job(job_id)

    return jsonify({"ok": True, "overlay_count": len(cleaned)})


@app.get("/api/download/<job_id>")
def download(job_id):
    job = load_job(job_id)
    if not job or not job.get("filled_path"):
        abort(404)
    # Download under the original PDF's name (basename only, .pdf enforced).
    name = Path(job.get("original_name") or "").name or "filled.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return send_file(job["filled_path"],
                     as_attachment=True,
                     download_name=name,
                     mimetype="application/pdf")


@app.get("/api/preview/<job_id>/<which>/<int:page>")
def preview(job_id, which, page):
    """which = 'page' (original) or 'filled'."""
    if which not in {"page", "filled"}:
        abort(404)
    if load_job(job_id) is None:
        abort(404)
    fname = f"{which}-{page}.png"
    fpath = OUTPUTS / job_id / fname
    if not fpath.exists():
        abort(404)
    return send_file(str(fpath), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)