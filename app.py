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
import io
import json
import os
import re
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import fitz
from flask import (Flask, jsonify, request, send_file, render_template,
                   abort, redirect, url_for, session, g)
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

import db
from json_utils import extract_json_object
from preprocess import preprocess_pdf
from multimodal_preprocess import multimodal_preprocess_pdf
from render import render_overlays_pdf, build_overlays_from_structure
from vision_preprocess import VISION_MODEL, VISION_DPI
from handwriting import font_store
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

# Finished jobs are kept (on disk and in memory) this many days after their
# last activity, then swept so neither outputs/ nor the in-memory JOBS dict
# grows without bound. A re-render or handwriting pass refreshes the clock.
JOB_RETENTION_DAYS = 7

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

PASSWORD_ADMIN = os.environ.get("ADMIN_PASSWORD", "alien")

# How much of the stored answer-key / reference text to feed the vision model
# when answering a hand-snipped question (see /api/snip).
SNIP_REF_MAX = 12000

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

# ---- Ad settings (file-backed, same pattern as the user password) -----------
# "Include Ads" shows a full-screen Google AdSense display ad while the worksheet
# fills. Needs the AdSense client (ca-pub-…) and an ad-unit slot ID. All default
# to off/empty so nothing changes until an admin opts in and supplies both.
ADS_ENABLED_PATH = BASE_DIR / "ads_enabled.txt"
VAST_TAGS_PATH = BASE_DIR / "vast_tags.txt"

# HilltopAds VAST tag URLs, tried in order. Used as the default until/unless
# overridden from the admin dashboard (one URL per line in vast_tags.txt).
DEFAULT_VAST_TAGS = [
    "https://surefootedpause.com/dgmwFAztd.G/NbvQZ/GEUY/Cecme9TuPZAUylMk/PtTccIxcMGzOc/3uOjDXkxtzNSzHEfz/NRz-cg5EMUwU",
    "https://surefootedpause.com/d.maFKz/dYG/NYv_ZKG/UY/pedmE9cuTZdUPlVkcPnTTc/xoMXz/cV3fO/Dpk-tXNnzHEXz/NEzpcc5/MjyyZLsIajWM1/p_d/DU0/xn",
]

def get_ads_enabled() -> bool:
    try:
        return ADS_ENABLED_PATH.read_text().strip() == "1"
    except OSError:
        return False

def set_ads_enabled(enabled: bool) -> None:
    ADS_ENABLED_PATH.write_text("1" if enabled else "0")

def get_vast_tags() -> list[str]:
    """VAST tag URLs (one per line). Falls back to the built-in defaults."""
    try:
        tags = [ln.strip() for ln in VAST_TAGS_PATH.read_text().splitlines() if ln.strip()]
    except OSError:
        tags = []
    return tags or list(DEFAULT_VAST_TAGS)

def set_vast_tags(raw: str) -> None:
    tags = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    VAST_TAGS_PATH.write_text("\n".join(tags))

# ---- Donate link -------------------------------------------------------------
# Optional "support this tool" link (Ko-fi / Buy Me a Coffee / PayPal.me / etc.).
# Leave empty to hide the donate button entirely.
DONATE_URL = ""

# All admin-dashboard data (sign-ins, filled assignments, devices) lives in a
# shared Supabase Postgres database — see db.py. A shared DB is the single
# source of truth across gunicorn workers, which is what finally kills the
# "two different tallies on refresh" bug that per-worker memory and per-file
# JSON both suffered from.
VALID_RATINGS = db.VALID_RATINGS

def _client_ip() -> str:
    return request.remote_addr or "unknown"

def _client_ua() -> str:
    return (request.user_agent.string or "")[:200]

def _record_signin(result: str):
    db.record_signin(_client_ip(), _client_ua(), result)

def _record_fill(job_id: str, name: str, style: str | None = None):
    db.record_fill(job_id, name, _client_ip(), style)

def _font_id_from_style(style_id: str | None) -> str | None:
    """If a style id names a user-built font (``font:<id>``) that exists on
    disk, return the font id; otherwise None."""
    if not style_id or not style_id.startswith("font:"):
        return None
    fid = style_id.split(":", 1)[1]
    return fid if font_store.font_path(fid) else None

def _style_label(style_id: str | None) -> str:
    """Human-readable description of the handwriting setting a user picked,
    for the admin dashboard."""
    if not style_id:
        return "Typed text"
    if style_id.startswith("font:"):
        fid = style_id.split(":", 1)[1]
        label = next((f["label"] for f in font_store.list_fonts()
                      if f["id"] == fid), fid)
        return f"{label} (your handwriting)"
    return str(style_id)

# ---- Device tracking -----------------------------------------------------
# A long-lived cookie identifies a browser/device. The first request without
# the cookie is counted as a brand-new device, inserted into the shared
# `devices` table so the count is consistent across workers and restarts.
DEVICE_COOKIE = "pf_device"

@app.before_request
def _track_device():
    g.new_device_id = None
    if request.cookies.get(DEVICE_COOKIE):
        return
    did = secrets.token_urlsafe(16)
    g.new_device_id = did
    db.record_device(did, _client_ip(), _client_ua())

@app.after_request
def _set_device_cookie(resp):
    did = getattr(g, "new_device_id", None)
    if did:
        resp.set_cookie(DEVICE_COOKIE, did,
                        max_age=60 * 60 * 24 * 365 * 2,  # 2 years
                        samesite="Lax")
    return resp

# OpenAI-compatible client. Uses the Hack Club AI proxy by default
# (free, no credit card), with an OpenRouter fallback on any failure. Reads
# HCAI_API_KEY / OPENROUTER_API_KEY from environment / .env.
_openai_client = None
def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from llm_client import build_client
        _openai_client = build_client()
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


def _write_hw_images(job_id: str, images: dict[str, bytes]) -> None:
    """Cache a {overlay_id: png_bytes} map to the job's hw dir, clearing any
    stale words from a prior fill first."""
    d = _hw_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()
    for ov_id, png in images.items():
        if png:
            (d / f"{ov_id}.png").write_bytes(png)


def _generate_hw_for_job(job_id: str) -> None:
    """If the job has a user-built handwriting font attached, render each
    overlay locally and cache it on disk. No-op otherwise (answers stay
    typeset)."""
    job = JOBS[job_id]
    font_id = _font_id_from_style(job.get("style_id"))
    if not font_id:
        return
    from handwriting.font_render import render_text_png
    otf = font_store.font_path(font_id)
    items = {ov["id"]: ov.get("text", "") for ov in job.get("overlays", [])}
    try:
        images = {ov_id: render_text_png(text, str(otf))
                  for ov_id, text in items.items() if str(text).strip()}
    except Exception as e:
        print(f"[handwriting] local font render failed, "
              f"falling back to text: {e}")
        return
    _write_hw_images(job_id, images)


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


def sweep_old_jobs() -> None:
    """Best-effort: drop jobs older than JOB_RETENTION_DAYS from disk and
    memory so outputs/ and the JOBS dict can't grow forever. A job's
    outputs/<id> mtime is bumped by every re-render/handwriting write, so this
    evicts by *last activity*, not creation time. Never raises."""
    cutoff = time.time() - JOB_RETENTION_DAYS * 86400
    try:
        dirs = [d for d in OUTPUTS.iterdir() if d.is_dir()]
    except OSError:
        return
    for d in dirs:
        try:
            if d.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(d, ignore_errors=True)
        (UPLOADS / f"{d.name}.pdf").unlink(missing_ok=True)
        JOBS.pop(d.name, None)


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
        "Always give the ACTUAL answer/definition. Never reply with meta or "
        "filler text such as 'Answer the prompt based on your situation' or "
        "'Complete the prompt with relevant information' — for a definition "
        "question, write the real definition.\n"
        "When a prompt is marked as a multi-part answer (point k of n), the "
        "units that share that question together form ONE list answer: write a "
        "DIFFERENT, specific point in each (e.g. the five components of SMART "
        "goals, or distinct functions of the Federal Reserve) and never repeat "
        "the same sentence across them.\n"
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
        model=os.environ.get("OPENAI_MODEL", "openai/gpt-5.5"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = extract_json_object(content)
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


def call_vision_for_answer(png_bytes: bytes, instructions: str = "") -> str:
    """
    Ask the vision model to answer a single worksheet item from a cropped
    screenshot — used when the AI left a question blank and the user snips it
    by hand. Returns just the answer string (no restated question).

    `instructions` is the same answer-key / reference text the original fill
    used (stored on the job), so a snipped answer stays consistent with the
    rest of the sheet.
    """
    system = (
        "You are helping a student fill in a worksheet. You are shown a cropped "
        "screenshot of ONE worksheet item (a fill-in-the-blank, a short "
        "question, or a prompt) that was left unanswered. Read it and return "
        "ONLY the answer that should be written in — do not restate the "
        "question, add a label, or explain. For a fill-in-the-blank give just "
        "the word or phrase; for a short-answer question give a concise answer "
        "(a few sentences at most). If the user supplied an answer key or notes, "
        "prefer them over your own knowledge. Return ONLY a JSON object: "
        "{\"answer\": \"<text>\"}. No prose, no markdown, no <think> tags. /no_think"
    )
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    user_content: list[dict] = [
        {"type": "text", "text": "Answer this worksheet item."},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    instructions = (instructions or "").strip()
    if instructions:
        user_content.insert(0, {
            "type": "text",
            "text": ("Answer key / reference material the user provided "
                     "(prefer it over your own knowledge):\n" + instructions[:SNIP_REF_MAX]),
        })

    response = get_openai_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    ans = extract_json_object(content).get("answer", "")
    if isinstance(ans, (int, float)):
        ans = str(ans)
    return ans.strip() if isinstance(ans, str) else ""


def call_openai_to_refine(text: str, mode: str, instruction: str = "",
                          ref: str = "") -> str:
    """
    Rewrite a single box's text per a quick edit request from the floating
    toolbar: 'shorten', 'lengthen', or 'else' (a free-text instruction the user
    typed). Returns just the replacement text.

    `ref` is the same answer-key / reference text the original fill used (stored
    on the job as `fill_instructions`), so a rewrite — especially "lengthen" —
    stays consistent with the source material instead of inventing new facts.
    """
    directive = {
        "shorten": "Make this text much more shorter and more concise while keeping the "
                   "same meaning and the same answer.",
        "lengthen": "Make this text longer and more detailed while keeping it "
                    "accurate and on-topic.",
    }.get(mode)
    if not directive:
        directive = (instruction or "").strip() or "Rewrite this text."

    system = (
        "You are editing a single answer a student wrote in one box of a "
        "worksheet. You are given the current text and an instruction for how "
        "to change it. Apply the instruction and return ONLY the rewritten "
        "text that should replace what's in the box — do not restate the "
        "question, add a label or quotes, or explain. Keep it factually "
        "correct. If the user supplied an answer key or notes, prefer them "
        "over your own knowledge. Return ONLY a JSON object: "
        "{\"text\": \"<rewritten text>\"}. No prose, no markdown, no <think> "
        "tags. /no_think"
    )

    parts = []
    ref = (ref or "").strip()
    if ref:
        parts.append("Answer key / reference material the answer is based on "
                     "(stay consistent with it):\n" + ref[:SNIP_REF_MAX])
    parts.append("Instruction: " + directive)
    parts.append("Current text:\n" + text)
    user = "\n\n".join(parts)

    response = get_openai_client().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "openai/gpt-5.5"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    out = extract_json_object(content).get("text", "")
    if isinstance(out, (int, float)):
        out = str(out)
    return out.strip() if isinstance(out, str) else ""


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


def _fmt_ts(iso: str | None) -> str:
    """Render a Postgres ISO timestamp as 'YYYY-MM-DD HH:MM:SS UTC' to match
    the dashboard's existing look."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError):
        return str(iso)


@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    # Read from the shared database so every worker shows the same numbers.
    signin_log = db.fetch_signins()
    activity_log = db.fetch_assignments()
    for e in signin_log:
        e["timestamp"] = _fmt_ts(e.get("ts"))
    for e in activity_log:
        e["timestamp"] = _fmt_ts(e.get("ts"))
    user_count = sum(1 for e in signin_log if e.get("result") == "user")
    admin_count = sum(1 for e in signin_log if e.get("result") == "admin")
    fail_count = sum(1 for e in signin_log if e.get("result") == "failed")
    rating_counts = {
        "green": sum(1 for e in activity_log if e.get("rating") == "green"),
        "yellow": sum(1 for e in activity_log if e.get("rating") == "yellow"),
        "red": sum(1 for e in activity_log if e.get("rating") == "red"),
    }
    return render_template(
        "admin.html",
        logs=signin_log,
        total=len(signin_log),
        user_count=user_count,
        admin_count=admin_count,
        fail_count=fail_count,
        activity=activity_log,
        activity_total=len(activity_log),
        rating_counts=rating_counts,
        device_count=db.device_count(),
        db_enabled=db.enabled(),
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        current_user_password=get_user_password(),
        pw_status=request.args.get("pw_status"),
        ads_enabled=get_ads_enabled(),
        vast_tags="\n".join(get_vast_tags()),
        ads_status=request.args.get("ads_status"),
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


@app.post("/admin/ads")
def change_ads():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    # Unchecked checkboxes don't submit, so absence means "off".
    set_ads_enabled(request.form.get("ads_enabled") == "on")
    set_vast_tags(request.form.get("vast_tags") or "")
    return redirect(url_for("admin", ads_status="ok"))


@app.route("/")
def index():
    # Gated: require a sign-in (user or admin) before the filler is shown.
    if not session.get("role"):
        return redirect(url_for("login"))
    # Ads only run when enabled AND at least one VAST tag is configured; the
    # template treats an empty tag list as "off" regardless of the flag.
    return render_template(
        "index.html",
        ads_enabled=get_ads_enabled(),
        vast_tags=get_vast_tags(),
        donate_url=DONATE_URL,
    )


@app.route("/handwriting")
def handwriting_onboarding():
    """Pro onboarding: print a template, photograph the filled page, and build
    a handwriting font from it (see /api/fonts)."""
    if not session.get("role"):
        return redirect(url_for("login"))
    return render_template("handwriting.html")


@app.route("/2d7883f358a775fc1a8f.txt")
def hilltopads_verify():
    # Public (no login gate) so HilltopAds' crawler can fetch it directly.
    # The homepage "/" redirects to /login, which would hide any token there.
    return send_file(BASE_DIR / "2d7883f358a775fc1a8f.txt", mimetype="text/plain")


@app.route("/0efb70ed5ecb5409945db6f7bb100589.html")
def site_verify_html():
    # Public (no login gate) so the verifying crawler can fetch it directly.
    return send_file(
        BASE_DIR / "0efb70ed5ecb5409945db6f7bb100589.html", mimetype="text/html"
    )


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

    sweep_old_jobs()  # opportunistic cleanup of stale jobs

    job_id = new_job_id()
    pdf_path = UPLOADS / f"{job_id}.pdf"
    f.save(pdf_path)

    # Which answer formats to detect — chosen by the user in the UI. A JSON
    # array of format ids; absent/invalid means "detect all".
    formats = None
    raw_formats = request.form.get("formats")
    if raw_formats:
        try:
            parsed = json.loads(raw_formats)
            if isinstance(parsed, list):
                formats = [str(x) for x in parsed]
        except (TypeError, json.JSONDecodeError):
            formats = None

    # Detector selection: deterministic (preprocess.py, default) vs the
    # multimodal vision path. Chosen per-request via the `detector` form field
    # or globally via the PAPERFILL_DETECTOR env var.
    detector_mode = (request.form.get("detector")
                     or os.environ.get("PAPERFILL_DETECTOR")
                     or "deterministic").strip().lower()
    use_multimodal = detector_mode in ("multimodal", "mm", "vision2")

    # Quick sanity check + preprocess
    try:
        if use_multimodal:
            structure = multimodal_preprocess_pdf(str(pdf_path), formats=formats)
        else:
            structure = preprocess_pdf(str(pdf_path), formats=formats)
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
    # Keep the answer key / reference text around so a hand-snipped question
    # (see /api/snip) is answered from the same source material.
    job["fill_instructions"] = instructions[:SNIP_REF_MAX]

    _generate_hw_for_job(job_id)          # no-op unless a style is attached

    try:
        _rerender_job(job_id)
    except Exception as e:
        return jsonify({"error": f"render failed: {e}"}), 500
    save_job(job_id)

    _record_fill(job_id, job.get("original_name"), _style_label(job.get("style_id")))

    return jsonify({
        "job_id": job_id,
        "answers": answers,
        "overlays": overlays,
        "page_count": job["page_count"],
        "page_sizes": job["page_sizes"],
    })


@app.post("/api/rate")
def rate():
    """Record a user's quality rating for a filled assignment.
    Body: {job_id, rating: 'green'|'yellow'|'red'}."""
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    rating = data.get("rating")
    if rating not in VALID_RATINGS:
        return jsonify({"error": "invalid rating"}), 400
    if db.set_rating(str(job_id), rating):
        return jsonify({"ok": True})
    return jsonify({"error": "unknown job_id"}), 404


@app.post("/api/feedback")
def submit_feedback():
    """Save a user's free-text feedback / bug report for a filled assignment.
    Body: {job_id, feedback}."""
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    text = str(data.get("feedback", "")).strip()[:2000]
    if not text:
        return jsonify({"error": "empty feedback"}), 400
    if db.set_feedback(str(job_id), text):
        return jsonify({"ok": True})
    return jsonify({"error": "unknown job_id"}), 404


@app.get("/api/fonts")
def list_fonts_route():
    """List the user's built handwriting fonts."""
    return jsonify({"fonts": font_store.list_fonts()})


@app.get("/api/fonts/<font_id>/sample.png")
def font_sample(font_id: str):
    """Render a sample in a built font (used by the onboarding page). Pass
    ?text=... to preview arbitrary text; defaults to a short word."""
    otf = font_store.font_path(font_id)
    if not otf:
        abort(404)
    text = (request.args.get("text") or "Sample").strip()[:120] or "Sample"
    from handwriting.font_render import render_text_png
    png = render_text_png(text, str(otf))
    if not png:
        abort(404)
    return send_file(io.BytesIO(png), mimetype="image/png")


@app.get("/api/fonts/template")
def download_template():
    """Serve the printable handwriting template (Pro onboarding step 1)."""
    if not session.get("role"):
        return redirect(url_for("login"))
    from handwriting import template as hw_template
    return send_file(io.BytesIO(hw_template.template_pdf_bytes()),
                     mimetype="application/pdf", as_attachment=True,
                     download_name="paperfill-handwriting-template.pdf")


@app.post("/api/fonts")
def build_font_route():
    """Build a handwriting font from a filled template and store it (Pro
    onboarding step 2). Multipart: 'template' = a multi-page PDF scan, or one
    file per page (in page order); 'name' = label. Returns {font_id, label}.
    The font then appears in /api/fonts."""
    if not session.get("role"):
        return jsonify({"error": "not signed in"}), 403
    files = [f for f in request.files.getlist("template") if f and f.filename]
    if not files:
        return jsonify({"error": "no template uploaded"}), 400
    name = (request.form.get("name") or "My handwriting").strip()[:40]

    import tempfile
    from handwriting.font_build import build_font
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, f in enumerate(files):
            ext = ".pdf" if f.filename.lower().endswith(".pdf") else ".img"
            p = os.path.join(d, f"page{i}{ext}")
            f.save(p)
            paths.append(p)
        otf_path = os.path.join(d, "font.otf")
        try:
            src = paths[0] if len(paths) == 1 else paths
            build_font(src, otf_path, family=name or "Paperfill Hand")
            otf_bytes = Path(otf_path).read_bytes()
        except Exception as e:
            return jsonify({"error": f"could not build font: {e}"}), 422
    font_id = font_store.save_font(name, otf_bytes)
    return jsonify({"ok": True, "font_id": font_id,
                    "style_id": f"font:{font_id}", "label": name})


@app.post("/api/style")
def upload_style():
    """Attach a user-built handwriting font to a job so the fill renders the
    answers in it. Body: {job_id, style: "font:<id>"}."""
    data = request.get_json(silent=True) or {}
    job_id = request.form.get("job_id") or data.get("job_id")
    job = load_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404

    style = request.form.get("style") or data.get("style") or ""
    if not _font_id_from_style(style):
        return jsonify({"error": f"unknown font '{style}'"}), 400
    job["style_id"] = style
    save_job(job_id)
    return jsonify({"ok": True})


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


@app.post("/api/snip")
def snip():
    """
    Answer a single question the user snipped by hand (a region the AI left
    blank). Body: {job_id, page, bbox:[x0,y0,x1,y1] in PDF points}.

    The selected region of the original page is rendered to a high-DPI crop and
    sent to the vision model, which returns just the answer text. The frontend
    drops that into a new, editable text box the user positions over the blank.
    """
    data = request.get_json(silent=True) or {}
    job = load_job(data.get("job_id"))
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404

    try:
        bbox = [float(x) for x in data.get("bbox", [])]
    except (TypeError, ValueError):
        bbox = []
    if len(bbox) != 4:
        return jsonify({"error": "bbox must be [x0,y0,x1,y1]"}), 400
    try:
        page_idx = int(data.get("page", 0))
    except (TypeError, ValueError):
        page_idx = -1
    if page_idx < 0 or page_idx >= job["page_count"]:
        return jsonify({"error": "page out of range"}), 400

    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return jsonify({"error": "selection too small"}), 400

    # Render just the selected region of the original page, padded a little so
    # edge text isn't clipped, at a DPI high enough for the model to read it.
    try:
        doc = fitz.open(job["pdf_path"])
        page = doc[page_idx]
        pad_pts = 4
        clip = fitz.Rect(
            max(0, x0 - pad_pts), max(0, y0 - pad_pts),
            min(page.rect.width, x1 + pad_pts), min(page.rect.height, y1 + pad_pts),
        )
        png_bytes = page.get_pixmap(dpi=VISION_DPI, clip=clip).tobytes("png")
        doc.close()
    except Exception as e:
        return jsonify({"error": f"could not crop page: {e}"}), 500

    try:
        answer = call_vision_for_answer(png_bytes, job.get("fill_instructions", ""))
    except Exception as e:
        return jsonify({"error": f"vision call failed: {e}"}), 502

    return jsonify({"answer": answer})


@app.post("/api/refine")
def refine():
    """
    Rewrite a single box's text per a quick edit from the floating toolbar.
    Body: {job_id, text, mode: 'shorten'|'lengthen'|'else', instruction?}.

    Returns {text} — the replacement. The frontend drops it back into the box
    and marks the job dirty; nothing is saved/re-rendered until the user saves.
    """
    data = request.get_json(silent=True) or {}
    job = load_job(data.get("job_id"))
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404

    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "no text to edit"}), 400
    mode = str(data.get("mode", "")).strip().lower()
    instruction = str(data.get("instruction", ""))[:2000]
    if mode not in ("shorten", "lengthen", "else"):
        return jsonify({"error": "invalid mode"}), 400
    if mode == "else" and not instruction.strip():
        return jsonify({"error": "describe how to edit the text"}), 400

    try:
        new_text = call_openai_to_refine(
            text, mode, instruction, job.get("fill_instructions", ""))
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {e}"}), 502
    if not new_text:
        return jsonify({"error": "couldn't rewrite that text"}), 502

    return jsonify({"text": new_text})


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