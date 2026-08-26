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
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import time
import zipfile
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
import fitz
from flask import (Flask, jsonify, request, send_file, render_template,
                   abort, redirect, url_for, session, g)
import smtplib
from email.message import EmailMessage

from openai import OpenAI
import requests
from authlib.integrations.flask_client import OAuth
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

from paperfill.paths import REPO_ROOT

# Load .env file if present (no external dependency)
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from paperfill.data import costs
from paperfill.data import db
from paperfill.data import stats
from paperfill.data import usage
from paperfill.utils.json_utils import extract_json_object
from paperfill.utils.plain_math import plain_math
from paperfill.ai.preprocess import preprocess_pdf
from paperfill.ai.multimodal_preprocess import multimodal_preprocess_pdf
from paperfill.ai.candidates import region_preprocess_pdf
from paperfill.ai.render import (render_overlays_pdf, build_overlays_from_structure,
                                 FILLED_MARKER)
from paperfill.ai.vision_preprocess import VISION_MODEL, VISION_DPI
from paperfill.ai.llm_client import call_context
from paperfill.handwriting import font_store
from paperfill.utils.context_sources import (extract_file_text, fetch_youtube_transcript,
                                             assemble_context)


# ---- Setup ---------------------------------------------------------------

BASE_DIR = REPO_ROOT
UPLOADS = BASE_DIR / "uploads"
OUTPUTS = BASE_DIR / "outputs"
# Snapshots of PDFs attached to user problem reports, one reports/<job_id>/
# directory per report. Deliberately outside uploads/ so sweep_old_jobs() can't
# delete the evidence for a report that hasn't been looked at yet.
REPORTS = BASE_DIR / "reports"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# Reports hold other people's documents, so they expire on their own clock
# rather than living until someone remembers to clear reports/.
REPORT_RETENTION_DAYS = 30

# Everything filed at or before the instant recorded here has already been
# handed to an admin by /admin/reports.zip. A dotfile so it can never collide
# with a job id.
REPORTS_MARKER = REPORTS / ".last_download"

MAX_UPLOAD_MB = 10
ALLOWED_EXT = {".pdf"}

# Finished jobs are kept (on disk and in memory) this many days after their
# last activity, then swept so neither outputs/ nor the in-memory JOBS dict
# grows without bound. A re-render or handwriting pass refreshes the clock.
JOB_RETENTION_DAYS = 7

# templates/ and static/ stay at the repo root (Flask convention), but this
# module now lives in src/paperfill/, so Flask's default root_path would look
# for them inside the package. Point it back at the repo root explicitly.
app = Flask(
    __name__,
    template_folder=str(REPO_ROOT / "templates"),
    static_folder=str(REPO_ROOT / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def _resolve_secret_key() -> str:
    """Flask session signing key. Prefer SECRET_KEY from the env. If it's
    unset, fall back to a key persisted on disk and SHARED across every
    gunicorn worker — a fresh per-process secrets.token_hex() would give each
    of the workers a *different* key, so a cookie signed by one worker fails
    HMAC validation on the next and users get randomly logged out. Persisting
    one key keeps sessions stable across workers and restarts. We still warn
    loudly because SECRET_KEY belongs in the production .env."""
    env_key = os.environ.get("SECRET_KEY", "").strip()
    if env_key:
        return env_key
    key_path = BASE_DIR / "secret_key.txt"
    try:
        existing = key_path.read_text().strip()
        if existing:
            print("[auth] WARNING: SECRET_KEY not set; using the generated key "
                  f"in {key_path.name}. Set SECRET_KEY in .env for production.")
            return existing
    except OSError:
        pass
    generated = secrets.token_hex(32)
    try:
        key_path.write_text(generated)
        os.chmod(key_path, 0o600)
    except OSError as e:
        # Can't persist (read-only FS): every worker will diverge. Warn hard.
        print(f"[auth] WARNING: SECRET_KEY unset and could not persist a shared "
              f"key ({e}); sessions may break across workers. Set SECRET_KEY in .env.")
    else:
        print("[auth] WARNING: SECRET_KEY not set; generated one and saved it to "
              f"{key_path.name}. Set SECRET_KEY in .env for production.")
    return generated


app.secret_key = _resolve_secret_key()

# Behind Caddy on Nest, TLS terminates at the proxy and gunicorn sees plain
# HTTP on the internal port. Without this, url_for(_external=True) builds an
# http://internal:8080 OAuth redirect_uri that won't match the https public one
# registered at Google. ProxyFix trusts Caddy's X-Forwarded-Proto/Host (one
# proxy hop) so Flask reconstructs the real https://<domain> URL.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Session cookie hardening. Secure (HTTPS-only) is opt-in via env so local
# plain-HTTP dev still works; set COOKIE_SECURE=1 in the production .env.
# HttpOnly keeps JS from reading the cookie; SameSite=Lax still allows the
# top-level GET redirect back from Google to carry the session.
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("COOKIE_SECURE", "0") == "1"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Email/password auth is off by default so production launches Google-only.
# Flip EMAIL_AUTH=1 in the env (once SMTP is configured) to expose it.
EMAIL_AUTH_ENABLED = os.environ.get("EMAIL_AUTH", "0") == "1"


@app.context_processor
def _inject_auth_flags():
    """Make auth/tier flags available to every template: login.html uses
    email_auth to show/hide the email forms; the filler and pricing pages use
    is_pro / is_admin / the account fields to render the right experience."""
    pro = _is_pro()
    return {
        "email_auth": EMAIL_AUTH_ENABLED,
        "is_authed": bool(session.get("role")),
        "is_admin": session.get("role") == "admin",
        "is_pro": pro,
        "has_font": bool(_current_font_id()),
        "acct_name": session.get("user_name", ""),
        "acct_email": session.get("user_email", ""),
        "acct_picture": session.get("user_picture", ""),
        "stripe_payment_link": STRIPE_PAYMENT_LINK,
        "pro_price": PRO_PRICE,
        "cancel_at_period_end": bool(session.get("cancel_at_period_end")),
        # Free-tier meter. Pro is unmetered, so credits_left is None there and
        # templates should check is_pro before showing a count.
        "free_daily_credits": usage.FREE_DAILY_CREDITS,
        "credits_left": None if pro else usage.remaining_credits(_user_key()),
    }

PASSWORD_ADMIN = os.environ.get("ADMIN_PASSWORD", "alien")

# Accounts whose email is in this allowlist get the "admin" role at login.
# Everyone else is a normal user (or pro, per their is_pro column). Read from
# env as a comma-separated list so it's configurable without a code change;
# defaults to the owner's address. Emails are lowercased so the check is
# case-insensitive and stored in a set (see _role_for below).
ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "karman.chhatwal@gmail.com").split(",")
    if e.strip()
}


def _role_for(email: str) -> str:
    """Map an email to its access role. Only allowlisted emails are admin."""
    return "admin" if (email or "").strip().lower() in ADMIN_EMAILS else "user"


def _vision_model(is_pro: bool) -> str:
    if is_pro:
        return os.environ.get("VISION_MODEL_PRO", VISION_MODEL)
    return VISION_MODEL


def _ai_model(is_pro: bool) -> str:
    normal = os.environ.get("AI_MODEL", "openai/gpt-5.5")
    if is_pro:
        return os.environ.get("AI_MODEL_PRO", normal)
    return normal


def _is_pro() -> bool:
    """True if the signed-in user is on the Pro tier. Admins are always Pro so
    the owner can use Pro features without paying themselves."""
    return bool(session.get("is_pro")) or session.get("role") == "admin"


def _user_key() -> str:
    """Stable per-account key for the daily credit budget. Prefers the session
    subject (Google sub, or "email:<addr>" for password accounts) and falls
    back to the address, so the balance follows the account rather than the
    browser."""
    return session.get("user_sub") or session.get("user_email") or ""


# The three things Pro actually buys, in one place: the pricing page, the
# upgrade card on the filler, and the limit message all read from this list so
# they can never drift out of sync.
def _pro_benefits() -> list[dict]:
    return [
        {"title": "Unlimited daily fills",
         "detail": f"Free gets {usage.FREE_DAILY_CREDITS} AI credits a day "
                   f"(1 credit = {usage.CREDIT_TOKENS:,} tokens, about "
                   "2 credits per page)."},
        {"title": "A smarter AI model",
         "detail": "Pro fills run on a stronger model, so answers are better."},
        {"title": "Fill in your own handwriting",
         "detail": "Turn your real handwriting into a font and write with it."},
    ]


@app.context_processor
def _inject_pro_benefits():
    return {"pro_benefits": _pro_benefits()}


# ---- Pro tier / billing --------------------------------------------------
# Pro is sold via a Stripe Payment Link — paste the link from the Stripe
# dashboard into STRIPE_PAYMENT_LINK and the upgrade buttons light up. Stripe
# collects the buyer's email at checkout; the webhook below matches it to the
# account and flips is_pro. STRIPE_WEBHOOK_SECRET is the signing secret for
# that endpoint (Stripe dashboard → Webhooks). Without the secret the webhook
# rejects everything, so an unauthenticated POST can never grant Pro.
STRIPE_PAYMENT_LINK = os.environ.get("STRIPE_PAYMENT_LINK", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
# Secret key (sk_...) for calling Stripe's API server-side — currently only
# used to cancel a subscription from /billing/cancel. Optional: without it,
# cancellation just tells the user to email support instead of erroring.
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
# Display price for the pricing page. Kept here (not hardcoded in the template)
# so it can be bumped without touching markup.
PRO_PRICE = os.environ.get("PRO_PRICE", "$5/yr")


def pro_required(f):
    """Gate an /api/* route behind the Pro tier. Returns 403 if not signed in,
    402 (Payment Required) with an upgrade URL if signed in but not Pro. This
    is the real enforcement — the UI also hides Pro controls, but a free user
    hitting the endpoint directly is stopped here."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("role"):
            return jsonify({"error": "authentication required"}), 403
        if not _is_pro():
            return jsonify({"error": "Pro required",
                            "upgrade_url": url_for("pricing")}), 402
        return f(*args, **kwargs)
    return wrapper

oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# A verification link is only good for this long. Short enough to limit the
# damage if an email is forwarded or leaks; long enough that a user who opens
# it the next morning still gets in.
VERIFY_TOKEN_TTL = timedelta(hours=24)


def send_verification_email(to_email: str, verify_url: str) -> None:
    """Email a verification link. If SMTP isn't configured (local dev), fall
    back to printing the link to the server log so the flow is still testable."""
    host = os.environ.get("SMTP_HOST")
    if not host:
        print(f"[auth] SMTP not configured. Verify link for {to_email}:\n  {verify_url}")
        return
    msg = EmailMessage()
    msg["Subject"] = "Verify your Paperfill account"
    msg["From"] = os.environ.get("SMTP_FROM", "no-reply@paperfill.app")
    msg["To"] = to_email
    msg.set_content(
        "Welcome to Paperfill!\n\n"
        f"Confirm your email to activate your account:\n{verify_url}\n\n"
        "This link expires in 24 hours. If you didn't sign up, ignore this email."
    )
    try:
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.starttls()
            user = os.environ.get("SMTP_USER")
            pw = os.environ.get("SMTP_PASS")
            if user and pw:
                s.login(user, pw)
            s.send_message(msg)
    except (smtplib.SMTPException, OSError) as e:
        # Never let a mail hiccup 500 the signup; the user can re-request later.
        print(f"[auth] send_verification_email failed for {to_email}: {e}")

# How much of the stored answer-key / reference text to feed the vision model
# when answering a hand-snipped question (see /api/snip).
SNIP_REF_MAX = 12000

# Answer-then-anchor fill: when on, /api/fill shows the worksheet page image(s)
# to the model so it answers from what it can SEE (answer banks, matching option
# lists, tables, diagrams) instead of from the transcribed text alone. Detection
# still owns where each answer is anchored. Set PAPERFILL_VISION_FILL=0 to revert
# to the text-only fill (call_openai_to_fill).
VISION_FILL = os.environ.get("PAPERFILL_VISION_FILL", "1") != "0"
# DPI for the page images sent to the vision fill.
VISION_FILL_DPI = int(os.environ.get("VISION_FILL_DPI", "150"))

# Hack Club's AI proxy (the "primary" provider) is free to us, but Hack Club
# funds it with a fixed daily allowance per project that resets at UTC
# midnight and does not roll over. The admin dashboard tracks estimated draw
# against today's cap so it's visible before the proxy runs dry for the day.
HACK_CLUB_BUDGET_USD = float(os.environ.get("HACK_CLUB_BUDGET_USD", "3.0"))

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


# Whether a brand-new account is created on Pro. Defaults to on (the behaviour
# every existing deployment has) until an admin turns it off from the console.
AUTO_PRO_PATH = BASE_DIR / "auto_pro.txt"


def get_auto_pro() -> bool:
    try:
        return AUTO_PRO_PATH.read_text().strip() == "1"
    except OSError:
        return True


def set_auto_pro(enabled: bool) -> None:
    AUTO_PRO_PATH.write_text("1" if enabled else "0")

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
        return "Your handwriting"
    return str(style_id)


def _current_font_id() -> str | None:
    """The font id owned by the signed-in user, if they've built one. A user
    can only ever address their own font (id derived from their subject)."""
    sub = session.get("user_sub", "")
    if not sub:
        return None
    fid = font_store.user_font_id(sub)
    return fid if font_store.font_path(fid) else None

# ---- Device tracking -----------------------------------------------------
# A long-lived cookie identifies a browser/device. The first request without
# the cookie is counted as a brand-new device, inserted into the shared
# `devices` table so the count is consistent across workers and restarts.
DEVICE_COOKIE = "pf_device"

# The whole product runs behind a sign-in. The page routes redirect to /login
# on their own, but every /api/* endpoint must be gated too — otherwise the
# login screen is cosmetic and anyone can drive the JSON API (and burn LLM
# credits) without an account. One guard covers them all so a newly-added route
# can't forget the check. Returns JSON 403 (not an HTML redirect) so API clients
# get a clean, parseable error.
_PUBLIC_API_PATHS = frozenset({"/api/fonts/template"})  # already gates itself


@app.before_request
def _require_login_for_api():
    path = request.path
    if not path.startswith("/api/"):
        return None
    if path in _PUBLIC_API_PATHS:
        return None
    if not session.get("role"):
        return jsonify({"error": "authentication required"}), 403
    return None


# ww2explained.com is a second hostname pointed at this same backend (see
# Caddyfile) that should only be usable by Pro accounts, while
# paperfill.hackclub.app stays open to everyone. Only the pages needed to
# sign in and buy Pro stay reachable without it; everything else redirects
# to the pricing page (or, for /api/*, returns JSON so fetch() callers don't
# choke on an HTML redirect).
_WW2_HOSTS = frozenset({"ww2explained.com", "www.ww2explained.com"})
_WW2_OPEN_PATHS = frozenset({
    "/pricing", "/login", "/signup", "/logout",
    "/auth/google", "/auth/google/callback", "/upgrade/success",
    "/stripe/webhook",
    "/2d7883f358a775fc1a8f.txt", "/0efb70ed5ecb5409945db6f7bb100589.html",
})


@app.before_request
def _require_pro_for_ww2explained():
    if request.host.split(":")[0].lower() not in _WW2_HOSTS:
        return None
    path = request.path
    if (path in _WW2_OPEN_PATHS or path.startswith("/static/")
            or path.startswith("/verify/")):
        return None
    if path.startswith("/api/"):
        if not _is_pro():
            return jsonify({"error": "Pro required on this domain"}), 402
        return None
    if not _is_pro():
        return redirect(url_for("pricing", ww2_only=1))
    return None


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
# AI_API_KEY / OPENROUTER_API_KEY from environment / .env.
_openai_client = None
def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from paperfill.ai.llm_client import build_client
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
        # No (or cleared) style → drop any cached handwriting so a re-render
        # falls back to typeset text instead of stamping stale PNGs.
        _write_hw_images(job_id, {})
        return
    from paperfill.handwriting.font_render import render_text_png
    from paperfill.ai.render import hw_wrap_width
    variants = font_store.font_variant_paths(font_id)
    settings = font_store.get_settings(font_id)
    # Keep each overlay's bbox so handwriting wraps to the slot width.
    items = {ov["id"]: (ov.get("text", ""), ov.get("bbox"))
             for ov in job.get("overlays", [])}
    try:
        images = {
            ov_id: render_text_png(text, variants,
                                   max_width_px=hw_wrap_width(bbox),
                                   settings=settings)
            for ov_id, (text, bbox) in items.items()
            if str(text).strip() and bbox
        }
    except Exception as e:
        print(f"[handwriting] local font render failed, "
              f"falling back to text: {e}")
        return
    _write_hw_images(job_id, images)


def save_job(job_id: str) -> None:
    """Mirror a job's metadata to disk so other workers can load it. Stamps the
    in-memory copy with the file's mtime so load_job can tell when another
    worker has written a newer version."""
    job = JOBS.get(job_id)
    if job is None:
        return
    (OUTPUTS / job_id).mkdir(exist_ok=True)
    path = _job_meta_path(job_id)
    # Don't persist the private mtime marker.
    path.write_text(json.dumps({k: v for k, v in job.items() if k != "_mtime"}))
    try:
        job["_mtime"] = path.stat().st_mtime_ns
    except OSError:
        pass


def load_job(job_id: str) -> dict | None:
    """Return a job, preferring the on-disk copy when it's newer than what this
    worker has cached. Under multiple gunicorn workers, an upload, fill and
    style change can each land on a different worker; without this freshness
    check a worker would serve its own stale copy (e.g. pre-fill overlays),
    which silently skipped handwriting regeneration."""
    if not job_id:
        return None
    path = _job_meta_path(job_id)
    cached = JOBS.get(job_id)
    try:
        disk_mtime = path.stat().st_mtime_ns
    except OSError:
        return cached  # no file on disk; best we have is the cache (maybe None)
    if cached is not None and cached.get("_mtime") == disk_mtime:
        return cached
    try:
        job = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return cached
    job["_mtime"] = disk_mtime
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
        # reports/<id>/ is intentionally left alone: a report outlives its job
        # so it's still reproducible when it's read, and expires on its own
        # REPORT_RETENTION_DAYS clock in sweep_old_reports().
        (UPLOADS / f"{d.name}.pdf").unlink(missing_ok=True)
        JOBS.pop(d.name, None)


# ---- Helpers -------------------------------------------------------------

def new_job_id() -> str:
    return secrets.token_urlsafe(12)


# Job ids are secrets.token_urlsafe() output, i.e. the URL-safe base64 alphabet.
# Anything else is refused before it can reach a filesystem path.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


REPORT_PDF_KINDS = ("original", "filled")


def reported_pdf_path(job_id: str, kind: str = "original") -> Path | None:
    """Path to a report's snapshotted PDF — the "original" the user uploaded or
    the "filled" one we produced from it. None if the id is malformed or that
    snapshot doesn't exist (a fill that never rendered has no filled PDF)."""
    if not _JOB_ID_RE.match(job_id or "") or kind not in REPORT_PDF_KINDS:
        return None
    path = REPORTS / job_id / f"{kind}.pdf"
    return path if path.is_file() else None


def _snapshot_reported_pdf(job_id: str) -> bool:
    """Copy a reported job's uploaded PDF and the filled PDF rendered from it
    into reports/<job_id>/. Best-effort: returns False (without raising) if the
    id is bad or the upload is already gone. A missing filled PDF is not a
    failure — the report is still worth keeping without it."""
    if not _JOB_ID_RE.match(job_id or ""):
        return False
    original = UPLOADS / f"{job_id}.pdf"
    if not original.is_file():
        return False
    dest = REPORTS / job_id
    try:
        dest.mkdir(exist_ok=True)
        shutil.copy2(original, dest / "original.pdf")
    except OSError as e:
        print(f"[report] snapshot failed for {job_id}: {e}")
        return False
    filled = (load_job(job_id) or {}).get("filled_path")
    if filled and Path(filled).is_file():
        try:
            shutil.copy2(filled, dest / "filled.pdf")
        except OSError as e:
            print(f"[report] filled snapshot failed for {job_id}: {e}")
    return True


def _report_job_ids() -> list[str]:
    """Job ids that currently have a snapshot bundle on this server's disk."""
    try:
        return [d.name for d in REPORTS.iterdir()
                if d.is_dir() and _JOB_ID_RE.match(d.name)]
    except OSError:
        return []


def reported_at(job_id: str) -> float:
    """When a report was filed, as a unix timestamp. Read from the sidecar
    rather than a file mtime: the snapshots are copy2'd, so their mtimes are
    the *upload's*, not the report's. Falls back to the directory's mtime for
    reports filed before the sidecar existed, and 0 when nothing is readable —
    which reads as "ancient", so such a bundle sweeps rather than sticking
    around forever."""
    d = REPORTS / job_id
    try:
        filed = json.loads((d / "report.json").read_text())["reported_at"]
        return datetime.fromisoformat(filed).timestamp()
    except (OSError, TypeError, ValueError, KeyError):
        pass
    try:
        return d.stat().st_mtime
    except OSError:
        return 0.0


def sweep_old_reports() -> None:
    """Best-effort: delete report bundles older than REPORT_RETENTION_DAYS so
    reports/ doesn't hold other people's documents indefinitely. Never raises."""
    cutoff = time.time() - REPORT_RETENTION_DAYS * 86400
    for job_id in _report_job_ids():
        if reported_at(job_id) >= cutoff:
            continue
        shutil.rmtree(REPORTS / job_id, ignore_errors=True)


def last_reports_download() -> float:
    """Unix timestamp of the newest report handed out by /admin/reports.zip.
    0 when the button has never been pressed, so the first press takes all."""
    try:
        return float(REPORTS_MARKER.read_text().strip())
    except (OSError, ValueError):
        return 0.0


def new_report_ids() -> list[str]:
    """Reports filed since the last /admin/reports.zip download, oldest first."""
    since = last_reports_download()
    return sorted((j for j in _report_job_ids() if reported_at(j) > since),
                  key=reported_at)


def _migrate_flat_reports() -> None:
    """Fold the pre-bundle reports/<id>.pdf and reports/<id>.json layout into
    reports/<id>/. Idempotent and cheap, so it just runs at import rather than
    living on as a second code path in every reader."""
    try:
        stale = [p for p in REPORTS.iterdir()
                 if p.is_file() and p.suffix in (".pdf", ".json")
                 and _JOB_ID_RE.match(p.stem)]
    except OSError:
        return
    for p in stale:
        dest = REPORTS / p.stem
        try:
            dest.mkdir(exist_ok=True)
            p.replace(dest / ("original.pdf" if p.suffix == ".pdf"
                              else "report.json"))
        except OSError as e:
            print(f"[report] could not migrate {p.name}: {e}")


_migrate_flat_reports()


# UI labels for the answer-format ids the picker sends. Kept next to the report
# code because that's the only place the raw ids are shown to a human.
_FORMAT_LABELS = {
    "inline_blanks": "Fill-in-the-blank",
    "open_response": "Open response",
    "bullet_answer": "Bullet list",
    "table": "Table / chart",
    "multiple_choice": "Multiple choice",
}


def report_settings_path(job_id: str) -> Path | None:
    """Path to the settings sidecar written beside a reported PDF, or None."""
    if not _JOB_ID_RE.match(job_id or ""):
        return None
    path = REPORTS / job_id / "report.json"
    return path if path.is_file() else None


def _snapshot_report_settings(job_id: str, text: str) -> bool:
    """Write the settings that produced a reported fill to the report bundle.

    A sidecar rather than a new DB column: it keeps the whole evidence bundle
    (PDF + settings) together and needs no schema migration. Best-effort, same
    as the PDF snapshot — a report is still worth keeping without it."""
    if not _JOB_ID_RE.match(job_id or ""):
        return False
    job = load_job(job_id) or {}
    meta = {
        "job_id": job_id,
        "reported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": text,
        "original_name": job.get("original_name"),
        # What the user picked in the UI.
        "detector": job.get("detector"),
        # None means the picker sent nothing, i.e. detect every format.
        "formats": job.get("formats"),
        # What the server actually did (vision fill can fall back to text).
        "fill_path": job.get("fill_path"),
        "vision_fill_enabled": VISION_FILL,
        # Only when the job is still on disk: _style_label(None) is "Typed text",
        # which would read as a recorded choice rather than "we don't know".
        "style": _style_label(job.get("style_id")) if job else None,
        "page_count": job.get("page_count"),
    }
    try:
        dest = REPORTS / job_id
        dest.mkdir(exist_ok=True)
        (dest / "report.json").write_text(json.dumps(meta, indent=2))
        return True
    except OSError as e:
        print(f"[report] settings snapshot failed for {job_id}: {e}")
        return False


def read_report_settings(job_id: str) -> dict | None:
    """Load a report's settings sidecar, with format ids mapped to their UI
    labels for display. None when there's no sidecar (every report filed
    before this was added)."""
    path = report_settings_path(job_id)
    if path is None:
        return None
    try:
        meta = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    formats = meta.get("formats")
    meta["format_labels"] = (
        [_FORMAT_LABELS.get(f, f) for f in formats]
        if isinstance(formats, list) else None)
    return meta


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
        elif u["type"] == "multiple_choice":
            # The model picks one option; the answer (its label) is keyed by
            # unit_id. Send the labels + option text so it can choose.
            clean["answer_key"] = u["unit_id"]
            clean["options"] = [{"label": o["label"], "text": o["text"]}
                                for o in (u.get("options") or [])]
        units_for_llm.append(clean)
    return {"units": units_for_llm}


def call_openai_to_fill(structure_for_llm: dict, instructions: str = "",
                        is_pro: bool = False, user_key: str = "") -> dict[str, str]:
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
        "  - 'multiple_choice': the prompt is a question with an 'options' list, "
        "    each option having a 'label' (a letter A, B, C… or a Roman numeral "
        "    I, II, III…) and text. Pick the ONE correct option and return just "
        "    its label EXACTLY as shown (e.g. \"C\" or \"III\") keyed by the "
        "    unit's answer_key.\n"
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
        "Write math the way it would be handwritten on the page: √, ∛, π, "
        "°, x², and a slash for fractions (5√6/√22). NEVER use LaTeX — no "
        "\\frac, no \\sqrt, no backslash commands, no $…$ and no \\(…\\) "
        "delimiters: the answer is drawn onto the paper exactly as you "
        "write it.\n"
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

    with call_context("text_fill", is_pro=is_pro, user_key=user_key):
        response = get_openai_client().chat.completions.create(
            model=_ai_model(is_pro),
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


def _render_page_pngs(pdf_path: str, dpi: int = VISION_FILL_DPI) -> list[bytes]:
    """Render each PDF page to PNG bytes for the vision fill."""
    out: list[bytes] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            out.append(page.get_pixmap(dpi=dpi).tobytes("png"))
    finally:
        doc.close()
    return out


_VISION_FILL_SYSTEM = (
    "You are filling in a worksheet. You are shown ONE worksheet page image AND "
    "a JSON list of the 'units' detected on that page. Each unit has an id and a "
    "prompt: 'inline_blanks'/'table' prompts contain {{slot_id}} placeholders "
    "(answer each slot_id); 'open_response' units are keyed by their answer_key.\n"
    "'multiple_choice' units carry an 'options' list (each with a 'label' — a "
    "letter A/B/C… or a Roman numeral I/II/III… — and text): pick the ONE correct "
    "option and return just its label EXACTLY as shown (e.g. \"C\" or \"III\") "
    "keyed by the unit's answer_key.\n"
    "Answer EVERY unit in the list — do not skip any. Read the PAGE IMAGE to "
    "understand each item: use any answer bank, word box, matching option list, "
    "table, diagram or worked example you can see. The image is authoritative; "
    "the unit list just tells you which id each answer belongs to.\n"
    "MATCHING items: when a term has a blank and there is a SEPARATE list of "
    "lettered or numbered options (e.g. 'A. to wake up', '1. nucleus'), the "
    "answer is the matching option's LABEL — the letter or number — NOT the "
    "option's text.\n"
    "'graph' units are a coordinate grid to plot on: answer with ONE STRING "
    "holding the points on the curve as \"(x, y)\" pairs, e.g. \"(-1, 7), "
    "(0, 1), (1, -1), (2, 1), (3, 7)\" — a string, never a JSON array or a "
    "nested list. Give 7 to 15 points spread across the visible x-range, "
    "including any intercepts and the vertex, and nothing else — the points are "
    "plotted literally where you put them.\n"
    "Give the ACTUAL answer in the language and format the item calls for (a "
    "word, phrase, conjugated verb, letter, number, date, …). Be accurate. "
    "Never reply with meta or filler text. For a multi-part answer (point k "
    "of n), write a DIFFERENT specific point in each, never repeating.\n"
    "If the user provides instructions or an answer key, treat those as "
    "authoritative and prefer them over your own knowledge.\n"
    "Write math the way it would be handwritten on the page: √, ∛, π, "
    "°, x², and a slash for fractions (5√6/√22). NEVER use LaTeX — no "
    "\\frac, no \\sqrt, no backslash commands, no $…$ and no \\(…\\) "
    "delimiters: the answer is drawn onto the paper exactly as you "
    "write it.\n"
    "Return ONLY a JSON object: {\"<slot_or_unit_id>\": \"<answer>\", ...}. "
    "No prose, no markdown, no <think> tags. /no_think"
)


def _vision_fill_one_page(structure_for_llm: dict, png: bytes,
                          instructions: str = "", is_pro: bool = False,
                          user_key: str = "",
                          missing_ids: list[str] | None = None) -> dict[str, str]:
    """One vision fill call for a single page's units + that page's image.

    `missing_ids` marks this as a second pass over the ids a first pass left
    blank; naming them is what stops the model from skipping the same hard
    items again."""
    structure_json = json.dumps(structure_for_llm, ensure_ascii=False)
    instructions = (instructions or "").strip()

    content: list[dict] = []
    if instructions:
        content.append({"type": "text", "text": (
            "User-provided answer key / instructions (authoritative — prefer "
            "over your own knowledge):\n" + instructions)})
    content.append({"type": "text",
                    "text": "Units detected on this page:\n" + structure_json})
    if missing_ids:
        content.append({"type": "text", "text": (
            "A previous pass left these ids unanswered. They are all "
            "answerable from the page — work each one out and return an "
            "answer for every id: " + ", ".join(missing_ids))})
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode()
    content.append({"type": "image_url", "image_url": {"url": data_uri}})

    with call_context("vision_fill", is_pro=is_pro, user_key=user_key):
        response = get_openai_client().chat.completions.create(
            model=_vision_model(is_pro),
            messages=[
                {"role": "system", "content": _VISION_FILL_SYSTEM},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
    raw = response.choices[0].message.content or "{}"
    flat = _flatten_answers(extract_json_object(raw))
    if not flat:
        print(f"[fill] WARNING: vision fill returned no usable answers. "
              f"Raw (first 400 chars):\n{raw[:400]}")
    return flat


def call_vision_to_fill(structure: dict, page_pngs: list[bytes],
                        instructions: str = "", is_pro: bool = False,
                        user_key: str = "") -> dict[str, str]:
    """
    Answer-then-anchor fill, ONE call per page (run in parallel).

    The model is shown a single page image plus the units detected on that page
    and returns {slot_or_unit_id: answer}. Per-page is deliberate: a single call
    spanning every page makes the model answer only the first page and stop, so a
    long study guide came back mostly blank. Splitting keeps each response small
    and complete, and the calls fan out across threads.

    Seeing the page lets the model use answer banks, matching option lists,
    tables and layout the transcribed structure can't convey. Detection still
    owns WHERE each answer is anchored; this owns only the answer text, keyed by
    the same slot/unit ids.

    A page whose response came back short gets one more call listing exactly
    the ids that are still blank. Models drop the hard items off the end of a
    long JSON object, and re-asking for just those recovers most of them; the
    extra call only happens on pages that need it.
    """
    from concurrent.futures import ThreadPoolExecutor

    by_page: dict[int, list] = {}
    for u in structure.get("units", []):
        by_page.setdefault(u.get("page", 0), []).append(u)

    def fill_page(pidx: int) -> dict[str, str]:
        if pidx < 0 or pidx >= len(page_pngs):
            return {}
        units = by_page[pidx]
        answered = _vision_fill_one_page(
            strip_bboxes_for_llm({"units": units}), page_pngs[pidx],
            instructions, is_pro, user_key)
        unanswered = [u for u in units
                      if any(not answered.get(i) for i in _answer_ids(u))]
        if unanswered:
            missing = [i for u in unanswered for i in _answer_ids(u)
                       if not answered.get(i)]
            retry = _vision_fill_one_page(
                strip_bboxes_for_llm({"units": unanswered}), page_pngs[pidx],
                instructions, is_pro, user_key, missing)
            answered.update({k: v for k, v in retry.items()
                             if v and not answered.get(k)})
        return answered

    pages = sorted(by_page)
    answers: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(pages) or 1)) as ex:
        for part in ex.map(fill_page, pages):
            answers.update(part)
    return answers


def _answer_ids(unit: dict) -> list[str]:
    """Every id the model owes an answer for in this unit."""
    if unit["type"] == "inline_blanks":
        return [s["slot_id"] for s in unit["slots"]]
    if unit["type"] == "table":
        return [s["slot_id"] for row in unit["table_cells"] for cell in row
                if cell for s in cell["slots"]]
    return [unit["unit_id"]]


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
        if isinstance(v, dict):
            for sk, sv in v.items():
                if text := _answer_text(sv):
                    out[_normalize_key(sk)] = text
        elif text := _answer_text(v):
            out[_normalize_key(k)] = text
    return out


def _answer_text(value) -> str:
    """One answer value as the text to write on the page.

    JSON mode returns a bare number for a numeric answer and occasionally a
    list for a multi-part one; both used to be dropped as off-spec, which left
    the blank empty on exactly the sheets — math — where it happens most."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(t for t in (_answer_text(v) for v in value) if t)
    if isinstance(value, str):
        return plain_math(value.strip())
    return ""


def call_vision_for_answer(png_bytes: bytes, instructions: str = "",
                           is_pro: bool = False, user_key: str = "") -> str:
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
        "prefer them over your own knowledge.\n"
        "Write math the way it would be handwritten on the page: √, ∛, π, "
        "°, x², and a slash for fractions (5√6/√22). NEVER use LaTeX — no "
        "\\frac, no \\sqrt, no backslash commands, no $…$ and no \\(…\\) "
        "delimiters: the answer is drawn onto the paper exactly as you "
        "write it.\n"
        "Return ONLY a JSON object: "
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

    with call_context("ask_ai", is_pro=is_pro, user_key=user_key):
        response = get_openai_client().chat.completions.create(
            model=_vision_model(is_pro),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
    content = response.choices[0].message.content or "{}"
    return _answer_text(extract_json_object(content).get("answer", ""))


def call_openai_to_refine(text: str, mode: str, instruction: str = "",
                          ref: str = "", is_pro: bool = False,
                          user_key: str = "") -> str:
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

    with call_context("refine", is_pro=is_pro, user_key=user_key):
        response = get_openai_client().chat.completions.create(
            model=_ai_model(is_pro),
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

def _start_user_session(user: dict) -> None:
    """Populate the session for a signed-in user. One place so Google and
    email/password logins can't drift apart on what they store."""
    email = user.get("email", "")
    session["role"] = _role_for(email)        # "admin" for the allowlist, else "user"
    session["is_pro"] = bool(user.get("is_pro"))  # the pro tier, independent of admin
    session["cancel_at_period_end"] = bool(user.get("cancel_at_period_end"))
    session["user_sub"] = user.get("google_sub") or f"email:{email}"
    session["user_email"] = email
    session["user_name"] = user.get("name", "")
    session["user_picture"] = user.get("picture", "")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if not EMAIL_AUTH_ENABLED:
            abort(404)
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        user = db.get_user_by_email(email)
        # Same generic message whether the email is unknown or the password is
        # wrong: telling an attacker "that email exists" leaks who has accounts.
        if not user or not user.get("password_hash") or \
                not check_password_hash(user["password_hash"], password):
            _record_signin("failed")
            return render_template("login.html",
                                   error="Incorrect email or password.")
        if not user.get("email_verified"):
            return render_template(
                "login.html",
                error="Please verify your email first. Check your inbox for the link.")
        _record_signin("user")
        _start_user_session(user)
        return redirect(url_for("index"))
    return render_template("login.html", error=None)


@app.post("/signup")
def signup():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    name = (request.form.get("name") or "").strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        return render_template("login.html", error="Enter a valid email address.", mode="signup")
    if len(password) < 8:
        return render_template("login.html",
                               error="Password must be at least 8 characters.", mode="signup")
    if db.get_user_by_email(email):
        return render_template("login.html",
                               error="An account with that email already exists.", mode="signup")

    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + VERIFY_TOKEN_TTL).isoformat()
    user = db.create_email_user(
        email=email,
        password_hash=generate_password_hash(password),
        name=name,
        token=token,
        token_expires=expires,
        is_pro=get_auto_pro(),
    )
    if user is None:
        return render_template("login.html",
                               error="Could not create account. Please try again.", mode="signup")
    verify_url = url_for("verify_email", token=token, _external=True)
    send_verification_email(email, verify_url)
    return render_template("login.html", notice=(
        "Account created. Check your email for a verification link to finish "
        "setting up your account."))


@app.route("/verify/<token>")
def verify_email(token):
    user = db.get_user_by_token(token)
    if user is None:
        return render_template("login.html",
                               error="That verification link is invalid or has already been used.")
    # Check expiry while the token (and its expiry) still exist on the row.
    expires = user.get("token_expires")
    if expires:
        try:
            if datetime.fromisoformat(expires.replace("Z", "+00:00")) < datetime.now(timezone.utc):
                return render_template("login.html",
                                       error="That verification link has expired. Please sign up again.")
        except (ValueError, TypeError):
            pass
    updated = db.mark_email_verified(token)
    _record_signin("user")
    _start_user_session(updated or user)
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/auth/google")
def auth_google():
    redirect_uri = url_for("auth_google_callback", _external=True)
    # Behind the Nest gateway the inbound scheme can arrive as http (chained
    # X-Forwarded-Proto that ProxyFix can't reliably collapse), so the callback
    # would be built as http:// and fail Google's exact-match against the https
    # URI we registered. Force https in production; local dev runs in debug over
    # plain http, so leave it alone there.
    if not app.debug and redirect_uri.startswith("http://"):
        redirect_uri = "https://" + redirect_uri[len("http://"):]
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def auth_google_callback():
    try:
        token = oauth.google.authorize_access_token()
    except Exception as e:
        print(f"[auth] Google OAuth error: {e}")
        return render_template("login.html", error="Google sign-in failed. Please try again.")
    userinfo = token.get("userinfo") or oauth.google.userinfo()
    google_sub = userinfo["sub"]
    user = db.get_or_create_user(
        google_sub=google_sub,
        email=userinfo.get("email", ""),
        name=userinfo.get("name", ""),
        picture=userinfo.get("picture", ""),
        is_pro=get_auto_pro(),
    )
    if user is None:
        return render_template("login.html", error="Could not create account. Please try again.")
    # Make sure the session carries fresh profile fields even when the DB is
    # off (the fallback row lacks name/picture); merge in what Google gave us.
    user = {**user, "email": userinfo.get("email", ""),
            "name": userinfo.get("name", ""),
            "picture": userinfo.get("picture", "")}
    _record_signin("user")
    _start_user_session(user)
    return redirect(url_for("index"))


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
    sweep_old_reports()  # so the "new reports" count below matches what's left

    # Window and display timezone are query params so the dashboard can be
    # re-sliced without a code change. Clamped: `days` feeds range() sizes.
    try:
        days = max(7, min(365, int(request.args.get("days", 30))))
    except (TypeError, ValueError):
        days = 30
    try:
        tz_offset = max(-14.0, min(14.0, float(request.args.get("tz", "0"))))
    except (TypeError, ValueError):
        tz_offset = 0.0

    # Eight independent HTTP round-trips to Supabase. Serially that's several
    # seconds of pure latency, so fan them out — they don't depend on each
    # other. Each fetch already swallows its own errors and returns [].
    from concurrent.futures import ThreadPoolExecutor
    jobs = {
        "signins": db.fetch_signins,
        "assignments": db.fetch_assignments,
        "users": db.fetch_users,
        "ai_calls": db.fetch_ai_calls,
        "payments": db.fetch_payments,
        "devices_daily": db.fetch_devices_daily,
        "devices_hourly": db.fetch_devices_hourly,
        "device_total": db.device_count,
    }
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {k: ex.submit(fn) for k, fn in jobs.items()}
        data = {k: f.result() for k, f in futures.items()}

    s = stats.build(
        signins=data["signins"], assignments=data["assignments"],
        users=data["users"], ai_calls=data["ai_calls"],
        payments=data["payments"], devices_daily=data["devices_daily"],
        devices_hourly=data["devices_hourly"], device_total=data["device_total"],
        rate_card=costs.load(), hack_club_cap_usd=HACK_CLUB_BUDGET_USD,
        days=days, tz_offset=tz_offset,
    )

    # Read from the shared database so every worker shows the same numbers.
    signin_log = data["signins"]
    activity_log = data["assignments"]
    for e in signin_log:
        e["timestamp"] = _fmt_ts(e.get("ts"))
    for e in activity_log:
        e["timestamp"] = _fmt_ts(e.get("ts"))
        # A non-empty `feedback` column *is* a report — the report text and the
        # PDF snapshot are written together by /api/report.
        e["reported"] = bool((e.get("feedback") or "").strip())
        e["has_pdf"] = reported_pdf_path(e.get("job_id") or "") is not None
        e["has_filled"] = reported_pdf_path(e.get("job_id") or "",
                                            "filled") is not None
        e["settings"] = (read_report_settings(e.get("job_id") or "")
                         if e["reported"] else None)
    user_count = sum(1 for e in signin_log if e.get("result") == "user")
    admin_count = sum(1 for e in signin_log if e.get("result") == "admin")
    fail_count = sum(1 for e in signin_log if e.get("result") == "failed")
    report_count = sum(1 for e in activity_log if e["reported"])
    # Snapshots live on this machine's disk while the rows come from the shared
    # DB, so a reported row can legitimately have no PDF here.
    pdf_count = sum(1 for e in activity_log if e["has_pdf"])
    last_download = last_reports_download()
    rate_card = costs.load()
    return render_template(
        "admin.html",
        logs=signin_log,
        total=len(signin_log),
        user_count=user_count,
        admin_count=admin_count,
        fail_count=fail_count,
        activity=activity_log,
        activity_total=len(activity_log),
        report_count=report_count,
        pdf_count=pdf_count,
        new_report_count=len(new_report_ids()),
        stored_report_count=len(_report_job_ids()),
        report_retention_days=REPORT_RETENTION_DAYS,
        last_reports_download=(
            datetime.fromtimestamp(last_download, timezone.utc)
            .strftime("%Y-%m-%d %H:%M UTC") if last_download else None),
        reports_status=request.args.get("reports_status"),
        device_count=data["device_total"],
        s=s,
        days=days,
        tz_offset=tz_offset,
        rate_card=rate_card,
        rate_models=sorted(set(costs.known_models())
                           | set(rate_card["models"])
                           | {r["name"] for r in s["money"]["by_model"]
                              if r["name"] != "—"}),
        rates_status=request.args.get("rates_status"),
        db_enabled=db.enabled(),
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        current_user_password=get_user_password(),
        pw_status=request.args.get("pw_status"),
        ads_enabled=get_ads_enabled(),
        vast_tags="\n".join(get_vast_tags()),
        ads_status=request.args.get("ads_status"),
        auto_pro=get_auto_pro(),
        pro_status=request.args.get("pro_status"),
        pro_email=request.args.get("pro_email", ""),
    )


@app.get("/admin/report/<job_id>/<kind>.pdf")
def admin_report_pdf(job_id: str, kind: str):
    """Download one PDF from a problem report — the user's original upload or
    the filled version we gave them back. Admin only, these are other people's
    documents."""
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    path = reported_pdf_path(job_id, kind)
    if path is None:
        abort(404)
    return send_file(path, mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"report-{job_id}-{kind}.pdf")


@app.get("/admin/reports.zip")
def admin_reports_zip():
    """Download report snapshots as one archive. Admin only.

    Only the reports filed since the last time this was pressed, so repeat
    presses don't re-download a growing pile of the same evidence; ?all=1
    takes everything still on disk, for when an archive gets lost.

    Built in memory: these are a handful of small PDFs, so a temp file buys
    nothing. Stored uncompressed because PDF content streams are already
    deflated — re-compressing costs CPU for ~no size win."""
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    sweep_old_reports()
    everything = request.args.get("all") == "1"
    job_ids = sorted(_report_job_ids(), key=reported_at) if everything \
        else new_report_ids()
    if not job_ids:
        return redirect(url_for("admin", reports_status="none") + "#logs")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for job_id in job_ids:
            for p in sorted((REPORTS / job_id).iterdir()):
                if p.is_file():
                    zf.write(p, arcname=f"report-{job_id}/{p.name}")
    buf.seek(0)
    # Mark the newest report we actually packed, not "now": a report filed
    # while this archive was being built would otherwise be skipped forever.
    _mark_reports_downloaded(reported_at(job_ids[-1]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"paperfill-reports-{stamp}.zip")


def _mark_reports_downloaded(ts: float) -> None:
    """Advance the "already handed over" marker; only ever moves forward."""
    if ts <= last_reports_download():
        return
    try:
        REPORTS_MARKER.write_text(f"{ts:.6f}")
    except OSError as e:
        print(f"[report] could not record download marker: {e}")


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


@app.post("/admin/grant-pro")
def grant_pro():
    """Manually grant or revoke Pro for an account by email. The Stripe webhook
    does this automatically on payment; this is the fallback (comps, refunds,
    or buyers whose checkout email didn't match their login)."""
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    email = (request.form.get("email") or "").strip().lower()
    grant = request.form.get("action") == "grant"
    if not email:
        return redirect(url_for("admin", pro_status="empty"))
    ok = db.set_user_pro(email, grant)
    status = ("granted" if grant else "revoked") if ok else "nouser"
    return redirect(url_for("admin", pro_status=status, pro_email=email))


@app.post("/admin/auto-pro")
def change_auto_pro():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    # Unchecked checkboxes don't submit, so absence means "off".
    set_auto_pro(request.form.get("auto_pro") == "on")
    return redirect(url_for("admin", pro_status="auto"))


@app.post("/admin/ai-rates")
def change_ai_rates():
    """Save the AI price list used to cost every future call.

    Anything saved here overrides the built-in rates in costs.BUILTIN_RATES,
    which only cover the models we actually route to. Editing rates does NOT
    rewrite past calls: each row stores the cost computed when it happened.
    """
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    models = {}
    # Fields arrive as rate_in[<model>] / rate_out[<model>].
    for key, val in request.form.items():
        if key.startswith("rate_in[") and key.endswith("]"):
            name = key[8:-1]
            models.setdefault(name, {})["in"] = val
        elif key.startswith("rate_out[") and key.endswith("]"):
            name = key[9:-1]
            models.setdefault(name, {})["out"] = val
    costs.save(models, request.form.get("primary_free") == "on")
    return redirect(url_for("admin", rates_status="ok") + "#money")


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
def handwriting_page():
    """The one handwriting page: build a font from a printed template, then
    tune how it looks (spacing, size, pen thickness) with a live preview.
    Setup and settings used to be two separate pages; they're one flow now,
    with the page showing whichever half applies to you."""
    if not session.get("role"):
        return redirect(url_for("login"))
    if not _is_pro():
        return redirect(url_for("pricing"))
    return render_template("handwriting.html")


@app.route("/handwriting/settings")
def handwriting_settings():
    """Legacy URL. The settings live on /handwriting itself now — kept as a
    redirect so old bookmarks and links don't 404."""
    return redirect(url_for("handwriting_page"))


@app.route("/pricing")
def pricing():
    """Free vs Pro comparison + the upgrade call-to-action. Viewable signed-out
    so it can double as a marketing page; the upgrade button routes to /login
    first when there's no session (we need to know who's buying).

    ww2_only=1 is set when the ww2explained.com gate redirects a non-Pro
    visitor here — shows a one-off notice. It's intentionally separate from
    _pro_benefits() so ww2explained.com access never shows up as a Pro perk
    on this page, the upgrade card, or the limit message."""
    return render_template("pricing.html", upgraded=False,
                            ww2_only=bool(request.args.get("ww2_only")))


@app.route("/upgrade/success")
def upgrade_success():
    """Stripe's post-payment redirect lands here. The webhook is what actually
    sets is_pro in the database; this route just re-reads the user's row so the
    *current session* reflects Pro without making them log out and back in."""
    if not session.get("role"):
        return redirect(url_for("login"))
    email = session.get("user_email", "")
    user = db.get_user_by_email(email) if email else None
    if user:
        session["is_pro"] = bool(user.get("is_pro"))
    return render_template("pricing.html", upgraded=True)


@app.post("/billing/cancel")
def billing_cancel():
    """Cancel the signed-in user's Pro subscription. Sets cancel_at_period_end
    on the Stripe subscription (they keep Pro until the period they already
    paid for runs out; the subscription.deleted webhook downgrades is_pro when
    it actually ends) rather than yanking access immediately.

    Needs both STRIPE_SECRET_KEY and a stripe_subscription_id on the account
    (stored from the checkout webhook) to actually call Stripe. Either being
    missing — Pro granted manually from /admin, or paid before this was wired
    up — falls back to telling the user to email support instead of silently
    downgrading them or claiming a cancellation that didn't happen."""
    if not session.get("role"):
        return jsonify({"error": "authentication required"}), 403
    if not _is_pro():
        return jsonify({"error": "You're not on Pro."}), 400
    email = session.get("user_email", "")
    user = db.get_user_by_email(email) if email else None
    subscription_id = (user or {}).get("stripe_subscription_id")
    if not STRIPE_SECRET_KEY or not subscription_id:
        support_email = sorted(ADMIN_EMAILS)[0] if ADMIN_EMAILS else "support"
        return jsonify({
            "error": f"We don't have a billing record on file to cancel automatically. "
                     f"Email {support_email} and we'll cancel it for you.",
        }), 400
    try:
        r = requests.post(
            f"https://api.stripe.com/v1/subscriptions/{subscription_id}",
            auth=(STRIPE_SECRET_KEY, ""),
            data={"cancel_at_period_end": "true"},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[stripe] cancel request failed for {email}: {e}")
        return jsonify({"error": "Couldn't reach Stripe. Please try again."}), 502
    if r.status_code >= 400:
        print(f"[stripe] cancel HTTP {r.status_code} for {email}: {r.text[:200]}")
        return jsonify({"error": "Stripe couldn't cancel that subscription. Please try again."}), 502
    db.set_cancel_at_period_end(email, True)
    session["cancel_at_period_end"] = True
    return jsonify({"ok": True,
                     "message": "Your subscription won't renew. You'll keep Pro until the "
                                "current period ends."})


def _verify_stripe_sig(payload: bytes, sig_header: str) -> bool:
    """Verify a Stripe webhook signature without the Stripe SDK. Stripe signs
    `"{timestamp}.{body}"` with HMAC-SHA256 keyed by the endpoint secret and
    sends it as `Stripe-Signature: t=...,v1=...`. No secret configured ⇒ reject,
    so a missing/misconfigured secret never silently trusts callers."""
    if not STRIPE_WEBHOOK_SECRET or not sig_header:
        return False
    try:
        parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
        ts, v1 = parts.get("t"), parts.get("v1")
        if not ts or not v1:
            return False
        if abs(time.time() - int(ts)) > 300:  # drop replays older than 5 min
            return False
        signed = ts.encode() + b"." + payload
        expected = hmac.new(STRIPE_WEBHOOK_SECRET.encode(), signed,
                            hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, v1)
    except (ValueError, TypeError):
        return False


@app.post("/stripe/webhook")
def stripe_webhook():
    """Stripe calls this when a checkout completes. We verify the signature,
    then flip the buyer's account to Pro by the email Stripe collected. Public
    (Stripe is unauthenticated) but signature-gated, and outside /api/ so the
    login guard doesn't intercept it."""
    payload = request.get_data()
    if not _verify_stripe_sig(payload, request.headers.get("Stripe-Signature", "")):
        return jsonify({"error": "bad signature"}), 400
    try:
        event = json.loads(payload or b"{}")
    except ValueError:
        return jsonify({"error": "bad payload"}), 400
    if event.get("type") == "checkout.session.completed":
        obj = event.get("data", {}).get("object", {}) or {}
        email = ((obj.get("customer_details") or {}).get("email")
                 or obj.get("customer_email") or "").strip().lower()
        # Book the revenue regardless of whether we can match the email to an
        # account: the money arrived either way, and a P&L that silently drops
        # unmatched sales is worse than useless. Keyed on the Stripe event id,
        # which is unique, so Stripe's retries can't double-count it.
        db.record_payment(
            email=email,
            amount_cents=obj.get("amount_total") or 0,
            currency=obj.get("currency") or "usd",
            event_id=str(event.get("id") or ""),
            livemode=bool(event.get("livemode")),
        )
        if email and db.set_user_pro(email, True):
            print(f"[stripe] upgraded {email} to Pro")
            # Stash the customer/subscription IDs so /billing/cancel has
            # something to call later. Best-effort — a missing ID here just
            # means cancellation falls back to "email support".
            customer_id = obj.get("customer") or ""
            subscription_id = obj.get("subscription") or ""
            if customer_id or subscription_id:
                db.set_stripe_ids(email, customer_id, subscription_id)
        else:
            print(f"[stripe] checkout completed but no matching user for "
                  f"'{email}' — grant Pro manually from /admin")
    elif event.get("type") == "customer.subscription.deleted":
        # Fires when a subscription actually ends (period ran out after a
        # cancel-at-period-end, or Stripe cancelled it directly). This is the
        # real end of billing, so downgrade the account here rather than at
        # the moment the user clicked Cancel.
        obj = event.get("data", {}).get("object", {}) or {}
        customer_id = obj.get("customer") or ""
        user = db.get_user_by_stripe_customer(customer_id) if customer_id else None
        if user and user.get("email"):
            db.set_user_pro(user["email"], False)
            db.set_cancel_at_period_end(user["email"], False)
            print(f"[stripe] {user['email']} subscription ended, downgraded to Free")
        else:
            print(f"[stripe] subscription.deleted for unknown customer '{customer_id}'")
    return jsonify({"received": True}), 200


@app.route("/2d7883f358a775fc1a8f.txt")
def hilltopads_verify():
    # Public (no login gate) so HilltopAds' crawler can fetch it directly.
    # The homepage "/" redirects to /login, which would hide any token there.
    return send_file(
        BASE_DIR / "verification" / "2d7883f358a775fc1a8f.txt", mimetype="text/plain"
    )


@app.route("/0efb70ed5ecb5409945db6f7bb100589.html")
def site_verify_html():
    # Public (no login gate) so the verifying crawler can fetch it directly.
    return send_file(
        BASE_DIR / "verification" / "0efb70ed5ecb5409945db6f7bb100589.html",
        mimetype="text/html",
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

    # Free tier: check the daily credit budget BEFORE doing any work. Unlike
    # the old per-upload quota, credits are spent as AI calls actually happen
    # (see llm_client._Method._record) rather than here — the true cost of a
    # fill isn't known until the tokens come back. This check just blocks
    # starting a new job once today's balance is already gone; the upgrade
    # card in the UI is only a hint.
    metered = not _is_pro()
    if metered and usage.remaining_credits(_user_key()) <= 0:
        return jsonify({
            "error": f"You've used all {usage.FREE_DAILY_CREDITS} free AI credits "
                     "for today. Upgrade to Pro for unlimited fills.",
            "limit_reached": True,
            "credits_left": 0,
            "upgrade_url": url_for("pricing"),
        }), 402

    sweep_old_jobs()  # opportunistic cleanup of stale jobs
    sweep_old_reports()

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

    # Detector selection: deterministic ("Standard", preprocess.py, default),
    # the "AI Vision" path (multimodal_preprocess.py), or "Regions"
    # (candidates.py), which proposes answer spaces geometrically and has a
    # model pick from them. Note all three are separate from the OCR path
    # (vision_preprocess.py), which fires automatically on scanned pages. Chosen
    # per-request via the `detector` form field or globally via the
    # PAPERFILL_DETECTOR env var.
    detector_mode = (request.form.get("detector")
                     or os.environ.get("PAPERFILL_DETECTOR")
                     or "deterministic").strip().lower()
    if detector_mode in ("regions", "region"):
        detector_name = "regions"
    elif detector_mode in ("multimodal", "mm", "vision2"):
        detector_name = "multimodal"
    else:
        detector_name = "deterministic"

    # Quick sanity check + preprocess
    try:
        if detector_name == "regions":
            structure = region_preprocess_pdf(str(pdf_path), formats=formats)
        elif detector_name == "multimodal":
            structure = multimodal_preprocess_pdf(str(pdf_path), formats=formats)
        else:
            structure = preprocess_pdf(str(pdf_path), formats=formats)
    except Exception as e:
        pdf_path.unlink(missing_ok=True)
        return jsonify({"error": f"could not parse PDF: {e}"}), 400

    # Render preview images of each page so the frontend can show
    # what was uploaded.
    doc = fitz.open(str(pdf_path))
    # A worksheet we filled once, downloaded and handed back. Its blanks are
    # still blank-looking, so it fills again — the answers just land on top of
    # the ones already there. The renderer refuses to restamp text that's
    # already in the slot; this is what lets the UI say why.
    already_filled = FILLED_MARKER in ((doc.metadata or {}).get("keywords") or "")
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
        # Kept so a problem report can say which settings actually produced the
        # fill. Recorded server-side rather than sent by the client with the
        # report: by then the user may have re-toggled the picker, and we want
        # what ran, not what the page currently shows. `formats` of None is
        # meaningful — it's the "detect all" case, not a missing value.
        "detector": detector_name,
        "formats": formats,
    }
    save_job(job_id)

    # Build a frontend-safe summary (no bboxes; they're huge and useless
    # to the UI).
    summary = {
        "job_id": job_id,
        "page_count": page_count,
        "unit_count": structure["unit_count"],
        "slot_count": structure["slot_count"],
        "already_filled": already_filled,
        # None for Pro (unmetered); the UI only shows a count on Free. Most of
        # the actual spend happens during /api/fill, not here, so this is
        # mainly accurate when the AI Vision detector ran above.
        "credits_left": None if _is_pro() else usage.remaining_credits(_user_key()),
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
    answers: dict[str, str] = {}
    # Primary path: answer-then-anchor — let the model see the page so it can use
    # answer banks, matching options and layout. Falls back to the text-only fill
    # if the vision call errors or comes back empty.
    if VISION_FILL:
        try:
            page_pngs = _render_page_pngs(job["pdf_path"])
            answers = call_vision_to_fill(job["structure"], page_pngs, instructions,
                                          _is_pro(), _user_key())
        except Exception as e:
            print(f"[fill] vision fill failed ({e}); falling back to text-only")
    used_vision = bool(answers)
    if not answers:
        try:
            answers = call_openai_to_fill(structure_for_llm, instructions,
                                          _is_pro(), _user_key())
        except Exception as e:
            return jsonify({"error": f"LLM call failed: {e}"}), 502

    overlays = build_overlays_from_structure(job["structure"], answers)
    # Which of the two fill paths above actually produced these answers. The
    # vision path can silently fall back to text-only, so the detector the user
    # picked doesn't tell you this on its own.
    job["fill_path"] = "vision" if used_vision else "text"
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
        # None for Pro (unmetered); the UI only shows a meter on Free. Read
        # fresh here since the fill above just spent some of it.
        "credits_left": None if _is_pro() else usage.remaining_credits(_user_key()),
    })


@app.post("/api/report")
def submit_report():
    """File a user's problem report for a filled assignment, attaching the PDF
    they uploaded so the fill can be reproduced. Body: {job_id, report}.

    The report text goes to the database; the PDF is snapshotted to reports/.
    A snapshot that fails to copy is not fatal — a report with no attachment
    still beats losing the user's description of what went wrong."""
    data = request.get_json(silent=True) or {}
    job_id = str(data.get("job_id", ""))
    text = str(data.get("report", "")).strip()[:2000]
    if not text:
        return jsonify({"error": "empty report"}), 400
    if not db.set_report(job_id, text):
        return jsonify({"error": "unknown job_id"}), 404
    _snapshot_reported_pdf(job_id)
    _snapshot_report_settings(job_id, text)
    return jsonify({"ok": True})


@app.get("/api/fonts")
@pro_required
def list_fonts_route():
    """The signed-in user's handwriting font (0 or 1 — one font per user)."""
    return jsonify({"fonts": font_store.list_fonts_for(session.get("user_sub", ""))})


def _settings_from_args(font_id: str) -> dict:
    """Merge any letter_spacing/font_size/word_spacing/pen_thickness query
    params over the font's stored settings, so the live preview reflects the
    slider positions without having to save first. Unknown/invalid params are
    ignored by coerce_settings."""
    stored = font_store.get_settings(font_id)
    overrides = {k: request.args.get(k) for k in font_store.SETTING_RANGES
                 if request.args.get(k) is not None}
    return font_store.coerce_settings({**stored, **overrides})


@app.get("/api/fonts/<font_id>/sample.png")
@pro_required
def font_sample(font_id: str):
    """Render a sample in the user's font (onboarding + settings preview). Pass
    ?text=... for arbitrary text, and any of the setting params (letter_spacing,
    font_size, word_spacing, pen_thickness) to preview slider positions live. A
    user may only sample their own font."""
    if font_id != _current_font_id():
        abort(404)
    text = (request.args.get("text") or "Sample").strip()[:120] or "Sample"
    from paperfill.handwriting.font_render import render_text_png
    png = render_text_png(text, font_store.font_variant_paths(font_id),
                          settings=_settings_from_args(font_id), apply_pen=True)
    if not png:
        abort(404)
    return send_file(io.BytesIO(png), mimetype="image/png")


@app.get("/api/fonts/<font_id>/settings")
@pro_required
def get_font_settings(font_id: str):
    """The font's tuned appearance settings plus the allowed slider ranges. A
    user may only read their own font's settings."""
    if font_id != _current_font_id():
        abort(404)
    return jsonify({"settings": font_store.get_settings(font_id),
                    "ranges": font_store.SETTING_RANGES,
                    "defaults": font_store.DEFAULT_SETTINGS})


@app.post("/api/fonts/<font_id>/settings")
@pro_required
def save_font_settings(font_id: str):
    """Persist the font's appearance settings (validated + clamped). They take
    effect the next time a job is filled or re-rendered with this font. Own
    font only."""
    if font_id != _current_font_id():
        abort(404)
    data = request.get_json(silent=True) or {}
    clean = font_store.save_settings(font_id, data)
    return jsonify({"ok": True, "settings": clean})


@app.get("/api/fonts/template")
def download_template():
    """Serve the printable handwriting template (Pro onboarding step 1)."""
    if not session.get("role"):
        return redirect(url_for("login"))
    if not _is_pro():
        return redirect(url_for("pricing"))
    from paperfill.handwriting import template as hw_template
    return send_file(io.BytesIO(hw_template.template_pdf_bytes()),
                     mimetype="application/pdf", as_attachment=True,
                     download_name="paperfill-handwriting-template.pdf")


@app.post("/api/fonts")
@pro_required
def build_font_route():
    """Build the user's handwriting font from 1–3 filled template copies and
    store it, replacing any previous font (one font per user). Each copy is a
    full filled template (a 2-page PDF, or its page images) sent as a separate
    multipart group: 'version1' (required), 'version2', 'version3' (optional).
    More copies → more variants → repeated letters look less stamped."""
    sub = session.get("user_sub", "")
    if not sub:
        return jsonify({"error": "not signed in"}), 403

    # Collect the version groups. Fall back to the legacy single 'template'
    # group so an older client still works (treated as one version).
    groups: list[list] = []
    for key in ("version1", "version2", "version3"):
        fs = [f for f in request.files.getlist(key) if f and f.filename]
        if fs:
            groups.append(fs)
    if not groups:
        legacy = [f for f in request.files.getlist("template") if f and f.filename]
        if legacy:
            groups.append(legacy)
    if not groups:
        return jsonify({"error": "no filled template uploaded"}), 400

    import tempfile
    from paperfill.handwriting.font_build import build_font
    otf_variants: list[bytes] = []
    with tempfile.TemporaryDirectory() as d:
        for vi, files in enumerate(groups):
            paths = []
            for fi, f in enumerate(files):
                ext = ".pdf" if f.filename.lower().endswith(".pdf") else ".img"
                p = os.path.join(d, f"v{vi}_page{fi}{ext}")
                f.save(p)
                paths.append(p)
            out = os.path.join(d, f"font{vi}.otf")
            try:
                build_font(paths[0] if len(paths) == 1 else paths, out,
                           family="Paperfill Hand")
                otf_variants.append(Path(out).read_bytes())
            except Exception as e:
                # One bad copy shouldn't sink the whole build if others worked.
                print(f"[fonts] version {vi + 1} failed to build: {e}")
        if not otf_variants:
            return jsonify({"error": "could not build a font from the upload — "
                            "make sure the four corner squares are visible and "
                            "the photo is sharp"}), 422

    font_id = font_store.save_user_font(sub, otf_variants)
    return jsonify({"ok": True, "font_id": font_id,
                    "style_id": f"font:{font_id}", "label": font_store.LABEL,
                    "variants": len(otf_variants)})


@app.post("/api/style")
@pro_required
def upload_style():
    """Attach a user-built handwriting font to a job so the fill renders the
    answers in it. Body: {job_id, style: "font:<id>"}."""
    data = request.get_json(silent=True) or {}
    job_id = request.form.get("job_id") or data.get("job_id")
    job = load_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404

    style = request.form.get("style") or data.get("style") or ""
    # Empty style is allowed: it clears handwriting back to typeset text.
    # Otherwise it must resolve to an existing font AND be THIS user's own —
    # no using anyone else's, and no setting a bogus style id.
    if style:
        fid = _font_id_from_style(style)
        if not fid or fid != _current_font_id():
            return jsonify({"error": "that handwriting font isn't available"}), 403
    job["style_id"] = style or None

    # If the worksheet has already been filled (style picked from the editor),
    # re-apply right away so the preview updates without a second fill pass.
    # Pre-fill (style picked before /api/fill) there's nothing to render yet.
    rerendered = False
    if job.get("overlays") and job.get("filled_path"):
        _generate_hw_for_job(job_id)
        try:
            _rerender_job(job_id)
            rerendered = True
        except Exception as e:
            return jsonify({"error": f"render failed: {e}"}), 500

    save_job(job_id)
    return jsonify({"ok": True, "rerendered": rerendered})


def _rerender_job(job_id: str) -> None:
    """Re-render the filled PDF + page PNG previews from the job's current overlays."""
    job = JOBS[job_id]
    font_id = _font_id_from_style(job.get("style_id"))
    pen_thickness = font_store.get_settings(font_id)["pen_thickness"] if font_id else None
    filled_path = OUTPUTS / f"{job_id}-filled.pdf"
    render_overlays_pdf(job["pdf_path"], job["overlays"], str(filled_path),
                        images=_load_hw_images(job_id),
                        pen_thickness_mm=pen_thickness)
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
    A "kind": "ink" overlay is a freehand pen stroke instead: {id, page, kind,
    points:[[x,y],...], color, width}.
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
        if ov.get("kind") == "ink":
            try:
                page = int(ov.get("page", 0))
                if page < 0 or page > max_page:
                    continue
                points = [[float(x), float(y)] for x, y in ov["points"]][:2000]
                if len(points) < 2:
                    continue
                color = str(ov.get("color", "")).lower()
                if not re.fullmatch(r"#[0-9a-f]{6}", color):
                    color = "#1a1a1a"
                width = max(0.5, min(20.0, float(ov.get("width", 2.0))))
                cleaned.append({
                    "id": str(ov.get("id", "")),
                    "page": page,
                    "kind": "ink",
                    "points": points,
                    "color": color,
                    "width": width,
                })
            except (KeyError, TypeError, ValueError, IndexError):
                pass
            continue
        if ov.get("kind") == "points":
            # A plotted graph. Without this branch it would fall through to the
            # text-box case below, lose its points and come back as an empty
            # box the next time the job re-renders.
            try:
                page = int(ov.get("page", 0))
                if page < 0 or page > max_page:
                    continue
                points = [[float(x), float(y)]
                          for x, y in ov.get("points") or []][:2000]
                plot = ov.get("plot", "points")
                if plot not in ("points", "curve", "none"):
                    plot = "points"
                cleaned.append({
                    "id": str(ov.get("id", "")),
                    "page": page,
                    "kind": "points",
                    "bbox": [float(x) for x in ov["bbox"]],
                    "points": points,
                    "plot": plot,
                })
            except (KeyError, TypeError, ValueError, IndexError):
                pass
            continue
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
            entry = {
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
            }
            # A "circle" overlay marks a multiple-choice answer — no text, drawn
            # as an oval. Preserve the kind so an edited/moved circle survives a
            # re-render instead of collapsing into an (empty) text box.
            if ov.get("kind") == "circle":
                entry["kind"] = "circle"
            cleaned.append(entry)
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
        answer = call_vision_for_answer(png_bytes, job.get("fill_instructions", ""),
                                        _is_pro(), _user_key())
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
            text, mode, instruction, job.get("fill_instructions", ""),
            _is_pro(), _user_key())
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


@app.get("/api/job/<job_id>")
def get_job(job_id):
    """Return a filled job's editor state so the front-end can restore the
    editor after a page refresh. The client keeps job state in memory only, so
    without this a refresh would orphan the (safely persisted) job on disk. The
    job_id is an unguessable token, same capability model as preview/download."""
    job = load_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    return jsonify({
        "job_id": job_id,
        "overlays": job.get("overlays", []),
        "page_count": job.get("page_count", 0),
        "page_sizes": job.get("page_sizes", []),
        "style_id": job.get("style_id"),
        "filled": bool(job.get("filled_path") and job.get("overlays")),
        "original_name": job.get("original_name", ""),
    })


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