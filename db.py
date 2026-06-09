"""
Supabase-backed storage for the admin dashboard data.

Why this exists: PaperFill runs under multiple gunicorn workers. Per-process
memory and even per-file JSON diverged between workers (the admin screen showed
two different tallies depending on which worker answered). A shared Postgres
database is the single source of truth, so every worker reads and writes the
same rows.

We talk to Supabase over its PostgREST HTTP API with a *secret* API key
(sb_secret_…), which runs as a trusted server and bypasses Row Level Security.
The browser never sees this key — only the Flask backend uses it. Set in the
environment:

    SUPABASE_URL         e.g. https://xxxx.supabase.co
    SUPABASE_SECRET_KEY  a secret key (Dashboard → Settings → API keys → secret)

If either is unset, `enabled()` is False and the callers degrade gracefully
(writes become no-ops, reads return empty) instead of crashing.
"""

import os

import requests

_TIMEOUT = 8  # seconds; keep short so a slow DB never hangs a web request

VALID_RATINGS = ("green", "yellow", "red")
VALID_SIGNIN_RESULTS = ("user", "admin", "failed")


def _base_url() -> str | None:
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    return url or None


def _secret_key() -> str | None:
    # Prefer the new secret-key name; fall back to the old one so existing
    # deployments don't break mid-migration.
    key = (os.environ.get("SUPABASE_SECRET_KEY")
           or os.environ.get("SUPABASE_SERVICE_KEY") or "").strip()
    return key or None


def enabled() -> bool:
    return bool(_base_url() and _secret_key())


def _headers(extra: dict | None = None) -> dict:
    key = _secret_key() or ""
    h = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def _rest(path: str) -> str:
    return f"{_base_url()}/rest/v1/{path}"


# ---- Writes --------------------------------------------------------------

def record_signin(ip: str, ua: str, result: str) -> None:
    """Append one sign-in attempt. Best-effort: never let a DB hiccup break login."""
    if not enabled():
        return
    try:
        requests.post(
            _rest("signins"),
            headers=_headers(),
            json={"ip": ip, "ua": ua, "result": result},
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        print(f"[db] record_signin failed: {e}")


def record_fill(job_id: str, name: str, ip: str, style: str | None = None) -> None:
    """Upsert one row per job. Omitting `rating`/`feedback` from the payload
    means an existing rating/feedback is preserved on re-fill (PostgREST only
    updates the columns present in the body)."""
    if not enabled():
        return
    body = {"job_id": job_id, "name": name or "Untitled", "ip": ip}
    if style is not None:
        body["style"] = style
    try:
        requests.post(
            _rest("assignments"),
            headers=_headers({"Prefer": "resolution=merge-duplicates"}),
            json=body,
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        print(f"[db] record_fill failed: {e}")


def set_rating(job_id: str, rating: str) -> bool:
    """Attach a rating to an existing assignment row.
    Returns True if a matching row was updated, False otherwise."""
    if rating not in VALID_RATINGS:
        return False
    if not enabled():
        return False
    try:
        r = requests.patch(
            _rest(f"assignments?job_id=eq.{requests.utils.quote(job_id, safe='')}"),
            headers=_headers({"Prefer": "return=representation"}),
            json={"rating": rating},
            timeout=_TIMEOUT,
        )
        if r.status_code >= 400:
            print(f"[db] set_rating HTTP {r.status_code}: {r.text[:200]}")
            return False
        return bool(r.json())
    except (requests.RequestException, ValueError) as e:
        print(f"[db] set_rating failed: {e}")
        return False


def set_feedback(job_id: str, feedback: str) -> bool:
    """Attach free-text feedback to an existing assignment row.
    Returns True if a matching row was updated, False otherwise."""
    if not enabled():
        return False
    try:
        r = requests.patch(
            _rest(f"assignments?job_id=eq.{requests.utils.quote(job_id, safe='')}"),
            headers=_headers({"Prefer": "return=representation"}),
            json={"feedback": feedback},
            timeout=_TIMEOUT,
        )
        if r.status_code >= 400:
            print(f"[db] set_feedback HTTP {r.status_code}: {r.text[:200]}")
            return False
        return bool(r.json())
    except (requests.RequestException, ValueError) as e:
        print(f"[db] set_feedback failed: {e}")
        return False


def record_device(device_id: str, ip: str, ua: str) -> None:
    """Insert a newly-seen device, ignoring the row if it somehow already exists."""
    if not enabled():
        return
    try:
        requests.post(
            _rest("devices"),
            headers=_headers({"Prefer": "resolution=ignore-duplicates"}),
            json={"device_id": device_id, "ip": ip, "ua": ua},
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        print(f"[db] record_device failed: {e}")


# ---- Reads (admin dashboard) ---------------------------------------------

def _get(path: str) -> list[dict]:
    if not enabled():
        return []
    try:
        r = requests.get(_rest(path), headers=_headers(), timeout=_TIMEOUT)
        if r.status_code >= 400:
            print(f"[db] GET {path} HTTP {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except (requests.RequestException, ValueError) as e:
        print(f"[db] GET {path} failed: {e}")
        return []


def fetch_signins() -> list[dict]:
    """Oldest-first, matching the previous file-based ordering."""
    return _get("signins?select=ts,ip,ua,result&order=ts.asc")


def fetch_assignments() -> list[dict]:
    return _get("assignments?select=job_id,name,ts,ip,rating,style,feedback&order=ts.asc")


def device_count() -> int:
    return len(_get("devices?select=device_id"))
