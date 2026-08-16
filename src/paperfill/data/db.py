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
    """Upsert one row per job. Omitting `feedback` from the payload means an
    existing report is preserved on re-fill (PostgREST only updates the
    columns present in the body)."""
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


def set_report(job_id: str, text: str) -> bool:
    """Attach a user's problem report to an existing assignment row. The text
    lands in the legacy `feedback` column, which is what the admin dashboard
    reads. Returns True if a matching row was updated, False otherwise."""
    if not enabled():
        return False
    try:
        r = requests.patch(
            _rest(f"assignments?job_id=eq.{requests.utils.quote(job_id, safe='')}"),
            headers=_headers({"Prefer": "return=representation"}),
            json={"feedback": text},
            timeout=_TIMEOUT,
        )
        if r.status_code >= 400:
            print(f"[db] set_report HTTP {r.status_code}: {r.text[:200]}")
            return False
        return bool(r.json())
    except (requests.RequestException, ValueError) as e:
        print(f"[db] set_report failed: {e}")
        return False


def record_ai_call(row: dict) -> None:
    """Append one metered LLM call. Called from llm_client's writer thread, so
    it must swallow everything — a telemetry failure must never surface."""
    if not enabled():
        return
    try:
        requests.post(_rest("ai_calls"), headers=_headers(), json=row,
                      timeout=_TIMEOUT)
    except requests.RequestException as e:
        print(f"[db] record_ai_call failed: {e}")


def record_payment(email: str, amount_cents: int, currency: str,
                   event_id: str, livemode: bool) -> None:
    """Record one completed checkout. `stripe_event_id` is UNIQUE and we ignore
    duplicates, which is what makes this safe against Stripe's retries — a
    webhook redelivery must not book the same revenue twice."""
    if not enabled():
        return
    try:
        requests.post(
            _rest("payments"),
            headers=_headers({"Prefer": "resolution=ignore-duplicates"}),
            json={"email": email, "amount_cents": int(amount_cents or 0),
                  "currency": (currency or "usd").lower(),
                  "stripe_event_id": event_id or None,
                  "livemode": bool(livemode)},
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        print(f"[db] record_payment failed: {e}")


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
    return _get("assignments?select=job_id,name,ts,ip,style,feedback&order=ts.asc")


def fetch_ai_calls(limit: int = 20000) -> list[dict]:
    """Metered LLM calls, newest first. Capped because this table grows with
    every fill and the dashboard only ever plots a window of it."""
    return _get("ai_calls?select=ts,purpose,model,provider,prompt_tokens,"
                f"output_tokens,total_tokens,cost_usd,latency_ms,ok,error,"
                f"is_pro,job_id&order=ts.desc&limit={int(limit)}")


def fetch_payments() -> list[dict]:
    return _get("payments?select=ts,email,amount_cents,currency,livemode"
                "&order=ts.asc")


def fetch_users() -> list[dict]:
    """Accounts, for signup-over-time and Free/Pro split. No secrets: password
    hashes and verification tokens are deliberately not selected."""
    return _get("users?select=email,name,is_pro,email_verified,created_at,"
                "google_sub&order=created_at.asc")


def fetch_devices_daily() -> list[dict]:
    """One row per day from the devices_daily view. The devices table has six
    figures of rows — aggregating in Postgres keeps them out of this process."""
    return _get("devices_daily?select=day,devices&order=day.asc")


def fetch_devices_hourly() -> list[dict]:
    return _get("devices_hourly?select=dow,hour,devices")


def count_rows(table: str) -> int:
    """Row count without transferring the rows. PostgREST returns the total in
    the Content-Range header when asked for an exact count, so this replaces a
    ~115k-row download that the admin page used to do on every load."""
    if not enabled():
        return 0
    try:
        r = requests.get(
            _rest(f"{table}?select=*&limit=0"),
            headers=_headers({"Prefer": "count=exact"}),
            timeout=_TIMEOUT,
        )
        # Header looks like "0-24/1234" or "*/1234".
        total = r.headers.get("Content-Range", "").split("/")[-1]
        return int(total) if total.isdigit() else 0
    except (requests.RequestException, ValueError) as e:
        print(f"[db] count_rows({table}) failed: {e}")
        return 0


def device_count() -> int:
    return count_rows("devices")


# ---- User accounts -------------------------------------------------------

def get_or_create_user(google_sub: str, email: str, name: str, picture: str) -> dict | None:
    """Look up a user by their Google subject ID; create if missing.
    Returns the user row dict, or None on error."""
    if not enabled():
        return {"google_sub": google_sub, "email": email, "name": name, "picture": picture, "is_pro": False}
    try:
        # Try to find existing user
        r = requests.get(
            _rest(f"users?google_sub=eq.{requests.utils.quote(google_sub, safe='')}"),
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if r.status_code < 400:
            rows = r.json()
            if rows:
                return rows[0]
        # Create new user. Google already verified the email it hands us, so
        # these accounts are email_verified from birth (no need to re-check).
        # New sign-ups start on Pro: every account is granted the paid tier at
        # creation rather than waiting on the Stripe webhook or an admin grant.
        # is_pro is sent explicitly so this doesn't depend on the column default.
        r = requests.post(
            _rest("users"),
            headers=_headers({"Prefer": "return=representation"}),
            json={"google_sub": google_sub, "email": email, "name": name,
                  "picture": picture, "email_verified": True, "is_pro": True},
            timeout=_TIMEOUT,
        )
        if r.status_code < 400:
            rows = r.json()
            return rows[0] if rows else None
    except (requests.RequestException, ValueError) as e:
        print(f"[db] get_or_create_user failed: {e}")
    return None


def get_user_by_email(email: str) -> dict | None:
    """Fetch a single user row by email (case-insensitive), or None."""
    if not enabled():
        return None
    # PostgREST `ilike` gives a case-insensitive exact match here (no wildcards).
    rows = _get(f"users?email=ilike.{requests.utils.quote(email, safe='')}")
    return rows[0] if rows else None


def create_email_user(email: str, password_hash: str, name: str,
                      token: str, token_expires: str) -> dict | None:
    """Create an email/password account, unverified, carrying a verification
    token and its expiry. Returns the new row, or None on error (including the
    unique-email collision Postgres raises if the address is already taken).

    Like the Google path, new accounts are created on Pro. The account still has
    to verify its email before it can sign in, so this grants the tier, not
    access."""
    if not enabled():
        return None
    try:
        r = requests.post(
            _rest("users"),
            headers=_headers({"Prefer": "return=representation"}),
            json={"email": email, "password_hash": password_hash, "name": name,
                  "email_verified": False, "verification_token": token,
                  "token_expires": token_expires, "is_pro": True},
            timeout=_TIMEOUT,
        )
        if r.status_code < 400:
            rows = r.json()
            return rows[0] if rows else None
        print(f"[db] create_email_user HTTP {r.status_code}: {r.text[:200]}")
    except (requests.RequestException, ValueError) as e:
        print(f"[db] create_email_user failed: {e}")
    return None


def get_user_by_token(token: str) -> dict | None:
    """Fetch the account holding this verification token (with its expiry so
    the caller can decide if it's still valid), or None if unknown."""
    if not enabled():
        return None
    rows = _get(f"users?verification_token=eq.{requests.utils.quote(token, safe='')}")
    return rows[0] if rows else None


def mark_email_verified(token: str) -> dict | None:
    """Flip the token's account to verified and clear the token so the link
    can't be replayed. Returns the updated row, or None on error."""
    if not enabled():
        return None
    try:
        r = requests.patch(
            _rest(f"users?verification_token=eq.{requests.utils.quote(token, safe='')}"),
            headers=_headers({"Prefer": "return=representation"}),
            json={"email_verified": True, "verification_token": None,
                  "token_expires": None},
            timeout=_TIMEOUT,
        )
        if r.status_code < 400:
            updated = r.json()
            return updated[0] if updated else None
    except (requests.RequestException, ValueError) as e:
        print(f"[db] mark_email_verified failed: {e}")
    return None


def set_user_pro(email: str, is_pro: bool) -> bool:
    """Flip a user's Pro flag by email (case-insensitive). Returns True only if
    a matching row was actually updated (so callers can tell 'no such user'
    apart from success). Used by the Stripe webhook and the admin grant form."""
    if not enabled():
        return False
    try:
        r = requests.patch(
            _rest(f"users?email=ilike.{requests.utils.quote(email, safe='')}"),
            headers=_headers({"Prefer": "return=representation"}),
            json={"is_pro": bool(is_pro)},
            timeout=_TIMEOUT,
        )
        if r.status_code < 400:
            return bool(r.json())
        print(f"[db] set_user_pro HTTP {r.status_code}: {r.text[:200]}")
    except (requests.RequestException, ValueError) as e:
        print(f"[db] set_user_pro failed: {e}")
    return False


def set_stripe_ids(email: str, customer_id: str, subscription_id: str) -> bool:
    """Attach the Stripe customer/subscription IDs from a completed checkout so
    a later cancel request has something to call. Blank IDs are skipped rather
    than overwriting an existing value with nothing."""
    if not enabled() or not email:
        return False
    patch = {}
    if customer_id:
        patch["stripe_customer_id"] = customer_id
    if subscription_id:
        patch["stripe_subscription_id"] = subscription_id
    if not patch:
        return False
    try:
        r = requests.patch(
            _rest(f"users?email=ilike.{requests.utils.quote(email, safe='')}"),
            headers=_headers({"Prefer": "return=representation"}),
            json=patch,
            timeout=_TIMEOUT,
        )
        if r.status_code < 400:
            return bool(r.json())
        print(f"[db] set_stripe_ids HTTP {r.status_code}: {r.text[:200]}")
    except (requests.RequestException, ValueError) as e:
        print(f"[db] set_stripe_ids failed: {e}")
    return False


def set_cancel_at_period_end(email: str, flag: bool) -> bool:
    """Mark whether a user's subscription is set to lapse at the end of the
    period they already paid for. Used by /billing/cancel and cleared when
    Stripe's subscription.deleted webhook confirms the period actually ended."""
    if not enabled() or not email:
        return False
    try:
        r = requests.patch(
            _rest(f"users?email=ilike.{requests.utils.quote(email, safe='')}"),
            headers=_headers({"Prefer": "return=representation"}),
            json={"cancel_at_period_end": bool(flag)},
            timeout=_TIMEOUT,
        )
        if r.status_code < 400:
            return bool(r.json())
        print(f"[db] set_cancel_at_period_end HTTP {r.status_code}: {r.text[:200]}")
    except (requests.RequestException, ValueError) as e:
        print(f"[db] set_cancel_at_period_end failed: {e}")
    return False


def get_user_by_stripe_customer(customer_id: str) -> dict | None:
    """Fetch the account tied to a Stripe customer ID. Used by the
    subscription.deleted webhook, whose payload carries the customer id but not
    the account email."""
    if not enabled() or not customer_id:
        return None
    rows = _get(f"users?stripe_customer_id=eq.{requests.utils.quote(customer_id, safe='')}")
    return rows[0] if rows else None
