"""
One-time backfill: push the old file-based logs into Supabase.

Run this ONCE on the server after deploying the Supabase-backed code, so the
sign-in history (and any filled-assignment / device data) collected before the
switch isn't lost:

    python migrate_logs.py

It reads signin_log.json / activity_log.json / devices.json from this folder
(whichever exist), preserves their original timestamps, and inserts them with
duplicate-ignoring upserts so it's safe to run more than once. Requires
SUPABASE_URL and SUPABASE_SERVICE_KEY in the environment / .env.
"""

import json
import os
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent

# Load .env exactly like app.py does, so this script works standalone.
_env = BASE_DIR / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import db  # noqa: E402  (after .env load so db sees the vars)


def _post(table: str, rows: list, prefer: str) -> None:
    if not rows:
        print(f"  {table}: nothing to migrate")
        return
    key = db._secret_key()
    r = requests.post(
        f"{os.environ['SUPABASE_URL'].rstrip('/')}/rest/v1/{table}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": prefer,
        },
        json=rows,
        timeout=30,
    )
    if r.status_code >= 400:
        print(f"  {table}: ERROR HTTP {r.status_code}: {r.text[:300]}")
    else:
        print(f"  {table}: migrated {len(rows)} row(s)")


def _load(name: str):
    p = BASE_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"  could not read {name}: {e}")
        return None


def main() -> None:
    if not db.enabled():
        raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — aborting.")

    print("Migrating file logs into Supabase…")

    signins = _load("signin_log.json") or []
    _post("signins", [
        {"ts": e.get("timestamp"), "ip": e.get("ip"),
         "ua": e.get("ua"), "result": e.get("result")}
        for e in signins if e.get("result") in db.VALID_SIGNIN_RESULTS
    ], prefer="return=minimal")

    activity = _load("activity_log.json") or []
    _post("assignments", [
        {"job_id": e.get("job_id"), "name": e.get("name") or "Untitled",
         "ts": e.get("timestamp"), "ip": e.get("ip"), "rating": e.get("rating")}
        for e in activity if e.get("job_id")
    ], prefer="resolution=merge-duplicates,return=minimal")

    devices = _load("devices.json") or {}
    _post("devices", [
        {"device_id": did, "first_seen": info.get("first_seen"),
         "ip": info.get("ip"), "ua": info.get("ua")}
        for did, info in (devices.items() if isinstance(devices, dict) else [])
    ], prefer="resolution=ignore-duplicates,return=minimal")

    print("Done.")


if __name__ == "__main__":
    main()
