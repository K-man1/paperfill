"""
Per-user daily upload quota — the one hard limit on the Free tier.

Free accounts get FREE_DAILY_UPLOADS worksheet uploads per UTC day; Pro is
unmetered. Counts live in a single JSON file next to the app rather than in the
database: this is on the hot path of every upload, and the numbers are cheap to
lose (a missing file just hands everyone a fresh day).

Two gunicorn workers share that file, so every read-modify-write runs under an
exclusive flock. Without it, two uploads landing at once both read "2 used" and
both write "3" — a lost update that quietly gives away a free upload.

Every entry is stamped with the day it belongs to, so a stale row is simply
ignored (and swept on the next write) instead of needing a reset job.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:                      # POSIX only; on anything else we run lock-free.
    import fcntl
except ImportError:       # pragma: no cover - Windows dev boxes
    fcntl = None

_PATH = Path(__file__).parent / "usage.json"

# How many uploads a Free account gets per day. Env-tunable so the limit can be
# loosened for a promo without a code change.
FREE_DAILY_UPLOADS = int(os.environ.get("FREE_DAILY_UPLOADS", "3"))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _mutate(fn):
    """Run ``fn(data) -> result`` against the usage map under an exclusive lock
    and persist whatever it leaves behind.

    Fails OPEN: if the file can't be read or written we return fn's view of an
    empty map, which means an unmetered upload. A broken quota file should cost
    us a few free fills, not break the product for everyone.
    """
    try:
        with open(_PATH, "a+", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read().strip()
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    data = {}
                if not isinstance(data, dict):
                    data = {}
                result = fn(data)
                # Drop rows from previous days so the file stays small.
                today = _today()
                data = {k: v for k, v in data.items()
                        if isinstance(v, dict) and v.get("date") == today}
                fh.seek(0)
                fh.truncate()
                json.dump(data, fh)
                return result
            finally:
                if fcntl is not None:
                    fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError as e:
        print(f"[usage] quota file unavailable ({e}) — not metering this request")
        return fn({})


def used_today(user_key: str) -> int:
    """How many uploads this user has spent today."""
    if not user_key:
        return 0

    def read(data: dict) -> int:
        row = data.get(user_key)
        if not isinstance(row, dict) or row.get("date") != _today():
            return 0
        try:
            return int(row.get("count", 0))
        except (TypeError, ValueError):
            return 0

    return _mutate(read)


def remaining(user_key: str) -> int:
    """Uploads left today for a Free account. Never negative."""
    return max(0, FREE_DAILY_UPLOADS - used_today(user_key))


def consume(user_key: str) -> int:
    """Spend one upload and return how many are left afterwards.

    Read and increment happen inside one locked pass, so concurrent uploads
    can't both spend the same slot.
    """
    if not user_key:
        return FREE_DAILY_UPLOADS

    def bump(data: dict) -> int:
        today = _today()
        row = data.get(user_key)
        count = 0
        if isinstance(row, dict) and row.get("date") == today:
            try:
                count = int(row.get("count", 0))
            except (TypeError, ValueError):
                count = 0
        count += 1
        data[user_key] = {"date": today, "count": count}
        return max(0, FREE_DAILY_UPLOADS - count)

    return _mutate(bump)
