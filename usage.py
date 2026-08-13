"""
Per-user daily AI-credit budget — the one hard limit on the Free tier.

Free accounts get FREE_DAILY_CREDITS credits per UTC day; Pro is unmetered.
1 credit = CREDIT_TOKENS tokens (prompt + output, straight from the
provider's own usage figures) spent answering that user's worksheets.
Counts live in a single JSON file next to the app rather than in the
database: credit consumption is on the hot path of every metered AI call,
and the numbers are cheap to lose (a missing file just hands everyone a
fresh day).

Two gunicorn workers share that file, so every read-modify-write runs under
an exclusive flock. Without it, two calls landing at once both read "5,000
tokens spent" and both write "5,800" instead of "6,600" — a lost update that
quietly gives away free credits.

Every entry is stamped with the day it belongs to, so a stale row is simply
ignored (and swept on the next write) instead of needing a reset job.

Rough conversion for the UI: the answer-filling call (vision_fill) averages
about 1,560 tokens per page it fills, so 2 credits (2,000 tokens) covers
roughly one page.
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

# Credits a Free account gets per day, and tokens per credit. Env-tunable so
# the limit can be loosened for a promo without a code change.
FREE_DAILY_CREDITS = int(os.environ.get("FREE_DAILY_CREDITS", "20"))
CREDIT_TOKENS = int(os.environ.get("CREDIT_TOKENS", "1000"))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _mutate(fn):
    """Run ``fn(data) -> result`` against the usage map under an exclusive lock
    and persist whatever it leaves behind.

    Fails OPEN: if the file can't be read or written we return fn's view of an
    empty map, which means an unmetered call. A broken quota file should cost
    us a few free credits, not break the product for everyone.
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


def tokens_used_today(user_key: str) -> int:
    """How many tokens this user has spent today, across all metered calls."""
    if not user_key:
        return 0

    def read(data: dict) -> int:
        row = data.get(user_key)
        if not isinstance(row, dict) or row.get("date") != _today():
            return 0
        try:
            return int(row.get("tokens", 0))
        except (TypeError, ValueError):
            return 0

    return _mutate(read)


def remaining_credits(user_key: str) -> float:
    """Credits left today for a Free account. Never negative."""
    used = tokens_used_today(user_key) / CREDIT_TOKENS
    return max(0.0, round(FREE_DAILY_CREDITS - used, 2))


def consume_tokens(user_key: str, tokens: int) -> float:
    """Spend `tokens` tokens' worth of credit and return what's left today.

    Read and increment happen inside one locked pass, so concurrent calls
    can't both spend the same credits.
    """
    if not user_key:
        return float(FREE_DAILY_CREDITS)
    if tokens <= 0:
        return remaining_credits(user_key)

    def bump(data: dict) -> float:
        today = _today()
        row = data.get(user_key)
        total = 0
        if isinstance(row, dict) and row.get("date") == today:
            try:
                total = int(row.get("tokens", 0))
            except (TypeError, ValueError):
                total = 0
        total += int(tokens)
        data[user_key] = {"date": today, "tokens": total}
        return max(0.0, round(FREE_DAILY_CREDITS - total / CREDIT_TOKENS, 2))

    return _mutate(bump)
