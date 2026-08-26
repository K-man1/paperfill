"""
Analytics for the admin dashboard.

Everything here is a pure function over rows already fetched from the database:
pass lists of dicts in, get plain numbers and series out. No network, no Flask,
no globals — which is what makes it testable and what keeps app.py's admin
route down to "fetch, aggregate, render".

Two conventions worth knowing before reading further:

* **Timestamps are stored in UTC** and shifted by `tz_offset` hours only for
  display. Every "by hour of day" number depends on that shift, so the page
  always states which zone it's showing.
* **Unknown is not zero.** A model with no configured rate produces a NULL
  cost, and those calls are counted and surfaced separately rather than being
  quietly averaged in as free. A profit figure that hides uncosted calls is a
  lie, so `money()` reports how many it had to ignore.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---- Time helpers --------------------------------------------------------

def parse_ts(value) -> datetime | None:
    """Parse a Postgres timestamptz into an aware UTC datetime, or None."""
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def shift(dt: datetime | None, tz_offset: float) -> datetime | None:
    """Move a UTC datetime into the dashboard's display zone."""
    if dt is None:
        return None
    return dt + timedelta(hours=tz_offset or 0)


def _stamps(rows: list[dict], field: str, tz_offset: float) -> list[datetime]:
    out = []
    for r in rows or []:
        dt = shift(parse_ts(r.get(field)), tz_offset)
        if dt is not None:
            out.append(dt)
    return out


# ---- Small numeric helpers ----------------------------------------------

def pct(part: float, whole: float) -> float:
    """Percentage, guarding the empty case so the template never divides by 0."""
    return round(100.0 * part / whole, 1) if whole else 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value at or below which at least
    p% of the data falls, i.e. rank ceil(p/100 * n), 1-indexed.

    Deliberately ceil() and not round() — with round(), p50 of ten samples
    lands on rank 6 instead of 5 and reports the 60th percentile as the median.
    No numpy for four numbers.
    """
    if not values:
        return 0.0
    s = sorted(values)
    k = max(1, math.ceil((p / 100.0) * len(s)))
    return s[min(k, len(s)) - 1]


def _fill_days(counts: dict, days: int, today: datetime) -> list[dict]:
    """Turn {date: n} into a dense, gap-free series ending today. Missing days
    must appear as zeros — a line chart that just skips them implies activity
    that never happened."""
    out = []
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).date()
        out.append({"label": d.isoformat(), "value": counts.get(d, 0)})
    return out


# ---- Sections ------------------------------------------------------------

def time_of_day(assignments, signins, tz_offset: float) -> dict:
    """The dow×hour heatmap plus its 1-D margins.

    Fills are the headline signal (someone actually used the product); sign-ins
    are shown alongside because they cover people who showed up and bounced.
    """
    grid = [[0] * 24 for _ in range(7)]
    by_hour = [0] * 24
    by_dow = [0] * 7
    for dt in _stamps(assignments, "ts", tz_offset):
        grid[dt.weekday()][dt.hour] += 1
        by_hour[dt.hour] += 1
        by_dow[dt.weekday()] += 1

    signin_by_hour = [0] * 24
    for dt in _stamps(signins, "ts", tz_offset):
        signin_by_hour[dt.hour] += 1

    peak_hour = max(range(24), key=lambda h: by_hour[h]) if any(by_hour) else None
    peak_dow = max(range(7), key=lambda d: by_dow[d]) if any(by_dow) else None
    return {
        "grid": grid,
        "max": max((max(row) for row in grid), default=0),
        "by_hour": by_hour,
        "by_dow": by_dow,
        "signin_by_hour": signin_by_hour,
        "peak_hour": peak_hour,
        "peak_hour_label": f"{peak_hour:02d}:00" if peak_hour is not None else "—",
        "peak_dow_label": DOW_NAMES[peak_dow] if peak_dow is not None else "—",
        "dow_names": DOW_NAMES,
    }


def activity(assignments, signins, users, devices_daily,
             days: int, tz_offset: float) -> dict:
    """Daily time series for the main volume charts."""
    today = shift(datetime.now(timezone.utc), tz_offset)

    fills = Counter(dt.date() for dt in _stamps(assignments, "ts", tz_offset))
    logins = Counter(dt.date() for dt in _stamps(signins, "ts", tz_offset)
                     if True)
    signups = Counter(dt.date() for dt in _stamps(users, "created_at", tz_offset))

    dev = {}
    for row in devices_daily or []:
        try:
            dev[datetime.fromisoformat(str(row.get("day"))).date()] = int(row.get("devices") or 0)
        except (ValueError, TypeError):
            continue

    return {
        "fills": _fill_days(fills, days, today),
        "signins": _fill_days(logins, days, today),
        "signups": _fill_days(signups, days, today),
        "devices": _fill_days(dev, days, today),
        "days": days,
    }


def money(ai_calls, payments, days: int, tz_offset: float) -> dict:
    """Revenue, AI cost and profit.

    Cost has two honest flavours and the dashboard shows both:

    * **cash** — what actually left the bank. PaperFill's primary provider is
      the free Hack Club proxy, so only fallback calls cost real money.
    * **uncosted** — calls whose model has no configured rate. Reported as a
      count, never folded into the totals as zero.
    """
    today = shift(datetime.now(timezone.utc), tz_offset)

    # Stripe test-mode events use the same webhook and shape as real ones —
    # only `livemode` tells them apart. Counting test checkouts as revenue
    # would make the P&L confidently wrong, so they're dropped here rather
    # than at the fetch layer (the raw rows may still be useful elsewhere).
    live_payments = [p for p in (payments or []) if p.get("livemode")]

    rev_by_day: dict = Counter()
    revenue_cents = 0
    for p in live_payments:
        dt = shift(parse_ts(p.get("ts")), tz_offset)
        cents = int(p.get("amount_cents") or 0)
        revenue_cents += cents
        if dt:
            rev_by_day[dt.date()] += cents

    cost_by_day: dict = defaultdict(float)
    cost_total = 0.0
    uncosted = 0
    by_purpose: dict = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})
    by_model: dict = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})
    tokens_in = tokens_out = 0

    for c in ai_calls or []:
        dt = shift(parse_ts(c.get("ts")), tz_offset)
        raw_cost = c.get("cost_usd")
        if raw_cost is None:
            uncosted += 1
            cost = 0.0
        else:
            try:
                cost = float(raw_cost)
            except (TypeError, ValueError):
                cost = 0.0
        p_tok = int(c.get("prompt_tokens") or 0)
        o_tok = int(c.get("output_tokens") or 0)
        tokens_in += p_tok
        tokens_out += o_tok
        cost_total += cost
        if dt:
            cost_by_day[dt.date()] += cost

        k = c.get("purpose") or "unknown"
        by_purpose[k]["calls"] += 1
        by_purpose[k]["tokens"] += p_tok + o_tok
        by_purpose[k]["cost"] += cost
        m = c.get("model") or "—"
        by_model[m]["calls"] += 1
        by_model[m]["tokens"] += p_tok + o_tok
        by_model[m]["cost"] += cost

    revenue = revenue_cents / 100.0
    profit = revenue - cost_total

    def _series(counts, scale=1.0):
        out = []
        for i in range(days - 1, -1, -1):
            d = (today - timedelta(days=i)).date()
            out.append({"label": d.isoformat(),
                        "value": round(counts.get(d, 0) * scale, 6)})
        return out

    return {
        "revenue": revenue,
        "revenue_cents": revenue_cents,
        "cost": cost_total,
        "profit": profit,
        "margin_pct": pct(profit, revenue) if revenue else 0.0,
        "payment_count": len(live_payments),
        "arpu": (revenue / len(live_payments)) if live_payments else 0.0,
        "uncosted_calls": uncosted,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "revenue_series": _series(rev_by_day, 0.01),
        "cost_series": _series(cost_by_day),
        "by_purpose": sorted(
            ({"name": k, **v} for k, v in by_purpose.items()),
            key=lambda r: -r["tokens"]),
        "by_model": sorted(
            ({"name": k, **v} for k, v in by_model.items()),
            key=lambda r: -r["tokens"]),
    }


def hack_club_budget(ai_calls, rate_card: dict, cap_usd: float) -> dict:
    """How much of *today's* Hack Club AI proxy credit has been drawn down.

    Hack Club funds the "primary" provider with a fixed daily allowance that
    resets at UTC midnight and does not carry over — unused credit from
    yesterday is just gone, and today starts back at the full cap. So this
    sums only calls timestamped on today's UTC date, not a running lifetime
    total. Deliberately UTC and not the dashboard's display tz_offset: the
    number needs to line up with when Hack Club's meter actually resets, not
    with how the admin prefers charts labeled.

    Calls on "primary" cost us $0 (that's the whole point of the proxy), but
    Hack Club is still paying real per-token rates on the other end. We
    estimate that draw with the same rate card used for fallback pricing,
    deliberately ignoring the "primary is free" flag here — that flag
    describes our bill, not theirs. A model missing from the rate card is
    counted as uncosted, same convention as money().
    """
    today = datetime.now(timezone.utc).date()
    models = (rate_card or {}).get("models") or {}
    spent = 0.0
    uncosted = 0
    for c in ai_calls or []:
        if (c.get("provider") or "") != "primary":
            continue
        ts = parse_ts(c.get("ts"))
        if ts is None or ts.date() != today:
            continue
        rate = models.get(c.get("model") or "")
        if not rate:
            uncosted += 1
            continue
        p_tok = int(c.get("prompt_tokens") or 0)
        o_tok = int(c.get("output_tokens") or 0)
        spent += (p_tok / 1_000_000.0 * float(rate.get("in", 0))
                  + o_tok / 1_000_000.0 * float(rate.get("out", 0)))
    remaining = max(0.0, cap_usd - spent)
    return {
        "cap": cap_usd,
        "spent": spent,
        "remaining": remaining,
        "pct_used": pct(spent, cap_usd),
        "uncosted_calls": uncosted,
        "exhausted": spent >= cap_usd,
    }


def reliability(ai_calls) -> dict:
    """Success rate, provider fallback rate and latency — the health of the AI
    layer, which is invisible from the fill logs alone."""
    total = len(ai_calls or [])
    ok = sum(1 for c in ai_calls or [] if c.get("ok"))
    fallback = sum(1 for c in ai_calls or [] if c.get("provider") == "fallback")
    lat = [float(c.get("latency_ms") or 0) for c in ai_calls or []
           if c.get("ok") and c.get("latency_ms")]

    per_purpose = defaultdict(list)
    for c in ai_calls or []:
        if c.get("ok") and c.get("latency_ms"):
            per_purpose[c.get("purpose") or "unknown"].append(float(c["latency_ms"]))

    errors = Counter((c.get("error") or "").split(":")[0]
                     for c in ai_calls or [] if not c.get("ok") and c.get("error"))

    return {
        "total": total,
        "ok": ok,
        "failed": total - ok,
        "success_pct": pct(ok, total),
        "fallback_calls": fallback,
        "fallback_pct": pct(fallback, total),
        "p50_ms": int(percentile(lat, 50)),
        "p95_ms": int(percentile(lat, 95)),
        "latency_by_purpose": sorted(
            ({"name": k,
              "p50": int(percentile(v, 50)),
              "p95": int(percentile(v, 95)),
              "calls": len(v)} for k, v in per_purpose.items()),
            key=lambda r: -r["calls"]),
        "top_errors": errors.most_common(6),
    }


def accounts(users, assignments, payments) -> dict:
    """Who's signed up and how far down the funnel they get."""
    total = len(users or [])
    pro = sum(1 for u in users or [] if u.get("is_pro"))
    verified = sum(1 for u in users or [] if u.get("email_verified"))
    google = sum(1 for u in users or [] if u.get("google_sub"))

    return {
        "total": total,
        "pro": pro,
        "free": total - pro,
        "pro_pct": pct(pro, total),
        "verified": verified,
        "verified_pct": pct(verified, total),
        "google": google,
        "email": total - google,
        "paid": len(payments or []),
        "paid_pct": pct(len(payments or []), total),
        "fills_per_user": round(len(assignments or []) / total, 1) if total else 0.0,
    }


# `style` on an assignment is a human-readable label, not a flag — a typed fill
# is stored as the string "Typed text", not as empty. So "did they use
# handwriting?" is "is the label anything other than the typed one", and an
# empty/missing label counts as typed too.
TYPED_STYLES = {"", "typed text", "none", "typed"}


def product(assignments) -> dict:
    """How the filler itself is being used."""
    total = len(assignments or [])
    reported = sum(1 for a in assignments or []
                   if (a.get("feedback") or "").strip())
    handwriting = sum(1 for a in assignments or []
                      if (a.get("style") or "").strip().lower() not in TYPED_STYLES)

    styles = Counter((a.get("style") or "").strip() or "Typed text"
                     for a in assignments or [])
    ips = Counter((a.get("ip") or "?") for a in assignments or [])

    return {
        "total": total,
        "reported": reported,
        "report_pct": pct(reported, total),
        "handwriting": handwriting,
        "handwriting_pct": pct(handwriting, total),
        "styles": styles.most_common(8),
        "unique_ips": len(ips),
        "top_ips": ips.most_common(8),
        "repeat_pct": pct(sum(n for _, n in ips.items() if n > 1), total),
    }


def clients(signins) -> dict:
    """Browser / OS / device split, parsed from user-agent strings.

    Deliberately crude: UA sniffing is guesswork, and a dashboard is the one
    place where "Other" is an acceptable answer.
    """
    browsers, oses, kinds = Counter(), Counter(), Counter()
    for s in signins or []:
        ua = (s.get("ua") or "")
        u = ua.lower()

        if "edg/" in u:
            browsers["Edge"] += 1
        elif "chrome" in u and "chromium" not in u:
            browsers["Chrome"] += 1
        elif "safari" in u and "chrome" not in u:
            browsers["Safari"] += 1
        elif "firefox" in u:
            browsers["Firefox"] += 1
        else:
            browsers["Other"] += 1

        if "iphone" in u or "ipad" in u:
            oses["iOS"] += 1
        elif "android" in u:
            oses["Android"] += 1
        elif "mac os" in u or "macintosh" in u:
            oses["macOS"] += 1
        elif "windows" in u:
            oses["Windows"] += 1
        elif "linux" in u:
            oses["Linux"] += 1
        else:
            oses["Other"] += 1

        kinds["Mobile" if ("mobile" in u or "iphone" in u or "android" in u)
              else "Desktop"] += 1

    return {
        "browsers": browsers.most_common(),
        "os": oses.most_common(),
        "kinds": kinds.most_common(),
    }


def devices_heatmap(rows) -> dict:
    """First-touch traffic by dow×hour, straight from the devices_hourly view.

    Comes pre-aggregated from Postgres because the devices table is six figures
    of rows and has no business being pulled into this process.
    """
    grid = [[0] * 24 for _ in range(7)]
    for r in rows or []:
        try:
            # Postgres DOW is 0=Sunday; the grid is 0=Monday.
            dow = (int(r.get("dow")) + 6) % 7
            grid[dow][int(r.get("hour"))] += int(r.get("devices") or 0)
        except (TypeError, ValueError, IndexError):
            continue
    return {"grid": grid, "max": max((max(row) for row in grid), default=0)}


def build(*, signins, assignments, users, ai_calls, payments,
          devices_daily, devices_hourly, device_total,
          rate_card: dict | None = None, hack_club_cap_usd: float = 3.0,
          days: int = 30, tz_offset: float = 0.0) -> dict:
    """Everything the dashboard needs, in one dict."""
    return {
        "time_of_day": time_of_day(assignments, signins, tz_offset),
        "activity": activity(assignments, signins, users, devices_daily,
                             days, tz_offset),
        "money": money(ai_calls, payments, days, tz_offset),
        "hack_club": hack_club_budget(ai_calls, rate_card or {}, hack_club_cap_usd),
        "reliability": reliability(ai_calls),
        "accounts": accounts(users, assignments, payments),
        "product": product(assignments),
        "clients": clients(signins),
        "device_heat": devices_heatmap(devices_hourly),
        "device_total": device_total,
        "tz_offset": tz_offset,
        "days": days,
    }
