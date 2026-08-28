"""
Analytics for the admin dashboard.

Everything here is a pure function over rows already fetched from the database:
pass lists of dicts in, get plain numbers and series out. No network, no Flask,
no globals — which is what makes it testable and what keeps app.py's admin
route down to "fetch, aggregate, render".

Three conventions worth knowing before reading further:

* **Timestamps are stored in UTC** and moved into the display zone only for
  display. Every "by hour of day" number depends on that shift, so the page
  always states which zone it's showing.
* **Unknown is not zero.** A model with no configured rate produces a NULL
  cost, and those calls are counted and surfaced separately rather than being
  quietly averaged in as free. A profit figure that hides uncosted calls is a
  lie, so `money()` reports how many it had to ignore.
* **The window filters events, not inventory.** `build()` slices the event
  streams — fills, sign-ins, AI calls, payments — to the chosen window, but
  account and device totals stay all-time: "how many accounts exist" is a
  stock, and windowing it would answer a question nobody asked.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo

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


def shift(dt: datetime | None, tz: tzinfo) -> datetime | None:
    """Move a UTC datetime into the dashboard's display zone."""
    if dt is None:
        return None
    return dt.astimezone(tz)


@dataclass(frozen=True)
class Window:
    """The slice of time the dashboard is showing.

    Half-open [start, end) in UTC so an event can never land in two buckets.
    `tz` is display-only — it decides which local day or hour a UTC timestamp
    is drawn in, never which rows are included. It is a zone rather than a
    fixed offset because a window can span a DST change (or sit entirely in a
    different regime from today's), and bucketing a December range with
    August's offset draws points outside the range that was asked for.

    Buckets are hourly for short spans: "last 6 hours" plotted as one daily
    point is a chart of nothing.
    """
    start: datetime
    end: datetime
    tz: tzinfo = timezone.utc
    label: str = ""

    @property
    def hourly(self) -> bool:
        return self.end - self.start <= timedelta(days=3)

    @property
    def tz_offset(self) -> float:
        """Hours east of UTC at the end of the window, for the zone label the
        page prints. Nothing is bucketed with this."""
        return self.end.astimezone(self.tz).utcoffset().total_seconds() / 3600.0

    def filter(self, rows: list[dict], field: str = "ts") -> list[dict]:
        """The rows whose timestamp falls in the window. Rows with an
        unparseable timestamp are dropped — a row we can't place in time can't
        honestly be claimed to be in the window."""
        out = []
        for r in rows or []:
            dt = parse_ts(r.get(field))
            if dt is not None and self.start <= dt < self.end:
                out.append(r)
        return out

    def key(self, dt: datetime):
        """The bucket a display-zone datetime belongs to. Hourly keys drop the
        zone so they stay comparable across a DST change, where the same wall
        clock hour carries two different offsets."""
        if self.hourly:
            return dt.replace(tzinfo=None, minute=0, second=0, microsecond=0)
        return dt.date()

    def bucket(self, dt: datetime | None):
        """The bucket key for a UTC timestamp, or None if it won't parse."""
        local = shift(dt, self.tz)
        return self.key(local) if local else None

    def series(self, counts: dict, scale: float = 1.0) -> list[dict]:
        """Dense {label, value} points, one per bucket, oldest first.

        Missing buckets must appear as zeros — a line chart that just skips
        them implies activity that never happened.
        """
        step = timedelta(hours=1) if self.hourly else timedelta(days=1)
        # Walk the local wall clock, not UTC: one point per local day/hour is
        # what the axis claims to show, and stepping in UTC would land 23:00 or
        # 01:00 on the far side of a DST change.
        cur = shift(self.start, self.tz).replace(tzinfo=None, minute=0,
                                                 second=0, microsecond=0)
        if not self.hourly:
            cur = cur.replace(hour=0)
        end = shift(self.end, self.tz).replace(tzinfo=None)
        out = []
        while cur < end:
            out.append({"label": self._point_label(cur),
                        "value": round(counts.get(self.key(cur), 0) * scale, 6)})
            cur += step
        return out

    def _point_label(self, dt: datetime) -> str:
        if not self.hourly:
            return dt.strftime("%m-%d")
        # Inside a single day the date is noise; across two or three it's the
        # only thing telling 09:00 Tuesday from 09:00 Wednesday.
        if self.end - self.start <= timedelta(days=1):
            return dt.strftime("%H:%M")
        return dt.strftime("%m-%d %H:%M")


def _stamps(rows: list[dict], field: str, tz) -> list[datetime]:
    """Display-zone timestamps. `tz` is a zone, or the fixed offset in hours
    that time_of_day still takes."""
    if not isinstance(tz, tzinfo):
        tz = timezone(timedelta(hours=tz or 0))
    out = []
    for r in rows or []:
        dt = shift(parse_ts(r.get(field)), tz)
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


def activity(assignments, signins, users, devices_daily, window: Window) -> dict:
    """Volume time series for the main charts, bucketed to fit the window."""
    tz = window.tz
    fills = Counter(window.key(dt) for dt in _stamps(assignments, "ts", tz))
    logins = Counter(window.key(dt) for dt in _stamps(signins, "ts", tz))
    signups = Counter(window.key(dt) for dt in _stamps(users, "created_at", tz))

    # devices_daily is pre-aggregated per calendar day in Postgres, so it can't
    # be split into hours. An hourly window gets an empty series rather than a
    # day total smeared across 24 buckets.
    #
    # Its days are UTC days; the chart's buckets are local ones. A row is a
    # span of 24 hours starting at its own UTC midnight, so resolve that
    # instant through the same bucket() every other count goes through.
    # Matching the bare UTC date against a local bucket key instead leaves the
    # newest row with nowhere to land for the hours between UTC midnight and
    # local midnight, and it drops off the chart every night.
    dev = {}
    if not window.hourly:
        for row in devices_daily or []:
            try:
                day = datetime.fromisoformat(str(row.get("day")))
                count = int(row.get("devices") or 0)
            except (ValueError, TypeError):
                continue
            key = window.bucket(day.replace(tzinfo=timezone.utc))
            if key is not None:
                dev[key] = count

    return {
        "fills": window.series(fills),
        "signins": window.series(logins),
        "signups": window.series(signups),
        "devices": window.series(dev),
        "devices_daily_only": window.hourly,
    }


def money(ai_calls, payments, window: Window) -> dict:
    """Revenue, AI cost and profit.

    Cost has two honest flavours and the dashboard shows both:

    * **cash** — what actually left the bank. PaperFill's primary provider is
      the free Hack Club proxy, so only fallback calls cost real money.
    * **uncosted** — calls whose model has no configured rate. Reported as a
      count, never folded into the totals as zero.
    """
    # Stripe test-mode events use the same webhook and shape as real ones —
    # only `livemode` tells them apart. Counting test checkouts as revenue
    # would make the P&L confidently wrong, so they're dropped here rather
    # than at the fetch layer (the raw rows may still be useful elsewhere).
    live_payments = [p for p in (payments or []) if p.get("livemode")]

    rev_by_bucket: dict = Counter()
    revenue_cents = 0
    for p in live_payments:
        k = window.bucket(parse_ts(p.get("ts")))
        cents = int(p.get("amount_cents") or 0)
        revenue_cents += cents
        if k is not None:
            rev_by_bucket[k] += cents

    cost_by_bucket: dict = defaultdict(float)
    cost_total = 0.0
    uncosted = 0
    by_purpose: dict = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})
    by_model: dict = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})
    tokens_in = tokens_out = 0

    for c in ai_calls or []:
        k = window.bucket(parse_ts(c.get("ts")))
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
        if k is not None:
            cost_by_bucket[k] += cost

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
        "revenue_series": window.series(rev_by_bucket, 0.01),
        "cost_series": window.series(cost_by_bucket),
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


FUNNEL_SOURCES = ("ad", "organic", "unknown")


def _source_bucket(value) -> str:
    """Fold a stored source into one of the funnel's three columns. Anything
    unrecognised, including the NULL on every row written before attribution
    existed, lands in `unknown` rather than being assumed organic."""
    source = (value or "").strip().lower()
    return source if source in ("ad", "organic") else "unknown"


def funnel(users, assignments, payments, device_counts) -> dict:
    """Visit, signup, fill, pay, split by where the visitor came from.

    Every stage after the first is keyed on the account, so one person is
    counted in the same column at each stage they reach and the funnel can
    only ever narrow. Devices are the exception: they're counted before anyone
    identifies themselves, so that row is a different population and the drop
    from it into signups is a rate, not a cohort.

    Fills and payments both attribute through the account rather than
    themselves: a fill has no source of its own, and it's the traffic that
    brought the *account* in that a funnel is asking about."""
    source_by_email = {}
    for user in users or []:
        email = (user.get("email") or "").strip().lower()
        if email:
            source_by_email[email] = _source_bucket(user.get("source"))

    def by_source(emails) -> Counter:
        return Counter(source_by_email.get(e, "unknown") for e in emails)

    # Sets, not row counts: someone who filled nine PDFs is still one person in
    # a funnel, and the payments table holds one row per charge.
    filled = {(a.get("user_email") or "").strip().lower() for a in assignments or []}
    paid = {(p.get("email") or "").strip().lower() for p in payments or []}
    filled.discard("")
    paid.discard("")

    counts = [
        ("Visited", {s: (device_counts or {}).get(s, 0) for s in FUNNEL_SOURCES}),
        ("Signed up", by_source(source_by_email)),
        ("Filled a PDF", by_source(filled)),
        ("Paid", by_source(paid)),
    ]
    return {
        "sources": list(FUNNEL_SOURCES),
        "stages": [{"name": name,
                    "values": {s: values.get(s, 0) for s in FUNNEL_SOURCES},
                    "total": sum(values.get(s, 0) for s in FUNNEL_SOURCES)}
                   for name, values in counts],
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

    ips = Counter((a.get("ip") or "?") for a in assignments or [])

    return {
        "total": total,
        "reported": reported,
        "report_pct": pct(reported, total),
        "handwriting": handwriting,
        "unique_ips": len(ips),
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
          window: Window, hack_club_calls: list[dict] | None = None,
          rate_card: dict | None = None, hack_club_cap_usd: float = 3.0,
          device_counts: dict | None = None) -> dict:
    """Everything the dashboard needs, in one dict.

    The window is applied here, once, so every section is slicing the same
    rows. Two deliberate exceptions: `accounts` stays all-time (see the module
    docstring), and `hack_club_budget` answers a fixed question — today's
    credit — that has nothing to do with whatever window the page is showing.
    `ai_calls` is fetched pre-windowed, so today's rows are handed in
    separately as `hack_club_calls`; without them a window that ends before
    today would report the day's proxy draw as $0.
    """
    win_assignments = window.filter(assignments)
    win_signins = window.filter(signins)
    win_calls = window.filter(ai_calls)
    win_payments = window.filter(payments)

    return {
        "time_of_day": time_of_day(win_assignments, win_signins, window.tz_offset),
        "activity": activity(win_assignments, win_signins, users,
                             devices_daily, window),
        "money": money(win_calls, win_payments, window),
        "hack_club": hack_club_budget(
            ai_calls if hack_club_calls is None else hack_club_calls,
            rate_card or {}, hack_club_cap_usd),
        "reliability": reliability(win_calls),
        "accounts": accounts(users, assignments, payments),
        # All-time, like `accounts`: attribution is a property of how an
        # account arrived, and windowing it would drop the signup that a
        # campaign is being judged on.
        "funnel": funnel(users, assignments, payments, device_counts),
        "product": product(win_assignments),
        "clients": clients(win_signins),
        "device_heat": devices_heatmap(devices_hourly),
        "device_total": device_total,
        "tz_offset": window.tz_offset,
        "window_label": window.label,
        "hourly": window.hourly,
    }
