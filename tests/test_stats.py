"""
Tests for the admin dashboard's aggregation layer.

stats.py is pure — rows in, numbers out — which is exactly why it's worth
testing: the dashboard is the one place where a silently wrong number looks
authoritative. These lock down the parts that are easy to get subtly wrong:
timezone shifting, "unknown vs zero" cost handling, and the style label that
already fooled one version of this code.
"""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from paperfill.data import costs
from paperfill.data import stats
from paperfill.data import usage
def ts(day: int, hour: int, minute: int = 0) -> str:
    """A UTC timestamp string shaped like Postgres returns."""
    return datetime(2026, 7, day, hour, minute, tzinfo=timezone.utc).isoformat()


def win(days: int = 7, tz=timezone.utc) -> stats.Window:
    """`days` whole days ending now, aligned to midnight the way the dashboard
    builds it. The section functions aggregate whatever rows they are handed —
    filtering happens once in build() — so a window here only decides the
    shape of the series."""
    now = datetime.now(timezone.utc)
    start = (now.replace(hour=0, minute=0, second=0, microsecond=0)
             - timedelta(days=days - 1))
    return stats.Window(start, now, tz, f"Last {days} days")


# America/New_York is UTC-5 in winter and UTC-4 in summer, which is the whole
# point of holding a zone rather than an offset.
ET = ZoneInfo("America/New_York")


# ---- Time of day ---------------------------------------------------------

def test_time_of_day_buckets_by_hour_and_weekday():
    # 2026-07-06 is a Monday.
    rows = [{"ts": ts(6, 14)}, {"ts": ts(6, 14)}, {"ts": ts(7, 9)}]
    out = stats.time_of_day(rows, [], tz_offset=0)
    assert out["grid"][0][14] == 2        # Monday 14:00
    assert out["grid"][1][9] == 1         # Tuesday 09:00
    assert out["by_hour"][14] == 2
    assert out["peak_hour_label"] == "14:00"
    assert out["peak_dow_label"] == "Mon"


def test_tz_offset_moves_events_across_the_day_boundary():
    """23:00 UTC is the previous evening in New York — and a different weekday.
    Getting this wrong would silently rotate every hour-of-day chart."""
    rows = [{"ts": ts(7, 2)}]             # Tuesday 02:00 UTC
    utc = stats.time_of_day(rows, [], tz_offset=0)
    est = stats.time_of_day(rows, [], tz_offset=-5)
    assert utc["grid"][1][2] == 1         # Tue 02:00
    assert est["grid"][0][21] == 1        # Mon 21:00


def test_fractional_offsets_work():
    """India is UTC+5:30 — a half-hour zone must not be truncated to +5."""
    rows = [{"ts": ts(6, 10, 45)}]        # 10:45 UTC -> 16:15 IST
    out = stats.time_of_day(rows, [], tz_offset=5.5)
    assert out["grid"][0][16] == 1


# ---- Money ---------------------------------------------------------------

def test_uncosted_calls_are_counted_not_treated_as_free():
    """A NULL cost means "no rate configured", which must never be averaged in
    as $0 — that would understate spend and overstate profit."""
    calls = [
        {"ts": ts(6, 1), "cost_usd": 0.25, "prompt_tokens": 10, "output_tokens": 5},
        {"ts": ts(6, 2), "cost_usd": None, "prompt_tokens": 99, "output_tokens": 99},
    ]
    out = stats.money(calls, [], win())
    assert out["cost"] == pytest.approx(0.25)
    assert out["uncosted_calls"] == 1
    assert out["tokens_total"] == 213      # tokens still count for both


def test_profit_and_margin():
    payments = [{"ts": ts(6, 3), "amount_cents": 500, "livemode": True},
                {"ts": ts(6, 4), "amount_cents": 500, "livemode": True}]
    calls = [{"ts": ts(6, 3), "cost_usd": 2.50, "prompt_tokens": 0, "output_tokens": 0}]
    out = stats.money(calls, payments, win())
    assert out["revenue"] == pytest.approx(10.0)
    assert out["profit"] == pytest.approx(7.5)
    assert out["margin_pct"] == pytest.approx(75.0)


def test_test_mode_payments_are_excluded_from_revenue():
    """Stripe test-mode checkouts hit the same webhook and look identical
    apart from `livemode` — they must not inflate the real P&L."""
    payments = [{"ts": ts(6, 3), "amount_cents": 500, "livemode": True},
                {"ts": ts(6, 4), "amount_cents": 99999, "livemode": False}]
    out = stats.money([], payments, win())
    assert out["revenue"] == pytest.approx(5.0)
    assert out["payment_count"] == 1
    assert out["arpu"] == pytest.approx(5.0)


def test_money_series_is_dense_and_ends_today():
    out = stats.money([], [], win(days=5))
    assert len(out["revenue_series"]) == 5
    today = datetime.now(timezone.utc).date()
    assert out["revenue_series"][-1]["label"] == today.strftime("%m-%d")


def test_short_windows_bucket_by_hour():
    """A six-hour window plotted as one daily point is a chart of nothing."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    w = stats.Window(now - timedelta(hours=5), now + timedelta(minutes=1))
    assert w.hourly
    out = stats.money([{"ts": now.isoformat(), "cost_usd": 1.0}], [], w)
    assert len(out["cost_series"]) == 6
    assert out["cost_series"][-1]["value"] == 1.0
    assert out["cost_series"][0]["label"] == (now - timedelta(hours=5)).strftime("%H:%M")


def test_window_filters_events_but_not_accounts():
    """The window slices event streams; account totals are a stock, not a flow,
    and windowing them would answer a question nobody asked."""
    now = datetime.now(timezone.utc)
    w = stats.Window(now - timedelta(days=2), now)
    recent = {"ts": (now - timedelta(hours=1)).isoformat(), "style": "Typed text"}
    old_fill = {"ts": (now - timedelta(days=40)).isoformat(), "style": "Typed text"}
    s = stats.build(signins=[], assignments=[recent, old_fill],
                    users=[{"created_at": (now - timedelta(days=90)).isoformat()}],
                    ai_calls=[], payments=[], devices_daily=[], devices_hourly=[],
                    device_total=7, window=w)
    assert s["product"]["total"] == 1
    assert s["accounts"]["total"] == 1


def test_empty_inputs_do_not_divide_by_zero():
    out = stats.money([], [], win())
    assert out["margin_pct"] == 0.0 and out["arpu"] == 0.0
    assert stats.accounts([], [], [])["pro_pct"] == 0.0
    assert stats.reliability([])["success_pct"] == 0.0


# ---- Windows -------------------------------------------------------------

def _et_window(first: datetime, last: datetime) -> stats.Window:
    """A Window over whole Eastern days, built the way _admin_window does."""
    return stats.Window(first.astimezone(timezone.utc),
                        last.astimezone(timezone.utc), ET)


def test_a_winter_window_buckets_at_the_winter_offset():
    """The window carries the zone, so it buckets in its own DST regime rather
    than whatever offset happened to be in force when the page was loaded. A
    December range shifted by August's -4 drew eleven points for a ten-day
    range, the last of them past the end of what was asked for."""
    w = _et_window(datetime(2026, 12, 1, tzinfo=ET), datetime(2026, 12, 11, tzinfo=ET))
    assert [p["label"] for p in w.series({})] == [f"12-{d:02d}" for d in range(1, 11)]
    # ET is UTC-5 in December: 04:59 UTC on the 11th is still the 10th locally.
    assert w.bucket(datetime(2026, 12, 11, 4, 59, tzinfo=timezone.utc)) == date(2026, 12, 10)
    assert w.bucket(datetime(2026, 12, 11, 5, 1, tzinfo=timezone.utc)) == date(2026, 12, 11)


def test_a_summer_window_buckets_at_the_summer_offset():
    """The other direction — a July range read in January must use UTC-4."""
    w = _et_window(datetime(2026, 7, 1, tzinfo=ET), datetime(2026, 7, 11, tzinfo=ET))
    assert [p["label"] for p in w.series({})] == [f"07-{d:02d}" for d in range(1, 11)]
    assert w.bucket(datetime(2026, 7, 11, 3, 59, tzinfo=timezone.utc)) == date(2026, 7, 10)
    assert w.bucket(datetime(2026, 7, 11, 4, 1, tzinfo=timezone.utc)) == date(2026, 7, 11)


def test_a_window_spanning_the_dst_change_draws_one_point_per_local_day():
    """ET falls back on 2026-11-01, making that local day 25 hours long. The
    axis still gets one point per calendar day, and a cost recorded that day
    lands on it."""
    w = _et_window(datetime(2026, 10, 30, tzinfo=ET), datetime(2026, 11, 4, tzinfo=ET))
    calls = [{"ts": "2026-11-01T18:00:00+00:00", "cost_usd": 2.0}]
    out = stats.money(calls, [], w)
    assert [p["label"] for p in out["cost_series"]] == [
        "10-30", "10-31", "11-01", "11-02", "11-03"]
    assert [p["value"] for p in out["cost_series"]] == [0, 0, 2.0, 0, 0]


def test_filter_includes_the_start_instant_and_excludes_the_end():
    """[start, end) — a row exactly at `start` is in, one exactly at `end`
    belongs to the next window. Anything else double-counts the boundary row
    when two windows are placed back to back."""
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 7, 8, tzinfo=timezone.utc)
    w = stats.Window(start, end)
    rows = [
        {"ts": (start - timedelta(microseconds=1)).isoformat(), "id": "before"},
        {"ts": start.isoformat(), "id": "at-start"},
        {"ts": (end - timedelta(microseconds=1)).isoformat(), "id": "before-end"},
        {"ts": end.isoformat(), "id": "at-end"},
    ]
    assert [r["id"] for r in w.filter(rows)] == ["at-start", "before-end"]


def test_devices_daily_lands_in_a_bucket_the_chart_actually_draws():
    """devices_daily is aggregated per UTC calendar day while the chart's
    buckets are local days. Matched on the bare UTC date, the newest row has
    no bucket at all between UTC midnight and midnight in ET, so it silently
    drops off the chart every night."""
    # 20:00 ET on the 10th is already 01:00 UTC on the 11th.
    w = _et_window(datetime(2026, 12, 5, tzinfo=ET),
                   datetime(2026, 12, 10, 20, tzinfo=ET))
    out = stats.activity([], [], [], [{"day": "2026-12-11", "devices": 9}], w)
    assert [p["label"] for p in out["devices"]][-1] == "12-10"
    assert out["devices"][-1]["value"] == 9


# ---- Product -------------------------------------------------------------

def test_typed_fills_are_not_counted_as_handwriting():
    """`style` is a human label, not a flag: typed fills are stored as the
    string "Typed text", so a truthiness check reports 100% handwriting."""
    rows = [{"style": "Typed text"}, {"style": ""}, {"style": None},
            {"style": "Your handwriting"}]
    out = stats.product(rows)
    assert out["handwriting"] == 1


def test_reports_are_counted_from_non_empty_feedback():
    rows = [{"feedback": "wrong answers"}, {"feedback": "   "}, {"feedback": None}]
    assert stats.product(rows)["reported"] == 1


# ---- Reliability ---------------------------------------------------------

def test_percentiles_and_fallback_rate():
    calls = [{"ok": True, "latency_ms": n * 100, "provider": "primary",
              "purpose": "text_fill"} for n in range(1, 11)]
    calls.append({"ok": False, "provider": "fallback", "purpose": "text_fill",
                  "error": "APIError: boom"})
    out = stats.reliability(calls)
    assert out["total"] == 11 and out["failed"] == 1
    assert out["success_pct"] == pytest.approx(90.9, abs=0.1)
    assert out["fallback_pct"] == pytest.approx(9.1, abs=0.1)
    assert out["p50_ms"] == 500 and out["p95_ms"] == 1000
    assert out["top_errors"][0][0] == "APIError"


# ---- Cost estimation -----------------------------------------------------

def test_free_primary_costs_nothing_but_unrated_fallback_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(costs, "_PATH", tmp_path / "rates.json")
    costs.save({"m/known": {"in": 1.0, "out": 3.0}}, primary_is_free=True)
    # 1M input + 1M output at $1/$3.
    assert costs.estimate("m/known", "fallback", 1_000_000, 1_000_000) == pytest.approx(4.0)
    assert costs.estimate("m/known", "primary", 1_000_000, 1_000_000) == 0.0
    # No rate -> None ("we don't know"), never 0.0 ("it was free").
    assert costs.estimate("m/unknown", "fallback", 1000, 1000) is None


def test_bad_rate_input_is_dropped_not_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(costs, "_PATH", tmp_path / "rates.json")
    saved = costs.save({"ok/m": {"in": "0.5", "out": "1"},
                        "bad/m": {"in": "abc", "out": "1"},
                        "neg/m": {"in": -5, "out": 1},
                        "": {"in": 1, "out": 1}}, primary_is_free=False)
    assert set(saved["models"]) == {"ok/m"}


# ---- Credit budget --------------------------------------------------------

def test_credits_consume_and_floor_at_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(usage, "_PATH", tmp_path / "usage.json")
    monkeypatch.setattr(usage, "FREE_DAILY_CREDITS", 2)
    monkeypatch.setattr(usage, "CREDIT_TOKENS", 1000)
    assert usage.remaining_credits("u1") == 2
    assert usage.consume_tokens("u1", 1000) == 1      # 1 credit spent
    assert usage.consume_tokens("u1", 1500) == 0       # over-spend floors at 0
    assert usage.consume_tokens("u1", 1000) == 0       # never negative
    assert usage.remaining_credits("u2") == 2          # per-user, not global


def test_credits_ignore_yesterdays_row(tmp_path, monkeypatch):
    import json
    p = tmp_path / "usage.json"
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    p.write_text(json.dumps({"u1": {"date": yesterday, "tokens": 99000}}))
    monkeypatch.setattr(usage, "_PATH", p)
    monkeypatch.setattr(usage, "FREE_DAILY_CREDITS", 20)
    monkeypatch.setattr(usage, "CREDIT_TOKENS", 1000)
    assert usage.tokens_used_today("u1") == 0
    assert usage.remaining_credits("u1") == 20


# ---- Acquisition funnel --------------------------------------------------
# The funnel is the one chart that mixes three tables, and the first version of
# it keyed fills on IP and reported more fillers than accounts. These pin the
# properties that made that version wrong.

def _users(*pairs):
    return [{"email": e, "source": s} for e, s in pairs]


def test_a_person_who_filled_ten_times_is_one_filler():
    f = stats.funnel(
        users=_users(("a@x.com", "ad")),
        assignments=[{"user_email": "a@x.com"} for _ in range(10)],
        payments=[], device_counts={},
    )
    assert f["stages"][2]["values"]["ad"] == 1


def test_repeat_charges_to_one_payer_are_one_paying_account():
    """`payments` holds a row per charge, so a renewing subscriber would
    otherwise inflate the bottom of the funnel every year."""
    f = stats.funnel(
        users=_users(("a@x.com", "organic")),
        assignments=[],
        payments=[{"email": "a@x.com"}, {"email": "a@x.com"}],
        device_counts={},
    )
    assert f["stages"][3]["total"] == 1


def test_the_account_stages_can_only_narrow():
    """Filling and paying are both subsets of signing up, because every stage
    below the first is keyed on the account. A funnel that widens is the bug
    this replaced."""
    f = stats.funnel(
        users=_users(("a@x.com", "ad"), ("b@x.com", "organic"), ("c@x.com", None)),
        assignments=[{"user_email": "a@x.com"}, {"user_email": "b@x.com"}],
        payments=[{"email": "a@x.com"}],
        device_counts={"ad": 10, "organic": 5, "unknown": 900},
    )
    totals = [s["total"] for s in f["stages"]]
    assert totals[1] >= totals[2] >= totals[3]
    assert totals == [915, 3, 2, 1]


def test_an_unattributed_row_is_unknown_and_never_organic():
    """Every row written before attribution shipped has a NULL source. Folding
    those into organic would silently credit organic with the whole back
    catalogue and make any ad comparison meaningless."""
    f = stats.funnel(users=_users(("a@x.com", None), ("b@x.com", "")),
                     assignments=[], payments=[], device_counts={})
    assert f["stages"][1]["values"] == {"ad": 0, "organic": 0, "unknown": 2}


def test_a_fill_attributes_to_its_accounts_source_not_its_own():
    f = stats.funnel(
        users=_users(("a@x.com", "ad")),
        assignments=[{"user_email": "A@X.com"}],   # casing varies at the boundary
        payments=[], device_counts={},
    )
    assert f["stages"][2]["values"]["ad"] == 1


def test_fills_with_no_account_are_dropped_not_counted_unknown():
    """Rows from before user_email existed carry NULL. They're real fills but
    unattributable to a person, and counting them would re-create the
    over-count the IP version had."""
    f = stats.funnel(users=_users(("a@x.com", "ad")),
                     assignments=[{"user_email": None}, {"user_email": ""}],
                     payments=[], device_counts={})
    assert f["stages"][2]["total"] == 0


def test_devices_come_from_the_counts_not_the_account_rows():
    """Devices are counted before anyone identifies themselves, so that stage
    is a separate population handed in from Postgres rather than derived."""
    f = stats.funnel(users=[], assignments=[], payments=[],
                     device_counts={"ad": 7, "organic": 3, "unknown": 1})
    assert f["stages"][0]["values"] == {"ad": 7, "organic": 3, "unknown": 1}
    assert f["stages"][0]["total"] == 11


def test_missing_device_counts_do_not_explode():
    f = stats.funnel(users=[], assignments=[], payments=[], device_counts=None)
    assert f["stages"][0]["total"] == 0
