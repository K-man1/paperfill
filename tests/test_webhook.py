"""
Tests for the Stripe webhook — the thing that actually records revenue.

This path can't be exercised by hand until Stripe is live, and by then a bug in
it means silently losing sales from the P&L. So it's tested with synthetic,
correctly-signed events instead. `db` is patched throughout: these tests must
never write a fake payment into the real database.
"""

import hashlib
import hmac
import json
import time

import pytest

from paperfill import app as A


SECRET = "whsec_test_not_a_real_secret"


def sign(payload: bytes, secret: str = SECRET, ts: int | None = None) -> str:
    """Build a Stripe-Signature header the way Stripe does: HMAC-SHA256 over
    "{timestamp}.{body}", keyed by the endpoint secret."""
    ts = int(time.time()) if ts is None else ts
    mac = hmac.new(secret.encode(), f"{ts}".encode() + b"." + payload,
                   hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def event(amount=500, currency="usd", email="buyer@example.com",
          eid="evt_test_1", etype="checkout.session.completed", livemode=False):
    return {
        "id": eid,
        "type": etype,
        "livemode": livemode,
        "data": {"object": {
            "amount_total": amount,
            "currency": currency,
            "customer_details": {"email": email},
        }},
    }


@pytest.fixture
def client(monkeypatch):
    A.app.config["TESTING"] = True
    monkeypatch.setattr(A, "STRIPE_WEBHOOK_SECRET", SECRET)
    return A.app.test_client()


@pytest.fixture
def recorded(monkeypatch):
    """Capture record_payment / set_user_pro instead of hitting Supabase."""
    calls = {"payments": [], "pro": []}
    monkeypatch.setattr(A.db, "record_payment",
                        lambda **kw: calls["payments"].append(kw))
    monkeypatch.setattr(A.db, "set_user_pro",
                        lambda email, flag: calls["pro"].append((email, flag)) or True)
    return calls


def post(client, ev, sig=None):
    body = json.dumps(ev).encode()
    return client.post("/stripe/webhook", data=body,
                       headers={"Stripe-Signature": sig or sign(body),
                                "Content-Type": "application/json"})


# ---- The happy path ------------------------------------------------------

def test_completed_checkout_books_revenue_and_grants_pro(client, recorded):
    r = post(client, event(amount=500, email="Buyer@Example.com"))
    assert r.status_code == 200
    assert len(recorded["payments"]) == 1
    p = recorded["payments"][0]
    assert p["amount_cents"] == 500
    assert p["currency"] == "usd"
    assert p["email"] == "buyer@example.com"     # lowercased for matching
    assert p["event_id"] == "evt_test_1"
    assert recorded["pro"] == [("buyer@example.com", True)]


def test_revenue_is_booked_even_when_the_email_matches_no_account(client, monkeypatch, recorded):
    """The money arrived whether or not we can tie it to a user. A P&L that
    drops unmatched sales is worse than useless."""
    monkeypatch.setattr(A.db, "set_user_pro", lambda email, flag: False)
    r = post(client, event(email="stranger@example.com"))
    assert r.status_code == 200
    assert len(recorded["payments"]) == 1


def test_amount_and_currency_are_taken_from_the_event(client, recorded):
    """Note the two normalisations happen at different layers: the webhook
    lowercases the *email* (it's a matching key for set_user_pro), while the
    currency is passed through and normalised by db.record_payment at the
    storage boundary, where every caller benefits from it."""
    post(client, event(amount=1299, currency="GBP"))
    p = recorded["payments"][0]
    assert p["amount_cents"] == 1299
    assert p["currency"] == "GBP"          # raw at this layer


def test_db_layer_normalises_currency(monkeypatch):
    """The lowercasing the webhook doesn't do, record_payment does."""
    sent = {}
    monkeypatch.setattr(A.db, "enabled", lambda: True)
    monkeypatch.setattr(A.db.requests, "post",
                        lambda url, **kw: sent.update(kw.get("json") or {}))
    A.db.record_payment(email="a@b.com", amount_cents=500, currency="GBP",
                        event_id="evt_x", livemode=False)
    assert sent["currency"] == "gbp"
    assert sent["amount_cents"] == 500


# ---- Rejections ----------------------------------------------------------

def test_bad_signature_is_rejected_and_books_nothing(client, recorded):
    body = json.dumps(event()).encode()
    r = client.post("/stripe/webhook", data=body,
                    headers={"Stripe-Signature": sign(body, secret="wrong")})
    assert r.status_code == 400
    assert recorded["payments"] == []


def test_missing_secret_rejects_everything(client, monkeypatch, recorded):
    """A misconfigured deployment must fail closed: without a secret, an
    unauthenticated POST could otherwise mint Pro accounts and fake revenue."""
    monkeypatch.setattr(A, "STRIPE_WEBHOOK_SECRET", "")
    assert post(client, event()).status_code == 400
    assert recorded["payments"] == []


def test_stale_timestamp_is_rejected(client, recorded):
    body = json.dumps(event()).encode()
    old = int(time.time()) - 3600
    r = client.post("/stripe/webhook", data=body,
                    headers={"Stripe-Signature": sign(body, ts=old)})
    assert r.status_code == 400
    assert recorded["payments"] == []


def test_other_event_types_are_acknowledged_but_book_nothing(client, recorded):
    r = post(client, event(etype="payment_intent.created"))
    assert r.status_code == 200        # 200 or Stripe retries forever
    assert recorded["payments"] == []


def test_unparseable_body_is_rejected(client, recorded):
    body = b"{not json"
    r = client.post("/stripe/webhook", data=body,
                    headers={"Stripe-Signature": sign(body)})
    assert r.status_code == 400
    assert recorded["payments"] == []


# ---- Idempotency ---------------------------------------------------------

def test_replayed_event_carries_the_same_id_for_dedupe(client, recorded):
    """Stripe retries on any non-2xx, so the same event can arrive twice. The
    handler forwards a stable event id; `payments.stripe_event_id` is UNIQUE
    with ignore-duplicates, which is what stops double-counted revenue."""
    ev = event(eid="evt_repeat")
    post(client, ev)
    post(client, ev)
    assert len(recorded["payments"]) == 2
    assert {p["event_id"] for p in recorded["payments"]} == {"evt_repeat"}
