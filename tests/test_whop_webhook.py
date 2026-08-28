"""
Tests for the Whop webhook — the path that actually turns a sale into Pro.

Same reasoning as test_webhook.py: this can't be exercised by hand until a real
purchase happens, and by then a bug means silently losing sales or handing out
Pro for free. So it's driven by synthetic, correctly-signed deliveries instead,
with `db` patched throughout so a test can never write a fake payment into the
real database.

Whop signs on a different scheme to Stripe: HMAC-SHA256 over
"{webhook-id}.{webhook-timestamp}.{body}", base64, sent as
`webhook-signature: v1,<sig>`.
"""

import base64
import hashlib
import hmac
import json
import time

import pytest

from paperfill import app as A


# Shaped like a real Whop secret: `ws_` plus 64 lowercase hex characters, so
# both key derivations in _whop_sig_keys are exercisable from the same fixture.
SECRET = "ws_" + "e1" * 32
WEBHOOK_ID = "msg_test_1"


def sign(body: bytes, key: bytes, webhook_id: str = WEBHOOK_ID,
         ts: int | None = None) -> dict[str, str]:
    ts = int(time.time()) if ts is None else ts
    signed = f"{webhook_id}.{ts}.".encode() + body
    digest = hmac.new(key, signed, hashlib.sha256).digest()
    return {
        "webhook-id": webhook_id,
        "webhook-timestamp": str(ts),
        "webhook-signature": "v1," + base64.b64encode(digest).decode(),
        "Content-Type": "application/json",
    }


def event(action="membership.activated", email="buyer@example.com", **data):
    return {"api_version": "v1", "account_id": "biz_test",
            "action": action, "data": {"user_email": email, **data}}


@pytest.fixture
def client(monkeypatch):
    A.app.config["TESTING"] = True
    monkeypatch.setattr(A, "WHOP_WEBHOOK_SECRET", SECRET)
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


def post(client, ev, key: bytes = None, headers=None):
    body = json.dumps(ev).encode()
    key = SECRET.encode() if key is None else key
    return client.post("/whop/webhook", data=body,
                       headers=headers or sign(body, key))


# ---- The happy path ------------------------------------------------------

def test_membership_activated_grants_pro(client, recorded):
    r = post(client, event(email="Buyer@Example.com"))
    assert r.status_code == 200
    assert recorded["pro"] == [("buyer@example.com", True)]  # lowercased to match


def test_membership_deactivated_revokes_pro(client, recorded):
    r = post(client, event(action="membership.deactivated"))
    assert r.status_code == 200
    assert recorded["pro"] == [("buyer@example.com", False)]


def test_legacy_went_valid_spelling_is_still_honoured(client, recorded):
    """A webhook created against the older generation sends went_valid rather
    than activated. Both must grant Pro."""
    post(client, event(action="membership.went_valid"))
    assert recorded["pro"] == [("buyer@example.com", True)]


def test_payment_succeeded_books_revenue(client, recorded):
    r = post(client, event(action="payment.succeeded", id="pay_1",
                           amount=10, currency="usd"))
    assert r.status_code == 200
    assert len(recorded["payments"]) == 1
    p = recorded["payments"][0]
    assert p["amount_cents"] == 1000        # Whop sends major units, we store cents
    assert p["currency"] == "usd"
    assert p["email"] == "buyer@example.com"
    assert p["event_id"] == "pay_1"


def test_revenue_is_booked_even_when_the_email_matches_no_account(client, monkeypatch, recorded):
    """Same stance as the Stripe path: the money arrived whether or not we can
    tie it to a user, and a P&L that drops unmatched sales is worse than
    useless."""
    monkeypatch.setattr(A.db, "set_user_pro", lambda email, flag: False)
    r = post(client, event(action="payment.succeeded", email="stranger@example.com",
                           id="pay_2", amount=10))
    assert r.status_code == 200
    assert len(recorded["payments"]) == 1


def test_a_sale_is_booked_once_even_though_it_fires_two_events(client, recorded):
    """Whop sends payment.succeeded *and* membership.activated for one
    purchase. Money must come off the payment only, access off the membership
    only, or every sale double-counts."""
    post(client, event(action="payment.succeeded", id="pay_3", amount=10))
    post(client, event(action="membership.activated"))
    assert len(recorded["payments"]) == 1
    assert recorded["pro"] == [("buyer@example.com", True)]


def test_payment_id_is_the_dedupe_key_so_a_retry_cannot_double_book(client, recorded):
    """record_payment leans on a UNIQUE event id to swallow redeliveries, so the
    id has to be stable across retries of the same payment."""
    ev = event(action="payment.succeeded", id="pay_4", amount=10)
    post(client, ev)
    post(client, ev)
    assert {p["event_id"] for p in recorded["payments"]} == {"pay_4"}


def test_delivery_id_stands_in_when_the_payload_carries_no_id(client, recorded):
    post(client, event(action="payment.succeeded", amount=10))
    assert recorded["payments"][0]["event_id"] == WEBHOOK_ID


# ---- Key derivation ------------------------------------------------------
# Whop's docs disagree on whether the HMAC key is the raw `ws_...` string or the
# hex tail decoded to bytes, so _whop_sig_keys accepts either. Both are
# pinned here: when a real delivery settles which one Whop actually uses, delete
# the loser from both the helper and this file.

def test_raw_secret_key_derivation_is_accepted(client, recorded):
    post(client, event(), key=SECRET.encode())
    assert recorded["pro"] == [("buyer@example.com", True)]


def test_hex_secret_key_derivation_is_accepted(client, recorded):
    post(client, event(), key=bytes.fromhex(SECRET.partition("_")[2]))
    assert recorded["pro"] == [("buyer@example.com", True)]


# ---- Rejections ----------------------------------------------------------

def test_bad_signature_is_rejected_and_changes_nothing(client, recorded):
    r = post(client, event(), key=b"wrong-key-entirely")
    assert r.status_code == 400
    assert recorded["pro"] == [] and recorded["payments"] == []


def test_missing_secret_rejects_everything(client, monkeypatch, recorded):
    """A misconfigured deployment must fail closed: without a secret, an
    unauthenticated POST could otherwise mint Pro accounts for free."""
    monkeypatch.setattr(A, "WHOP_WEBHOOK_SECRET", "")
    assert post(client, event()).status_code == 400
    assert recorded["pro"] == []


def test_a_replayed_delivery_is_rejected_once_it_goes_stale(client, recorded):
    body = json.dumps(event()).encode()
    old = sign(body, SECRET.encode(), ts=int(time.time()) - 3600)
    r = client.post("/whop/webhook", data=body, headers=old)
    assert r.status_code == 400
    assert recorded["pro"] == []


def test_signature_over_a_different_body_is_rejected(client, recorded):
    """The body is inside the signed string, so swapping it after signing must
    not verify — otherwise anyone could replay one real delivery's headers with
    a payload naming their own email."""
    headers = sign(json.dumps(event()).encode(), SECRET.encode())
    r = client.post("/whop/webhook",
                    data=json.dumps(event(email="attacker@example.com")).encode(),
                    headers=headers)
    assert r.status_code == 400
    assert recorded["pro"] == []


def test_missing_signature_headers_are_rejected(client, recorded):
    r = client.post("/whop/webhook", data=json.dumps(event()).encode(),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert recorded["pro"] == []


def test_an_unmatched_email_never_grants_pro(client, recorded):
    """No email in the payload must mean no grant, rather than a call to
    set_user_pro with an empty key."""
    ev = {"api_version": "v1", "action": "membership.activated", "data": {}}
    r = post(client, ev)
    assert r.status_code == 200
    assert recorded["pro"] == []
