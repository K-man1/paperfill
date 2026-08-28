"""
Tests for the one invariant the money depends on: an account is Pro because it
paid or because an admin said so, and for no other reason.

test_webhook.py and test_whop_webhook.py cover each processor's endpoint in
isolation — signatures, dedupe, payload shapes. This file is the other half:
that a verified payment actually lands as Pro on the *live session* (the cache
in front of the DB is what breaks this in practice), and that nothing outside
the two webhooks and the admin console can flip the flag.

`db` is faked with an in-memory dict throughout, so nothing here can touch
Supabase.
"""

import hashlib
import hmac
import json
import time
from base64 import b64encode

import pytest

from paperfill import app as A


STRIPE_SECRET = "whsec_test_not_a_real_secret"
WHOP_SECRET = "ws_" + "e1" * 32

BUYER = "buyer@example.com"
ADMIN = sorted(A.ADMIN_EMAILS)[0]


@pytest.fixture
def users(monkeypatch):
    """Stand in for the users table. Returns the dict so a test can seed rows
    and read back what a webhook did to them."""
    rows: dict[str, dict] = {}

    def set_user_pro(email, is_pro):
        row = rows.get((email or "").strip().lower())
        if row is None:
            return False
        row["is_pro"] = bool(is_pro)
        return True

    monkeypatch.setattr(A.db, "enabled", lambda: True)
    monkeypatch.setattr(A.db, "get_user_by_email",
                        lambda email: rows.get((email or "").strip().lower()))
    monkeypatch.setattr(A.db, "set_user_pro", set_user_pro)
    monkeypatch.setattr(A.db, "record_payment", lambda **kw: None)
    monkeypatch.setattr(A.db, "set_stripe_ids", lambda *a, **kw: True)
    A._pro_cache.clear()
    return rows


@pytest.fixture
def client(monkeypatch):
    A.app.config["TESTING"] = True
    monkeypatch.setattr(A, "STRIPE_WEBHOOK_SECRET", STRIPE_SECRET)
    monkeypatch.setattr(A, "WHOP_WEBHOOK_SECRET", WHOP_SECRET)
    return A.app.test_client()


def sign_in(client, email, role="user", is_pro=False):
    with client.session_transaction() as s:
        s["role"] = role
        s["user_email"] = email
        s["user_sub"] = f"email:{email}"
        s["is_pro"] = is_pro


def is_pro_now(client) -> bool:
    """Ask the app itself, through a Pro-gated endpoint, rather than reading
    the flag we're trying to test."""
    return client.get("/api/fonts").status_code == 200


def pay_stripe(client, email=BUYER, amount=1000):
    body = json.dumps({
        "id": f"evt_{email}", "type": "checkout.session.completed",
        "livemode": True,
        "data": {"object": {"amount_total": amount, "currency": "usd",
                            "customer_details": {"email": email}}},
    }).encode()
    ts = int(time.time())
    mac = hmac.new(STRIPE_SECRET.encode(), f"{ts}".encode() + b"." + body,
                   hashlib.sha256).hexdigest()
    return client.post("/stripe/webhook", data=body,
                       headers={"Stripe-Signature": f"t={ts},v1={mac}",
                                "Content-Type": "application/json"})


def whop_event(client, action, email=BUYER, **data):
    body = json.dumps({"action": action,
                       "data": {"user_email": email, **data}}).encode()
    ts = int(time.time())
    digest = hmac.new(WHOP_SECRET.encode(), f"msg_1.{ts}.".encode() + body,
                      hashlib.sha256).digest()
    return client.post("/whop/webhook", data=body, headers={
        "webhook-id": "msg_1", "webhook-timestamp": str(ts),
        "webhook-signature": "v1," + b64encode(digest).decode(),
        "Content-Type": "application/json"})


# ---- Paying gets you Pro -------------------------------------------------

def test_whop_purchase_makes_a_live_free_session_pro(client, users):
    """The one that matters. _is_pro memoises the tier for a minute, so a sale
    that lands mid-session is only real if the webhook drops that entry —
    otherwise the buyer pays and stays Free for up to a minute."""
    users[BUYER] = {"email": BUYER, "is_pro": False}
    sign_in(client, BUYER)
    assert not is_pro_now(client)          # primes the cache as Free

    assert whop_event(client, "membership.activated").status_code == 200
    assert users[BUYER]["is_pro"] is True
    assert is_pro_now(client)


def test_stripe_purchase_makes_a_live_free_session_pro(client, users):
    users[BUYER] = {"email": BUYER, "is_pro": False}
    sign_in(client, BUYER)
    assert not is_pro_now(client)

    assert pay_stripe(client).status_code == 200
    assert users[BUYER]["is_pro"] is True
    assert is_pro_now(client)


def test_pro_survives_the_next_request(client, users):
    """Paying isn't a one-request fluke: the tier is read off the row, so it
    holds across requests without another webhook."""
    users[BUYER] = {"email": BUYER, "is_pro": False}
    sign_in(client, BUYER)
    whop_event(client, "membership.activated")
    assert is_pro_now(client) and is_pro_now(client)


def test_a_purchase_under_a_different_email_does_not_upgrade_you(client, users):
    """Whop matches the checkout email to an account. Someone else's sale must
    not land on this session."""
    users[BUYER] = {"email": BUYER, "is_pro": False}
    users["other@example.com"] = {"email": "other@example.com", "is_pro": False}
    sign_in(client, BUYER)
    whop_event(client, "membership.activated", email="other@example.com")
    assert users[BUYER]["is_pro"] is False
    assert not is_pro_now(client)


def test_membership_ending_takes_pro_away_from_a_live_session(client, users):
    """Same cache, other direction: a refund or lapsed renewal has to land
    immediately, not a minute later."""
    users[BUYER] = {"email": BUYER, "is_pro": True}
    sign_in(client, BUYER, is_pro=True)
    assert is_pro_now(client)

    assert whop_event(client, "membership.deactivated").status_code == 200
    assert users[BUYER]["is_pro"] is False
    assert not is_pro_now(client)


# ---- An admin grant gets you Pro ----------------------------------------

def test_admin_can_grant_and_revoke_pro(client, users):
    users[BUYER] = {"email": BUYER, "is_pro": False}
    sign_in(client, ADMIN, role="admin")

    client.post("/admin/grant-pro", data={"email": BUYER, "action": "grant"})
    assert users[BUYER]["is_pro"] is True

    client.post("/admin/grant-pro", data={"email": BUYER, "action": "revoke"})
    assert users[BUYER]["is_pro"] is False


def test_a_granted_account_is_pro_on_its_next_request(client, users):
    users[BUYER] = {"email": BUYER, "is_pro": False}
    sign_in(client, BUYER)
    assert not is_pro_now(client)

    admin = A.app.test_client()
    sign_in(admin, ADMIN, role="admin")
    admin.post("/admin/grant-pro", data={"email": BUYER, "action": "grant"})
    assert is_pro_now(client)


# ---- Nothing else does ---------------------------------------------------

def test_a_free_account_is_locked_out_of_pro_features(client, users):
    users[BUYER] = {"email": BUYER, "is_pro": False}
    sign_in(client, BUYER)
    assert client.get("/api/fonts").status_code == 402
    assert client.post("/api/style").status_code == 402
    assert client.get("/handwriting").status_code == 302
    assert client.get("/api/fonts/template").status_code == 302


def test_a_signed_out_visitor_is_locked_out(client, users):
    assert client.get("/api/fonts").status_code == 403
    assert client.post("/api/fonts").status_code == 403


def test_a_free_user_cannot_grant_themselves_pro_from_the_admin_form(client, users):
    """The grant form is the only manual path in. It has to be admin-only —
    a normal session posting to it gets bounced to /login, unchanged."""
    users[BUYER] = {"email": BUYER, "is_pro": False}
    sign_in(client, BUYER)
    r = client.post("/admin/grant-pro", data={"email": BUYER, "action": "grant"})
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert users[BUYER]["is_pro"] is False
    assert not is_pro_now(client)


def test_a_free_user_cannot_flip_the_auto_pro_toggle(client, users, monkeypatch):
    flipped = []
    monkeypatch.setattr(A, "set_auto_pro", lambda on: flipped.append(on))
    sign_in(client, BUYER)
    r = client.post("/admin/auto-pro", data={"auto_pro": "on"})
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert flipped == []


def test_visiting_the_success_page_does_not_grant_pro(client, users):
    """/upgrade/success is just where the processor redirects after checkout —
    anyone can navigate to it directly, so it must only re-read the row."""
    users[BUYER] = {"email": BUYER, "is_pro": False}
    sign_in(client, BUYER)
    assert client.get("/upgrade/success").status_code == 200
    assert users[BUYER]["is_pro"] is False
    assert not is_pro_now(client)


def test_an_unsigned_webhook_grants_nothing_on_either_processor(client, users):
    users[BUYER] = {"email": BUYER, "is_pro": False}
    body = json.dumps({"action": "membership.activated",
                       "data": {"user_email": BUYER}}).encode()
    assert client.post("/whop/webhook", data=body).status_code == 400
    assert client.post("/stripe/webhook", data=body).status_code == 400
    assert users[BUYER]["is_pro"] is False


def test_a_stale_pro_session_cookie_does_not_outlive_the_revoke(client, users):
    """session["is_pro"] is written at login and would otherwise keep a revoked
    account on Pro until it signed out. The row is the authority."""
    users[BUYER] = {"email": BUYER, "is_pro": False}
    sign_in(client, BUYER, is_pro=True)   # cookie says Pro, row says Free
    assert not is_pro_now(client)


# ---- New signups ---------------------------------------------------------

def test_new_signups_are_free_unless_auto_pro_was_switched_on(client, monkeypatch, tmp_path):
    """auto_pro.txt is gitignored, so it's absent on a fresh deploy. Absent has
    to read as off: defaulting to on hands Pro to every signup for free."""
    monkeypatch.setattr(A, "AUTO_PRO_PATH", tmp_path / "auto_pro.txt")
    assert A.get_auto_pro() is False

    (tmp_path / "auto_pro.txt").write_text("1")
    assert A.get_auto_pro() is True


def test_signup_creates_the_account_on_the_tier_auto_pro_says(client, monkeypatch):
    created = {}
    monkeypatch.setattr(A, "EMAIL_AUTH_ENABLED", True)
    monkeypatch.setattr(A, "get_auto_pro", lambda: False)
    monkeypatch.setattr(A, "send_verification_email", lambda *a: None)
    monkeypatch.setattr(A.db, "get_user_by_email", lambda email: None)
    monkeypatch.setattr(A.db, "device_source", lambda cookie: None, raising=False)
    monkeypatch.setattr(A.db, "create_email_user",
                        lambda **kw: created.update(kw) or {"email": kw["email"]})

    client.post("/signup", data={"email": "new@example.com",
                                 "password": "hunter2hunter2", "name": "New"})
    assert created["is_pro"] is False
