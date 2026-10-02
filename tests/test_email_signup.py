"""
Tests for the gate on /signup: who is allowed to open an email account.

The rules that matter here are the ones a burner would walk straight through
otherwise — a disposable domain, a wildcard subdomain of one — plus the
ordering that keeps a failed verification email from stranding a row nobody can
sign in to or sign up over.

DNS is faked in every test, so nothing here does a live lookup.
"""

import pytest

from paperfill import app as A
from paperfill.utils import email_domains


@pytest.fixture(autouse=True)
def fresh_rate_limits():
    """The limiter is module state, so one test's attempts would otherwise
    count against the next one's quota."""
    A._RATE_BUCKETS.clear()


@pytest.fixture
def deliverable(monkeypatch):
    """Every domain accepts mail, so tests exercise the blocklist alone."""
    monkeypatch.setattr(email_domains, "_accepts_mail", lambda domain: True)


@pytest.fixture
def signup(monkeypatch, deliverable):
    """A test client with an empty users table. Returns (client, rows, sent)."""
    rows: dict[str, dict] = {}
    sent: list[tuple[str, str]] = []

    def create_email_user(email, password_hash, name, token, token_expires,
                          is_pro=False, source=None):
        rows[email] = {"email": email, "name": name, "is_pro": is_pro}
        return rows[email]

    monkeypatch.setattr(A.db, "enabled", lambda: True)
    monkeypatch.setattr(A.db, "get_user_by_email", lambda email: rows.get(email))
    monkeypatch.setattr(A.db, "create_email_user", create_email_user)
    monkeypatch.setattr(A.db, "refresh_verification_token",
                        lambda email, token, expires: email in rows)
    monkeypatch.setattr(A, "EMAIL_AUTH_ENABLED", True)
    monkeypatch.setattr(A, "send_verification_email",
                        lambda to, url: bool(sent.append((to, url))) or True)
    A.app.config["TESTING"] = True
    return A.app.test_client(), rows, sent


def post(client, email, password="hunter2hunter2"):
    return client.post("/signup", data={"email": email, "password": password,
                                        "name": "Test"})


@pytest.mark.parametrize("domain", ["mailinator.com", "yopmail.com",
                                    "10minutemail.com", "guerrillamail.com"])
def test_known_throwaway_domains_are_rejected(deliverable, domain):
    assert email_domains.rejection_reason(f"burner@{domain}") is not None


def test_wildcard_subdomain_of_a_throwaway_is_rejected(deliverable):
    # The list carries mailinator.com; the service hands out anything under it.
    assert email_domains.is_disposable("team17.mailinator.com")


def test_real_providers_are_accepted(deliverable):
    for address in ("a@gmail.com", "b@outlook.com",
                    "c@proton.me", "d@hackclub.com"):
        assert email_domains.rejection_reason(address) is None


def test_malformed_addresses_are_rejected(deliverable):
    for address in ("", "nope", "a@b", "a b@example.com", "a@@example.com"):
        assert email_domains.rejection_reason(address) is not None


def test_undeliverable_domain_is_rejected(monkeypatch):
    monkeypatch.setattr(email_domains, "_accepts_mail", lambda domain: False)
    assert "gmail.con" in email_domains.rejection_reason("a@gmail.con")


def test_dns_failure_lets_the_signup_through(monkeypatch):
    """A resolver outage must not become a signup outage."""
    import dns.exception

    def blow_up(*a, **kw):
        raise dns.exception.Timeout()

    monkeypatch.setattr(email_domains.dns.resolver, "Resolver", blow_up)
    assert email_domains._accepts_mail("gmail.com") is True


def test_signup_refuses_a_disposable_address(signup):
    client, rows, sent = signup
    r = post(client, "burner@mailinator.com")
    assert b"disposable" in r.data.lower()
    assert rows == {} and sent == []


def test_signup_mails_a_verification_link(signup):
    client, rows, sent = signup
    r = post(client, "student@gmail.com")
    assert b"Check your email" in r.data
    assert "student@gmail.com" in rows
    assert len(sent) == 1 and "/verify/" in sent[0][1]


def test_no_account_is_created_when_the_mail_fails(signup, monkeypatch):
    """Otherwise the address is taken by a row that can never be verified."""
    client, rows, sent = signup
    monkeypatch.setattr(A, "send_verification_email", lambda to, url: False)
    r = post(client, "student@gmail.com")
    assert b"send the verification email" in r.data
    assert rows == {}


def test_short_passwords_are_rejected(signup):
    client, rows, _ = signup
    r = post(client, "student@gmail.com", password="short")
    assert b"at least 8 characters" in r.data
    assert rows == {}


# ---- Resend ---------------------------------------------------------------

def unverified(rows, email="student@gmail.com"):
    rows[email] = {"email": email, "email_verified": False,
                   "password_hash": "pbkdf2:fake"}
    return email


def test_resend_mails_a_fresh_link(signup):
    client, rows, sent = signup
    email = unverified(rows)
    r = client.post("/resend", data={"email": email})
    assert b"on its way" in r.data
    assert len(sent) == 1 and "/verify/" in sent[0][1]


def test_resend_says_the_same_thing_for_an_unknown_address(signup):
    """The reply must not reveal who has an account."""
    client, rows, sent = signup
    known = client.post("/resend", data={"email": unverified(rows)}).data
    A._RATE_BUCKETS.clear()
    unknown = client.post("/resend", data={"email": "nobody@gmail.com"}).data
    assert b"on its way" in unknown
    assert known == unknown
    assert len(sent) == 1          # nothing was mailed for the unknown address


def test_resend_ignores_an_already_verified_account(signup):
    client, rows, sent = signup
    rows["done@gmail.com"] = {"email": "done@gmail.com", "email_verified": True,
                              "password_hash": "pbkdf2:fake"}
    client.post("/resend", data={"email": "done@gmail.com"})
    assert sent == []


def test_resend_ignores_a_google_account(signup):
    """No password to come back to, so a verification link buys them nothing."""
    client, rows, sent = signup
    rows["g@gmail.com"] = {"email": "g@gmail.com", "email_verified": False,
                           "google_sub": "12345"}
    client.post("/resend", data={"email": "g@gmail.com"})
    assert sent == []


# ---- Rate limits ----------------------------------------------------------

def test_an_address_cannot_be_mail_bombed_through_resend(signup):
    client, rows, sent = signup
    email = unverified(rows)
    for _ in range(A.VERIFY_MAIL_LIMIT + 3):
        r = client.post("/resend", data={"email": email})
        assert b"on its way" in r.data     # refusal stays indistinguishable
    assert len(sent) == A.VERIFY_MAIL_LIMIT


def test_signup_stops_after_enough_accounts_from_one_network(signup):
    client, rows, sent = signup
    for i in range(A.SIGNUP_LIMIT):
        assert b"Check your email" in post(client, f"student{i}@gmail.com").data
    r = post(client, "onemore@gmail.com")
    assert b"Too many accounts" in r.data
    assert len(rows) == A.SIGNUP_LIMIT


def test_the_limiter_refills_as_the_window_slides(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(A.time, "monotonic", lambda: clock[0])

    assert not A._rate_limited("t", "key", 2, 60)
    assert not A._rate_limited("t", "key", 2, 60)
    assert A._rate_limited("t", "key", 2, 60)

    # Hammering while blocked must not push the window forward, or an attacker
    # could hold themselves out indefinitely and so could anyone spoofing them.
    clock[0] += 30
    for _ in range(50):
        assert A._rate_limited("t", "key", 2, 60)
    clock[0] += 31                    # first two attempts are now 61s old
    assert not A._rate_limited("t", "key", 2, 60)


def test_unverified_signin_offers_the_resend_form(signup):
    """The dead end this replaces: "verify your email first" with no way to get
    another link if the first one never arrived."""
    client, rows, _ = signup
    email = unverified(rows)
    rows[email]["password_hash"] = A.generate_password_hash("hunter2hunter2")
    r = client.post("/login", data={"email": email,
                                    "password": "hunter2hunter2"})
    assert b"send yourself a new one" in r.data
    assert f'value="{email}"'.encode() in r.data   # prefilled, not retyped


# ---- Sign-in throttle -----------------------------------------------------

def verified(rows, email="student@gmail.com", password="hunter2hunter2"):
    rows[email] = {"email": email, "email_verified": True,
                   "password_hash": A.generate_password_hash(password)}
    return email


def signin(client, email, password="hunter2hunter2"):
    return client.post("/login", data={"email": email, "password": password})


def test_sign_in_stops_after_enough_wrong_passwords(signup):
    client, rows, _ = signup
    email = verified(rows)
    for _ in range(A.LOGIN_LIMIT):
        assert b"Incorrect email or password" in signin(client, email, "wrong").data
    blocked = signin(client, email, "wrong")
    assert b"Too many failed sign-in attempts" in blocked.data

    # The throttle outranks the password: a brute forcer who lands on the right
    # one during the block still doesn't get in.
    assert b"Too many failed sign-in attempts" in signin(client, email).data


def test_the_throttle_does_not_reveal_which_addresses_exist(signup):
    """The block message is keyed to the address, not the account, so hammering
    an unknown address has to look exactly like hammering a real one."""
    client, rows, _ = signup
    verified(rows, "real@gmail.com")
    for _ in range(A.LOGIN_LIMIT):
        signin(client, "real@gmail.com", "wrong")
        signin(client, "ghost@gmail.com", "wrong")
    assert (signin(client, "real@gmail.com", "wrong").data
            == signin(client, "ghost@gmail.com", "wrong").data)


def test_successful_sign_ins_do_not_count_against_the_limit(signup):
    """Otherwise signing in often enough would lock you out of your own account."""
    client, rows, _ = signup
    email = verified(rows)
    for _ in range(A.LOGIN_LIMIT + 5):
        assert signin(client, email).status_code == 302
    assert A._RATE_BUCKETS.get(("login_email", email), []) == []


def test_signing_in_clears_the_failures_before_it(signup):
    client, rows, _ = signup
    email = verified(rows)
    for _ in range(A.LOGIN_LIMIT - 1):
        signin(client, email, "wrong")
    assert signin(client, email).status_code == 302
    assert ("login_email", email) not in A._RATE_BUCKETS
