"""
The OpenRouter fallback costs money, so only admin requests may reach it: a
non-admin call that fails on the primary must fail, not quietly bill the paid
key. The flag rides a contextvar, so these also check it survives the
off-request threads that detection and fill run on.
"""

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from paperfill import app as A
from paperfill.ai import llm_client as L


class _Endpoint:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _fake(outcome):
    ep = _Endpoint(outcome)
    return L._Namespace(chat=L._Namespace(completions=ep)), ep


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(L._Method, "_record", lambda *a, **k: None)
    primary, p_ep = _fake(RuntimeError("primary down"))
    fallback, fb_ep = _fake("served by fallback")
    return L.FallbackClient(primary, fallback, "OpenRouter"), p_ep, fb_ep


def _call(c):
    return c.chat.completions.create(model="m", messages=[])


def test_non_admin_never_reaches_fallback(client):
    c, p_ep, fb_ep = client
    contextvars.copy_context().run(L.allow_fallback, True)  # not this context
    with pytest.raises(RuntimeError, match="primary down"):
        _call(c)
    assert p_ep.calls == 1 and fb_ep.calls == 0


def test_admin_falls_back(client):
    c, _, fb_ep = client

    def run():
        L.allow_fallback(True)
        return _call(c)

    assert contextvars.copy_context().run(run) == "served by fallback"
    assert fb_ep.calls == 1


def test_flag_follows_copied_context_into_threads(client):
    c, _, fb_ep = client
    results = []

    def request():
        L.allow_fallback(True)
        t = threading.Thread(target=contextvars.copy_context().run,
                             args=(lambda: results.append(_call(c)),))
        t.start()
        t.join()
        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=2) as ex:
            results.extend(ex.map(lambda _: ctx.copy().run(_call, c), range(3)))

    contextvars.copy_context().run(request)
    assert results == ["served by fallback"] * 4
    assert fb_ep.calls == 4


def test_before_request_sets_flag_from_role():
    def flag_for(role):
        with A.app.test_request_context():
            if role:
                A.session["role"] = role
            A.app.preprocess_request()
            return L._fallback_allowed.get()

    for role, expected in (("admin", True), ("user", False), (None, False)):
        assert contextvars.copy_context().run(flag_for, role) is expected, role
