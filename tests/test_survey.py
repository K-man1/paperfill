"""
Tests for the returning-user survey.

The rule that matters is who gets asked: someone who has filled at least one
worksheet and hasn't answered or dismissed yet, never an admin, and never on a
database error (a prompt whose answer can't be saved is worse than no prompt).
`db` is faked in memory, so nothing here reaches Supabase.
"""

import pytest

from paperfill import app as A
from paperfill.data import db, stats


EMAIL = "mine@example.com"


@pytest.fixture
def client():
    A.app.config["TESTING"] = True
    A._RATE_BUCKETS.clear()
    return A.app.test_client()


def sign_in(client, email=EMAIL, role="user"):
    with client.session_transaction() as sess:
        sess["role"] = role
        sess["user_email"] = email
        sess["user_sub"] = f"email:{email}"


@pytest.fixture
def table(monkeypatch):
    """In-memory survey_responses and assignments behind the db functions the
    routes call. `saved` maps email to the row save_survey was given."""
    saved: dict[str, dict] = {}
    fillers = {EMAIL}

    monkeypatch.setattr(A.db, "enabled", lambda: True)
    monkeypatch.setattr(
        A.db, "survey_due",
        lambda email: email in fillers and email not in saved)

    def save_survey(email, answers):
        saved.setdefault(email, answers)
        return True

    monkeypatch.setattr(A.db, "save_survey", save_survey)
    return saved


# ---- Who is asked ---------------------------------------------------------

def test_returning_user_is_asked(client, table):
    sign_in(client)
    assert client.get("/api/survey").get_json() == {"show": True}


def test_user_with_no_fills_is_not_asked(client, table):
    sign_in(client, "new@example.com")
    assert client.get("/api/survey").get_json() == {"show": False}


def test_admin_is_not_asked(client, table):
    sign_in(client, role="admin")
    assert client.get("/api/survey").get_json() == {"show": False}


def test_signed_out_is_refused(client, table):
    assert client.get("/api/survey").status_code == 403
    assert client.post("/api/survey", json={"dismissed": True}).status_code == 403


def test_not_asked_again_after_answering(client, table):
    sign_in(client)
    assert client.post("/api/survey", json={"rating": 4}).status_code == 200
    assert client.get("/api/survey").get_json() == {"show": False}


def test_not_asked_again_after_dismissing(client, table):
    sign_in(client)
    client.post("/api/survey", json={"dismissed": True})
    assert client.get("/api/survey").get_json() == {"show": False}


# ---- What is stored -------------------------------------------------------

def test_answers_are_stored_trimmed_and_capped(client, table):
    sign_in(client)
    client.post("/api/survey", json={
        "rating": 5, "recommend": 9, "good": "  fast  ", "bad": "x" * 2000,
        "heard_from": " a friend "})
    row = table[EMAIL]
    assert row["rating"] == 5 and row["recommend"] == 9
    assert row["good"] == "fast"
    assert len(row["bad"]) == 1000
    assert row["heard_from"] == "a friend"
    assert row["dismissed"] is False


def test_empty_submit_is_stored_as_a_dismissal(client, table):
    sign_in(client)
    client.post("/api/survey", json={"rating": None, "good": "   "})
    assert table[EMAIL]["dismissed"] is True


@pytest.mark.parametrize("body", [
    {"rating": 0}, {"rating": 6}, {"rating": "5"}, {"rating": True},
    {"recommend": -1}, {"recommend": 11}, {"recommend": 3.5},
])
def test_out_of_range_scores_are_rejected(client, table, body):
    sign_in(client)
    assert client.post("/api/survey", json=body).status_code == 400
    assert EMAIL not in table


def test_failed_save_is_reported(client, table, monkeypatch):
    monkeypatch.setattr(A.db, "save_survey", lambda email, answers: False)
    sign_in(client)
    assert client.post("/api/survey", json={"rating": 3}).status_code == 502


def test_rate_limit_trips(client, table):
    sign_in(client)
    codes = [client.post("/api/survey", json={"dismissed": True}).status_code
             for _ in range(A.SURVEY_LIMIT + 1)]
    assert codes[-1] == 429


# ---- db.survey_due fails closed -------------------------------------------

def fake_get(monkeypatch, answers_by_table):
    """Back _get_checked with canned rows keyed by table name. A value of None
    stands for a failed request."""
    monkeypatch.setattr(db, "enabled", lambda: True)
    monkeypatch.setattr(
        db, "_get_checked",
        lambda path: answers_by_table[path.split("?")[0]])


def test_survey_due_true_for_filler_with_no_row(monkeypatch):
    fake_get(monkeypatch, {"survey_responses": [], "assignments": [{"job_id": "a"}]})
    assert db.survey_due(EMAIL) is True


def test_survey_due_false_when_row_exists(monkeypatch):
    fake_get(monkeypatch, {"survey_responses": [{"user_email": EMAIL}],
                           "assignments": [{"job_id": "a"}]})
    assert db.survey_due(EMAIL) is False


def test_survey_due_false_with_no_fills(monkeypatch):
    fake_get(monkeypatch, {"survey_responses": [], "assignments": []})
    assert db.survey_due(EMAIL) is False


def test_survey_due_false_when_table_lookup_fails(monkeypatch):
    fake_get(monkeypatch, {"survey_responses": None, "assignments": [{"job_id": "a"}]})
    assert db.survey_due(EMAIL) is False


def test_survey_due_false_when_fill_lookup_fails(monkeypatch):
    fake_get(monkeypatch, {"survey_responses": [], "assignments": None})
    assert db.survey_due(EMAIL) is False


def test_survey_due_false_without_database(monkeypatch):
    monkeypatch.setattr(db, "enabled", lambda: False)
    assert db.survey_due(EMAIL) is False


# ---- Admin summary --------------------------------------------------------

def test_summary_excludes_dismissals_from_averages():
    rows = [
        {"dismissed": False, "rating": 5, "recommend": 10, "heard_from": "Hack Club"},
        {"dismissed": False, "rating": 3, "recommend": 6, "heard_from": " hack club "},
        {"dismissed": False, "rating": None, "recommend": 8, "heard_from": ""},
        {"dismissed": True, "rating": None, "recommend": None, "heard_from": None},
    ]
    out = stats.survey_summary(rows)
    assert out["answered"] == 3
    assert out["dismissed"] == 1
    assert out["avg_rating"] == 4.0
    # One promoter (10), one detractor (6), one passive (8): (1 - 1) / 3.
    assert out["nps"] == 0
    assert out["heard_from"] == [("hack club", 2)]


def test_summary_of_nothing_has_no_averages():
    out = stats.survey_summary([])
    assert out["avg_rating"] is None and out["nps"] is None
    assert out["answered"] == 0 and out["heard_from"] == []
