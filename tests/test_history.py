"""
Tests for /history — the last 30 days of an account's own fills.

Two things decide whether the page is correct. It must show a user their own
fills and nobody else's, and a delete must take the documents with it while
staying refused to anyone who merely holds the job's link. The rest (dates,
thumbnails) is presentation.

`db` is faked with in-memory rows throughout, and OUTPUTS/UPLOADS are pointed
at tmp_path, so nothing here reaches Supabase or the real outputs/ directory.
"""

from datetime import datetime, timedelta, timezone

import pytest

from paperfill import app as A


MINE = "mine@example.com"
THEIRS = "theirs@example.com"


def iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


@pytest.fixture
def fills(monkeypatch, tmp_path):
    """Stand in for the assignments table plus the job files on disk. Seed a row
    with `add`; the returned list is what the fake queries read."""
    rows: list[dict] = []
    outputs, uploads = tmp_path / "outputs", tmp_path / "uploads"
    outputs.mkdir()
    uploads.mkdir()
    monkeypatch.setattr(A, "OUTPUTS", outputs)
    monkeypatch.setattr(A, "UPLOADS", uploads)

    def fetch_fills_for(email, since):
        email = (email or "").strip().lower()
        if not email:
            return []
        return sorted(
            (r for r in rows
             if (r.get("user_email") or "").strip().lower() == email
             and A.stats.parse_ts(r["ts"]) >= since),
            key=lambda r: r["ts"], reverse=True)

    def fill_owner(job_id):
        for r in rows:
            if r["job_id"] == job_id:
                return (r.get("user_email") or "").strip().lower() or None
        return None

    monkeypatch.setattr(A.db, "enabled", lambda: True)
    monkeypatch.setattr(A.db, "fetch_fills_for", fetch_fills_for)
    monkeypatch.setattr(A.db, "fill_owner", fill_owner)

    def add(job_id, email=MINE, days_ago=1, name="Worksheet.pdf",
            on_disk=True, style="Typed text"):
        rows.append({"job_id": job_id, "name": name, "ts": iso(days_ago),
                     "style": style, "user_email": email})
        if on_disk:
            (outputs / f"{job_id}-filled.pdf").write_bytes(b"%PDF-1.4\n")
            (outputs / job_id).mkdir(exist_ok=True)
            (outputs / job_id / "filled-0.png").write_bytes(b"png")
            (uploads / f"{job_id}.pdf").write_bytes(b"%PDF-1.4\n")
        return job_id

    add.rows = rows
    add.outputs = outputs
    add.uploads = uploads
    return add


@pytest.fixture
def client():
    A.app.config["TESTING"] = True
    return A.app.test_client()


def sign_in(client, email):
    with client.session_transaction() as s:
        s["role"] = "user"
        s["user_email"] = email
        s["user_sub"] = f"email:{email}"


def listed(client) -> list[str]:
    r = client.get("/api/history")
    assert r.status_code == 200
    return [f["job_id"] for f in r.get_json()["fills"]]


# ---- Whose fills you see -------------------------------------------------

def test_your_own_fills_come_back_newest_first(client, fills):
    fills("old", days_ago=9)
    fills("recent", days_ago=1)
    fills("middle", days_ago=4)
    sign_in(client, MINE)
    assert listed(client) == ["recent", "middle", "old"]


def test_another_accounts_fills_are_not_listed(client, fills):
    fills("theirs", email=THEIRS)
    fills("mine")
    sign_in(client, MINE)
    assert listed(client) == ["mine"]


def test_a_fill_past_the_window_is_not_listed(client, fills):
    fills("expired", days_ago=A.JOB_RETENTION_DAYS + 1)
    sign_in(client, MINE)
    assert listed(client) == []


def test_a_fill_whose_pdf_is_gone_is_not_listed(client, fills):
    """The sweep and the database expire on different clocks. An entry offering
    a view, an edit and a download that would all 404 is worse than no entry."""
    fills("swept", on_disk=False)
    sign_in(client, MINE)
    assert listed(client) == []


def test_the_page_and_the_api_need_a_session(client, fills):
    assert client.get("/api/history").status_code == 403
    r = client.get("/history")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


# ---- Deleting ------------------------------------------------------------

def delete(client, job_id):
    return client.post("/api/history/delete", json={"job_id": job_id})


def test_deleting_removes_the_documents_and_the_entry(client, fills):
    fills("gone")
    sign_in(client, MINE)
    assert delete(client, "gone").status_code == 200
    assert not (fills.outputs / "gone-filled.pdf").exists()
    assert not (fills.outputs / "gone").exists()
    assert not (fills.uploads / "gone.pdf").exists()
    assert listed(client) == []


def test_deleting_leaves_the_row_the_dashboard_counts(client, fills):
    """The fill happened. Clearing a user's list must not rewrite the numbers
    the admin dashboard reports."""
    fills("gone")
    sign_in(client, MINE)
    delete(client, "gone")
    assert [r["job_id"] for r in fills.rows] == ["gone"]


def test_you_cannot_delete_someone_elses_fill(client, fills):
    """A job id is unguessable but it travels, in the ?job= link the editor puts
    in the address bar. Holding it must not be enough to destroy the fill."""
    fills("theirs", email=THEIRS)
    sign_in(client, MINE)
    assert delete(client, "theirs").status_code == 404
    assert (fills.outputs / "theirs-filled.pdf").exists()


def test_a_signed_out_visitor_cannot_delete(client, fills):
    fills("mine")
    assert delete(client, "mine").status_code == 403
    assert (fills.outputs / "mine-filled.pdf").exists()


def test_a_job_id_that_is_not_one_is_refused_before_touching_disk(client, fills):
    sign_in(client, MINE)
    for bad in ("../../etc/passwd", "a/b", "", "x" * 65):
        assert delete(client, bad).status_code == 400
