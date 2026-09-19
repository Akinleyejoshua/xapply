"""Scan results that survive a restart, and a way to try an application again.

A scan over thirty boards takes minutes. Keeping its results only in the server's
memory meant a restart threw them away, and there was no way to remove a posting you
were not interested in. Both now live in the database, which is also what the terminal
reads, so the two views cannot disagree.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import create_app
from config import Settings
from database import STATUS_FAILED, STATUS_PENDING, STATUS_SUBMITTED, Database
from models import GREENHOUSE, LEVER, JobPosting

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=tmp_path / "t.db", output_dir=tmp_path / "out",
                    log_dir=tmp_path / "logs", audit_dir=tmp_path / "logs" / "a",
                    user_data_dir=tmp_path / "p", template_dir=ROOT / "templates",
                    company_file=tmp_path / "companies.json",
                    overrides_path=tmp_path / "settings.local.json",
                    admin_token="change-me")


@pytest.fixture
def db(settings: Settings) -> Database:
    d = Database(settings.db_path)
    d.init()
    return d


@pytest.fixture
def client(settings: Settings, db: Database) -> TestClient:
    return TestClient(create_app(settings, db))


def _jobs() -> list[JobPosting]:
    return [
        JobPosting(job_id="gh-1", url="https://job-boards.greenhouse.io/acme/jobs/1",
                   apply_url="https://job-boards.greenhouse.io/acme/jobs/1",
                   title="Data Analyst", company="Acme", location="Remote",
                   description="x" * 300, source=GREENHOUSE, ats=GREENHOUSE, relevance=0.91),
        JobPosting(job_id="lv-2", url="https://jobs.lever.co/beta/2",
                   apply_url="https://jobs.lever.co/beta/2/apply",
                   title="Analytics Engineer", company="Beta", location="Lagos",
                   description="y" * 300, source=LEVER, ats=LEVER, relevance=0.44),
    ]


# ---- the results outlive the process --------------------------------------

def test_scan_results_survive_a_restart(settings: Settings, db: Database) -> None:
    db.save_discovered(_jobs())

    reopened = Database(settings.db_path)        # as a fresh process would see it
    rows = reopened.list_discovered()

    assert [r["title"] for r in rows] == ["Data Analyst", "Analytics Engineer"]
    assert rows[0]["relevance"] == 0.91, "the better match is listed first"


def test_scanning_twice_refreshes_rather_than_duplicates(db: Database) -> None:
    db.save_discovered(_jobs())
    db.save_discovered(_jobs())

    assert db.count_discovered() == 2


def test_the_same_job_id_from_two_boards_is_two_postings(db: Database) -> None:
    """Board ids are only unique within a board, so the source is part of the key."""
    a = JobPosting(job_id="7", url="https://job-boards.greenhouse.io/a/jobs/7", source=GREENHOUSE)
    b = JobPosting(job_id="7", url="https://jobs.lever.co/b/7", source=LEVER)
    db.save_discovered([a, b])

    assert db.count_discovered() == 2


# ---- removing them ---------------------------------------------------------

def test_a_single_scan_result_can_be_removed(client: TestClient, db: Database) -> None:
    db.save_discovered(_jobs())

    assert client.delete("/api/discovered/gh-1").json()["deleted"] == 1
    assert [r["job_id"] for r in db.list_discovered()] == ["lv-2"]
    assert client.delete("/api/discovered/gh-1").status_code == 404


def test_scan_results_can_be_removed_in_bulk(client: TestClient, db: Database) -> None:
    db.save_discovered(_jobs())

    out = client.post("/api/discovered/delete", json={"job_ids": ["gh-1", "lv-2"]})

    assert out.json()["deleted"] == 2 and db.count_discovered() == 0


def test_the_whole_scan_list_can_be_cleared(client: TestClient, db: Database) -> None:
    db.save_discovered(_jobs())

    assert client.post("/api/discovered/delete", json={"all": True}).json()["deleted"] == 2
    assert db.count_discovered() == 0


def test_a_bulk_delete_has_to_say_what_to_delete(client: TestClient, db: Database) -> None:
    db.save_discovered(_jobs())

    assert client.post("/api/discovered/delete", json={}).status_code == 400
    assert db.count_discovered() == 2, "an ambiguous request deletes nothing"


def test_the_listing_is_searchable(client: TestClient, db: Database) -> None:
    db.save_discovered(_jobs())

    rows = client.get("/api/discovered", params={"search": "Beta"}).json()

    assert [r["company"] for r in rows] == ["Beta"]


def test_the_listing_leaves_out_the_description(client: TestClient, db: Database) -> None:
    """The list view shows dozens of rows; whole job descriptions would dwarf it."""
    db.save_discovered(_jobs())

    assert "description" not in client.get("/api/discovered").json()[0]


# ---- trying again ----------------------------------------------------------

@pytest.fixture
def no_real_run(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture what a retry would run, without needing a browser or an LLM key."""
    started: list[dict] = []

    class StubPipeline:
        def __init__(self, *_a, **_k) -> None:
            self.gate = None

        async def run(self, urls=None, limit=None):
            started.append({"urls": list(urls or []), "limit": limit})
            return {"submitted": 0}

    import pipeline
    monkeypatch.setattr(pipeline, "Pipeline", StubPipeline)
    return started


def _record(db: Database, job_id: str, status: str, url: str, apply_url: str = "") -> int:
    job = JobPosting(job_id=job_id, url=url, apply_url=apply_url, company="Acme",
                     title="Data Analyst", source="urls", ats=LEVER)
    return db.record(job, status)


def test_failed_applications_can_be_tried_again(client: TestClient, db: Database,
                                                no_real_run: list[dict]) -> None:
    _record(db, "f1", STATUS_FAILED, "https://jobs.lever.co/acme/f1")
    _record(db, "ok", STATUS_SUBMITTED, "https://jobs.lever.co/acme/ok")

    out = client.post("/api/applications/retry", json={"status": "failed"}).json()

    assert out["retrying"] == 1
    assert no_real_run[0]["urls"] == ["https://jobs.lever.co/acme/f1"]


def test_applications_waiting_on_you_can_be_tried_again(client: TestClient, db: Database,
                                                        no_real_run: list[dict]) -> None:
    _record(db, "p1", STATUS_PENDING, "https://jobs.lever.co/acme/p1")

    assert client.post("/api/applications/retry",
                       json={"status": "pending_human_review"}).json()["retrying"] == 1


def test_a_retry_clears_the_old_row_so_it_is_not_skipped(client: TestClient, db: Database,
                                                         no_real_run: list[dict]) -> None:
    """Discovery skips anything already recorded, so a retry that left the row behind
    would quietly do nothing at all."""
    _record(db, "f1", STATUS_FAILED, "https://jobs.lever.co/acme/f1")

    client.post("/api/applications/retry", json={"status": "failed"})

    assert db.list(status=STATUS_FAILED) == []
    assert not db.has_job("urls", "f1")


def test_the_apply_link_is_preferred_over_the_posting_link(client: TestClient, db: Database,
                                                           no_real_run: list[dict]) -> None:
    _record(db, "f1", STATUS_FAILED, "https://jobs.lever.co/acme/f1",
            apply_url="https://jobs.lever.co/acme/f1/apply")

    client.post("/api/applications/retry", json={"status": "failed"})

    assert no_real_run[0]["urls"] == ["https://jobs.lever.co/acme/f1/apply"]


def test_specific_applications_can_be_tried_again_by_id(client: TestClient, db: Database,
                                                        no_real_run: list[dict]) -> None:
    a = _record(db, "f1", STATUS_FAILED, "https://jobs.lever.co/acme/f1")
    _record(db, "f2", STATUS_FAILED, "https://jobs.lever.co/acme/f2")

    out = client.post("/api/applications/retry", json={"ids": [a]}).json()

    assert out["retrying"] == 1
    assert no_real_run[0]["urls"] == ["https://jobs.lever.co/acme/f1"]
    assert len(db.list(status=STATUS_FAILED)) == 1, "the other one is untouched"


def test_a_retry_that_matches_nothing_says_so(client: TestClient, db: Database,
                                              no_real_run: list[dict]) -> None:
    assert client.post("/api/applications/retry", json={"status": "failed"}).status_code == 404
    assert client.post("/api/applications/retry", json={}).status_code == 400
    assert no_real_run == []


def test_only_failed_and_waiting_applications_may_be_retried(client: TestClient,
                                                             db: Database) -> None:
    """Retrying a submitted application would apply to the same job twice."""
    _record(db, "ok", STATUS_SUBMITTED, "https://jobs.lever.co/acme/ok")

    assert client.post("/api/applications/retry", json={"status": "submitted"}).status_code == 422


def test_a_missing_api_key_is_reported_not_a_crash(client: TestClient, db: Database,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing Retry with no key configured used to return a 500 with nothing to act on."""
    from llm import LLMError

    class NoKey:
        def __init__(self, *_a, **_k) -> None:
            raise LLMError("GEMINI_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=nvidia.")

    import pipeline
    monkeypatch.setattr(pipeline, "Pipeline", NoKey)
    _record(db, "f1", STATUS_FAILED, "https://jobs.lever.co/acme/f1")

    out = client.post("/api/applications/retry", json={"status": "failed"})

    assert out.status_code == 400
    assert "GEMINI_API_KEY" in out.json()["detail"]
