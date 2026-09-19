"""The activity feed: what the run is doing, in enough detail to read back afterwards.

It used to carry only the handful of moments the API itself knew about, as plain
strings. The interesting part of a run happens inside the pipeline and the form filler,
so the feed is fed from their logs instead, and each line carries what kind of thing it
is so the dashboard can colour it.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import FEED_LOGGERS, LOG_LINES, create_app  # noqa: E402
from config import Settings  # noqa: E402
from database import Database  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=tmp_path / "t.db", output_dir=tmp_path / "o",
                    log_dir=tmp_path / "l", audit_dir=tmp_path / "l" / "a",
                    user_data_dir=tmp_path / "p", company_file=tmp_path / "c.json",
                    overrides_path=tmp_path / "s.json", template_dir=ROOT / "templates")


@pytest.fixture
def client(settings: Settings, tmp_path: Path) -> TestClient:
    db = Database(settings.db_path)
    db.init()
    return TestClient(create_app(settings, db))


def feed(client: TestClient) -> list[dict]:
    return client.get("/api/run").json()["log"]


def test_what_the_run_does_reaches_the_feed(client: TestClient) -> None:
    """None of these go through the API. They are what the run logs as it works."""
    logging.getLogger("appliers").info("Job gh-1 | Data Analyst @ Acme")
    logging.getLogger("browser_bot").info("Filled Email: you@example.com  [profile]")
    logging.getLogger("pipeline").info("SUBMITTED: Emailed to careers@acme.com")

    texts = [row["text"] for row in feed(client)]

    assert "Job gh-1 | Data Analyst @ Acme" in texts
    assert "Filled Email: you@example.com  [profile]" in texts
    assert "SUBMITTED: Emailed to careers@acme.com" in texts


def test_every_line_says_what_kind_of_thing_it_is(client: TestClient) -> None:
    logging.getLogger("pipeline").info("a normal step")
    logging.getLogger("pipeline").warning("something needs attention")
    logging.getLogger("discovery").error("something failed")

    kinds = {row["text"]: row["kind"] for row in feed(client)}

    assert kinds["a normal step"] == "info"
    assert kinds["something needs attention"] == "warn"
    assert kinds["something failed"] == "error"


def test_library_chatter_is_kept_out(client: TestClient) -> None:
    """Playwright and httpx produce more lines than the run does, and none of them are
    about your application."""
    for noisy in ("httpx", "urllib3", "asyncio", "playwright"):
        logging.getLogger(noisy).warning("noise from " + noisy)

    assert not any("noise from" in row["text"] for row in feed(client))


def test_info_lines_are_not_filtered_before_they_reach_the_feed(client: TestClient) -> None:
    """A logger filters by level before any handler runs, so without raising it the
    feed showed only warnings and failures, which is the opposite of the whole flow."""
    for name in FEED_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() <= logging.INFO, name


def test_each_line_carries_a_time_and_a_source(client: TestClient) -> None:
    logging.getLogger("resume_builder").info("Resume built")

    row = [r for r in feed(client) if r["text"] == "Resume built"][0]

    assert row["from"] == "resume_builder"
    assert len(row["at"]) == 8 and row["at"].count(":") == 2


def test_notes_from_the_api_sit_in_the_same_feed(client: TestClient, settings) -> None:
    client.patch("/api/config", json={"remote_only": True})

    rows = feed(client)

    assert any("settings saved" in r["text"] for r in rows)
    assert all({"at", "kind", "text"} <= set(r) for r in rows)


def test_the_feed_does_not_grow_without_end(client: TestClient) -> None:
    """A long run produces thousands of lines and the server keeps them in memory."""
    for i in range(LOG_LINES + 200):
        logging.getLogger("pipeline").info("line %d", i)

    assert len(client.app.state.log) <= LOG_LINES


def test_a_very_long_line_is_cut_down(client: TestClient) -> None:
    logging.getLogger("ai_agent").info("x" * 5000)

    assert max(len(r["text"]) for r in feed(client)) <= 500


def test_filling_a_field_is_logged_with_where_the_answer_came_from() -> None:
    """This is the part of a run worth reading back: what went into the form."""
    import browser_bot

    source = Path(browser_bot.__file__).read_text(encoding="utf-8")

    assert 'log.info("Filled %s: %s  [%s]"' in source
