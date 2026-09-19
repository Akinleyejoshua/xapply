"""Accounts the browser profile is signed in to, and for how long.

Signing in is the one thing only you can do, so no password is handled here. What is
added is being able to see the state of every account at once: without it, whether a run
can read X or send from Gmail is only discoverable by starting one and watching it fail.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import create_app  # noqa: E402
from config import Settings  # noqa: E402
from database import Database  # noqa: E402
from integrations import (BY_NAME, INTEGRATIONS, cookie_db, forget, signed_in,  # noqa: E402
                          status)

ROOT = Path(__file__).resolve().parents[1]
#: Chrome counts microseconds from 1601.
EPOCH_OFFSET = 11_644_473_600


def chrome_time(seconds_from_now: float) -> int:
    return int((time.time() + seconds_from_now + EPOCH_OFFSET) * 1_000_000)


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    """A browser profile with a cookie store shaped like Chrome's."""
    store = tmp_path / "Default"
    store.mkdir(parents=True)
    conn = sqlite3.connect(store / "Cookies")
    conn.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, expires_utc INTEGER)")
    conn.commit()
    conn.close()
    return tmp_path


def add_cookie(profile: Path, host: str, name: str, expires: int | None = None) -> None:
    conn = sqlite3.connect(profile / "Default" / "Cookies")
    conn.execute("INSERT INTO cookies VALUES (?,?,?)",
                 (host, name, chrome_time(3600) if expires is None else expires))
    conn.commit()
    conn.close()


def test_a_profile_with_no_cookies_is_signed_in_to_nothing(profile: Path) -> None:
    assert all(row["signed_in"] is False for row in status(profile))


def test_a_session_cookie_means_signed_in(profile: Path) -> None:
    add_cookie(profile, ".x.com", "auth_token")

    assert signed_in(profile, BY_NAME["x"]) is True
    assert signed_in(profile, BY_NAME["linkedin"]) is False


def test_an_expired_session_is_not_a_session(profile: Path) -> None:
    """A stale cookie would say signed in right up until the run failed."""
    add_cookie(profile, ".linkedin.com", "li_at", expires=chrome_time(-3600))

    assert signed_in(profile, BY_NAME["linkedin"]) is False


def test_a_cookie_that_lives_for_the_browser_session_still_counts(profile: Path) -> None:
    """Zero means it dies when the browser does, not that it is already dead."""
    add_cookie(profile, "mail.google.com", "SID", expires=0)

    assert signed_in(profile, BY_NAME["gmail"]) is True


def test_a_cookie_from_another_site_is_not_yours(profile: Path) -> None:
    add_cookie(profile, ".notlinkedin.com", "li_at")

    assert signed_in(profile, BY_NAME["linkedin"]) is False


def test_tracking_cookies_are_not_a_session(profile: Path) -> None:
    """A profile collects hundreds of these and none of them signs you in."""
    for host in (".adnxs.com", ".doubleclick.net", ".x.com"):
        add_cookie(profile, host, "uuid")

    assert signed_in(profile, BY_NAME["x"]) is False


def test_it_says_which_accounts_the_current_settings_need(profile: Path) -> None:
    settings = Settings(_env_file=None, email_apply=True, email_transport="gmail",
                        search_x=True, sources=["greenhouse"])

    needed = {row["name"] for row in status(profile, settings) if row["needed_now"]}

    assert needed == {"gmail", "x"}


def test_nothing_is_needed_when_nothing_uses_it(profile: Path) -> None:
    settings = Settings(_env_file=None, email_apply=False, search_x=False,
                        sources=["greenhouse"])

    assert not any(row["needed_now"] for row in status(profile, settings))


# ---- signing out -----------------------------------------------------------

def test_signing_out_removes_only_that_account(profile: Path) -> None:
    add_cookie(profile, ".x.com", "auth_token")
    add_cookie(profile, ".linkedin.com", "li_at")

    forget(profile, "x")

    assert signed_in(profile, BY_NAME["x"]) is False
    assert signed_in(profile, BY_NAME["linkedin"]) is True


def test_signing_out_of_something_unknown_is_an_error(profile: Path) -> None:
    with pytest.raises(KeyError):
        forget(profile, "myspace")


def test_a_profile_that_has_never_run_is_handled(tmp_path: Path) -> None:
    assert cookie_db(tmp_path) is None
    assert forget(tmp_path, "x") == 0
    assert all(row["signed_in"] is False for row in status(tmp_path))


# ---- through the API -------------------------------------------------------

@pytest.fixture
def client(profile: Path, tmp_path: Path) -> TestClient:
    settings = Settings(_env_file=None, db_path=tmp_path / "t.db",
                        overrides_path=tmp_path / "s.json", log_dir=tmp_path / "l",
                        audit_dir=tmp_path / "l" / "a", output_dir=tmp_path / "o",
                        user_data_dir=profile, company_file=tmp_path / "c.json",
                        template_dir=ROOT / "templates", search_x=True)
    db = Database(settings.db_path)
    db.init()
    return TestClient(create_app(settings, db))


def test_the_dashboard_can_list_them(client: TestClient, profile: Path) -> None:
    add_cookie(profile, ".x.com", "auth_token")

    rows = client.get("/api/integrations").json()

    assert {r["name"] for r in rows} == {i.name for i in INTEGRATIONS}
    assert [r for r in rows if r["name"] == "x"][0]["signed_in"] is True


def test_signing_out_through_the_dashboard(client: TestClient, profile: Path) -> None:
    add_cookie(profile, ".x.com", "auth_token")

    out = client.post("/api/integrations/x/logout").json()

    assert out["forgotten"] == 1
    assert [r for r in client.get("/api/integrations").json()
            if r["name"] == "x"][0]["signed_in"] is False


def test_an_unknown_account_is_a_404(client: TestClient) -> None:
    assert client.post("/api/integrations/myspace/logout").status_code == 404
    assert client.post("/api/integrations/myspace/login").status_code == 404


def test_every_account_says_what_it_is_for() -> None:
    for item in INTEGRATIONS:
        assert item.used_for, item.name
        assert item.session_cookies and item.domains, item.name
        assert item.sign_in_url.startswith("https://"), item.name
