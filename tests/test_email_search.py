"""Finding roles that are advertised with an address instead of a form.

A great deal of hiring never reaches an applicant tracking system. Someone writes "we
are hiring a data analyst, send your CV to careers@example.com" on their own site or on
X, and that is the whole process. No board API can see those, because there is no board.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from database import Database  # noqa: E402
from discovery import SOURCE_REGISTRY, EmailSearchSource  # noqa: E402
from models import UNKNOWN  # noqa: E402

POSTING_TEXT = ("We are hiring a Data Analyst to join our small team in Lagos. You will "
                "build reporting for the finance group, own the weekly numbers and help "
                "us decide what to measure. Experience with SQL and dashboards is what "
                "matters most to us, more than any particular degree. To apply, send "
                "your CV to careers@example-corp.com and tell us what you have built.")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=tmp_path / "t.db", log_dir=tmp_path,
                    audit_dir=tmp_path, output_dir=tmp_path, user_data_dir=tmp_path,
                    overrides_path=tmp_path / "s.json", company_file=tmp_path / "c.json",
                    search_queries=["Data Analyst"], search_location="Remote",
                    email_apply=True, title_match_threshold=0.3)


@pytest.fixture
def db(settings: Settings) -> Database:
    d = Database(settings.db_path)
    d.init()
    return d


class FakePage:
    def __init__(self, text: str = POSTING_TEXT, title: str = "Data Analyst | Example Corp",
                 url: str = "https://example-corp.com/careers/analyst"):
        self._text, self._title, self.url = text, title, url
        self.visited: list[str] = []

    async def goto(self, url: str, **_: Any) -> None:
        self.visited.append(url)
        self.url = url

    async def evaluate(self, script: str, *_: Any) -> Any:
        return self._text

    async def title(self) -> str:
        return self._title

    async def eval_on_selector_all(self, sel: str, script: str) -> list:
        return []


class FakeBrowser:
    def __init__(self) -> None:
        self.gate = None

    async def sleep(self, *_a: Any, **_k: Any) -> None:
        return None

    async def goto(self, page: Any, url: str) -> str:
        await page.goto(url)
        return url


def source(settings: Settings, db: Database) -> EmailSearchSource:
    src = EmailSearchSource(settings, db, None)
    src.b = FakeBrowser()
    return src


def test_the_source_is_registered() -> None:
    assert SOURCE_REGISTRY["emails"] is EmailSearchSource


def test_it_needs_a_browser_page() -> None:
    from job_search import BROWSER_SOURCES

    assert "emails" in BROWSER_SOURCES, "it opens pages, so a run must give it one"


@pytest.mark.asyncio
async def test_it_does_nothing_useful_without_somewhere_to_send(settings, db) -> None:
    """Finding these is pointless if the tool cannot then write to anyone."""
    settings.email_apply = False

    assert await source(settings, db).discover(FakePage()) == []


@pytest.mark.asyncio
async def test_a_page_that_names_an_address_becomes_a_posting(settings, db) -> None:
    job = await source(settings, db).read_posting(FakePage(), "https://example-corp.com/x")

    assert job is not None
    assert job.title == "Data Analyst"
    assert job.company == "Example-Corp"
    assert job.ats == UNKNOWN, "no form behind it, so it goes the email route"
    assert job.apply_url == "", "there is nothing to open, only someone to write to"
    assert "careers@example-corp.com" in job.description


@pytest.mark.asyncio
async def test_a_page_with_no_address_is_not_a_posting(settings, db) -> None:
    page = FakePage(text=POSTING_TEXT.replace("careers@example-corp.com", "our website"))

    assert await source(settings, db).read_posting(page, "https://x.com/y") is None


@pytest.mark.asyncio
async def test_a_thin_page_is_not_a_posting(settings, db) -> None:
    """A listing index names an address too, and is not a job."""
    page = FakePage(text="Jobs. Contact careers@example-corp.com")

    assert await source(settings, db).read_posting(page, "https://x.com/y") is None


@pytest.mark.asyncio
async def test_the_search_engines_own_links_are_ignored(settings, db) -> None:
    src = source(settings, db)

    class ResultsPage(FakePage):
        async def eval_on_selector_all(self, sel: str, script: str) -> list:
            return ["https://www.google.com/search?q=x",
                    "https://accounts.google.com/signin",
                    "https://www.youtube.com/watch?v=1",
                    "https://example-corp.com/careers/analyst",
                    "https://another.example/jobs/2"]

    links = await src.result_links(ResultsPage())

    assert links == ["https://example-corp.com/careers/analyst",
                     "https://another.example/jobs/2"]


@pytest.mark.asyncio
async def test_being_sent_to_the_x_login_page_is_reported(settings, db) -> None:
    """X refuses anonymous searches, and silence there looks like no results."""
    src = source(settings, db)
    page = FakePage(url="https://x.com/login")

    found = await src.search_engine(page, "https://x.com/search?q={q}", "hiring", "x")

    assert found == []


@pytest.mark.asyncio
async def test_the_seniority_you_picked_reaches_this_search_too(settings, db) -> None:
    settings.seniority_levels = ["intern"]

    assert "intern" in source(settings, db).level_clause()
