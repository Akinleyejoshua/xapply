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


# ---- only pages that actually name somebody to write to --------------------

BOARD_INDEX = ("Indeed - Data Analyst jobs in Lagos. Create an account to apply. "
               "Questions about your account? Write to contact@indeed.com. Interested in "
               "working at Indeed itself? Send your CV to careers@indeed.com and we will "
               "be in touch about our own openings. Browse thousands of listings today.")

BLOG = ("Ten things recruiters wish you knew. Most people send your CV to the wrong "
        "person entirely. " + "Filler sentence about job hunting. " * 30 +
        "You can reach the author at hello@somebody.blog for speaking enquiries.")


@pytest.mark.asyncio
async def test_a_job_board_index_page_is_not_a_posting(settings, db) -> None:
    """Results came back that were job boards, with nobody to write to. A board's own
    hiring address is not the posting's."""
    page = FakePage(text=BOARD_INDEX, url="https://www.indeed.com/q-data-analyst")

    assert await source(settings, db).read_posting(
        page, "https://www.indeed.com/q-data-analyst") is None


@pytest.mark.asyncio
async def test_a_recruiters_post_is_kept(settings, db) -> None:
    """These were being dropped for their domain, which is where they all live."""
    post = ("URGENT: we are hiring a Data Analyst for a remote team. Send your CV to "
            "jane@ascendion.com and mention where you saw this. We are looking for "
            "someone comfortable with SQL and dashboards who can start soon.")
    page = FakePage(text=post, title="Yevgeniya on LinkedIn",
                    url="https://www.linkedin.com/posts/y_urgent")

    job = await source(settings, db).read_posting(
        page, "https://www.linkedin.com/posts/y_urgent")

    assert job is not None and job.email_to == "jane@ascendion.com"


def test_a_single_post_is_kept_and_a_profile_is_not() -> None:
    """A profile or a feed is not an advert; one post is."""
    from urllib.parse import urlparse

    from discovery import is_social, is_social_post

    def kept(url: str) -> bool:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        return (not is_social(host)) or is_social_post(host, u.path)

    assert kept("https://x.com/r/status/2101178396546089178") is True
    assert kept("https://www.linkedin.com/posts/y_urgent-hiring") is True
    assert kept("https://x.com/Yuj_recruit") is False
    assert kept("https://www.linkedin.com/feed/") is False
    assert kept("https://northwind.com/jobs/1") is True


@pytest.mark.asyncio
async def test_an_article_that_merely_mentions_cvs_is_not_a_posting(settings, db) -> None:
    page = FakePage(text=BLOG, url="https://somebody.blog/post")

    assert await source(settings, db).read_posting(page, "https://somebody.blog/post") is None


@pytest.mark.asyncio
async def test_a_real_posting_carries_the_address_it_named(settings, db) -> None:
    """Every result now has somebody to write to, recorded on the posting itself."""
    job = await source(settings, db).read_posting(
        FakePage(), "https://example-corp.com/careers/analyst")

    assert job is not None
    assert job.email_to == "careers@example-corp.com"


@pytest.mark.asyncio
async def test_every_posting_a_scan_returns_has_an_address(settings, db) -> None:
    """The guarantee the source exists to make."""
    src = source(settings, db)

    class Results(FakePage):
        async def eval_on_selector_all(self, sel: str, script: str) -> list:
            if "google" in (self.url or ""):
                return ["https://example-corp.com/careers/analyst"]
            return []

    found = await src.discover(Results(url="https://www.google.com/search"))

    assert found, "the one real posting should have been kept"
    assert all(job.email_to for job in found)


def test_an_address_is_carried_through_to_the_database(tmp_path: Path) -> None:
    from database import Database
    from models import JobPosting

    db = Database(tmp_path / "t.db")
    db.init()
    db.save_discovered([JobPosting(job_id="a", url="https://nw.com/j", source="emails",
                                   title="Data Analyst", company="Northwind",
                                   email_to="careers@nw.com")])

    assert db.list_discovered()[0]["email_to"] == "careers@nw.com"


def test_a_database_made_before_the_column_existed_is_brought_up_to_date(tmp_path: Path) -> None:
    """Nobody should have to delete their database to get a new field."""
    import sqlite3

    from database import Database

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE discovered (job_id TEXT NOT NULL, source TEXT NOT NULL,"
                 " ats TEXT, company TEXT, title TEXT, location TEXT, url TEXT,"
                 " apply_url TEXT, description TEXT, relevance REAL, found_at TEXT,"
                 " PRIMARY KEY (source, job_id))")
    conn.commit()
    conn.close()

    Database(path).init()

    columns = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(discovered)")}
    assert "email_to" in columns


# ---- judging the advert, not the page title -------------------------------

def _job(title: str, body: str):
    from models import JobPosting

    return JobPosting(job_id="j", url="https://x.com/j", title=title, description=body)


def test_a_role_named_only_in_the_body_still_counts(settings, db) -> None:
    """A recruiter's post is titled "Recruit NG on X" or "Abuja Jobs". The role is in
    the words underneath, and matching the title alone threw those away."""
    src = source(settings, db)

    assert src.mentions_the_role(
        _job("Abuja Jobs", "WE ARE HIRING a Data Analyst. Send your CV to hr@x.com")) is True
    assert src.mentions_the_role(
        _job("Recruit NG on X", "Data Analyst needed, send CV")) is True


def test_a_title_that_is_the_role_still_counts(settings, db) -> None:
    assert source(settings, db).mentions_the_role(_job("Data Analyst", "anything")) is True


def test_something_genuinely_unrelated_is_still_turned_away(settings, db) -> None:
    """The filter has to keep doing its job, or a scan returns every advert alive."""
    src = source(settings, db)

    assert src.mentions_the_role(
        _job("My Engineers", "We build bridges. Send your CV to hr@x.com")) is False


def test_which_engines_are_searched(settings, db) -> None:
    """X is only searched when you have asked for it, because it refuses anonymous
    searches and would otherwise produce nothing but a warning every run."""
    src = source(settings, db)

    assert src.GOOGLE.startswith("https://www.google.com/search")
    assert src.X_SEARCH.startswith("https://x.com/search")
    assert settings.search_x is False, "off unless you turn it on"


# ---- the emails source walks pages too ------------------------------------

@pytest.mark.asyncio
async def test_more_than_one_page_of_results_is_read(settings, db) -> None:
    src = source(settings, db)
    settings.search_result_pages = 3
    asked: list[int] = []

    async def watched(page, template, terms, engine, number=0):
        asked.append(number)
        return [] if number >= 2 else [f"https://example{number}.com/job"]

    src.search_engine = watched
    src.read_posting = lambda page, url: _none()

    async def _none():
        return None

    await src.discover(FakePage())

    assert asked[:3] == [0, 1, 2], "each page in turn until one comes back empty"


@pytest.mark.asyncio
async def test_it_stops_opening_pages_at_your_limit(settings, db) -> None:
    """Opening a result costs a page load, which is what makes a scan long."""
    settings.max_pages_opened = 2
    settings.search_result_pages = 5
    src = source(settings, db)
    opened: list[str] = []

    async def watched(page, template, terms, engine, number=0):
        return [f"https://example.com/job/{number}/{i}" for i in range(5)]

    async def read(page, url):
        opened.append(url)
        return None

    src.search_engine = watched
    src.read_posting = read

    await src.discover(FakePage())

    assert len(opened) <= 2


def test_the_limits_are_yours_to_set(settings) -> None:
    from config import PERSISTED_KEYS

    assert settings.search_result_pages == 1, "one page unless you ask for more"
    assert settings.max_pages_opened == 12
    assert "search_result_pages" in PERSISTED_KEYS
    assert "max_pages_opened" in PERSISTED_KEYS
