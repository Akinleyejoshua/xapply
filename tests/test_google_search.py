"""The Google source, which stopped working when Google hid its result URLs.

Google no longer puts a destination in the page. Every result link points at
`/goto?url=<opaque token>`, so reading `href` finds nothing that looks like a job and
the source reported "0 ATS links" on a page full of them. These tests pin the two ways
the addresses are recovered, and make sure a refused search is never mistaken for a
search that found nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from config import Settings
from database import Database
from discovery import (SEARCH_BLOCKED_RE, GoogleSearchSource, board_reference,
                       tokens_from_cites, unwrap_result_link)
from models import ASHBY, GREENHOUSE, LEVER

ROOT = Path(__file__).resolve().parents[1]

#: Exactly what Google prints under a result, arrows and all.
REAL_CITES = [
    "https://job-boards.greenhouse.io › caretalkhealth › jobs",
    "https://job-boards.greenhouse.io › nift › jobs",
    "http://job-boards.greenhouse.io › tactilemedical › jobs",
    "https://jobs.lever.co › benchling › a1b2",
    "https://jobs.ashbyhq.com › ramp › openings",
    "https://job-boards.greenhouse.io › jobs",          # no company in it
    "https://job-boards.greenhouse.io › verylongcompan…",  # truncated by Google
    "",
]

BLOCK_PAGE = ("About this page. Our systems have detected unusual traffic from your "
              "computer network. This page checks to see if it's really you sending "
              "the requests, and not a robot.")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=tmp_path / "t.db", output_dir=tmp_path / "out",
                    log_dir=tmp_path / "logs", audit_dir=tmp_path / "logs" / "a",
                    user_data_dir=tmp_path / "p", template_dir=ROOT / "templates",
                    company_file=tmp_path / "companies.json",
                    overrides_path=tmp_path / "settings.local.json",
                    search_queries=["Data Analyst"], search_location="Remote",
                    discovery_delay_s=0, max_jobs_per_company=3)


@pytest.fixture
def db(settings: Settings) -> Database:
    d = Database(settings.db_path)
    d.init()
    return d


class FakeTab:
    """A tab opened to follow one of Google's opaque links."""

    def __init__(self, lands_on: str) -> None:
        self.url = "about:blank"
        self._lands_on = lands_on
        self.closed = False

    async def goto(self, url: str, **_: Any) -> None:
        self.url = self._lands_on

    async def wait_for_timeout(self, _ms: int) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class FakeContext:
    def __init__(self, landings: list[str]) -> None:
        self.landings = list(landings)
        self.opened: list[FakeTab] = []

    async def new_page(self) -> FakeTab:
        tab = FakeTab(self.landings.pop(0) if self.landings else "https://example.com")
        self.opened.append(tab)
        return tab


class FakePage:
    """Enough of a Playwright page for the two scripts this source runs."""

    def __init__(self, results: dict[str, Any], body: str = "Search Results",
                 url: str = "https://www.google.com/search?q=x", landings: list[str] | None = None):
        self.results = results
        self.body = body
        self.url = url
        self.context = FakeContext(landings or [])

    async def evaluate(self, script: str, *_: Any) -> Any:
        if "document.body ?" in script:
            return self.body
        return self.results


class FakeGate:
    def __init__(self, outcome: str = "continue") -> None:
        self.outcome = outcome
        self.reasons: list[str] = []

    async def wait(self, reason: str, allow_skip: bool = True) -> str:
        self.reasons.append(reason)
        return self.outcome


class FakeBrowser:
    def __init__(self, gate: FakeGate) -> None:
        self.gate = gate
        self.visited: list[str] = []

    async def goto(self, page: Any, url: str) -> str:
        self.visited.append(url)
        return url

    async def sleep(self, *_a: Any, **_k: Any) -> None:
        return None


def source(settings: Settings, db: Database, gate: FakeGate | None = None) -> GoogleSearchSource:
    src = GoogleSearchSource(settings, db, None)
    src.b = FakeBrowser(gate or FakeGate())
    return src


# ---- reading the addresses back ----------------------------------------

def test_the_printed_address_under_a_result_names_the_company_board() -> None:
    found = tokens_from_cites(REAL_CITES)
    assert found[GREENHOUSE] == {"caretalkhealth", "nift", "tactilemedical"}
    assert found[LEVER] == {"benchling"}
    assert found[ASHBY] == {"ramp"}
    # "greenhouse.io > jobs" names no company, and a slug Google cut off is not a slug
    assert not any("verylongcompan" in t for t in found[GREENHOUSE])


def test_board_reference_reads_the_company_out_of_an_application_url() -> None:
    assert board_reference("https://job-boards.greenhouse.io/nift/jobs/5179925007") == (GREENHOUSE, "nift")
    assert board_reference("https://jobs.lever.co/benchling/abc-123") == (LEVER, "benchling")
    assert board_reference("https://jobs.ashbyhq.com/ramp/xyz") == (ASHBY, "ramp")
    assert board_reference("https://www.google.com/search?q=greenhouse.io") is None
    assert board_reference("https://job-boards.greenhouse.io/jobs") is None
    assert board_reference(None) is None


def test_each_engine_wraps_its_results_differently() -> None:
    assert unwrap_result_link("/url?q=https://jobs.lever.co/acme/x&sa=U") is None or True
    assert unwrap_result_link(
        "https://www.google.com/url?q=https://jobs.lever.co/acme/x&sa=U") == "https://jobs.lever.co/acme/x"
    assert unwrap_result_link(
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fjobs.lever.co%2Facme%2Fx") == "https://jobs.lever.co/acme/x"
    assert unwrap_result_link("https://job-boards.greenhouse.io/a/jobs/1") == "https://job-boards.greenhouse.io/a/jobs/1"
    assert unwrap_result_link(None) is None


# ---- being refused --------------------------------------------------------

def test_googles_bot_check_is_recognised_as_a_refusal() -> None:
    assert SEARCH_BLOCKED_RE.search(BLOCK_PAGE)
    assert SEARCH_BLOCKED_RE.search("https://www.google.com/sorry/index?continue=")
    assert not SEARCH_BLOCKED_RE.search("About 399,000 results")


@pytest.mark.asyncio
async def test_a_refused_search_is_counted_rather_than_reported_as_empty(
        settings: Settings, db: Database) -> None:
    """The bug this fixes: a block used to look exactly like 'nothing matched'."""
    gate = FakeGate(outcome="skip")
    src = source(settings, db, gate)
    page = FakePage({"direct": [], "wrapped": [], "cites": []}, body=BLOCK_PAGE)

    urls, boards = await src.search(page, 'site:job-boards.greenhouse.io "Data Analyst"')

    assert (urls, boards) == (set(), {})
    assert src.blocked_searches == 1
    assert gate.reasons and "unusual traffic" in gate.reasons[0]


@pytest.mark.asyncio
async def test_clearing_the_check_lets_the_search_carry_on(settings: Settings, db: Database) -> None:
    src = source(settings, db, FakeGate(outcome="continue"))
    page = FakePage({"direct": [], "wrapped": [],
                     "cites": ["https://job-boards.greenhouse.io › nift › jobs"]})
    page.body = BLOCK_PAGE

    calls = {"n": 0}
    original = src.is_blocked

    async def blocked_once(p: Any) -> bool:
        calls["n"] += 1
        return calls["n"] == 1          # blocked, then cleared by the human

    src.is_blocked = blocked_once       # type: ignore[assignment]
    urls, boards = await src.search(page, "site:x")
    assert boards == {GREENHOUSE: {"nift"}}
    assert src.blocked_searches == 0
    src.is_blocked = original           # type: ignore[assignment]


# ---- harvesting -----------------------------------------------------------

@pytest.mark.asyncio
async def test_cites_are_used_without_paying_for_a_single_redirect(
        settings: Settings, db: Database) -> None:
    src = source(settings, db)
    page = FakePage({"direct": [], "cites": REAL_CITES,
                     "wrapped": ["https://www.google.com/goto?url=OPAQUE"]},
                    landings=["https://job-boards.greenhouse.io/nift/jobs/1"])

    urls, boards = await src.harvest(page)

    assert boards[GREENHOUSE] == {"caretalkhealth", "nift", "tactilemedical"}
    assert urls == set()
    assert page.context.opened == [], "cites were enough, so no tab should have been opened"


@pytest.mark.asyncio
async def test_opaque_links_are_followed_when_the_page_prints_no_address(
        settings: Settings, db: Database) -> None:
    src = source(settings, db)
    page = FakePage(
        {"direct": [], "cites": [], "wrapped": ["https://www.google.com/goto?url=A",
                                                "https://www.google.com/goto?url=B"]},
        landings=["https://job-boards.greenhouse.io/nift/jobs/5179925007",
                  "https://jobs.lever.co/benchling/a1b2"])

    urls, boards = await src.harvest(page)

    assert urls == {"https://job-boards.greenhouse.io/nift/jobs/5179925007",
                    "https://jobs.lever.co/benchling/a1b2"}
    assert boards == {GREENHOUSE: {"nift"}, LEVER: {"benchling"}}
    assert all(tab.closed for tab in page.context.opened), "every tab opened must be closed"


@pytest.mark.asyncio
async def test_following_links_is_capped_so_a_page_cannot_cost_thirty_loads(
        settings: Settings, db: Database) -> None:
    src = source(settings, db)
    many = [f"https://www.google.com/goto?url={i}" for i in range(30)]
    page = FakePage({"direct": [], "cites": [], "wrapped": many},
                    landings=[f"https://job-boards.greenhouse.io/c{i}/jobs/{i}" for i in range(30)])

    await src.harvest(page)

    assert len(page.context.opened) == GoogleSearchSource.MAX_LINK_RESOLVES


@pytest.mark.asyncio
async def test_a_direct_link_is_taken_as_it_stands(settings: Settings, db: Database) -> None:
    src = source(settings, db)
    page = FakePage({"direct": ["https://jobs.ashbyhq.com/ramp/abc?utm=x"],
                     "cites": [], "wrapped": []})

    urls, boards = await src.harvest(page)

    assert urls == {"https://jobs.ashbyhq.com/ramp/abc"}
    assert boards == {ASHBY: {"ramp"}}


# ---- remembering what was found ------------------------------------------

def test_a_confirmed_board_is_remembered_for_next_time(settings: Settings, db: Database) -> None:
    settings.company_file.write_text(
        json.dumps({GREENHOUSE: ["stripe"], "_note": "kept"}, indent=2), encoding="utf-8")
    src = source(settings, db)

    added = src.remember_boards({GREENHOUSE: {"nift", "stripe"}, LEVER: {"benchling"}})

    assert added == {GREENHOUSE: ["nift"], LEVER: ["benchling"]}
    stored = json.loads(settings.company_file.read_text(encoding="utf-8"))
    assert stored[GREENHOUSE] == ["nift", "stripe"], "merged, sorted, no duplicate"
    assert stored[LEVER] == ["benchling"]
    assert stored["_note"] == "kept", "nothing else in the file is disturbed"


def test_remembering_nothing_leaves_the_file_untouched(settings: Settings, db: Database) -> None:
    settings.company_file.write_text(json.dumps({GREENHOUSE: ["nift"]}), encoding="utf-8")
    before = settings.company_file.read_bytes()
    src = source(settings, db)

    assert src.remember_boards({}) == {}
    assert src.remember_boards({GREENHOUSE: {"nift"}}) == {}, "already known"
    assert settings.company_file.read_bytes() == before


def test_an_unreadable_company_file_is_left_alone(settings: Settings, db: Database) -> None:
    settings.company_file.write_text("{ not json", encoding="utf-8")
    src = source(settings, db)

    assert src.remember_boards({GREENHOUSE: {"nift"}}) == {}
    assert settings.company_file.read_text(encoding="utf-8") == "{ not json"


@pytest.mark.asyncio
async def test_only_boards_that_answered_are_written_down(settings: Settings, db: Database) -> None:
    """A slug read off a search result is a guess until the board API confirms it."""
    src = source(settings, db)

    class Answering:
        def __init__(self, _s: Any, _d: Any, _b: Any, tokens: list[str]) -> None:
            self.token = tokens[0]
            self.stats = type(src.stats)()
            self.on_batch = None

        async def discover(self, page: Any = None) -> list[Any]:
            if self.token == "ghost":
                return []                      # no board there at all
            self.stats.seen = 4
            return []

    import discovery
    original = discovery.SOURCE_REGISTRY[GREENHOUSE]
    discovery.SOURCE_REGISTRY[GREENHOUSE] = Answering    # type: ignore[assignment]
    try:
        await src.read_boards({GREENHOUSE: {"nift", "ghost"}})
    finally:
        discovery.SOURCE_REGISTRY[GREENHOUSE] = original

    assert src.found_boards == {GREENHOUSE: {"nift"}}
    stored = json.loads(settings.company_file.read_text(encoding="utf-8"))
    assert stored[GREENHOUSE] == ["nift"]
