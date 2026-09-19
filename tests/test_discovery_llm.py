"""Offline tests for the discovery layer and the pluggable LLM backends.

No network: HTTP is stubbed, so these run in CI and on a plane.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from database import Database  # noqa: E402
from discovery import (  # noqa: E402
    ATS_URL_RE,
    AshbyBoardSource,
    GreenhouseBoardSource,
    LeverBoardSource,
    load_company_tokens,
    location_matches,
    query_tokens,
    strip_html,
    title_matches,
)
from llm import LLMError, extract_json, schema_of  # noqa: E402
from models import ASHBY, GREENHOUSE, LEVER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "t.db", output_dir=tmp_path / "out", log_dir=tmp_path / "logs",
                    audit_dir=tmp_path / "logs" / "a", user_data_dir=tmp_path / "p",
                    template_dir=ROOT / "templates", company_file=ROOT / "companies.json",
                    search_queries=["Machine Learning Engineer", "Backend Engineer"],
                    search_location="Remote", max_jobs_per_company=5, discovery_delay_s=0)


@pytest.fixture
def db(settings: Settings) -> Database:
    d = Database(settings.db_path)
    d.init()
    return d


def stub(routes: dict[str, Any]):
    """An httpx transport that answers from a dict of url-substring -> payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        for fragment, payload in routes.items():
            if fragment in str(request.url):
                if isinstance(payload, int):
                    return httpx.Response(payload, json={})
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": "not stubbed"})

    return httpx.MockTransport(handler)


def patch_client(monkeypatch, source, routes: dict[str, Any]) -> None:
    async def _client(self):  # noqa: ANN001
        return httpx.AsyncClient(transport=stub(routes), timeout=5, follow_redirects=True)

    monkeypatch.setattr(type(source), "_client", _client)


# ---- title / location filtering -------------------------------------------


@pytest.mark.parametrize("title,expected", [
    ("Machine Learning Engineer", True),
    ("Senior Machine Learning Engineer, Ads", True),
    ("Machine Learning Scientist", True),          # 2 of 3 words
    ("Backend Engineer, Billing", True),
    ("Backend Software Engineer", True),
    ("Sales Engineer", False),                      # 'engineer' alone is not enough
    ("Solutions Engineer", False),
    ("Data Engineer", False),
    ("Abuse Investigator", False),
    ("Engineering Manager", False),
])
def test_title_matches_requires_a_majority(title, expected) -> None:
    toks = query_tokens(["Machine Learning Engineer", "Backend Engineer"])
    assert title_matches(title, toks) is expected


def test_title_matches_without_queries_accepts_everything() -> None:
    assert title_matches("Anything At All", []) is True


def test_query_tokens_drops_seniority_noise() -> None:
    assert query_tokens(["Senior Backend Engineer"]) == [{"backend", "engineer"}]


@pytest.mark.parametrize("text,remote_only,is_remote,expected", [
    ("Remote - United States", True, None, True),
    ("San Francisco, CA", True, None, False),
    ("San Francisco, CA", True, True, True),        # explicit flag wins
    ("Remote", True, False, False),
    ("Anywhere", True, None, True),
    ("Berlin", False, None, True),                  # not filtering on remote
])
def test_location_matches(text, remote_only, is_remote, expected) -> None:
    assert location_matches(text, "Remote", remote_only, is_remote) is expected


# ---- html cleanup + url recognition ---------------------------------------


def test_strip_html_handles_double_escaping() -> None:
    raw = "&lt;h2&gt;About&lt;/h2&gt;&lt;p&gt;We use &amp;amp; love Python&lt;/p&gt;&lt;li&gt;Ship code&lt;/li&gt;"
    out = strip_html(raw)
    assert "<" not in out and "&lt;" not in out
    assert "About" in out and "Python" in out and "Ship code" in out


def test_strip_html_empty() -> None:
    assert strip_html("") == ""


@pytest.mark.parametrize("url,matches", [
    ("https://job-boards.greenhouse.io/anthropic/jobs/5421031008", True),
    ("https://boards.greenhouse.io/stripe/jobs/8172503", True),
    ("https://jobs.lever.co/palantir/10dfc8bc-99ad-4ca2-ab76-853cb90a92c2", True),
    ("https://jobs.ashbyhq.com/openai/240d459b-696d-43eb-8497-fab3e56ecd9b", True),
    ("https://example.com/careers/42", False),
    ("https://www.linkedin.com/jobs/view/123/", False),
])
def test_ats_url_regex(url, matches) -> None:
    assert bool(ATS_URL_RE.fullmatch(url)) is matches


def test_company_file_is_valid() -> None:
    tokens = load_company_tokens(ROOT / "companies.json")
    assert set(tokens) >= {GREENHOUSE, LEVER, ASHBY}
    assert all(isinstance(t, str) and t for group in tokens.values() for t in group)


# ---- board sources --------------------------------------------------------


@pytest.mark.asyncio
async def test_greenhouse_source(monkeypatch, settings, db) -> None:
    src = GreenhouseBoardSource(settings, db, None, tokens=["acme"])
    patch_client(monkeypatch, src, {
        "/boards/acme/jobs/1": {"id": 1, "title": "Machine Learning Engineer", "company_name": "Acme",
                                "location": {"name": "Remote"}, "content": "&lt;p&gt;" + "Build models. " * 30 + "&lt;/p&gt;",
                                "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"},
        "/boards/acme/jobs": {"jobs": [
            {"id": 1, "title": "Machine Learning Engineer", "location": {"name": "Remote"},
             "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"},
            {"id": 2, "title": "Office Manager", "location": {"name": "NYC"},
             "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/2"},
        ]},
    })
    jobs = await src.discover()
    assert len(jobs) == 1                                  # 'Office Manager' filtered out by title
    job = jobs[0]
    assert job.ats == GREENHOUSE and job.company == "Acme"
    assert job.job_id == "gh-acme-1"
    assert job.apply_url == "https://job-boards.greenhouse.io/acme/jobs/1"
    assert "Build models." in job.description and "<p>" not in job.description


@pytest.mark.asyncio
async def test_greenhouse_rewrites_company_hosted_urls(monkeypatch, settings, db) -> None:
    """When absolute_url points at the company site, apply at the real Greenhouse form."""
    src = GreenhouseBoardSource(settings, db, None, tokens=["acme"])
    patch_client(monkeypatch, src, {
        "/boards/acme/jobs/9": {"id": 9, "title": "Backend Engineer", "company_name": "Acme",
                               "location": {"name": "Remote"}, "content": "Ship services. " * 30,
                               "absolute_url": "https://acme.com/careers?gh_jid=9"},
        "/boards/acme/jobs": {"jobs": [{"id": 9, "title": "Backend Engineer", "location": {"name": "Remote"},
                                        "absolute_url": "https://acme.com/careers?gh_jid=9"}]},
    })
    job = (await src.discover())[0]
    assert job.apply_url == "https://job-boards.greenhouse.io/acme/jobs/9"
    assert job.url == "https://acme.com/careers?gh_jid=9"


@pytest.mark.asyncio
async def test_ashby_source(monkeypatch, settings, db) -> None:
    src = AshbyBoardSource(settings, db, None, tokens=["acme"])
    patch_client(monkeypatch, src, {"posting-api/job-board/acme": {"jobs": [
        {"id": "abc", "title": "Backend Engineer", "isListed": True, "isRemote": True,
         "location": "Remote", "secondaryLocations": [],
         "descriptionPlain": "Own the API. " * 30,
         "jobUrl": "https://jobs.ashbyhq.com/acme/abc"},
        {"id": "hid", "title": "Backend Engineer", "isListed": False, "isRemote": True,
         "location": "Remote", "descriptionPlain": "x" * 500, "jobUrl": "https://jobs.ashbyhq.com/acme/hid"},
    ]}})
    jobs = await src.discover()
    assert len(jobs) == 1                                   # unlisted posting skipped
    assert jobs[0].apply_url == "https://jobs.ashbyhq.com/acme/abc/application"
    assert jobs[0].ats == ASHBY


@pytest.mark.asyncio
async def test_lever_source(monkeypatch, settings, db) -> None:
    src = LeverBoardSource(settings, db, None, tokens=["acme"])
    patch_client(monkeypatch, src, {"postings/acme": [
        {"id": "u-1", "text": "Backend Engineer", "categories": {"location": "Remote"},
         "descriptionPlain": "Build APIs. " * 30, "additionalPlain": "Nice to have: Go",
         "hostedUrl": "https://jobs.lever.co/acme/u-1"},
    ]})
    jobs = await src.discover()
    assert len(jobs) == 1
    assert jobs[0].apply_url == "https://jobs.lever.co/acme/u-1/apply"
    assert "Nice to have: Go" in jobs[0].description
    assert jobs[0].ats == LEVER


@pytest.mark.asyncio
async def test_sources_skip_jobs_already_in_the_database(monkeypatch, settings, db) -> None:
    src = AshbyBoardSource(settings, db, None, tokens=["acme"])
    routes = {"posting-api/job-board/acme": {"jobs": [
        {"id": "abc", "title": "Backend Engineer", "isListed": True, "isRemote": True, "location": "Remote",
         "descriptionPlain": "Own the API. " * 30, "jobUrl": "https://jobs.ashbyhq.com/acme/abc"}]}}
    patch_client(monkeypatch, src, routes)
    first = await src.discover()
    assert len(first) == 1
    db.record(first[0], "submitted")
    assert await src.discover() == []


@pytest.mark.asyncio
async def test_board_source_survives_a_dead_board(monkeypatch, settings, db) -> None:
    src = GreenhouseBoardSource(settings, db, None, tokens=["missing"])
    patch_client(monkeypatch, src, {})            # every request 404s
    assert await src.discover() == []


# ---- llm helpers ----------------------------------------------------------


def test_extract_json_variants() -> None:
    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! Here: {"a": 1} hope that helps') == '{"a": 1}'
    assert json.loads(extract_json('{"a": {"b": "}"}}'))["a"]["b"] == "}"


@pytest.mark.parametrize("bad", ["", "no json here", "{unbalanced"])
def test_extract_json_rejects_garbage(bad) -> None:
    with pytest.raises(LLMError):
        extract_json(bad)


def test_schema_of_inlines_refs() -> None:
    from ai_agent import JobAnalysis

    schema = schema_of(JobAnalysis)
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    bullets = schema["properties"]["tailored_bullets"]["items"]
    assert "properties" in bullets and "bullets" in bullets["properties"]


def test_build_provider_rejects_unknown(tmp_path: Path) -> None:
    from llm import build_provider

    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        build_provider(Settings(llm_provider="chatgpt-please", db_path=tmp_path / "x.db"))


def test_providers_require_their_key(tmp_path: Path) -> None:
    from llm import build_provider

    with pytest.raises(LLMError, match="NVIDIA_API_KEY"):
        build_provider(Settings(llm_provider="nvidia", nvidia_api_key="", db_path=tmp_path / "x.db"))
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        build_provider(Settings(llm_provider="gemini", gemini_api_key="", db_path=tmp_path / "x.db"))


def test_active_model_follows_provider(tmp_path: Path) -> None:
    s = Settings(llm_provider="nvidia", nvidia_model="nv/x", gemini_model="gem/y", db_path=tmp_path / "x.db")
    assert s.active_model == "nv/x"
    s.llm_provider = "gemini"
    assert s.active_model == "gem/y"
