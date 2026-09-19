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
                    overrides_path=tmp_path / "settings.local.json",
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
    ("Machine Learning Scientist", True),
    ("Backend Engineer, Billing", True),
    ("Backend Software Engineer", True),
    ("Sales Engineer", False),                      # 'engineer' alone is not enough
    ("Solutions Engineer", False),
    ("Abuse Investigator", False),
    ("Engineering Manager", False),
])
def test_title_matches_on_the_distinctive_words(title, expected) -> None:
    assert title_matches(title, ["Machine Learning Engineer", "Backend Engineer"]) is expected


def test_title_matches_without_queries_accepts_everything() -> None:
    assert title_matches("Anything At All", []) is True


def test_query_tokens_keeps_the_terms_as_typed() -> None:
    assert query_tokens(["Senior Backend Engineer", "  ", "Data Analytics"]) == \
        ["Senior Backend Engineer", "Data Analytics"]


@pytest.mark.parametrize("text,remote_only,workplace_type,expected", [
    ("Remote - United States", True, None, True),
    ("San Francisco, CA", True, None, False),
    ("San Francisco, CA", True, "Remote", True),     # workplaceType is authoritative
    ("Remote", True, "Hybrid", False),               # ...even against the location text
    ("Anywhere", True, None, True),
    ("Berlin", False, None, True),                   # not filtering on remote at all
])
def test_location_matches(text, remote_only, workplace_type, expected) -> None:
    assert location_matches(text, "Remote", remote_only, workplace_type) is expected


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


def test_a_stray_brace_before_the_object_is_stepped_over() -> None:
    """Seen live from NVIDIA: the model opens a brace, then starts the object again."""
    raw = '{\n{\n  "answer": "Corvendra",\n  "confidence": 0.9,\n  "needs_human": false'
    assert json.loads(extract_json(raw)) == {
        "answer": "Corvendra", "confidence": 0.9, "needs_human": False}


def test_an_object_cut_off_between_fields_is_closed_and_kept() -> None:
    """Four retries of a half-written response leave the field blank; closing it does not."""
    assert json.loads(extract_json('{"answer": "Full answer.", "confidence": 0.8,')) == {
        "answer": "Full answer.", "confidence": 0.8}
    assert json.loads(extract_json('{"a": {"b": 1}, "c": 2')) == {"a": {"b": 1}, "c": 2}


def test_an_answer_cut_off_mid_sentence_is_refused() -> None:
    """Half a sentence must not reach someone's application, so this one is retried."""
    with pytest.raises(LLMError):
        extract_json('{"answer": "I defined a metric that')


def test_repair_never_degrades_to_an_empty_object() -> None:
    """Stepping back far enough always reaches '{}', which parses and says nothing."""
    with pytest.raises(LLMError):
        extract_json("{ {")


def test_a_missing_reasoning_does_not_cost_a_retry() -> None:
    """Seen live: the model answered well but left out the field used only for logging."""
    from ai_agent import FieldAnswer

    fa = FieldAnswer.model_validate(
        {"answer": "I defined a data-quality score.", "confidence": 0.95, "needs_human": False})
    assert fa.answer.startswith("I defined") and fa.reasoning == ""

    # The two decisions are still required: guessing either one answers for the candidate.
    with pytest.raises(Exception):
        FieldAnswer.model_validate({"answer": "x", "confidence": 0.9})


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


# ---- remote detection -----------------------------------------------------


@pytest.mark.parametrize("location,workplace_type,expected", [
    ("US - Remote", "Remote", True),
    ("San Francisco", "Remote", True),          # workplaceType is authoritative
    ("San Francisco", "Hybrid", False),         # Ashby flags hybrid roles isRemote=true
    ("Sao Paulo", "Hybrid", False),
    ("London", "OnSite", False),
    ("Remote, United States", None, True),
    ("Remote, Bangalore", None, True),
    ("Anywhere", None, True),
    ("Hybrid - NYC", None, False),              # the word hybrid overrides
    ("San Francisco", None, False),
    ("", None, False),
])
def test_looks_remote(location, workplace_type, expected) -> None:
    from discovery import looks_remote

    assert looks_remote(location, workplace_type) is expected


def test_location_matches_without_remote_only() -> None:
    assert location_matches("Berlin", "Berlin", False) is True
    assert location_matches("Berlin", "Lisbon", False) is False
    assert location_matches("Berlin", "", False) is True            # no preference set
    assert location_matches("Remote, EU", "Lisbon", False) is True  # remote satisfies any location


# ---- seniority ------------------------------------------------------------


@pytest.mark.parametrize("title,level", [
    ("Software Engineering Intern", "intern"),
    ("Summer 2026 Intern, Backend", "intern"),
    ("Junior Developer", "junior"),
    ("Jr. Data Analyst", "junior"),
    ("Associate Product Manager", "junior"),        # junior prefix beats "manager"
    ("New Grad Software Engineer", "junior"),
    ("Entry-Level Developer", "junior"),
    ("Backend Engineer", "mid"),
    ("Full Stack Developer", "mid"),
    ("Software Engineer II", "mid"),
    ("Senior Backend Engineer", "senior"),
    ("Sr. Data Scientist", "senior"),
    ("Staff Software Engineer", "lead"),
    ("Senior Staff Engineer", "lead"),
    ("Principal Engineer", "lead"),
    ("Engineering Manager", "lead"),
    ("Senior Engineering Manager", "lead"),
    ("Head of Platform", "lead"),
    ("Solutions Architect", "lead"),
    ("Software Engineer, Ads Manager", "mid"),      # "Manager" is a product name here
    ("Software Engineer, Fleet Manager", "mid"),
    ("Senior Software Engineer, Package Manager", "senior"),
])
def test_seniority_of(title, level) -> None:
    from discovery import seniority_of

    assert seniority_of(title) == level


def test_seniority_matches_empty_means_any() -> None:
    from discovery import seniority_matches

    assert seniority_matches("Junior Developer", None) is True
    assert seniority_matches("Junior Developer", []) is True
    assert seniority_matches("Junior Developer", ["senior", "lead"]) is False
    assert seniority_matches("Senior Developer", ["senior", "lead"]) is True
    assert seniority_matches("Senior Developer", ["SENIOR"]) is True    # case-insensitive


@pytest.mark.asyncio
async def test_ashby_source_respects_remote_and_seniority(monkeypatch, tmp_path: Path) -> None:
    """A hybrid role flagged isRemote must not survive a remote-only scan."""
    base = dict(output_dir=tmp_path, log_dir=tmp_path, audit_dir=tmp_path, user_data_dir=tmp_path,
                template_dir=ROOT / "templates", company_file=ROOT / "companies.json",
                overrides_path=tmp_path / "settings.local.json",
                search_queries=["Software Engineer"], max_jobs_per_company=50, discovery_delay_s=0)
    payload = {"jobs": [
        {"id": "a", "title": "Senior Software Engineer", "isListed": True, "isRemote": True,
         "location": "San Francisco", "workplaceType": "Hybrid",
         "descriptionPlain": "x " * 200, "jobUrl": "https://jobs.ashbyhq.com/acme/a"},
        {"id": "b", "title": "Software Engineer", "isListed": True, "isRemote": True,
         "location": "US - Remote", "workplaceType": "Remote",
         "descriptionPlain": "y " * 200, "jobUrl": "https://jobs.ashbyhq.com/acme/b"},
        {"id": "c", "title": "Staff Software Engineer", "isListed": True, "isRemote": True,
         "location": "US - Remote", "workplaceType": "Remote",
         "descriptionPlain": "z " * 200, "jobUrl": "https://jobs.ashbyhq.com/acme/c"},
    ]}

    async def discover(settings: Settings):
        d = Database(settings.db_path)
        d.init()
        src = AshbyBoardSource(settings, d, None, tokens=["acme"])
        patch_client(monkeypatch, src, {"posting-api/job-board/acme": payload})
        return await src.discover()

    everything = await discover(Settings(db_path=tmp_path / "1.db", **base))
    assert {j.job_id[-1] for j in everything} == {"a", "b", "c"}

    remote = await discover(Settings(db_path=tmp_path / "2.db", remote_only=True, **base))
    assert {j.job_id[-1] for j in remote} == {"b", "c"}          # the hybrid one is gone

    leads = await discover(Settings(db_path=tmp_path / "3.db", seniority_levels=["lead"], **base))
    assert {j.job_id[-1] for j in leads} == {"c"}

    both = await discover(Settings(db_path=tmp_path / "4.db", remote_only=True,
                                   seniority_levels=["mid"], **base))
    assert {j.job_id[-1] for j in both} == {"b"}


def test_settings_seniority_accepts_csv(monkeypatch) -> None:
    monkeypatch.setenv("SENIORITY_LEVELS", "mid, senior ,lead")
    assert Settings(_env_file=None).seniority_levels == ["mid", "senior", "lead"]


# ---- title matching ---------------------------------------------------------
#
# The scorer itself lives in matching.py and is covered by tests/test_matching.py.
# What matters here is how discovery uses it.


@pytest.mark.parametrize("title,expected", [
    # unmistakable hits
    ("Data Analyst", True),
    ("Senior Data Analyst", True),
    ("Financial Data Analyst", True),
    ("Data Analytics Lead", True),
    # adjacent data roles: kept on purpose, because the model scores them properly
    # afterwards and a wrongly dropped posting is never seen again
    ("Data Engineer", True),
    ("Data Scientist, Fraud", True),
    ("Analytics Engineer Intern", True),
    # a sales role that happens to say "Data" scores 0.50 and survives discovery.
    # That is the intended trade: the model reads the description and rejects it,
    # which is cheap, whereas a wrongly dropped posting is never seen again.
    ("Account Executive, Product Sales (Data)", True),
    # genuinely unrelated
    ("Software Engineer", False),
    ("Accounting Intern", False),
    ("University Recruiter", False),
])
def test_data_analytics_query_at_the_default_threshold(title, expected) -> None:
    assert title_matches(title, ["Data Analytics", "Data Analysis"]) is expected


def test_a_stricter_threshold_narrows_to_exact_roles() -> None:
    queries = ["Data Analytics", "Data Analysis"]
    assert title_matches("Data Analyst", queries, 0.8) is True
    assert title_matches("Data Engineer", queries, 0.8) is False
    assert title_matches("Analytics Engineer Intern", queries, 0.8) is False


def test_generic_words_do_not_widen_a_search() -> None:
    assert title_matches("Backend Engineer", ["Backend Engineer"]) is True
    assert title_matches("Sales Engineer", ["Backend Engineer"]) is False
    assert title_matches("Engineering Manager", ["Backend Engineer"]) is False


# ---- scan diagnostics -------------------------------------------------------


def test_scan_stats_summary_and_reasons() -> None:
    from discovery import ScanStats

    s = ScanStats(seen=100, dropped_title=90, dropped_seniority=8, kept=2)
    assert s.dropped == 98
    assert dict(s.reasons()) == {"search terms": 90, "seniority": 8}
    assert "2 kept of 100" in s.summary()

    empty = ScanStats()
    assert "no postings" in empty.summary()
    assert empty.reasons() == []


def test_scan_stats_add() -> None:
    from discovery import ScanStats

    total = ScanStats()
    total += ScanStats(seen=10, dropped_title=8, kept=2)
    total += ScanStats(seen=5, dropped_location=4, kept=1)
    assert total.seen == 15 and total.kept == 3
    assert total.dropped_title == 8 and total.dropped_location == 4


def test_explain_empty_scan_names_the_responsible_filter(settings: Settings) -> None:
    from discovery import ScanStats, explain_empty_scan

    settings.search_queries = ["Data Analytics"]
    settings.seniority_levels = ["intern"]
    settings.countries = ["Nigeria"]

    tips = explain_empty_scan(ScanStats(seen=2146, dropped_title=2136, dropped_seniority=10), settings)
    assert any("Data Analytics" in t and "search terms" in t for t in tips)
    assert any("intern" in t for t in tips)

    tips = explain_empty_scan(ScanStats(seen=100, dropped_location=100), settings)
    assert any("Nigeria" in t for t in tips)

    # a thin result is explained too: one posting out of two thousand looks like the
    # search terms when it is usually the country filter
    thin = explain_empty_scan(ScanStats(seen=2708, kept=1, dropped_location=1238), settings)
    assert any("Only 1 of 2708" in t for t in thin)
    assert any("Nigeria" in t for t in thin)

    # nothing to explain once a scan returns a useful number
    assert explain_empty_scan(ScanStats(seen=2708, kept=227, dropped_title=1469), settings) == []
    # or when no board answered at all
    assert "companies --probe" in explain_empty_scan(ScanStats(), settings)[0]


@pytest.mark.asyncio
async def test_board_source_records_why_postings_dropped(monkeypatch, settings, db) -> None:
    settings.search_queries = ["Data Analytics"]
    settings.seniority_levels = ["senior"]
    src = GreenhouseBoardSource(settings, db, None, tokens=["acme"])
    patch_client(monkeypatch, src, {
        "/boards/acme/jobs/1": {"id": 1, "title": "Senior Data Analyst", "company_name": "Acme",
                                "location": {"name": "Remote"}, "content": "Own reporting. " * 30,
                                "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"},
        "/boards/acme/jobs": {"jobs": [
            {"id": 1, "title": "Senior Data Analyst", "location": {"name": "Remote"},
             "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"},
            {"id": 2, "title": "Data Analyst", "location": {"name": "Remote"},       # wrong level
             "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/2"},
            {"id": 3, "title": "Office Manager", "location": {"name": "Remote"},     # wrong title
             "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/3"},
        ]},
    })
    jobs = await src.discover()
    assert len(jobs) == 1
    assert src.stats.seen == 3
    assert src.stats.kept == 1
    assert src.stats.dropped_title == 1
    assert src.stats.dropped_seniority == 1


@pytest.mark.asyncio
async def test_stats_count_jobs_already_in_the_database(monkeypatch, settings, db) -> None:
    settings.search_queries = ["Data Analytics"]
    routes = {
        "/boards/acme/jobs/1": {"id": 1, "title": "Data Analyst", "company_name": "Acme",
                                "location": {"name": "Remote"}, "content": "Own reporting. " * 30,
                                "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"},
        "/boards/acme/jobs": {"jobs": [{"id": 1, "title": "Data Analyst", "location": {"name": "Remote"},
                                        "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"}]},
    }
    src = GreenhouseBoardSource(settings, db, None, tokens=["acme"])
    patch_client(monkeypatch, src, routes)
    first = await src.discover()
    db.record(first[0], "submitted")

    again = GreenhouseBoardSource(settings, db, None, tokens=["acme"])
    patch_client(monkeypatch, again, routes)
    assert await again.discover() == []
    assert again.stats.dropped_seen_before == 1


def test_dashboard_shows_scan_diagnostics() -> None:
    html = (ROOT / "static" / "index.html").read_text()
    for marker in ('id="scanDiag"', "function renderScanDiag()", "/api/scan-stats", "LAST_KIND"):
        assert marker in html, marker


# ---- results as they arrive, and a scan you stop ---------------------------
#
# Regression: a scan over thirty company boards takes minutes, and stopping it threw
# away everything it had gathered. Results are now published board by board.


@pytest.mark.asyncio
async def test_sources_report_each_board_as_it_finishes(monkeypatch, settings, db) -> None:
    settings.search_queries = ["Data Analytics"]
    batches: list[tuple[str, int]] = []

    def on_batch(name, jobs):
        batches.append((name, len(jobs)))

    # The stub matches on a URL fragment, and the list path is a prefix of the detail
    # path, so the detail routes have to be registered first.
    routes = {}
    for token, ids in (("acme", [1, 2]), ("beta", [3])):
        for i in ids:
            routes[f"/boards/{token}/jobs/{i}"] = {
                "id": i, "title": "Data Analyst", "company_name": token.title(),
                "location": {"name": "Remote"}, "content": "Own reporting. " * 30,
                "absolute_url": f"https://job-boards.greenhouse.io/{token}/jobs/{i}"}
    for token, ids in (("acme", [1, 2]), ("beta", [3])):
        routes[f"/boards/{token}/jobs"] = {"jobs": [
            {"id": i, "title": "Data Analyst", "location": {"name": "Remote"},
             "absolute_url": f"https://job-boards.greenhouse.io/{token}/jobs/{i}"} for i in ids]}

    src = GreenhouseBoardSource(settings, db, None, tokens=["acme", "beta"])
    src.on_batch = on_batch
    patch_client(monkeypatch, src, routes)
    jobs = await src.discover()

    assert len(jobs) == 3
    assert batches == [("greenhouse", 2), ("greenhouse", 1)], \
        "each board must report as soon as it finishes, not all at the end"
    assert sum(n for _, n in batches) == len(jobs)


@pytest.mark.asyncio
async def test_a_board_with_no_matches_reports_nothing(monkeypatch, settings, db) -> None:
    settings.search_queries = ["Data Analytics"]
    calls: list[int] = []
    src = GreenhouseBoardSource(settings, db, None, tokens=["acme"])
    src.on_batch = lambda name, jobs: calls.append(len(jobs))
    patch_client(monkeypatch, src, {"/boards/acme/jobs": {"jobs": [
        {"id": 1, "title": "Office Manager", "location": {"name": "Remote"},
         "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"}]}})
    assert await src.discover() == []
    assert calls == [], "an empty board must not fire an empty batch"


@pytest.mark.asyncio
async def test_a_failing_progress_callback_does_not_stop_the_scan(monkeypatch, settings, db) -> None:
    settings.search_queries = ["Data Analytics"]
    src = GreenhouseBoardSource(settings, db, None, tokens=["acme"])

    def explode(name, jobs):
        raise RuntimeError("the UI went away")

    src.on_batch = explode
    patch_client(monkeypatch, src, {
        "/boards/acme/jobs/1": {"id": 1, "title": "Data Analyst", "company_name": "Acme",
                                "location": {"name": "Remote"}, "content": "Own reporting. " * 30,
                                "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"},
        "/boards/acme/jobs": {"jobs": [{"id": 1, "title": "Data Analyst",
                                        "location": {"name": "Remote"},
                                        "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1"}]}})
    assert len(await src.discover()) == 1


def test_api_publishes_partial_results_and_survives_a_stop() -> None:
    src = (ROOT / "api.py").read_text()
    go = src[src.index("async def _go() -> None:\n            from browser_bot import HumanGate"):]
    go = go[:go.index("start(\"discover\"")]
    assert "def publish()" in go
    assert "src.on_batch = on_batch" in go
    assert "except asyncio.CancelledError:" in go
    assert "keeping the" in go, "a stopped scan must say what it kept"
    # the counters shown must include the source still running
    assert "def live_totals()" in go


def test_dashboard_refreshes_while_a_scan_runs() -> None:
    html = (ROOT / "static" / "index.html").read_text()
    poll = html[html.index("async function poll()"):]
    poll = poll[:poll.index("async function release()")]
    assert "st.discovered !== FOUND.length" in poll, "results must refresh as the count moves"
    assert "found so far" in html
    assert "still scanning" in html
