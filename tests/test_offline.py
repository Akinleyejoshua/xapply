"""Offline unit tests: no network, no browser, no Gemini calls.

Run with: make test   (or .venv/bin/python -m pytest -q tests)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_agent import JobAnalysis, ScreeningAnswer, TailoredBulletGroup  # noqa: E402
from browser_bot import (  # noqa: E402
    AnswerResolver,
    FormField,
    ResolveContext,
    choose_option,
    extract_number,
    format_number,
    is_placeholder_option,
)
from config import Settings  # noqa: E402
from database import STATUS_PENDING, STATUS_SKIPPED, STATUS_SUBMITTED, Database  # noqa: E402
from models import ASHBY, GREENHOUSE, LEVER, LINKEDIN, JobPosting, detect_ats, job_id_from_url  # noqa: E402
from resume_builder import ResumeBuilder, slugify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile() -> dict:
    return json.loads((ROOT / "profile.example.json").read_text())


@pytest.fixture
def analysis() -> JobAnalysis:
    return JobAnalysis(
        job_title="Senior Python Engineer",
        company_name="Acme Corp",
        match_score=82,
        match_rationale="Strong Python and AWS overlap.",
        missing_requirements=["Go"],
        tailored_summary="Backend engineer with six years of Python. Ships distributed services on AWS.",
        highlighted_skills=["Python", "FastAPI", "PostgreSQL", "AWS (ECS, Lambda, S3, RDS)", "Rust"],
        tailored_bullets=[
            TailoredBulletGroup(kind="experience", name="Acme Analytics", title="Senior Backend Engineer",
                                bullets=["Built FastAPI services handling 40M requests/day."]),
            TailoredBulletGroup(kind="experience", name="Ghost Company", title="Imaginary",
                                bullets=["Invented something that never happened."]),
        ],
        answers=[
            ScreeningAnswer(question="How many years of experience do you have with Python?", answer="6"),
            ScreeningAnswer(question="Do you require visa sponsorship?", answer="No"),
            ScreeningAnswer(question="What is your notice period?", answer="2 weeks"),
        ],
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(gemini_api_key="test-key", db_path=tmp_path / "t.db", output_dir=tmp_path / "out",
                    log_dir=tmp_path / "logs", audit_dir=tmp_path / "logs" / "apps",
                    user_data_dir=tmp_path / "prof", template_dir=ROOT / "templates")


# ---- models ---------------------------------------------------------------


@pytest.mark.parametrize("url,ats", [
    ("https://www.linkedin.com/jobs/view/4012345678/", LINKEDIN),
    ("https://boards.greenhouse.io/acme/jobs/5551234", GREENHOUSE),
    ("https://job-boards.greenhouse.io/acme/jobs/5551234", GREENHOUSE),
    ("https://jobs.lever.co/acme/1a2b3c4d-5e6f", LEVER),
    ("https://jobs.ashbyhq.com/acme/9f8e7d6c", ASHBY),
    ("https://careers.acme.com/job/123?gh_jid=778899", GREENHOUSE),
    ("https://example.com/careers/42", "unknown"),
])
def test_detect_ats(url: str, ats: str) -> None:
    assert detect_ats(url) == ats


def test_job_id_is_stable_and_unique() -> None:
    a = job_id_from_url("https://www.linkedin.com/jobs/view/4012345678/?refId=x")
    assert a == "4012345678"
    lever = job_id_from_url("https://jobs.lever.co/acme/1a2b3c4d")
    assert lever == "lever-1a2b3c4d"
    assert job_id_from_url("https://x.com/a") != job_id_from_url("https://x.com/b")
    assert job_id_from_url("https://x.com/a/") == job_id_from_url("https://x.com/a")


# ---- option matching ------------------------------------------------------


@pytest.mark.parametrize("answer,options,expected", [
    ("Yes", ["Select an option", "Yes", "No"], "Yes"),
    ("no", ["Yes", "No"], "No"),
    ("I do not wish to answer", ["Yes", "No", "I do not wish to answer"], "I do not wish to answer"),
    ("6", ["1", "2", "5", "6", "7"], "6"),
    ("7", ["1", "2", "5", "10"], "5"),                                   # nearest single number
    ("Bachelor's Degree", ["High School", "Bachelor's Degree", "Master's Degree"], "Bachelor's Degree"),
    ("Remote", ["On-site", "Hybrid", "Remote"], "Remote"),
    ("something unrelated zzz", ["Yes", "No"], None),
])
def test_choose_option(answer, options, expected) -> None:
    assert choose_option(answer, options) == expected


@pytest.mark.parametrize("answer,options,expected", [
    ("6", ["0-2 years", "3-5 years", "5-10 years"], "5-10 years"),       # falls inside the range
    ("2", ["0-2 years", "3-5 years", "5-10 years"], "0-2 years"),        # boundary belongs to the lower band
    ("4", ["0-2 years", "3-5 years", "5-10 years"], "3-5 years"),
    ("12", ["0-2 years", "3-5 years", "10+ years"], "10+ years"),        # open-ended upper band
    ("1", ["Less than 2 years", "2-5 years", "5+ years"], "Less than 2 years"),
    ("20", ["0-2 years", "3-5 years"], "3-5 years"),                     # out of range -> nearest band
])
def test_choose_option_numeric_ranges(answer, options, expected) -> None:
    assert choose_option(answer, options) == expected


@pytest.mark.parametrize("option,is_ph", [
    ("Select an option", True), ("Please choose one", True), ("---", True), ("N/A", True),
    ("", True), ("Select...", True), ("Yes", False), ("No", False), ("Bachelor's Degree", False),
])
def test_placeholder_detection(option, is_ph) -> None:
    assert is_placeholder_option(option) is is_ph


def test_choose_option_ignores_placeholder() -> None:
    assert choose_option("Yes", ["Select...", "Yes", "No"]) == "Yes"
    assert choose_option("anything", ["Select..."]) is None


def test_number_helpers() -> None:
    assert extract_number("about 6 years") == 6
    assert extract_number("no digits") is None
    assert format_number(6.0) == "6"
    assert format_number(2.5) == "2.5"


# ---- answer resolution ----------------------------------------------------


def _field(label: str, kind: str = "text", options: list[str] | None = None, required: bool = True) -> FormField:
    return FormField(kind=kind, label=label, idx="x0", required=required,
                     options=[{"label": o, "value": o, "idx": "", "dom_id": ""} for o in (options or [])])


@pytest.mark.asyncio
async def test_resolver_uses_profile_fields(profile, analysis, settings) -> None:
    r = AnswerResolver(None, settings)
    ctx = ResolveContext(profile, JobPosting.from_url("https://jobs.lever.co/a/b"), analysis)
    assert (await r.resolve(_field("First name"), ctx)).value == "Jane"
    assert (await r.resolve(_field("Last Name"), ctx)).value == "Doe"
    assert (await r.resolve(_field("Email address"), ctx)).value == "jane.doe@example.com"
    assert (await r.resolve(_field("LinkedIn Profile"), ctx)).value == profile["linkedin"]
    assert (await r.resolve(_field("Current company"), ctx)).value == "Acme Analytics"


@pytest.mark.asyncio
async def test_resolver_years_of_experience(profile, analysis, settings) -> None:
    r = AnswerResolver(None, settings)
    ctx = ResolveContext(profile, JobPosting.from_url("https://jobs.lever.co/a/b"), analysis)
    assert (await r.resolve(_field("How many years of experience do you have with Python?"), ctx)).value == "6"
    assert (await r.resolve(_field("Years of experience with Kubernetes"), ctx)).value == "2"


@pytest.mark.asyncio
async def test_resolver_screening_defaults(profile, analysis, settings) -> None:
    r = AnswerResolver(None, settings)
    ctx = ResolveContext(profile, JobPosting.from_url("https://jobs.lever.co/a/b"), analysis)
    sponsor = await r.resolve(_field("Will you now or in the future require sponsorship?", "radio", ["Yes", "No"]), ctx)
    assert sponsor.value == "No"
    notice = await r.resolve(_field("What is your notice period?"), ctx)
    assert notice.value == "2 weeks"


@pytest.mark.asyncio
async def test_resolver_escalates_when_nothing_matches(profile, analysis, settings) -> None:
    r = AnswerResolver(None, settings)  # no AI agent configured
    ctx = ResolveContext(profile, JobPosting.from_url("https://jobs.lever.co/a/b"), analysis)
    out = await r.resolve(_field("Describe a time you resolved a production incident"), ctx)
    assert out.needs_human and out.value is None


# ---- hallucination guard --------------------------------------------------


def test_resume_context_drops_invented_content(profile, analysis, settings) -> None:
    ctx = ResumeBuilder(settings).build_context(profile, analysis)
    companies = [e["company"] for e in ctx["experience"]]
    assert "Ghost Company" not in companies                      # invented company dropped
    acme = next(e for e in ctx["experience"] if e["company"] == "Acme Analytics")
    assert acme["bullets"] == ["Built FastAPI services handling 40M requests/day."]
    bright = next(e for e in ctx["experience"] if e["company"] == "Bright Labs")
    assert bright["bullets"] == profile["experience"][1]["bullets"]  # untouched entries keep originals
    core = next(g for g in ctx["skill_groups"] if g["label"] == "Core")
    assert "Rust" not in core["items"]                            # invented skill dropped
    assert "Python" in core["items"]


def test_resume_renders_html(profile, analysis, settings) -> None:
    html = ResumeBuilder(settings).render_html(profile, analysis)
    assert "Jane Doe" in html and "Acme Analytics" in html
    assert "Ghost Company" not in html and "Rust" not in html


def test_slugify() -> None:
    assert slugify("Acme, Inc.") == "Acme_Inc"
    assert slugify("Senior Python Engineer (Remote)") == "Senior_Python_Engineer_Remote"
    assert slugify("") == "Unknown"


# ---- database -------------------------------------------------------------


def test_database_dedupes_and_updates(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init()
    job = JobPosting(job_id="1", url="https://www.linkedin.com/jobs/view/1/", title="Eng",
                     company="Acme", source=LINKEDIN, ats=LINKEDIN)
    assert not db.has_job(LINKEDIN, "1")
    first = db.record(job, STATUS_PENDING, match_score=80, answers=[{"label": "Email", "value": "a@b.c"}])
    assert db.has_job(LINKEDIN, "1")
    second = db.record(job, STATUS_SUBMITTED, notes="done")
    assert first == second                                        # one row per job, not two
    row = db.get(first)
    assert row["status"] == STATUS_SUBMITTED
    assert row["match_score"] == 80                               # preserved, not overwritten by None
    assert row["answers"][0]["label"] == "Email"
    assert len(db.list()) == 1


def test_database_stats_and_filters(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init()
    for i, (st, score, ats) in enumerate([(STATUS_SUBMITTED, 90, LEVER), (STATUS_SKIPPED, 30, LINKEDIN),
                                          (STATUS_SUBMITTED, 70, GREENHOUSE)]):
        db.record(JobPosting(job_id=str(i), url=f"https://x/{i}", company=f"C{i}", title="Dev",
                             source="urls", ats=ats), st, match_score=score)
    s = db.stats()
    assert s["total"] == 3
    assert s["by_status"][STATUS_SUBMITTED] == 2
    assert s["avg_match_score"] == pytest.approx(63.3, abs=0.1)
    assert s["submitted_by_ats"] == {LEVER: 1, GREENHOUSE: 1}
    assert len(db.list(status=STATUS_SUBMITTED)) == 2
    assert len(db.list(search="C1")) == 1


# ---- config ---------------------------------------------------------------


def test_settings_parse_csv_lists(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_QUERIES", "Python Developer, Backend Engineer ,ML Engineer")
    monkeypatch.setenv("SOURCES", "linkedin,urls")
    monkeypatch.setenv("AUTO_SUBMIT", "true")
    s = Settings(_env_file=None)
    assert s.search_queries == ["Python Developer", "Backend Engineer", "ML Engineer"]
    assert s.sources == ["linkedin", "urls"]
    assert s.auto_submit is True


# ---- api ------------------------------------------------------------------


def test_admin_api(tmp_path: Path, settings: Settings) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    db = Database(settings.db_path)
    db.init()
    job = JobPosting(job_id="7", url="https://jobs.lever.co/acme/7", company="Acme", title="Dev",
                     source="urls", ats=LEVER)
    app_id = db.record(job, STATUS_SUBMITTED, match_score=88,
                       answers=[{"label": "Email", "value": "jane@x.com", "source": "profile", "ok": True}],
                       analysis={"match_score": 88, "tailored_summary": "ok"})
    client = TestClient(create_app(settings, db))
    assert client.get("/health").json()["ok"] is True
    assert client.get("/api/stats").json()["by_status"]["submitted"] == 1
    rows = client.get("/api/applications").json()
    assert len(rows) == 1 and rows[0]["company"] == "Acme"
    detail = client.get(f"/api/applications/{app_id}").json()
    assert detail["answers"][0]["value"] == "jane@x.com"
    assert detail["analysis"]["match_score"] == 88
    assert client.get("/api/applications/9999").status_code == 404
    patched = client.patch(f"/api/applications/{app_id}", json={"status": "failed", "notes": "manual"})
    assert patched.json()["status"] == "failed"
    assert "text/csv" in client.get("/api/export.csv").headers["content-type"]
    assert "XApply" in client.get("/").text


def test_admin_api_requires_token(settings: Settings) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    settings.admin_token = "s3cret"
    db = Database(settings.db_path)
    db.init()
    client = TestClient(create_app(settings, db))
    assert client.get("/api/stats").status_code == 401
    assert client.get("/api/stats", headers={"X-Admin-Token": "s3cret"}).status_code == 200
    assert client.get("/api/stats?token=s3cret").status_code == 200


# ---- admin api: control + settings endpoints ------------------------------


def test_api_config_roundtrip(settings: Settings) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    db = Database(settings.db_path)
    db.init()
    client = TestClient(create_app(settings, db))

    cfg = client.get("/api/config").json()
    assert cfg["llm_provider"] in ("gemini", "nvidia")
    assert "greenhouse" in cfg["known_sources"] and "linkedin" in cfg["known_sources"]

    patched = client.patch("/api/config", json={
        "llm_provider": "nvidia", "nvidia_model": "nvidia/test-model",
        "match_threshold": 80, "auto_submit": True,
        "sources": ["greenhouse", "ashby"], "search_queries": ["Backend Engineer"],
    }).json()
    assert patched["llm_provider"] == "nvidia"
    assert patched["active_model"] == "nvidia/test-model"
    assert patched["match_threshold"] == 80
    assert patched["auto_submit"] is True
    assert patched["sources"] == ["greenhouse", "ashby"]
    assert settings.auto_submit is True      # the live Settings object really changed

    assert client.patch("/api/config", json={"match_threshold": 500}).status_code == 422
    assert client.patch("/api/config", json={"llm_provider": "hal9000"}).status_code == 422


def test_api_profile_read_write(settings: Settings, tmp_path: Path, profile: dict) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    settings.profile_path = tmp_path / "profile.json"
    settings.profile_path.write_text(json.dumps(profile), encoding="utf-8")
    db = Database(settings.db_path)
    db.init()
    client = TestClient(create_app(settings, db))

    assert client.get("/api/profile").json()["name"] == "Jane Doe"
    updated = {**profile, "headline": "Staff Engineer"}
    assert client.put("/api/profile", json=updated).json()["saved"] is True
    assert json.loads(settings.profile_path.read_text())["headline"] == "Staff Engineer"
    assert (tmp_path / "profile.json.bak").exists()           # previous version kept
    assert client.put("/api/profile", json={"name": ""}).status_code == 400


def test_api_companies_crud(settings: Settings, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    settings.company_file = tmp_path / "companies.json"
    settings.company_file.write_text(json.dumps({"greenhouse": ["acme"]}), encoding="utf-8")
    db = Database(settings.db_path)
    db.init()
    client = TestClient(create_app(settings, db))

    assert client.get("/api/companies").json() == {"greenhouse": ["acme"]}
    after_add = client.post("/api/companies", json={"ats": "ashby", "token": "widgets"}).json()
    assert after_add["ashby"] == ["widgets"]
    assert client.post("/api/companies", json={"ats": "ashby", "token": "widgets"}).json()["ashby"] == ["widgets"]
    assert client.delete("/api/companies/ashby/widgets").json()["ashby"] == []
    assert client.post("/api/companies", json={"ats": "workday", "token": "x"}).status_code == 422


def test_api_run_status_and_stop(settings: Settings) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    db = Database(settings.db_path)
    db.init()
    client = TestClient(create_app(settings, db))
    st = client.get("/api/run").json()
    assert st["running"] is False and st["kind"] == ""
    assert "provider" in st and "model" in st and isinstance(st["log"], list)
    assert client.post("/admin/stop").json()["stopped"] is False
    assert client.get("/api/gate").json()["paused"] is False
    assert client.get("/api/discovered").json() == []
    assert client.post("/admin/apply-selected", json={"urls": []}).status_code == 400


def test_dashboard_is_served(settings: Settings) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    db = Database(settings.db_path)
    db.init()
    html = TestClient(create_app(settings, db)).get("/").text
    for marker in ('id="view-dash"', 'id="view-scan"', 'id="view-settings"',
                   'id="runModel"', 'id="cfgModel"', "/admin/discover"):
        assert marker in html, marker
