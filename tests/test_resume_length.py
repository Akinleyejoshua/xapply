"""How much of a CV survives, and whether you are told when something did not.

Regression: fitting every resume onto one page silently dropped 2 of 5 roles and 7 of 8
projects. Losing a job from a CV is not a formatting decision, so the page budget is now
a setting, work history is protected, and anything left out is recorded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_agent import JobAnalysis, ScreeningAnswer, TailoredBulletGroup  # noqa: E402
from config import Settings  # noqa: E402
from models import JobPosting  # noqa: E402
from pipeline import load_profile  # noqa: E402
from resume_builder import ResumeBuilder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(db_path=tmp_path / "t.db", output_dir=tmp_path, log_dir=tmp_path,
                    audit_dir=tmp_path, user_data_dir=tmp_path, overrides_path=tmp_path / "s.json",
                    template_dir=ROOT / "templates", profile_path=ROOT / "profile.example.json",
                    **kw)


@pytest.fixture
def profile() -> dict:
    """A career long enough to force the trimming decisions this file is about.

    The shipped example profile has two roles and fits anywhere, so it would never
    exercise the paths that drop content.
    """
    base = load_profile(ROOT / "profile.example.json")
    return {
        **base,
        "experience": [
            {"company": f"Company {i}", "title": "Senior Engineer", "location": "Remote",
             "start": str(2024 - 2 * i), "end": str(2026 - 2 * i),
             "bullets": [f"Delivered project {i}.{j} end to end, raising throughput by {j}0%."
                         for j in range(1, 6)]}
            for i in range(6)
        ],
        "projects": [
            {"name": f"Project {i}", "url": f"https://example.com/{i}",
             "tech": ["Python", "FastAPI"],
             "bullets": [f"Built component {i}.{j} handling real traffic." for j in range(1, 4)]}
            for i in range(10)
        ],
    }


def _analysis(profile: dict) -> JobAnalysis:
    return JobAnalysis(
        job_title="Senior Engineer", company_name="Acme", match_score=80, match_rationale="r",
        missing_requirements=[], tailored_summary="s",
        highlighted_skills=["Python", "FastAPI"],
        tailored_bullets=[TailoredBulletGroup(kind="experience", name=e["company"],
                                              title=e["title"], bullets=e["bullets"])
                          for e in profile["experience"]],
        answers=[ScreeningAnswer(question="Notice period?", answer="2 weeks")])


# ---- the setting -----------------------------------------------------------


def test_two_pages_is_the_default(tmp_path: Path) -> None:
    """One page costs whole roles for anyone past a few years' experience."""
    assert _settings(tmp_path).resume_max_pages == 2


def test_work_history_is_protected_by_default(tmp_path: Path) -> None:
    assert _settings(tmp_path).resume_may_drop_experience is False


def test_page_budget_is_bounded(tmp_path: Path) -> None:
    from pydantic import ValidationError

    for good in (1, 2, 3):
        assert _settings(tmp_path, resume_max_pages=good).resume_max_pages == good
    for bad in (0, 4, -1):
        with pytest.raises(ValidationError):
            _settings(tmp_path, resume_max_pages=bad)


def test_both_settings_are_persisted() -> None:
    from config import PERSISTED_KEYS

    assert "resume_max_pages" in PERSISTED_KEYS
    assert "resume_may_drop_experience" in PERSISTED_KEYS


# ---- what gets trimmed, and in what order ----------------------------------


def test_trimming_costs_the_least_first(tmp_path: Path, profile) -> None:
    builder = ResumeBuilder(_settings(tmp_path))
    context = builder.build_context(profile, _analysis(profile))
    names = [name for name, _ in builder._variants(context)]

    assert names[0] == "nothing trimmed"
    bullets_at = next(i for i, n in enumerate(names) if "bullets per entry" in n)
    projects_at = next(i for i, n in enumerate(names) if "project" in n)
    assert bullets_at < projects_at, "shortening bullets must be tried before dropping projects"


def test_roles_are_never_dropped_unless_allowed(tmp_path: Path, profile) -> None:
    builder = ResumeBuilder(_settings(tmp_path))          # resume_may_drop_experience False
    context = builder.build_context(profile, _analysis(profile))
    full = len(context["experience"])
    for name, variant in builder._variants(context):
        assert len(variant["experience"]) == full, f"variant {name!r} dropped a role"


def test_roles_may_be_dropped_when_you_allow_it(tmp_path: Path, profile) -> None:
    builder = ResumeBuilder(_settings(tmp_path, resume_may_drop_experience=True))
    context = builder.build_context(profile, _analysis(profile))
    variants = builder._variants(context)
    shortest = variants[-1][1]
    assert len(shortest["experience"]) < len(context["experience"])
    # and the variant says which roles it gave up
    assert any("dropped" in name for name, _ in variants)


def test_every_variant_is_labelled(tmp_path: Path, profile) -> None:
    """The label becomes the note on the application, so it has to read plainly."""
    builder = ResumeBuilder(_settings(tmp_path, resume_may_drop_experience=True))
    context = builder.build_context(profile, _analysis(profile))
    for name, _ in builder._variants(context):
        assert name and not name.startswith("_")
        assert "<=" not in name, "labels are shown to the user, not to a developer"


# ---- rendering -------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_pages_keeps_the_whole_profile(tmp_path: Path, profile) -> None:
    from pypdf import PdfReader

    builder = ResumeBuilder(_settings(tmp_path, resume_max_pages=2))
    job = JobPosting(job_id="x", url="https://job-boards.greenhouse.io/acme/jobs/1",
                     title="Senior Engineer", company="Acme")
    pdf = await builder.build(profile, _analysis(profile), job)
    reader = PdfReader(str(pdf))
    assert len(reader.pages) <= 2
    text = "".join(page.extract_text() for page in reader.pages)
    for entry in profile["experience"]:
        assert entry["company"].split(" (")[0] in text, f"{entry['company']} is missing"


@pytest.mark.asyncio
async def test_one_page_still_keeps_every_role(tmp_path: Path, profile) -> None:
    """Even squeezed onto one page, the work history survives; projects go first."""
    from pypdf import PdfReader

    builder = ResumeBuilder(_settings(tmp_path, resume_max_pages=1))
    job = JobPosting(job_id="x", url="https://job-boards.greenhouse.io/acme/jobs/1",
                     title="Senior Engineer", company="Acme")
    pdf = await builder.build(profile, _analysis(profile), job)
    reader = PdfReader(str(pdf))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    for entry in profile["experience"]:
        assert entry["company"].split(" (")[0] in text, f"{entry['company']} was dropped"


@pytest.mark.asyncio
async def test_a_shortened_resume_says_so(tmp_path: Path, profile) -> None:
    builder = ResumeBuilder(_settings(tmp_path, resume_max_pages=1))
    job = JobPosting(job_id="x", url="https://job-boards.greenhouse.io/acme/jobs/1",
                     title="Senior Engineer", company="Acme")
    await builder.build(profile, _analysis(profile), job)
    assert builder.last_trim != "nothing trimmed"
    assert builder.last_trim != "overflow"


def test_the_pipeline_records_what_was_left_out() -> None:
    src = (ROOT / "pipeline.py").read_text()
    assert "self.resumes.last_trim" in src
    assert '"label": "Resume shortened"' in src


def test_the_dashboard_exposes_the_page_budget() -> None:
    html = (ROOT / "static" / "index.html").read_text()
    assert 'id="cfgPages"' in html and 'id="cfgDropExp"' in html
    assert "resume_max_pages" in html
    api = (ROOT / "api.py").read_text()
    assert "resume_max_pages" in api and "resume_may_drop_experience" in api
