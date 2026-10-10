"""Sites that are not Greenhouse, Lever, Ashby or LinkedIn.

`apply.remote.com` carries the whole advert and no form at all: the application is a
dialog the page's Apply button opens, so a bot that waits for a form to appear on its
own waits forever. Every other careers site with no driver of its own is reached the
same way, through one fallback that opens the posting, follows the Apply trigger and
treats whatever the site calls an application as the form. A page with nowhere to
apply is skipped rather than filled, which is what keeps the email route open for it.

Run with: make test   (or .venv/bin/python -m pytest -q tests)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_agent import JobAnalysis, ScreeningAnswer, TailoredBulletGroup  # noqa: E402
from appliers import (NO_FORM_NOTE, AnySiteApplier, GreenhouseApplier,  # noqa: E402
                      RemoteApplier, get_applier)
from browser_bot import AnswerResolver, FormFiller, HumanGate, StealthBrowser  # noqa: E402
from config import Settings  # noqa: E402
from database import STATUS_SKIPPED  # noqa: E402
from models import (ASHBY, GREENHOUSE, LEVER, LINKEDIN, REMOTE, UNKNOWN,  # noqa: E402
                    ApplyResult, JobPosting, detect_ats, job_id_from_url)
from pipeline import load_profile  # noqa: E402
from resume_builder import ResumeBuilder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = ("We are hiring an engineer to build reliable systems. " * 8)

#: A page shaped like apply.remote.com: the advert, an Apply button, and the
#: application itself only inside the dialog that button opens.
REMOTE_HTML = """<!doctype html><html><body>
<h1>Senior Backend Engineer</h1>
<p>%(desc)s</p>
<button id="apply">Apply</button>
<div id="dialog" role="dialog" aria-label="Apply for Senior Backend Engineer" hidden>
  <form id="application-form">
    <label for="first">First name</label><input id="first" name="first">
    <label for="last">Last name</label><input id="last" name="last">
    <label for="email">Email address</label><input id="email" name="email" type="email">
    <label for="resume">Resume</label>
    <input type="file" id="resume" name="resume" accept=".pdf">
    <span id="taken"></span>
    <button type="button" id="sub">Submit application</button>
  </form>
</div>
<script>
document.getElementById('apply').onclick = function () {
  document.getElementById('dialog').hidden = false;
};
document.getElementById('resume').onchange = function () {
  document.getElementById('taken').textContent = this.files[0].name;
};
document.getElementById('sub').onclick = function () {
  document.body.innerHTML = '<h1>Thank you for applying!</h1>';
};
</script></body></html>""" % {"desc": DESCRIPTION}

#: A site that renders no <form> element anywhere, which is what Ashby does and what
#: plenty of hand-rolled career pages do too. The form has to be found by climbing the
#: DOM, and only after the Apply button has been pressed.
NO_FORM_TAG_HTML = """<!doctype html><html><body>
<h1>Data Analyst</h1>
<p>%(desc)s</p>
<button id="apply">Apply</button>
<div id="application" hidden>
  <label for="first">First name</label><input id="first" name="first">
  <label for="last">Last name</label><input id="last" name="last">
  <label for="email">Email address</label><input id="email" name="email" type="email">
  <label for="resume">Resume</label>
  <input type="file" id="resume" name="resume" accept=".pdf">
  <span id="taken"></span>
  <button type="button" id="sub">Submit application</button>
</div>
<script>
document.getElementById('apply').onclick = function () {
  document.getElementById('application').hidden = false;
};
document.getElementById('resume').onchange = function () {
  document.getElementById('taken').textContent = this.files[0].name;
};
document.getElementById('sub').onclick = function () {
  document.body.innerHTML = '<h1>Application received, thanks!</h1>';
};
</script></body></html>""" % {"desc": DESCRIPTION}

#: Nowhere on this page to apply: no form, no button, just words.
NO_WHERE_TO_APPLY_HTML = """<!doctype html><html><body>
<h1>Field Engineer</h1>
<p>%(desc)s</p>
<p>Send your CV by post. There is no form here.</p>
</body></html>""" % {"desc": DESCRIPTION}


def _settings(tmp_path: Path, **kwargs) -> Settings:
    base = dict(db_path=tmp_path / "t.db", log_dir=tmp_path / "logs",
                audit_dir=tmp_path / "logs" / "apps", output_dir=tmp_path / "out",
                user_data_dir=tmp_path / "prof", overrides_path=tmp_path / "s.json",
                profile_path=ROOT / "profile.example.json", template_dir=ROOT / "templates",
                headless=True, browser_channel="", delay_mean_s=0.05, delay_std_s=0.02)
    base.update(kwargs)
    return Settings(_env_file=None, **base)


def _analysis() -> JobAnalysis:
    return JobAnalysis(
        job_title="Senior Backend Engineer", company_name="Acme", match_score=80,
        match_rationale="r", missing_requirements=[], tailored_summary="s",
        highlighted_skills=["Python"],
        tailored_bullets=[TailoredBulletGroup(kind="experience", name="Acme",
                                              title="Backend Engineer",
                                              bullets=["Shipped the service."])],
        answers=[ScreeningAnswer(question="Notice period?", answer="2 weeks")])


# ---- what the bot recognises ------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://apply.remote.com/jobs/74376551-7545-40fa-9b5e-26560627f4cf",
    "https://apply.remote.com/jobs/74376551-7545-40fa-9b5e-26560627f4cf?gh_src=x",
    "https://remote.com/jobs/74376551-7545-40fa-9b5e-26560627f4cf",
])
def test_remote_postings_are_recognised(url: str) -> None:
    assert detect_ats(url) == REMOTE, "an in-house ATS is still an ATS"


def test_other_sites_stay_unknown() -> None:
    assert detect_ats("https://example.com/careers/42") == UNKNOWN
    assert detect_ats("https://remote.com/careers") == UNKNOWN


def test_a_remote_posting_is_identified_by_its_uuid() -> None:
    """The same posting is reachable at more than one address; it is one posting."""
    a = job_id_from_url("https://apply.remote.com/jobs/74376551-7545-40fa-9b5e-26560627f4cf")
    b = job_id_from_url("https://apply.remote.com/jobs/74376551-7545-40fa-9b5e-26560627f4cf?x=1")
    assert a == b == f"{REMOTE}-74376551-7545-40fa-9b5e-26560627f4cf"
    assert a != job_id_from_url("https://apply.remote.com/jobs/11111111-2222-3333-4444-555555555555")


@pytest.mark.parametrize("ats,cls", [
    (REMOTE, RemoteApplier),
    (UNKNOWN, AnySiteApplier),
    (GREENHOUSE, GreenhouseApplier),
])
def test_every_recognised_site_gets_a_driver(ats: str, cls) -> None:
    assert isinstance(get_applier(ats, None, None, None, None), cls)


def test_the_fallback_driver_may_still_be_emailed() -> None:
    """It has to say so, or the pipeline would never look for an address."""
    assert getattr(AnySiteApplier, "email_fallback", False) is True
    assert getattr(RemoteApplier, "email_fallback", False) is False


def test_a_page_with_no_form_is_skipped_not_failed() -> None:
    """A site we guessed at having a form is a posting we cannot apply to, not a crash."""
    assert AnySiteApplier.no_form_status == STATUS_SKIPPED
    assert AnySiteApplier.no_form_note == NO_FORM_NOTE


def test_the_known_ats_still_report_a_missing_form_as_a_failure() -> None:
    assert GreenhouseApplier.no_form_status != STATUS_SKIPPED
    assert GreenhouseApplier.no_form_note == "Application form not found"


# ---- driving the real pages in a browser -----------------------------------


async def _run(settings: Settings, cls, html: str, job: JobPosting, act):
    profile = load_profile(settings.profile_path)
    analysis = _analysis()
    page_file = settings.log_dir / "posting.html"
    page_file.parent.mkdir(parents=True, exist_ok=True)
    page_file.write_text(html, encoding="utf-8")
    job.url = page_file.as_uri()
    builder = ResumeBuilder(settings)
    resume = await builder.build(profile, analysis, job)
    gate = HumanGate("api", settings.log_dir / "CONTINUE")
    async with StealthBrowser(settings, gate) as browser:
        filler = FormFiller(browser, AnswerResolver(None, settings), settings)
        applier = cls(browser, filler, gate, settings)
        result, seen = await asyncio.gather(
            applier.apply(browser.page, job, analysis, resume, profile),
            act(browser, gate) if act is not None else _settle())
    return result, seen


async def _settle() -> dict:
    await asyncio.sleep(1)
    return {}


async def _wait_then(browser, gate, action):
    """Wait for the agent to hand over, record what it managed, then carry on."""
    for _ in range(60):
        await asyncio.sleep(1)
        if gate.paused:
            break
    page = browser.page
    typed = await page.evaluate(
        "() => [...document.querySelectorAll('input:not([type=file]),textarea')]"
        ".filter(n => n.value).map(n => n.id)")
    files = await page.evaluate(
        "() => [...document.querySelectorAll('input[type=file]')]"
        ".filter(n => n.files.length).map(n => n.name)")
    await action(page)
    await asyncio.sleep(2.5)
    gate.release()
    return {"typed": typed, "files": files,
            "reason": gate.history[-1]["reason"] if gate.history else ""}


def _job(ats: str, source_id: str) -> JobPosting:
    return JobPosting(job_id=source_id, url="", title="Senior Backend Engineer",
                      company="Acme", location="Remote", description=DESCRIPTION,
                      source="urls", ats=ats)


@pytest.mark.asyncio
async def test_a_remote_dialog_form_is_filled_and_submitted(tmp_path: Path) -> None:
    """The posting in the log: apply.remote.com, where the form is behind Apply."""
    settings = _settings(tmp_path, fill_mode="assisted")

    async def act(browser, gate):
        async def submit(page):
            await page.click("#sub")

        return await _wait_then(browser, gate, submit)

    result, seen = await _run(settings, RemoteApplier, REMOTE_HTML,
                              _job(REMOTE, "remote-74376551"), act)

    assert seen["files"] == ["resume"], "the tailored resume must be attached"
    assert "Acme_Senior_Backend_Engineer.pdf" in seen["reason"]
    assert "first" in seen["typed"] and "email" in seen["typed"], seen
    assert "attached" in seen["reason"].lower()
    assert result.status == "submitted"
    assert "submitted it" in result.note.lower(), result.note


@pytest.mark.asyncio
async def test_a_site_with_no_form_tag_is_reached_through_the_apply_button(tmp_path: Path) -> None:
    """The fallback driver: no <form> element, so the DOM has to be climbed."""
    settings = _settings(tmp_path, fill_mode="documents")

    async def act(browser, gate):
        async def submit(page):
            await page.click("#sub")

        return await _wait_then(browser, gate, submit)

    result, seen = await _run(settings, AnySiteApplier, NO_FORM_TAG_HTML,
                              _job(UNKNOWN, "url-abc"), act)

    assert seen["files"] == ["resume"], "the resume must reach the form"
    assert result.status == "submitted"
    assert "submitted it" in result.note.lower(), result.note


@pytest.mark.asyncio
async def test_a_page_with_nowhere_to_apply_is_skipped(tmp_path: Path) -> None:
    """Nothing to fill and no button to press: say so, and touch nothing."""
    settings = _settings(tmp_path, fill_mode="auto")

    result, _ = await _run(settings, AnySiteApplier, NO_WHERE_TO_APPLY_HTML,
                           _job(UNKNOWN, "url-def"), None)

    assert result.status == STATUS_SKIPPED
    assert result.note == NO_FORM_NOTE
    assert not result.answers


# ---- the pipeline keeps the email route open -------------------------------


class _Page:
    async def evaluate(self, *_args, **_kwargs) -> str:
        return "We are hiring. Send your CV to Careers@Example-Corp.com."


class _Browser:
    page = _Page()


class _Source:
    async def hydrate(self, _page, job: JobPosting) -> JobPosting:
        return job


class _StubAI:
    async def analyze_job(self, _profile, _job) -> JobAnalysis:
        return _analysis()


class _StubResumes:
    last_trim = "nothing trimmed"

    async def build(self, _profile, _analysis, _job) -> Path:
        return Path("resume.pdf")


class _StubApplier:
    """A driver for a site with no form, which is all the pipeline needs to know."""

    email_fallback = True
    cover_letter = None

    def __init__(self) -> None:
        self.result = ApplyResult(status=STATUS_SKIPPED, note=NO_FORM_NOTE,
                                  apply_url="https://example.com/jobs/1")

    async def apply(self, _page, _job, _analysis, _resume, _profile) -> ApplyResult:
        return self.result


async def _process_without_a_form(tmp_path: Path, email_apply: bool, monkeypatch):
    import pipeline

    settings = _settings(tmp_path, email_apply=email_apply, match_threshold=65,
                         gemini_api_key="test-key")
    pipe = pipeline.Pipeline(settings)
    pipe.filler = None
    pipe.ai = _StubAI()
    pipe.resumes = _StubResumes()
    applier = _StubApplier()
    monkeypatch.setattr(pipeline, "get_applier", lambda *_a, **_k: applier)

    sent: dict[str, str] = {}

    async def by_email(job, analysis, resume_path, trimmed, email_to, cover_letter,
                       browser=None):
        sent["to"] = email_to
        return "submitted"

    pipe._apply_by_email = by_email
    job = JobPosting(job_id="url-1", url="https://example.com/jobs/1", title="Data Analyst",
                     company="Example Corp",
                     description="We are hiring a Data Analyst. " * 10,
                     source="urls", ats=UNKNOWN)
    status = await pipe._process(_Browser(), _Source(), job)
    return status, applier, sent


@pytest.mark.asyncio
async def test_a_formless_posting_falls_back_to_the_address_it_gives(tmp_path: Path,
                                                                     monkeypatch) -> None:
    """The posting offers no form but an address: the application still goes in."""
    status, _applier, sent = await _process_without_a_form(tmp_path, True, monkeypatch)

    assert status == "submitted"
    assert sent["to"] == "careers@example-corp.com"


@pytest.mark.asyncio
async def test_without_email_apply_the_posting_says_how_to_enable_it(tmp_path: Path,
                                                                     monkeypatch) -> None:
    status, applier, sent = await _process_without_a_form(tmp_path, False, monkeypatch)

    assert status == STATUS_SKIPPED
    assert not sent, "nothing may be sent when the feature is off"
    assert "Turn on 'Apply by email'" in applier.result.note
