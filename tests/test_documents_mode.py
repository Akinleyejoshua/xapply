"""Documents mode: the agent supplies the paperwork, you fill in the rest.

Its job is the tailored resume and, when a form asks for one, a cover letter. Every
other field is left alone, and the submission you make yourself is still recorded.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_agent import CoverLetter, JobAnalysis, ScreeningAnswer, TailoredBulletGroup  # noqa: E402
from appliers import COVER_LETTER_RE, COVER_PROMPT_RE, GreenhouseApplier  # noqa: E402
from browser_bot import AnswerResolver, FormFiller, HumanGate, StealthBrowser  # noqa: E402
from config import Settings  # noqa: E402
from models import JobPosting  # noqa: E402
from pipeline import load_profile  # noqa: E402
from resume_builder import ResumeBuilder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FORM = """<html><body><h1>Senior Engineer</h1><form id="application-form">
<label for="first">First name</label><input id="first" name="first">
<label for="last">Last name</label><input id="last" name="last">
<label for="email">Email</label><input id="email" name="email" type="email">
<label for="phone">Phone</label><input id="phone" name="phone">
<label for="resume">Resume</label><input type="file" name="resume" id="resume">
{cover}
<button type="button" id="sub">Submit application</button></form>
<script>document.getElementById('sub').onclick = () =>
  document.body.innerHTML = '<h1>Thank you for applying!</h1>';</script></body></html>"""
COVER_UPLOAD = '<label for="cl">Cover letter</label><input type="file" name="cover_letter" id="cl">'
COVER_BOX = '<label for="cl">Cover letter</label><textarea id="cl" name="cover_letter"></textarea>'


def _analysis() -> JobAnalysis:
    return JobAnalysis(
        job_title="Senior Engineer", company_name="Acme", match_score=80, match_rationale="r",
        missing_requirements=[], tailored_summary="s", highlighted_skills=["Python"],
        tailored_bullets=[TailoredBulletGroup(kind="experience", name="Corvendra",
                                              title="Full Stack Developer",
                                              bullets=["Owned delivery."])],
        answers=[ScreeningAnswer(question="Notice period?", answer="2 weeks")])


def _letter() -> CoverLetter:
    return CoverLetter(greeting="Dear Hiring Team,", opening="Opening. " * 4,
                       body=["Body one. " * 15], closing="Closing.",
                       signature="Sincerely,\nJoshua Akinleye")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "d.db", log_dir=tmp_path, audit_dir=tmp_path,
                    output_dir=tmp_path, user_data_dir=tmp_path / "p",
                    overrides_path=tmp_path / "s.json", headless=True, browser_channel="",
                    delay_mean_s=0.05, delay_std_s=0.02, fill_mode="documents",
                    profile_path=ROOT / "profile.example.json", template_dir=ROOT / "templates")


# ---- the mode itself -------------------------------------------------------


@pytest.mark.parametrize("mode,fills,submits", [
    ("documents", False, False),
    ("assisted", True, False),
    ("auto", True, True),
])
def test_modes(tmp_path: Path, mode, fills, submits) -> None:
    s = Settings(_env_file=None, fill_mode=mode, db_path=tmp_path / "x.db")
    assert s.fills_every_field is fills
    assert s.auto_submit is submits


def test_legacy_auto_submit_env_still_means_auto(tmp_path: Path) -> None:
    s = Settings(_env_file=None, auto_submit=True, db_path=tmp_path / "x.db")
    assert s.fill_mode == "auto"


def test_mode_is_the_single_source_of_truth(tmp_path: Path) -> None:
    """Switching away from auto must actually clear auto_submit."""
    s = Settings(_env_file=None, fill_mode="auto", db_path=tmp_path / "x.db")
    assert s.auto_submit is True
    s.fill_mode = "documents"
    assert s.auto_submit is False and s.fills_every_field is False
    s.fill_mode = "assisted"
    assert s.auto_submit is False and s.fills_every_field is True


def test_mode_is_persisted_not_auto_submit() -> None:
    from config import PERSISTED_KEYS

    assert "fill_mode" in PERSISTED_KEYS
    assert "auto_submit" not in PERSISTED_KEYS, "two sources of truth would drift apart"


# ---- spotting a request for a cover letter ---------------------------------


@pytest.mark.parametrize("label", [
    "Cover letter", "Cover Letter (optional)", "Letter of interest",
    "Letter of introduction", "Motivation letter",
])
def test_cover_letter_labels(label) -> None:
    assert COVER_LETTER_RE.search(label), label


@pytest.mark.parametrize("label", [
    "Why do you want to join us?", "Why are you interested in this role?",
    "Tell us why you applied", "What draws you to this company?",
])
def test_cover_letter_prompts(label) -> None:
    assert COVER_PROMPT_RE.search(label), label


@pytest.mark.parametrize("label", [
    "Resume", "First name", "LinkedIn URL", "How many years of Python?", "Salary expectation",
])
def test_other_fields_are_not_mistaken_for_a_letter(label) -> None:
    assert not COVER_LETTER_RE.search(label)
    assert not COVER_PROMPT_RE.search(label)


# ---- end to end on a page we control ---------------------------------------


class LocalGreenhouse(GreenhouseApplier):
    """The real applier, pointed at a local page instead of a live board."""

    html = FORM.format(cover=COVER_UPLOAD)

    async def open_form(self, page, job):
        await page.goto("https://example.com/careers/1", wait_until="domcontentloaded")
        await page.set_content(self.html)
        return await self.find_form(page)


async def _run(settings: Settings, html: str, act):
    profile = load_profile(settings.profile_path)
    analysis = _analysis()
    job = JobPosting(job_id="gh-acme-1", url="https://job-boards.greenhouse.io/acme/jobs/1",
                     apply_url="https://job-boards.greenhouse.io/acme/jobs/1",
                     title="Senior Engineer", company="Acme", source="urls", ats="greenhouse")
    builder = ResumeBuilder(settings)
    resume = await builder.build(profile, analysis, job)
    letter = _letter()
    cache = {"r": (await builder.build_cover_letter(profile, letter, job, analysis), letter.as_text())}

    async def cover():
        return cache["r"]

    cover.cache = cache
    gate = HumanGate("api", settings.log_dir / "CONTINUE")
    async with StealthBrowser(settings, gate) as browser:
        filler = FormFiller(browser, AnswerResolver(None, settings), settings)
        applier = LocalGreenhouse(browser, filler, gate, settings, cover_letter=cover)
        applier.html = html
        result, extra = await asyncio.gather(
            applier.apply(browser.page, job, analysis, resume, profile), act(browser, gate))
    return result, extra


async def _wait_then(browser, gate, action):
    for _ in range(60):
        await asyncio.sleep(1)
        if gate.paused:
            break
    page = browser.page
    typed = await page.evaluate(
        "() => [...document.querySelectorAll('input:not([type=file]),textarea')]"
        ".filter(n => n.value).map(n => n.name)")
    files = await page.evaluate(
        "() => [...document.querySelectorAll('input[type=file]')]"
        ".filter(n => n.files.length).map(n => n.name)")
    await action(page)
    await asyncio.sleep(2.5)
    gate.release()
    return {"typed": typed, "files": files, "reason": gate.history[-1]["reason"] if gate.history else ""}


@pytest.mark.asyncio
async def test_documents_mode_attaches_both_and_fills_nothing_else(settings) -> None:
    async def act(browser, gate):
        async def submit(page):
            await page.fill("#first", "Joshua")        # the human does this part
            await page.click("#sub")

        return await _wait_then(browser, gate, submit)

    result, seen = await _run(settings, FORM.format(cover=COVER_UPLOAD), act)

    assert seen["files"] == ["resume", "cover_letter"], "both documents must be attached"
    assert seen["typed"] == [], "documents mode must not fill anything else"
    assert "Attached" in seen["reason"] and "cover_letter" in " ".join(seen["files"])
    # and the submission the human made is recorded
    assert result.status == "submitted"
    assert "you submitted it" in result.note
    assert {a["label"] for a in result.answers} == {"Resume", "Cover letter"}


@pytest.mark.asyncio
async def test_no_cover_letter_when_the_form_does_not_ask(settings) -> None:
    async def act(browser, gate):
        return await _wait_then(browser, gate, lambda page: page.click("#sub"))

    result, seen = await _run(settings, FORM.format(cover=""), act)
    assert seen["files"] == ["resume"]
    assert [a["label"] for a in result.answers] == ["Resume"]


@pytest.mark.asyncio
async def test_cover_letter_goes_into_a_textarea_whole(settings) -> None:
    """A long letter used to be typed key by key and truncated at the action timeout."""
    async def act(browser, gate):
        for _ in range(60):
            await asyncio.sleep(1)
            if gate.paused:
                break
        value = await browser.page.input_value("#cl")
        gate.release()
        return value

    result, value = await _run(settings, FORM.format(cover=COVER_BOX), act)
    assert value.startswith("Dear Hiring Team,")
    assert value.strip().endswith("Joshua Akinleye")
    assert len(value) == len(_letter().as_text()), "the whole letter must land, not a prefix"


@pytest.mark.asyncio
async def test_documents_mode_is_honest_when_you_do_not_submit(settings) -> None:
    async def act(browser, gate):
        return await _wait_then(browser, gate, lambda page: asyncio.sleep(0))

    result, _ = await _run(settings, FORM.format(cover=COVER_UPLOAD), act)
    assert result.status == "pending_human_review"
    assert "the rest is yours" in result.note


# ---- saying what actually happened ----------------------------------------

DOCS = [{"label": "Resume", "value": "Joshua_Resume.pdf"},
        {"label": "Cover letter", "value": "cover_letter.pdf"}]
FIELDS = [{"label": "Email"}, {"label": "First name"}]


def test_the_review_prompt_says_what_was_actually_filled() -> None:
    """It used to say "Form filled." even when the agent had filled nothing at all."""
    from appliers import BaseApplier as B

    both = B.review_prompt(B, DOCS, FIELDS)
    assert "Filled 2 fields" in both and "Joshua_Resume.pdf" in both

    docs_only = B.review_prompt(B, DOCS, [])
    assert "no other field" in docs_only
    assert "Form filled" not in docs_only and "Filled 0" not in docs_only

    nothing = B.review_prompt(B, [], [])
    assert "Nothing on this form could be filled" in nothing


def test_the_review_prompt_keeps_the_filename_as_it_is() -> None:
    from appliers import BaseApplier as B

    assert "Joshua_Resume.pdf" in B.review_prompt(B, DOCS, FIELDS)


def test_one_filled_field_is_not_described_as_fields() -> None:
    from appliers import BaseApplier as B

    assert "filled 1 field and" in B.what_was_done(DOCS, [{"label": "Email"}])


def test_the_saved_note_records_what_was_done() -> None:
    from appliers import BaseApplier as B

    assert B.what_was_done(DOCS, FIELDS).startswith("filled 2 fields and attached")
    assert B.what_was_done(DOCS, []) == "attached Joshua_Resume.pdf, cover_letter.pdf"
    assert B.what_was_done([], []) == "filled nothing"
