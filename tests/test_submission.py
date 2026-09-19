"""Noticing a submit that the person performed themselves.

In assisted mode the human presses Submit, and the agent has to record that honestly.
Checking once after they say "continue" missed it whenever a site worded its confirmation
unusually, so an application that really went in was filed as still waiting for them.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from appliers import CONFIRM_TEXT_RE, CONFIRM_URL_HINTS, SubmissionEvidence, SubmissionWatcher  # noqa: E402
from browser_bot import HumanGate, StealthBrowser  # noqa: E402
from config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FORM_HTML = """<html><body><h1>Senior Engineer</h1>
<form id="f">
 <input name="first" placeholder="First name"><input name="last" placeholder="Last name">
 <input name="email" type="email"><input name="phone"><input type="file" name="resume">
 <textarea name="cover"></textarea><select name="src"><option>LinkedIn</option></select>
 <button type="button" id="sub">Submit application</button>
</form>
<script>document.getElementById('sub').onclick = () => { REACTION };</script>
</body></html>"""


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    tmp = tmp_path_factory.mktemp("sub")
    return Settings(db_path=tmp / "d.db", log_dir=tmp, audit_dir=tmp, output_dir=tmp,
                    user_data_dir=tmp / "p", overrides_path=tmp / "s.json",
                    headless=True, browser_channel="", delay_mean_s=0.05, delay_std_s=0.01)


# ---- the phrases and paths sites use ---------------------------------------


@pytest.mark.parametrize("text", [
    "Thank you for applying!",
    "Thanks for applying to Acme",
    "Thank you for your interest in this role",
    "Your application has been submitted",
    "Application received",
    "We have received your application",
    "You're all set",
    "We'll be in touch",
    "Successfully submitted",
    "Application complete",
])
def test_confirmation_phrases(text) -> None:
    assert CONFIRM_TEXT_RE.search(text), text


@pytest.mark.parametrize("text", [
    "Submit your application", "Please complete the form", "An error occurred",
    "Apply for this job", "Upload your resume",
])
def test_non_confirmation_text_is_not_matched(text) -> None:
    assert not CONFIRM_TEXT_RE.search(text), text


def test_confirmation_url_hints() -> None:
    for good in ("https://x.com/careers/thank-you", "https://x.com/apply/success",
                 "https://x.com/jobs/submitted", "https://x.com/confirmation"):
        assert any(h in good.lower() for h in CONFIRM_URL_HINTS), good
    for bad in ("https://x.com/jobs/123", "https://boards.greenhouse.io/acme/jobs/9"):
        assert not any(h in bad.lower() for h in CONFIRM_URL_HINTS), bad


# ---- watching a real page --------------------------------------------------


async def _watch(browser, page, reaction: str, click: bool = True) -> SubmissionEvidence:
    await page.goto("https://example.com/careers/job/1", wait_until="domcontentloaded")
    await page.set_content(FORM_HTML.replace("REACTION", reaction))
    watcher = SubmissionWatcher(browser, page)
    await watcher.start()
    await asyncio.sleep(0.3)
    if click:
        await page.click("#sub")
    await asyncio.sleep(2.2)
    return await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_sees_every_way_a_site_confirms(settings) -> None:
    gate = HumanGate("api", settings.log_dir / "CONTINUE")
    async with StealthBrowser(settings, gate) as browser:
        page = browser.page

        # 1. the usual wording
        ev = await _watch(browser, page,
                          "document.body.innerHTML = '<h1>Thank you for applying!</h1>'")
        assert ev.submitted and "confirmation text" in ev.signal

        # 2. wording the old check did not know
        ev = await _watch(browser, page,
                          "document.body.innerHTML = '<h1>You are all set. Speak soon.</h1>'")
        assert ev.submitted, "an unusual confirmation must still be noticed"

        # 3. no words at all: the form simply goes away
        ev = await _watch(browser, page,
                          "document.body.innerHTML = '<h1>Senior Engineer</h1><p>Done.</p>'")
        assert ev.submitted and "form is gone" in ev.signal

        # 4. a confirmation URL
        ev = await _watch(browser, page,
                          "history.pushState({}, '', '/careers/thank-you')")
        assert ev.submitted and "URL" in ev.signal

        # 5. nothing happened: the person looked and left it
        ev = await _watch(browser, page, "void 0", click=False)
        assert not ev.submitted
        assert ev.describe() == "no submission detected"


@pytest.mark.asyncio
async def test_watcher_does_not_fire_while_the_form_is_being_filled(settings) -> None:
    """Typing into the form must never look like a submission."""
    gate = HumanGate("api", settings.log_dir / "CONTINUE")
    async with StealthBrowser(settings, gate) as browser:
        page = browser.page
        await page.goto("https://example.com/careers/job/2", wait_until="domcontentloaded")
        await page.set_content(FORM_HTML.replace("REACTION", "void 0"))
        watcher = SubmissionWatcher(browser, page)
        await watcher.start()
        for name in ("first", "last", "phone"):
            await page.fill(f"input[name={name}]", "something")
            await asyncio.sleep(0.5)
        assert not (await watcher.stop()).submitted


# ---- the appliers use it ---------------------------------------------------


def test_every_human_handoff_watches_for_a_submit() -> None:
    """Each place the agent hands over is a place the person may submit."""
    src = (ROOT / "appliers.py").read_text()
    # one watcher per pause that a person could act on
    assert src.count("SubmissionWatcher(self.b, page)") >= 5
    assert src.count("await watcher.stop()") >= 5
    # and the outcome is driven by the evidence, not by a single late check
    assert "if evidence.submitted:" in src or "evidence.submitted else STATUS_PENDING" in src


def test_assisted_mode_records_what_was_seen() -> None:
    src = (ROOT / "appliers.py").read_text()
    block = src[src.index("if not self.s.auto_submit:"):]
    block = block[:block.index("for attempt in range(1, 4):")]
    assert "SubmissionWatcher" in block
    assert "STATUS_SUBMITTED" in block and "evidence.describe()" in block
    assert "STATUS_PENDING" in block          # still honest when nothing was submitted


def test_evidence_describes_itself() -> None:
    ev = SubmissionEvidence(True, "confirmation URL", "https://x/thanks", 1.0)
    assert "confirmation URL" in ev.describe() and "https://x/thanks" in ev.describe()
    assert SubmissionEvidence().describe() == "no submission detected"
