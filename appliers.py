"""ATS-specific application drivers.

All drivers share `FormFiller` (generic field discovery/filling) and `HumanGate`.
They differ only in how the form is reached, where the submit button lives,
and how a successful submission is confirmed.

  LinkedInEasyApplyApplier - multi-step modal (Next -> Review -> Submit)
  GreenhouseApplier        - single page form under the job description
  LeverApplier             - /apply page
  AshbyApplier             - /application page
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import asyncio
import time
from dataclasses import dataclass

from playwright.async_api import Locator, Page, TimeoutError as PlaywrightTimeout

from ai_agent import JobAnalysis
from browser_bot import FormField, FormFiller, HumanGate, ResolveContext, StealthBrowser
from config import Settings
from database import STATUS_FAILED, STATUS_PENDING, STATUS_SKIPPED, STATUS_SUBMITTED
from models import (
    ASHBY,
    GREENHOUSE,
    LEVER,
    LINKEDIN,
    UNKNOWN,
    ApplyResult,
    JobPosting,
    detect_ats,
)

log = logging.getLogger(__name__)

#: Every phrasing a careers site uses to say "we got it". Kept broad on purpose: a
#: submission the agent fails to notice is recorded as still waiting for you, which is
#: worse than a rare false positive you can correct from the Applications tab.
CONFIRM_TEXT_RE = re.compile(
    r"(thank(s| you)[^.]{0,30}(for )?(applying|your (application|interest|submission))"
    r"|application (has been |was |is )?(submitted|sent|received|complete)"
    r"|we('ve| have) received your application"
    r"|your application (has been |was )?(submitted|sent|received)"
    r"|successfully (submitted|applied)|applied successfully|submission received"
    r"|you'?re all set|all set!|we'?ll be in touch|we will be in touch"
    r"|we'?ll review your application|your application is on its way"
    r"|application complete|thanks for your interest)",
    re.I,
)
CONFIRM_URL_HINTS = ("/thanks", "/thank-you", "/thankyou", "/confirmation", "/confirmed",
                     "/submitted", "/success", "/complete", "/received", "/applied")

#: A snapshot of the page, used to notice that the form has gone away.
PAGE_STATE_JS = r"""
() => {
  const sel = 'input:not([type=hidden]):not([type=submit]):not([type=button]), textarea, select';
  const visible = n => {
    const st = getComputedStyle(n);
    if (st.visibility === 'hidden' || st.display === 'none') return false;
    const r = n.getBoundingClientRect();
    return r.width > 0 || r.height > 0 || n.type === 'file';
  };
  const fields = [...document.querySelectorAll(sel)].filter(visible).length;
  const submit = [...document.querySelectorAll('button, input[type=submit], a[role=button]')]
    .filter(b => /submit|apply/i.test((b.innerText || b.value || '')) && visible(b)).length;
  return {fields, submit, text: (document.body ? document.body.innerText : '').slice(0, 6000)};
}
"""


@dataclass
class SubmissionEvidence:
    """What the agent saw that says the application went in."""

    submitted: bool = False
    signal: str = ""
    url: str = ""
    at: float = 0.0

    def describe(self) -> str:
        return f"{self.signal} at {self.url}" if self.submitted else "no submission detected"


class SubmissionWatcher:
    """Watches a page while a human works on it, and notices a manual submit.

    In assisted mode the person presses Submit themselves, and the agent has to record
    that accurately. Checking once after they say "continue" misses it whenever the site
    phrases its confirmation unusually, or redirects somewhere unexpected. So the page is
    polled throughout, and any one of four independent signals counts: a confirmation
    phrase, a confirmation URL, the form disappearing, or the submit button going away
    after the URL changed.
    """

    POLL_SECONDS = 1.2

    def __init__(self, browser: "StealthBrowser", page: Page):
        self.b = browser
        self.page = page
        self.evidence = SubmissionEvidence()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self.baseline: dict[str, Any] = {}

    async def _snapshot(self) -> dict[str, Any]:
        try:
            state = await self.page.evaluate(PAGE_STATE_JS)
            state["url"] = self.page.url
            return state
        except Exception:
            return {}

    async def start(self) -> None:
        self.baseline = await self._snapshot()
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="submission-watcher")

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.POLL_SECONDS)
                return
            except asyncio.TimeoutError:
                pass
            if self.evidence.submitted:
                continue
            found = await self.check()
            if found.submitted:
                self.evidence = found
                log.info("Submission detected while you worked: %s", found.describe())

    async def check(self) -> SubmissionEvidence:
        """One look at the page for any sign the application went through."""
        now = await self._snapshot()
        if not now:
            return SubmissionEvidence()
        url = now.get("url", "")
        text = now.get("text", "") or ""
        base_fields = self.baseline.get("fields", 0)

        match = CONFIRM_TEXT_RE.search(text)
        if match:
            return SubmissionEvidence(True, f"confirmation text {match.group(0)!r}", url, time.time())
        low = url.lower()
        if any(hint in low for hint in CONFIRM_URL_HINTS):
            return SubmissionEvidence(True, "confirmation URL", url, time.time())
        # The form vanishing is the one signal every ATS shares, whatever it says.
        if base_fields >= 5 and now.get("fields", 0) <= max(2, base_fields * 0.3):
            return SubmissionEvidence(True, "the application form is gone", url, time.time())
        if (url != self.baseline.get("url") and self.baseline.get("submit", 0)
                and not now.get("submit", 0) and now.get("fields", 0) < base_fields):
            return SubmissionEvidence(True, "navigated away and the submit button is gone", url,
                                      time.time())
        return SubmissionEvidence()

    async def stop(self) -> SubmissionEvidence:
        """Stop watching and return the best evidence, checking once more first."""
        self._stop.set()
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if not self.evidence.submitted:
            self.evidence = await self.check()
        return self.evidence
ERROR_SELECTOR = (
    '.artdeco-inline-feedback--error, [role="alert"], [class*="error-message" i], [class*="form-error" i], '
    '[class*="field-error" i], [class*="errorMessage" i], .error-text, [class*="_error" i]:not(input)'
)
FORM_SHAPE_JS = r"""
(el) => {
  const sel = 'input:not([type=hidden]):not([type=submit]):not([type=button]), textarea, select';
  const visible = n => {
    const st = getComputedStyle(n);
    if (st.visibility === 'hidden' || st.display === 'none') return false;
    const r = n.getBoundingClientRect();
    return r.width > 0 || r.height > 0 || n.type === 'file';
  };
  const nodes = [...el.querySelectorAll(sel)];
  const shown = nodes.filter(visible);
  const typeOf = n => (n.getAttribute('type') || n.tagName).toLowerCase();
  const blob = n => ((n.name || '') + ' ' + (n.id || '') + ' ' + (n.placeholder || '') + ' ' +
                     (n.getAttribute('aria-label') || '')).toLowerCase();
  return {
    total: shown.length,
    file: nodes.filter(n => typeOf(n) === 'file').length,
    email: shown.filter(n => typeOf(n) === 'email' || /e-?mail/.test(blob(n))).length,
    password: shown.filter(n => typeOf(n) === 'password').length,
    search: shown.filter(n => typeOf(n) === 'search' || /\bsearch\b|\bquery\b/.test(blob(n))).length,
    name: shown.filter(n => /first.?name|last.?name|full.?name|\bname\b/.test(blob(n))).length,
  };
}
"""

FORM_CONTAINER_JS = r"""
() => {
  const sel = 'input:not([type=hidden]):not([type=submit]):not([type=button]), textarea, select';
  const visible = el => {
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 || r.height > 0 || el.type === 'file';
  };
  const inputs = [...document.querySelectorAll(sel)].filter(visible);
  if (inputs.length < 2) return 0;
  // Climb from the first input until the ancestor holds (almost) all of them,
  // then stop: that is the tightest wrapper around the application.
  let el = inputs[0], best = null;
  while (el && el !== document.body) {
    el = el.parentElement;
    if (!el) break;
    const n = [...el.querySelectorAll(sel)].filter(visible).length;
    if (n >= Math.max(2, Math.ceil(inputs.length * 0.8))) { best = el; break; }
  }
  best = best || document.body;
  document.querySelectorAll('[data-xapply-form]').forEach(e => e.removeAttribute('data-xapply-form'));
  best.setAttribute('data-xapply-form', '1');
  return [...best.querySelectorAll(sel)].filter(visible).length;
}
"""

#: Hosts that really serve application forms. Matching on the host, not on the URL
#: text, matters: a Google API proxy iframe carries `greenhouse.io` in its query string
#: and would otherwise be followed instead of the form.
ATS_IFRAME_HOSTS = (
    "greenhouse.io", "lever.co", "ashbyhq.com", "workable.com", "smartrecruiters.com",
    "myworkdayjobs.com", "icims.com", "jobvite.com", "breezy.hr", "bamboohr.com",
    "recruitee.com", "teamtailor.com", "personio.de", "rippling.com",
)


def is_ats_iframe(src: Optional[str]) -> bool:
    """Whether an iframe src points at an applicant tracking system, by host."""
    if not src:
        return False
    try:
        host = (urlparse(src).hostname or "").lower()
    except ValueError:
        return False
    return any(host == h or host.endswith("." + h) for h in ATS_IFRAME_HOSTS)


#: Buttons and links that open an application form on a company-hosted careers page.
APPLY_TRIGGER_RE = re.compile(
    r"^\s*(apply|apply now|apply for this (job|role|position)|apply to this job|"
    r"submit (an )?application|start (your )?application|i'?m interested)\s*$", re.I
)

#: A cover letter can be asked for as an upload or as a free-text box.
COVER_LETTER_RE = re.compile(
    r"cover.?letter|letter of (interest|introduction|motivation)|motivation letter", re.I)
#: Prompts that are really "write us a short letter", answered the same way.
COVER_PROMPT_RE = re.compile(
    r"why (do you want|are you interested|would you like).{0,40}(join|work|role|position|us|company)"
    r"|tell us (about yourself|why)|what (draws|attracts) you"
    r"|why (this|our) (role|company|team)", re.I)
COVER_FILE_INPUTS = (
    'input[type="file"][name*="cover" i], input[type="file"][id*="cover" i], '
    'input[type="file"][name*="letter" i], input[type="file"][id*="letter" i]'
)

RESUME_FILE_INPUTS = (
    'input[type="file"][name*="resume" i], input[type="file"][id*="resume" i], '
    'input[type="file"][name*="cv" i], input[type="file"][id*="cv" i], '
    'input[type="file"][accept*="pdf" i], input[type="file"]'
)


def _short_exc(exc: BaseException, limit: int = 300) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:limit] if str(exc) else ''}".strip()


class BaseApplier:
    ats = "unknown"

    def __init__(self, browser: StealthBrowser, filler: FormFiller, gate: HumanGate,
                 settings: Settings, cover_letter: Optional[Callable[[], Any]] = None):
        self.b = browser
        self.filler = filler
        self.gate = gate
        self.s = settings
        #: Awaitable returning (pdf_path, text) for this posting. Called only when a form
        #: actually asks for a letter, so no API call is spent when none is wanted.
        self.cover_letter = cover_letter

    async def attach_documents(self, page: Page, scope: Locator,
                               resume_path: Path) -> list[dict[str, Any]]:
        """Upload the resume, and a cover letter when the form asks for one.

        This is the whole of the agent's job in documents mode, and the first step of
        every other mode.
        """
        attached: list[dict[str, Any]] = []
        if await self.upload_resume(page, scope, resume_path):
            attached.append({"label": "Resume", "kind": "file", "value": resume_path.name,
                             "source": "resume_builder", "confidence": 1.0, "ok": True})
        entry = await self.attach_cover_letter(page, scope)
        if entry:
            attached.append(entry)
        return attached

    async def wants_cover_letter(self, scope: Locator) -> Optional[FormField]:
        """The field a cover letter belongs in, if the form has one."""
        try:
            fields = await self.filler.discover(scope)
        except Exception as exc:
            log.debug("could not inspect the form for a cover letter: %s", exc)
            return None
        for f in fields:
            if f.kind == "file" and COVER_LETTER_RE.search(f.label or ""):
                return f
            if f.kind == "textarea" and (COVER_LETTER_RE.search(f.label or "")
                                         or COVER_PROMPT_RE.search(f.label or "")):
                return f
        return None

    async def attach_cover_letter(self, page: Page, scope: Locator) -> Optional[dict[str, Any]]:
        """Write and attach a cover letter, but only if this form asked for one."""
        if not self.cover_letter:
            return None
        field = await self.wants_cover_letter(scope)
        if field is None:
            return None
        try:
            pdf_path, text = await self.cover_letter()
        except Exception as exc:
            log.warning("Could not write a cover letter: %s", exc)
            return None
        if field.kind == "file":
            uploads = scope.locator(COVER_FILE_INPUTS)
            if not await uploads.count():
                return None
            await uploads.first.set_input_files(str(pdf_path))
            await self.b.sleep(1.2, 0.3)
            log.info("Attached cover letter %s", pdf_path.name)
            return {"label": field.label or "Cover letter", "kind": "file",
                    "value": pdf_path.name, "source": "cover_letter", "confidence": 1.0, "ok": True}
        box = scope.locator(f'[data-xapply-idx="{field.idx}"]')
        try:
            await self.b.human_type(box, text)
        except Exception as exc:
            log.warning("Could not type the cover letter into %r: %s", field.label, exc)
            return None
        log.info("Wrote the cover letter into %r", field.label)
        return {"label": field.label or "Cover letter", "kind": "textarea",
                "value": text[:120] + ("..." if len(text) > 120 else ""),
                "source": "cover_letter", "confidence": 1.0, "ok": True}

    async def apply(self, page: Page, job: JobPosting, analysis: JobAnalysis,
                    resume_path: Path, profile: dict[str, Any]) -> ApplyResult:  # pragma: no cover
        raise NotImplementedError

    # ---- shared helpers -------------------------------------------------
    async def confirmed(self, page: Page, timeout: int = 8000) -> bool:
        if any(h in page.url.lower() for h in CONFIRM_URL_HINTS):
            return True
        try:
            await page.get_by_text(CONFIRM_TEXT_RE).first.wait_for(state="visible", timeout=timeout)
            return True
        except PlaywrightTimeout:
            return any(h in page.url.lower() for h in CONFIRM_URL_HINTS)
        except Exception:
            return False

    async def errors(self, scope: Locator) -> list[str]:
        out: list[str] = []
        try:
            loc = scope.locator(ERROR_SELECTOR)
            n = min(await loc.count(), 30)
            for i in range(n):
                el = loc.nth(i)
                if await el.is_visible():
                    t = re.sub(r"\s+", " ", (await el.inner_text()).strip())
                    if t and t not in out and len(t) < 300:
                        out.append(t)
        except Exception as exc:
            log.debug("error scan failed: %s", exc)
        return out

    async def upload_resume(self, page: Page, scope: Locator, resume_path: Path) -> bool:
        """Attach the tailored PDF and wait until the site has taken it."""
        inputs = scope.locator(RESUME_FILE_INPUTS)
        if not await inputs.count():
            return False
        file_input = inputs.first
        await file_input.set_input_files(str(resume_path))
        try:
            await page.wait_for_function(
                "el => el.files && el.files.length > 0", arg=await file_input.element_handle(), timeout=5000
            )
        except Exception:
            pass
        # 1) the file name shows up somewhere on the page
        stem = resume_path.stem[:24]
        try:
            await scope.get_by_text(stem, exact=False).first.wait_for(state="visible", timeout=20_000)
            log.info("Upload confirmed: %s", resume_path.name)
        except PlaywrightTimeout:
            log.warning("Upload confirmation text not found for %s, relying on spinner check", resume_path.name)
        # 2) no upload spinner / progress indicator still running
        spinner = page.locator(
            '.artdeco-spinner, [class*="uploading" i], [class*="progress" i][aria-busy="true"], progress, '
            '[class*="spinner" i], [aria-label*="uploading" i]'
        )
        try:
            if await spinner.count() and await spinner.first.is_visible():
                await spinner.first.wait_for(state="hidden", timeout=30_000)
        except Exception:
            pass
        await self.b.sleep(1.5, 0.4)
        return True

    def _result(self, status: str, note: str, answers: list[dict[str, Any]], shot: str, url: str) -> ApplyResult:
        log.info("Apply result [%s]: %s", status, note)
        return ApplyResult(status=status, note=note, answers=answers, screenshot_path=shot, apply_url=url)


# --------------------------------------------------------------------------
# LinkedIn Easy Apply (multi-step modal)
# --------------------------------------------------------------------------


class LinkedInEasyApplyApplier(BaseApplier):
    ats = LINKEDIN
    EASY_APPLY_RE = re.compile(r"^\s*easy apply", re.I)
    NEXT_RE = re.compile(r"^(continue to next step|next)\s*$", re.I)
    REVIEW_RE = re.compile(r"^(review your application|review)\s*$", re.I)
    SUBMIT_RE = re.compile(r"^\s*submit application\s*$", re.I)
    APPLIED_RE = re.compile(r"^\s*applied\b", re.I)

    async def modal(self, page: Page) -> Optional[Locator]:
        """The Easy Apply dialog, by its known ids and then by role."""
        for sel in (".jobs-easy-apply-modal", 'div[data-test-modal-id*="easy-apply" i]', 'div[role="dialog"]'):
            loc = page.locator(sel).filter(has=page.locator("form, input, select, textarea, button"))
            n = await loc.count()
            for i in range(n):
                if await loc.nth(i).is_visible():
                    return loc.nth(i)
        return None

    async def already_applied(self, page: Page) -> bool:
        try:
            loc = page.locator(".jobs-s-apply__application-link, .post-apply-timeline, .artdeco-inline-feedback--success")
            if await loc.count() and await loc.first.is_visible():
                return True
            txt = page.get_by_text(self.APPLIED_RE)
            n = min(await txt.count(), 5)
            for i in range(n):
                if await txt.nth(i).is_visible():
                    return True
        except Exception:
            pass
        return False

    async def _button(self, scope: Locator, pattern: re.Pattern) -> Optional[Locator]:
        btn = scope.get_by_role("button", name=pattern)
        n = await btn.count()
        for i in range(n):
            if await btn.nth(i).is_visible():
                return btn.nth(i)
        return None

    async def _uncheck_follow(self, modal: Locator) -> None:
        if self.s.follow_companies:
            return
        try:
            cb = modal.locator('input#follow-company-checkbox, input[type="checkbox"][id*="follow" i]')
            if await cb.count() and await cb.first.is_checked():
                dom_id = await cb.first.get_attribute("id")
                lbl = modal.locator(f'label[for="{dom_id}"]')
                if await lbl.count():
                    await self.b.human_click(lbl.first)
                else:
                    await cb.first.set_checked(False, force=True)
        except Exception as exc:
            log.debug("follow checkbox: %s", exc)

    async def _dismiss_post_apply(self, page: Page) -> None:
        try:
            btn = page.get_by_role("button", name=re.compile(r"^(done|dismiss|not now|continue|close)$", re.I))
            if await btn.count() and await btn.first.is_visible():
                await self.b.human_click(btn.first)
        except Exception:
            pass

    async def _discard(self, page: Page) -> None:
        """Close the modal without submitting (LinkedIn asks to discard)."""
        try:
            modal = await self.modal(page)
            if modal is None:
                return
            close = modal.get_by_role("button", name=re.compile(r"dismiss|close", re.I))
            if await close.count():
                await self.b.human_click(close.first)
                discard = page.get_by_role("button", name=re.compile(r"^discard", re.I))
                try:
                    await discard.first.wait_for(state="visible", timeout=4000)
                    await self.b.human_click(discard.first)
                except PlaywrightTimeout:
                    pass
        except Exception as exc:
            log.debug("discard failed: %s", exc)

    async def apply(self, page: Page, job: JobPosting, analysis: JobAnalysis,
                    resume_path: Path, profile: dict[str, Any]) -> ApplyResult:
        answers: list[dict[str, Any]] = []
        shot = ""
        try:
            await self.b.goto(page, job.url)
            await self.b.human_scroll(page, 500)
            if await self.already_applied(page):
                return self._result(STATUS_SKIPPED, "Already applied on LinkedIn", answers, shot, job.url)
            easy = page.get_by_role("button", name=self.EASY_APPLY_RE).first
            try:
                await easy.wait_for(state="visible", timeout=8000)
            except PlaywrightTimeout:
                return self._result(STATUS_SKIPPED, "No Easy Apply button on this posting", answers, shot, job.url)
            await self.b.human_click(easy)
            await self.b.guard(page)
            ctx = ResolveContext(profile, job, analysis)
            uploaded = False
            error_rounds = 0
            for step in range(1, self.s.max_form_steps + 1):
                await self.b.guard(page)
                modal = await self.modal(page)
                if modal is None:
                    if await self.confirmed(page, timeout=3000) or await self.already_applied(page):
                        return self._result(STATUS_SUBMITTED, "Application sent", answers, shot, job.url)
                    return self._result(STATUS_FAILED, "Easy Apply modal disappeared", answers, shot, job.url)
                if not uploaded:
                    attached = await self.attach_documents(page, modal, resume_path)
                    uploaded = bool(attached)
                    answers.extend(attached)
                if not self.s.fills_every_field:
                    watcher = SubmissionWatcher(self.b, page)
                    await watcher.start()
                    shot = await self.b.screenshot(page, f"documents_linkedin_{job.job_id}")
                    outcome = await self.gate.wait(
                        "Resume attached. Fill in the rest of the Easy Apply steps and submit "
                        "them yourself, then continue.")
                    evidence = await watcher.stop()
                    if evidence.submitted or await self.modal(page) is None \
                            or await self.already_applied(page):
                        await self._dismiss_post_apply(page)
                        return self._result(STATUS_SUBMITTED,
                                            "Documents attached; you submitted it", answers, shot, job.url)
                    if outcome == HumanGate.SKIP:
                        await self._discard(page)
                        return self._result(STATUS_SKIPPED, "Skipped by you", answers, shot, job.url)
                    await self._discard(page)
                    return self._result(STATUS_PENDING, "Resume attached; the rest is yours",
                                        answers, shot, job.url)
                res = await self.filler.fill_step(modal, ctx)
                answers.extend(res.filled)
                if res.unresolved:
                    await self.gate.wait("Could not confidently answer required field(s): "
                                         + "; ".join(res.unresolved) + ". Fill them in the browser, then continue.")
                    modal = await self.modal(page) or modal
                submit = await self._button(modal, self.SUBMIT_RE)
                if submit is not None:
                    await self._uncheck_follow(modal)
                    if await self.b.guard(page) == HumanGate.SKIP:
                        await self._discard(page)
                        return self._result(STATUS_SKIPPED, "Skipped by you at the challenge",
                                            answers, shot, job.url)
                    shot = await self.b.screenshot(page, f"review_linkedin_{job.job_id}")
                    if self.s.auto_submit:
                        await self.b.human_click(submit)
                        if await self.confirmed(page, timeout=12_000):
                            await self._dismiss_post_apply(page)
                            return self._result(STATUS_SUBMITTED, "Auto-submitted", answers, shot, job.url)
                        await self.b.guard(page)
                        errs = await self.errors(modal)
                        if errs and error_rounds < 2:
                            error_rounds += 1
                            res = await self.filler.fill_step(modal, ctx, only_errors=True, force_ai=True)
                            answers.extend(res.filled)
                            continue
                        await self.gate.wait("Clicked Submit but no confirmation was detected"
                                             + (f" (errors: {errs})" if errs else "")
                                             + ". Check the browser and submit manually if needed, then continue.")
                        if await self.confirmed(page, timeout=3000) or await self.already_applied(page):
                            await self._dismiss_post_apply(page)
                            return self._result(STATUS_SUBMITTED, "Submitted by human after bot attempt", answers, shot, job.url)
                        await self._discard(page)
                        return self._result(STATUS_PENDING, "Submission not confirmed", answers, shot, job.url)
                    watcher = SubmissionWatcher(self.b, page)
                    await watcher.start()
                    outcome = await self.gate.wait("Review page reached. Check it and press "
                                                   "'Submit application' yourself, then continue.")
                    evidence = await watcher.stop()
                    # On LinkedIn the modal closing is itself proof the application went in.
                    modal_gone = await self.modal(page) is None
                    if evidence.submitted or modal_gone or await self.already_applied(page):
                        await self._dismiss_post_apply(page)
                        reason = evidence.describe() if evidence.submitted else "the Easy Apply dialog closed"
                        return self._result(STATUS_SUBMITTED, f"You submitted it: {reason}",
                                            answers, shot, job.url)
                    if outcome == HumanGate.SKIP:
                        await self._discard(page)
                        return self._result(STATUS_SKIPPED, "Skipped by you", answers, shot, job.url)
                    await self._discard(page)
                    return self._result(STATUS_PENDING, "Reached the review page; no submission seen",
                                        answers, shot, job.url)
                nav = await self._button(modal, self.REVIEW_RE) or await self._button(modal, self.NEXT_RE)
                if nav is None:
                    shot = await self.b.screenshot(page, f"no_nav_{job.job_id}")
                    await self.gate.wait("Could not find a Next/Review/Submit button. Advance the form manually, then continue.")
                    continue
                await self.b.human_click(nav)
                await self.b.sleep(1.0, 0.3)
                modal = await self.modal(page)
                errs = await self.errors(modal) if modal is not None else []
                if errs:
                    error_rounds += 1
                    log.warning("Validation errors on step %d: %s", step, errs)
                    res = await self.filler.fill_step(modal, ctx, only_errors=True, force_ai=True)
                    answers.extend(res.filled)
                    if error_rounds >= 2:
                        await self.gate.wait(f"Form validation errors persist: {errs}. Fix them in the browser, then continue.")
                        error_rounds = 0
            return self._result(STATUS_FAILED, f"Exceeded {self.s.max_form_steps} form steps", answers, shot, job.url)
        except Exception as exc:
            log.exception("LinkedIn Easy Apply failed for %s", job.url)
            shot = await self.b.screenshot(page, f"error_linkedin_{job.job_id}") or shot
            await self._discard(page)
            return self._result(STATUS_FAILED, _short_exc(exc), answers, shot, job.url)


# --------------------------------------------------------------------------
# Single-page ATS forms (Greenhouse, Lever, Ashby)
# --------------------------------------------------------------------------


class SinglePageApplier(BaseApplier):
    form_selectors: tuple[str, ...] = ("form",)
    submit_re = re.compile(r"submit( application| your application)?$", re.I)

    async def open_form(self, page: Page, job: JobPosting) -> Optional[Locator]:
        """Navigate so the application form is on screen and return its scope."""
        await self.b.goto(page, self.apply_url(job))
        form = await self.find_in_frames_or_page(page, timeout=12_000)
        if form is None:
            form = await self.follow_apply_trigger(page)
        return form

    def apply_url(self, job: JobPosting) -> str:
        return job.apply_url or job.url

    async def find_form(self, page: Page, timeout: int = 15_000) -> Optional[Locator]:
        """Locate the application form, falling back to the DOM when no selector matches.

        Not every ATS wraps its application in a `<form>`: Ashby renders plain divs.
        So after the configured selectors, the smallest element that still contains
        every visible input is tagged and used as the scope.

        Every candidate is checked against `is_application_form` first. Without that,
        a company careers page hands back its search box, which looks like a form,
        contains one input, and stops the search before the real form is ever reached.
        """
        per_selector = max(2000, timeout // max(1, len(self.form_selectors)))
        for sel in self.form_selectors:
            loc = page.locator(sel).filter(has=page.locator('input:not([type="hidden"]), textarea, select'))
            try:
                await loc.first.wait_for(state="visible", timeout=per_selector)
            except PlaywrightTimeout:
                continue
            for i in range(min(await loc.count(), 4)):
                candidate = loc.nth(i)
                if await self.is_application_form(candidate):
                    return candidate
                log.debug("Ignoring %s[%d]: does not look like an application form", sel, i)
        container = await self.find_form_container(page)
        if container is not None and await self.is_application_form(container):
            return container
        return None

    async def is_application_form(self, scope: Locator) -> bool:
        """Does this element hold an application, or just a search box or newsletter signup?"""
        try:
            counts = await scope.evaluate(FORM_SHAPE_JS)
        except Exception as exc:
            log.debug("form shape check failed: %s", exc)
            return False
        total = counts.get("total", 0)
        if counts.get("password"):
            return False                                    # a sign-in form
        if counts.get("file"):
            return True                                     # a resume upload settles it
        if counts.get("search") and total <= 3:
            return False                                    # a site search box
        if counts.get("email") and counts.get("name"):
            return True                                     # asks who you are and how to reach you
        if total >= 4 and (counts.get("email") or counts.get("name")):
            return True
        # A lone email box is a newsletter signup, which is what a careers page usually
        # offers next to the job description. Two fields are never an application.
        return total >= 6

    async def find_form_container(self, page: Page) -> Optional[Locator]:
        """Tag the smallest element holding most of the page's inputs and return it."""
        try:
            found = await page.evaluate(FORM_CONTAINER_JS)
        except Exception as exc:
            log.debug("container fallback failed: %s", exc)
            return None
        if not found:
            return None
        log.info("Using DOM fallback for the form scope (%s inputs)", found)
        return page.locator("[data-xapply-form]").first

    async def find_in_frames_or_page(self, page: Page, timeout: int = 10_000) -> Optional[Locator]:
        """Look for the form on the page, then inside any embedded ATS iframe.

        Company careers pages frequently embed the real form in an iframe. Navigating
        straight to the iframe's source is more reliable than driving it in place.
        """
        form = await self.find_form(page, timeout=timeout)
        if form is not None:
            return form
        try:
            frames = page.locator("iframe[src]")
            for i in range(min(await frames.count(), 12)):
                src = await frames.nth(i).get_attribute("src")
                if not is_ats_iframe(src):
                    continue
                log.info("Following embedded application iframe to %s", src)
                await self.b.goto(page, src)
                found = await self.find_form(page, timeout=timeout)
                if found is not None:
                    return found
                await page.go_back(wait_until="domcontentloaded")
        except Exception as exc:
            log.debug("iframe follow failed: %s", exc)
        return None

    async def follow_apply_trigger(self, page: Page) -> Optional[Locator]:
        """Click the Apply button or link a company page shows instead of a form.

        A link is followed by its href when it points at a known ATS, because many
        careers pages open the real form in a new tab.
        """
        try:
            link = page.get_by_role("link", name=APPLY_TRIGGER_RE)
            for i in range(min(await link.count(), 4)):
                el = link.nth(i)
                if not await el.is_visible():
                    continue
                href = await el.get_attribute("href")
                if href and detect_ats(href) != UNKNOWN:
                    log.info("Following the Apply link to %s", href)
                    await self.b.goto(page, href)
                else:
                    await self.b.human_click(el)
                form = await self.find_in_frames_or_page(page, timeout=10_000)
                if form is not None:
                    return form
        except Exception as exc:
            log.debug("apply link handling failed: %s", exc)
        try:
            btn = page.get_by_role("button", name=APPLY_TRIGGER_RE)
            for i in range(min(await btn.count(), 4)):
                el = btn.nth(i)
                if not await el.is_visible():
                    continue
                log.info("Clicking the Apply button on %s", page.url)
                await self.b.human_click(el)
                form = await self.find_in_frames_or_page(page, timeout=10_000)
                if form is not None:
                    return form
        except Exception as exc:
            log.debug("apply button handling failed: %s", exc)
        return None

    async def find_submit(self, scope: Locator, page: Page) -> Optional[Locator]:
        for root in (scope, page.locator("body")):
            for cand in (
                root.get_by_role("button", name=self.submit_re),
                root.locator('button[type="submit"], input[type="submit"]'),
            ):
                n = await cand.count()
                for i in range(n):
                    el = cand.nth(i)
                    if await el.is_visible():
                        return el
        return None

    async def apply(self, page: Page, job: JobPosting, analysis: JobAnalysis,
                    resume_path: Path, profile: dict[str, Any]) -> ApplyResult:
        url = self.apply_url(job)
        answers: list[dict[str, Any]] = []
        shot = ""
        try:
            scope = await self.open_form(page, job)
            if scope is None:
                shot = await self.b.screenshot(page, f"noform_{self.ats}_{job.job_id}")
                return self._result(STATUS_FAILED, "Application form not found", answers, shot, url)
            ctx = ResolveContext(profile, job, analysis)
            answers.extend(await self.attach_documents(page, scope, resume_path))
            await self.b.human_scroll(page, 700)

            if not self.s.fills_every_field:
                # Documents mode: the attachments were the job. Everything else is yours.
                watcher = SubmissionWatcher(self.b, page)
                await watcher.start()
                shot = await self.b.screenshot(page, f"documents_{self.ats}_{job.job_id}")
                attached = ", ".join(a["value"] for a in answers) or "nothing"
                outcome = await self.gate.wait(
                    f"Attached {attached}. Fill in the rest of the form and submit it yourself, "
                    "then continue.")
                evidence = await watcher.stop()
                if evidence.submitted:
                    return self._result(STATUS_SUBMITTED,
                                        f"Documents attached; you submitted it: {evidence.describe()}",
                                        answers, shot, url)
                if outcome == HumanGate.SKIP:
                    return self._result(STATUS_SKIPPED, "Skipped by you", answers, shot, url)
                return self._result(STATUS_PENDING,
                                    f"Documents attached ({attached}); the rest is yours",
                                    answers, shot, url)

            res = await self.filler.fill_step(scope, ctx)
            answers.extend(res.filled)
            if res.unresolved:
                watcher = SubmissionWatcher(self.b, page)
                await watcher.start()
                outcome = await self.gate.wait(
                    "Could not answer required field(s) truthfully: " + "; ".join(res.unresolved)
                    + ". Fill them in the browser, then continue.")
                evidence = await watcher.stop()
                if evidence.submitted:
                    return self._result(STATUS_SUBMITTED,
                                        f"You filled the rest and submitted it: {evidence.describe()}",
                                        answers, shot, url)
                if outcome == HumanGate.SKIP:
                    return self._result(STATUS_SKIPPED, "Skipped by you at the review step",
                                        answers, shot, url)
            # The form is filled by now, so this is the right moment to deal with a CAPTCHA.
            if await self.b.guard(page) == HumanGate.SKIP:
                return self._result(STATUS_SKIPPED, "Skipped by you at the challenge", answers, shot, url)
            shot = await self.b.screenshot(page, f"review_{self.ats}_{job.job_id}")
            submit = await self.find_submit(scope, page)
            if submit is None:
                watcher = SubmissionWatcher(self.b, page)
                await watcher.start()
                await self.gate.wait("No Submit button found. If the form looks complete, "
                                     "submit it yourself, then continue.")
                evidence = await watcher.stop()
                return self._result(
                    STATUS_SUBMITTED if evidence.submitted else STATUS_PENDING,
                    f"You submitted it: {evidence.describe()}" if evidence.submitted
                    else "No Submit button found on the page", answers, shot, url)
            if not self.s.auto_submit:
                # Watch the page for the whole time you are on it, so a submit you make
                # yourself is recorded even if this site words its confirmation oddly.
                watcher = SubmissionWatcher(self.b, page)
                await watcher.start()
                outcome = await self.gate.wait(
                    "Form filled. Check it in the browser and press Submit yourself, then continue.")
                evidence = await watcher.stop()
                if evidence.submitted:
                    shot = await self.b.screenshot(page, f"submitted_{self.ats}_{job.job_id}") or shot
                    return self._result(STATUS_SUBMITTED,
                                        f"You submitted it: {evidence.describe()}", answers, shot, url)
                if outcome == HumanGate.SKIP:
                    return self._result(STATUS_SKIPPED, "Skipped by you", answers, shot, url)
                return self._result(STATUS_PENDING, "Filled and left for you; no submission seen",
                                    answers, shot, url)
            for attempt in range(1, 4):
                await self.b.human_click(submit)
                if await self.confirmed(page, timeout=15_000):
                    return self._result(STATUS_SUBMITTED, "Auto-submitted", answers, shot, url)
                if await self.b.guard(page) == HumanGate.SKIP:   # a CAPTCHA may follow Submit
                    return self._result(STATUS_SKIPPED, "Skipped by you after submitting",
                                        answers, shot, url)
                if await self.confirmed(page, timeout=4000):
                    return self._result(STATUS_SUBMITTED, "Submitted after human solved challenge", answers, shot, url)
                errs = await self.errors(scope)
                if not errs:
                    watcher = SubmissionWatcher(self.b, page)
                    await watcher.start()
                    await self.gate.wait("Clicked Submit but saw no confirmation. "
                                         "Check the browser and submit it yourself if needed, then continue.")
                    evidence = await watcher.stop()
                    return self._result(
                        STATUS_SUBMITTED if evidence.submitted else STATUS_PENDING,
                        f"You finished it: {evidence.describe()}" if evidence.submitted
                        else "Submit clicked but no confirmation appeared",
                        answers, shot, url)
                log.warning("Validation errors (attempt %d): %s", attempt, errs)
                res = await self.filler.fill_step(scope, ctx, only_errors=True, force_ai=True)
                answers.extend(res.filled)
                if res.unresolved or attempt >= 2:
                    watcher = SubmissionWatcher(self.b, page)
                    await watcher.start()
                    await self.gate.wait(f"The form reports errors: {errs}. "
                                         "Fix them and press Submit yourself, then continue.")
                    evidence = await watcher.stop()
                    if evidence.submitted:
                        return self._result(STATUS_SUBMITTED,
                                            f"You submitted it after fixing the form: {evidence.describe()}",
                                            answers, shot, url)
                submit = await self.find_submit(scope, page) or submit
            return self._result(STATUS_FAILED, "Validation errors persisted after retries", answers, shot, url)
        except Exception as exc:
            log.exception("%s application failed for %s", self.ats, url)
            shot = await self.b.screenshot(page, f"error_{self.ats}_{job.job_id}") or shot
            return self._result(STATUS_FAILED, _short_exc(exc), answers, shot, url)


#: Half of all Greenhouse boards redirect their own job URL to the company's careers
#: site, which shows the description and an "Apply" button but never the form itself.
#: Greenhouse always serves the real form at this embed URL, and it never redirects.
GREENHOUSE_EMBED = "https://job-boards.greenhouse.io/embed/job_app?for={token}&token={job_id}"
GH_BOARD_RE = re.compile(r"(?:job-boards|boards)\.greenhouse\.io/(?!embed)([A-Za-z0-9_.-]+)", re.I)
GH_JOB_IN_PATH_RE = re.compile(r"greenhouse\.io/[A-Za-z0-9_.-]+/jobs/(\d+)", re.I)
GH_JID_RE = re.compile(r"[?&]gh_jid=(\d+)", re.I)
GH_JOB_ID_RE = re.compile(r"^gh-([A-Za-z0-9_.-]+)-(\d+)$")


def greenhouse_embed_url(job: JobPosting) -> Optional[str]:
    """Build the always-working Greenhouse form URL for a posting, if we can identify it."""
    token = job_id = None
    m = GH_JOB_ID_RE.match(job.job_id or "")
    if m:
        token, job_id = m.group(1), m.group(2)
    for url in (job.apply_url, job.url):
        if not url:
            continue
        if token is None:
            b = GH_BOARD_RE.search(url)
            if b:
                token = b.group(1)
        if job_id is None:
            j = GH_JOB_IN_PATH_RE.search(url) or GH_JID_RE.search(url)
            if j:
                job_id = j.group(1)
    if token and job_id:
        return GREENHOUSE_EMBED.format(token=token, job_id=job_id)
    return None


class GreenhouseApplier(SinglePageApplier):
    ats = GREENHOUSE
    form_selectors = ("form#application-form", "form#application_form", "#application",
                      "#app_body", "form")

    async def open_form(self, page: Page, job: JobPosting) -> Optional[Locator]:
        """Reach the application form, whatever the company has done to its careers page.

        The Greenhouse embed form is tried first when the board and job can be
        identified, because it always renders the real form and never redirects.
        Half of all Greenhouse boards bounce their own job URL to a company careers
        page that shows only the description, and those pages often carry their own
        marketing CAPTCHA, which would stop the run before the form is ever reached.

        If the embed is unavailable, fall back to the posting itself, any embedded
        ATS iframe, and finally an Apply button or link on the page.
        """
        embed = greenhouse_embed_url(job)
        if embed:
            log.info("Opening the Greenhouse application form directly: %s", embed)
            await self.b.goto(page, embed)
            form = await self.find_form(page, timeout=15_000)
            if form is not None:
                return form
            log.info("Embed form did not render; falling back to the posting page")

        await self.b.goto(page, self.apply_url(job))
        form = await self.find_in_frames_or_page(page, timeout=8000)
        if form is not None:
            return form
        form = await self.follow_apply_trigger(page)
        if form is not None:
            return form
        log.warning("Could not reach an application form for %s", job.url)
        return None


class LeverApplier(SinglePageApplier):
    ats = LEVER
    form_selectors = ("form#application-form", "form[action*='apply']", "form")
    submit_re = re.compile(r"submit application|submit", re.I)

    def apply_url(self, job: JobPosting) -> str:
        url = (job.apply_url or job.url).split("?")[0].rstrip("/")
        return url if url.endswith("/apply") else url + "/apply"


class AshbyApplier(SinglePageApplier):
    ats = ASHBY
    # Ashby renders no <form> element; the application lives in a plain div#form.
    form_selectors = ("div#form", ".ashby-job-posting-right-pane",
                      "[class*='ashby-application-form' i]", "form", "main")
    submit_re = re.compile(r"submit application|submit", re.I)

    def apply_url(self, job: JobPosting) -> str:
        url = (job.apply_url or job.url).split("?")[0].rstrip("/")
        return url if url.endswith("/application") else url + "/application"




APPLIERS: dict[str, type[BaseApplier]] = {
    LINKEDIN: LinkedInEasyApplyApplier,
    GREENHOUSE: GreenhouseApplier,
    LEVER: LeverApplier,
    ASHBY: AshbyApplier,
}


def get_applier(ats: str, browser: StealthBrowser, filler: FormFiller, gate: HumanGate,
                settings: Settings,
                cover_letter: Optional[Callable[[], Any]] = None) -> Optional[BaseApplier]:
    cls = APPLIERS.get(ats)
    return cls(browser, filler, gate, settings, cover_letter) if cls else None
