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
from typing import Any, Optional

from playwright.async_api import Locator, Page, TimeoutError as PlaywrightTimeout

from ai_agent import JobAnalysis
from browser_bot import FormFiller, HumanGate, ResolveContext, StealthBrowser
from config import Settings
from database import STATUS_FAILED, STATUS_PENDING, STATUS_SKIPPED, STATUS_SUBMITTED
from models import ASHBY, GREENHOUSE, LEVER, LINKEDIN, ApplyResult, JobPosting

log = logging.getLogger(__name__)

CONFIRM_TEXT_RE = re.compile(
    r"(thank you for applying|thanks for applying|application (has been |was )?(submitted|sent|received)"
    r"|we('ve| have) received your application|successfully submitted|your application was sent"
    r"|application submitted|applied successfully)",
    re.I,
)
CONFIRM_URL_HINTS = ("/thanks", "/thank-you", "/confirmation", "/submitted", "/success")
ERROR_SELECTOR = (
    '.artdeco-inline-feedback--error, [role="alert"], [class*="error-message" i], [class*="form-error" i], '
    '[class*="field-error" i], [class*="errorMessage" i], .error-text, [class*="_error" i]:not(input)'
)
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

RESUME_FILE_INPUTS = (
    'input[type="file"][name*="resume" i], input[type="file"][id*="resume" i], '
    'input[type="file"][name*="cv" i], input[type="file"][id*="cv" i], '
    'input[type="file"][accept*="pdf" i], input[type="file"]'
)


def _short_exc(exc: BaseException, limit: int = 300) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:limit] if str(exc) else ''}".strip()


class BaseApplier:
    ats = "unknown"

    def __init__(self, browser: StealthBrowser, filler: FormFiller, gate: HumanGate, settings: Settings):
        self.b = browser
        self.filler = filler
        self.gate = gate
        self.s = settings

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
                    uploaded = await self.upload_resume(page, modal, resume_path)
                    if uploaded:
                        answers.append({"label": "Resume", "kind": "file", "value": resume_path.name,
                                        "source": "resume_builder", "confidence": 1.0, "ok": True})
                res = await self.filler.fill_step(modal, ctx)
                answers.extend(res.filled)
                if res.unresolved:
                    await self.gate.wait("Could not confidently answer required field(s): "
                                         + "; ".join(res.unresolved) + ". Fill them in the browser, then continue.")
                    modal = await self.modal(page) or modal
                submit = await self._button(modal, self.SUBMIT_RE)
                if submit is not None:
                    await self._uncheck_follow(modal)
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
                    await self.gate.wait("Final review page reached. Verify the form and click "
                                         "'Submit application' yourself, then continue.")
                    if await self.confirmed(page, timeout=3000) or await self.already_applied(page) \
                            or await self.modal(page) is None:
                        await self._dismiss_post_apply(page)
                        return self._result(STATUS_SUBMITTED, "Submitted by human (assisted mode)", answers, shot, job.url)
                    await self._discard(page)
                    return self._result(STATUS_PENDING, "Reached review page; human chose not to submit", answers, shot, job.url)
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
        return await self.find_form(page)

    def apply_url(self, job: JobPosting) -> str:
        return job.apply_url or job.url

    async def find_form(self, page: Page, timeout: int = 15_000) -> Optional[Locator]:
        """Locate the form scope, falling back to the DOM when no selector matches.

        Not every ATS wraps its application in a `<form>`: Ashby renders plain divs.
        So after the configured selectors, the smallest element that still contains
        every visible input is tagged and used as the scope.
        """
        per_selector = max(2000, timeout // max(1, len(self.form_selectors)))
        for sel in self.form_selectors:
            loc = page.locator(sel).filter(has=page.locator('input:not([type="hidden"]), textarea, select'))
            try:
                await loc.first.wait_for(state="visible", timeout=per_selector)
                return loc.first
            except PlaywrightTimeout:
                continue
        return await self.find_form_container(page)

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
            if await self.upload_resume(page, scope, resume_path):
                answers.append({"label": "Resume", "kind": "file", "value": resume_path.name,
                                "source": "resume_builder", "confidence": 1.0, "ok": True})
            await self.b.human_scroll(page, 700)
            res = await self.filler.fill_step(scope, ctx)
            answers.extend(res.filled)
            if res.unresolved:
                await self.gate.wait("Could not confidently answer required field(s): "
                                     + "; ".join(res.unresolved) + ". Fill them in the browser, then continue.")
            await self.b.guard(page)
            shot = await self.b.screenshot(page, f"review_{self.ats}_{job.job_id}")
            submit = await self.find_submit(scope, page)
            if submit is None:
                await self.gate.wait("Submit button not found. If the form is complete, submit it manually, then continue.")
                ok = await self.confirmed(page, timeout=3000)
                return self._result(STATUS_SUBMITTED if ok else STATUS_PENDING,
                                    "Submitted manually" if ok else "Submit button not found", answers, shot, url)
            if not self.s.auto_submit:
                await self.gate.wait("Form filled. Verify it in the browser and click Submit yourself, then continue.")
                ok = await self.confirmed(page, timeout=3000)
                return self._result(STATUS_SUBMITTED if ok else STATUS_PENDING,
                                    "Submitted by human (assisted mode)" if ok else "Human chose not to submit",
                                    answers, shot, url)
            for attempt in range(1, 4):
                await self.b.human_click(submit)
                if await self.confirmed(page, timeout=15_000):
                    return self._result(STATUS_SUBMITTED, "Auto-submitted", answers, shot, url)
                await self.b.guard(page)  # a CAPTCHA may appear only after Submit
                if await self.confirmed(page, timeout=4000):
                    return self._result(STATUS_SUBMITTED, "Submitted after human solved challenge", answers, shot, url)
                errs = await self.errors(scope)
                if not errs:
                    await self.gate.wait("Clicked Submit but no confirmation was detected. "
                                         "Check the browser and submit manually if needed, then continue.")
                    ok = await self.confirmed(page, timeout=3000)
                    return self._result(STATUS_SUBMITTED if ok else STATUS_PENDING,
                                        "Submitted by human after bot attempt" if ok else "No confirmation after submit",
                                        answers, shot, url)
                log.warning("Validation errors (attempt %d): %s", attempt, errs)
                res = await self.filler.fill_step(scope, ctx, only_errors=True, force_ai=True)
                answers.extend(res.filled)
                if res.unresolved or attempt >= 2:
                    await self.gate.wait(f"Form validation errors: {errs}. Fix them and click Submit yourself, then continue.")
                    if await self.confirmed(page, timeout=3000):
                        return self._result(STATUS_SUBMITTED, "Submitted by human after validation errors", answers, shot, url)
                submit = await self.find_submit(scope, page) or submit
            return self._result(STATUS_FAILED, "Validation errors persisted after retries", answers, shot, url)
        except Exception as exc:
            log.exception("%s application failed for %s", self.ats, url)
            shot = await self.b.screenshot(page, f"error_{self.ats}_{job.job_id}") or shot
            return self._result(STATUS_FAILED, _short_exc(exc), answers, shot, url)


class GreenhouseApplier(SinglePageApplier):
    ats = GREENHOUSE
    form_selectors = ("form#application-form", "form#application_form", "#application", "form")

    async def open_form(self, page: Page, job: JobPosting) -> Optional[Locator]:
        await self.b.goto(page, self.apply_url(job))
        # Embedded boards on company sites load the real form in an iframe: go to its source directly.
        frame = page.locator("iframe#grnhse_iframe")
        if await frame.count():
            src = await frame.first.get_attribute("src")
            if src:
                await self.b.goto(page, src)
        # Some boards hide the form behind an "Apply" button.
        form = await self.find_form(page, timeout=6000)
        if form is None:
            btn = page.get_by_role("button", name=re.compile(r"^apply( now| for this job)?$", re.I))
            if await btn.count():
                await self.b.human_click(btn.first)
                form = await self.find_form(page)
        return form


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

    async def open_form(self, page: Page, job: JobPosting) -> Optional[Locator]:
        await self.b.goto(page, self.apply_url(job))
        form = await self.find_form(page, timeout=8000)
        if form is None:
            tab = page.get_by_role("link", name=re.compile(r"^application|^apply", re.I))
            if await tab.count():
                await self.b.human_click(tab.first)
                form = await self.find_form(page)
        return form


APPLIERS: dict[str, type[BaseApplier]] = {
    LINKEDIN: LinkedInEasyApplyApplier,
    GREENHOUSE: GreenhouseApplier,
    LEVER: LeverApplier,
    ASHBY: AshbyApplier,
}


def get_applier(ats: str, browser: StealthBrowser, filler: FormFiller, gate: HumanGate,
                settings: Settings) -> Optional[BaseApplier]:
    cls = APPLIERS.get(ats)
    return cls(browser, filler, gate, settings) if cls else None
