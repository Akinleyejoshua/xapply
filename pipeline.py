"""Orchestration: discover -> analyze -> score -> tailor resume -> apply -> record.

Every job produces one database row plus one JSON audit file under
`logs/applications/` containing the full job description, the AI analysis and
every answer the bot typed, so an application can be reviewed after the fact.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_agent import AIAgent, JobAnalysis
from appliers import get_applier
from browser_bot import (AnswerResolver, BrowserRevealed, FormFiller, HumanGate,
                         StealthBrowser)
from config import Settings
from config import settings as default_settings
from database import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SKIPPED,
    STATUS_SUBMITTED,
    Database,
)
from job_search import build_sources
from llm import ModelUnavailable
from models import JobPosting
from resume_builder import ResumeBuilder

log = logging.getLogger(__name__)


def load_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}. Copy profile.example.json to profile.json.")
    profile = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("name", "email", "experience") if not profile.get(k)]
    if missing:
        raise ValueError(f"profile.json is missing required keys: {missing}")
    return {k: v for k, v in profile.items() if not k.startswith("_")}


class Pipeline:
    def __init__(self, settings: Settings = default_settings, db: Optional[Database] = None,
                 gate: Optional[HumanGate] = None):
        self.s = settings
        self.s.ensure_dirs()
        self.db = db or Database(settings.db_path)
        self.db.init()
        self.gate = gate or HumanGate(settings.human_gate_mode, settings.log_dir / "CONTINUE")
        self.profile = load_profile(settings.profile_path)
        self.ai = AIAgent(settings)
        self.resumes = ResumeBuilder(settings)
        self.stats: dict[str, int] = {}

    # ---- audit ---------------------------------------------------------
    def write_audit(self, job: JobPosting, analysis: Optional[JobAnalysis], status: str,
                    note: str, answers: list[dict[str, Any]], resume_path: str, shot: str) -> Path:
        self.s.audit_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = self.s.audit_dir / f"{stamp}_{job.source}_{job.job_id}.json"
        payload = {
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": status,
            "note": note,
            "job": job.to_dict(),
            "analysis": analysis.model_dump() if analysis else None,
            "answers_submitted": answers,
            "resume_path": resume_path,
            "resume_trimmed": getattr(self.resumes, "last_trim", ""),
            "screenshot_path": shot,
            "auto_submit": self.s.auto_submit,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _bump(self, status: str) -> None:
        self.stats[status] = self.stats.get(status, 0) + 1

    # ---- single job ----------------------------------------------------
    async def process(self, browser: StealthBrowser, source: Any, job: JobPosting,
                      _retry: bool = False) -> str:
        """Work one posting. Restarted once if a window has to be opened part way."""
        try:
            return await self._process(browser, source, job)
        except BrowserRevealed as exc:
            if _retry:
                note = f"Still blocked after opening a window: {exc}"
                self.db.record(job, STATUS_FAILED, notes=note)
                self._bump(STATUS_FAILED)
                log.error(note)
                return STATUS_FAILED
            log.info("Starting this posting again now there is a window: %s", job.url)
            return await self.process(browser, source, job, _retry=True)

    async def _process(self, browser: StealthBrowser, source: Any, job: JobPosting) -> str:
        page = browser.page
        log.info("=" * 70)
        log.info("Job %s | %s @ %s | %s", job.job_id, job.title or "?", job.company or "?", job.url)
        try:
            job = await source.hydrate(page, job)
        except Exception as exc:
            log.warning("Could not load posting %s: %s", job.url, exc)
            self.db.record(job, STATUS_FAILED, notes=f"Could not load posting: {exc}")
            self._bump(STATUS_FAILED)
            return STATUS_FAILED

        if len(job.description) < 120:
            note = "Job description too short to analyze"
            self.db.record(job, STATUS_SKIPPED, notes=note)
            self.write_audit(job, None, STATUS_SKIPPED, note, [], "", "")
            self._bump(STATUS_SKIPPED)
            log.info("SKIP: %s", note)
            return STATUS_SKIPPED

        cover_letter = self._cover_letter_factory(job)
        applier = get_applier(job.ats, browser, self.filler, self.gate, self.s,
                              cover_letter=cover_letter)
        email_to = None
        if applier is None and self.s.email_apply:
            # No form to fill, but the posting may simply be asking to be emailed.
            from email_apply import find_address

            page_text = await self._page_text(browser)
            email_to = find_address(job, page_text)
            if email_to:
                log.info("No application form here, but the posting says to write to %s",
                         email_to)
        if applier is None and not email_to:
            note = f"No applier for ATS {job.ats!r} ({job.apply_url or job.url})"
            if not self.s.email_apply:
                note += ". Turn on 'Apply by email' to write to an address the posting gives."
            self.db.record(job, STATUS_SKIPPED, notes=note)
            self.write_audit(job, None, STATUS_SKIPPED, note, [], "", "")
            self._bump(STATUS_SKIPPED)
            log.info("SKIP: %s", note)
            return STATUS_SKIPPED

        analysis = await self.ai.analyze_job(self.profile, job)
        job.title = job.title or analysis.job_title
        job.company = job.company or analysis.company_name

        if analysis.match_score < self.s.match_threshold:
            note = f"Match {analysis.match_score} < threshold {self.s.match_threshold}: {analysis.match_rationale}"
            self.db.record(job, STATUS_SKIPPED, match_score=analysis.match_score, notes=note,
                           analysis=analysis.model_dump(), answers=[])
            self.write_audit(job, analysis, STATUS_SKIPPED, note, [], "", "")
            self._bump(STATUS_SKIPPED)
            log.info("SKIP: %s", note)
            return STATUS_SKIPPED

        resume_path = await self.resumes.build(self.profile, analysis, job)
        trimmed = self.resumes.last_trim
        cover_letter.cache["analysis"] = analysis
        if applier is None:
            return await self._apply_by_email(job, analysis, resume_path, trimmed,
                                              email_to, cover_letter, browser=browser)
        if applier.cover_letter is not None:
            applier.cover_letter.cache["analysis"] = analysis
        mode = {"documents": "attaching documents only", "assisted": "filling the form",
                "auto": "filling and submitting"}[self.s.fill_mode]
        log.info("Match %d -> %s via %s", analysis.match_score, mode, job.ats)
        result = await applier.apply(page, job, analysis, resume_path, self.profile)
        # Record what the resume had to leave out, so a shortened CV is never a surprise.
        if trimmed not in ("nothing trimmed", "overflow"):
            result.answers.insert(0, {
                "label": "Resume shortened", "kind": "note", "value": trimmed,
                "source": f"fits {self.s.resume_max_pages} page(s)", "confidence": 1.0, "ok": True})
        audit = self.write_audit(job, analysis, result.status, result.note, result.answers,
                                 str(resume_path), result.screenshot_path)
        self.db.record(
            job, result.status, match_score=analysis.match_score,
            notes=f"{result.note} | audit: {audit.name}", resume_path=str(resume_path),
            screenshot_path=result.screenshot_path, analysis=analysis.model_dump(), answers=result.answers,
        )
        self._bump(result.status)
        log.info("%s: %s", result.status.upper(), result.note)
        return result.status

    @staticmethod
    async def _page_text(browser: StealthBrowser) -> str:
        """The visible text of whatever is open, for finding an address in it."""
        try:
            return await browser.page.evaluate(
                "() => document.body ? document.body.innerText.slice(0, 20000) : ''")
        except Exception:
            return ""

    async def _apply_by_email(self, job: JobPosting, analysis: JobAnalysis,
                              resume_path: Path, trimmed: str, email_to: str,
                              cover_letter: Any, browser: Any = None) -> str:
        """Send the documents to the address the posting gives, once you have read it."""
        from email_apply import EmailApplier

        mailer = EmailApplier(self.s, self.gate, browser=browser)
        missing = mailer.missing_settings()
        if missing:
            note = ("Apply by email is on, but " + ", ".join(missing) +
                    " is not set in .env, so nothing was sent.")
            self.db.record(job, STATUS_FAILED, match_score=analysis.match_score, notes=note,
                           analysis=analysis.model_dump(), resume_path=str(resume_path))
            self._bump(STATUS_FAILED)
            log.error(note)
            return STATUS_FAILED

        letter_pdf, letter_text = await cover_letter()
        draft = mailer.compose(job, email_to, self.profile, letter_text,
                               [resume_path, letter_pdf])
        preview = mailer.save_preview(draft, job)
        log.info("Draft saved to %s", preview)

        status, note = STATUS_PENDING, ""
        if not self.s.email_auto_send:
            outcome = await self.gate.wait(
                f"About to email your application to {email_to}.\n"
                f"{draft.describe()[:600]}\n"
                f"The full message is at {preview}. Continue to send it, or skip.")
            if outcome == HumanGate.SKIP:
                note = f"You chose not to email {email_to}"
                self.db.record(job, STATUS_SKIPPED, match_score=analysis.match_score,
                               notes=note, analysis=analysis.model_dump(),
                               resume_path=str(resume_path))
                self._bump(STATUS_SKIPPED)
                log.info("SKIP: %s", note)
                return STATUS_SKIPPED
        try:
            await mailer.send(draft)
            status = STATUS_SUBMITTED
            note = f"Emailed to {email_to} with {len(draft.attachments)} attachment(s)"
        except Exception as exc:          # SMTP failures come in many shapes
            status, note = STATUS_FAILED, f"Could not send to {email_to}: {exc}"
            log.error(note)

        answers = [{"label": "Sent to", "kind": "email", "value": email_to,
                    "source": "posting", "confidence": 1.0, "ok": status == STATUS_SUBMITTED},
                   {"label": "Subject", "kind": "email", "value": draft.subject,
                    "source": "composed", "confidence": 1.0, "ok": True}]
        if trimmed not in ("nothing trimmed", "overflow"):
            answers.insert(0, {"label": "Resume shortened", "kind": "note", "value": trimmed,
                               "source": f"fits {self.s.resume_max_pages} page(s)",
                               "confidence": 1.0, "ok": True})
        audit = self.write_audit(job, analysis, status, note, answers, str(resume_path), "")
        self.db.record(job, status, match_score=analysis.match_score,
                       notes=f"{note} | audit: {audit.name}", resume_path=str(resume_path),
                       analysis=analysis.model_dump(), answers=answers)
        self._bump(status)
        log.info("%s: %s", status.upper(), note)
        return status

    def _cover_letter_factory(self, job: JobPosting):
        """Write this posting's cover letter at most once, and only if a form asks.

        Most applications never request one, so generating eagerly would spend an API
        call per job for nothing.
        """
        cache: dict[str, Any] = {}

        async def make() -> tuple[Path, str]:
            if "result" in cache:
                return cache["result"]
            analysis = cache.get("analysis")
            letter = await self.ai.write_cover_letter(self.profile, job, analysis)
            pdf = await self.resumes.build_cover_letter(self.profile, letter, job, analysis)
            cache["result"] = (pdf, letter.as_text())
            log.info("Cover letter ready: %s", pdf.name)
            return cache["result"]

        make.cache = cache          # the pipeline drops the analysis in before applying
        return make

    # ---- run -----------------------------------------------------------
    async def preflight(self) -> None:
        """Confirm the configured model answers before any posting is touched.

        A model that cannot be reached fails identically for every job, so checking
        once turns a run that would record dozens of useless failures into a single
        clear message.
        """
        from llm import ModelUnavailable, check_model

        result = await check_model(self.s, self.s.llm_provider, self.s.active_model)
        if result.get("ok") or result.get("status") == 503:
            log.info("Model check passed: %s / %s", self.s.llm_provider, self.s.active_model)
            return
        if result.get("transient"):
            # The provider was never reached, so nothing was learned about the model.
            # Stopping here would blame a model that may be perfectly fine.
            log.warning("Could not verify the model: %s. Carrying on; a posting that "
                        "cannot be scored will say so itself.", result.get("detail", ""))
            return
        raise ModelUnavailable(
            f"{self.s.llm_provider} cannot use {self.s.active_model!r}, so nothing can be "
            f"scored.\n  {result.get('detail', '')}\n"
            f"  Open Settings and press 'Check which models work', or run: "
            f"python main.py models --verify"
        )

    async def run(self, urls: Optional[list[str]] = None, limit: Optional[int] = None) -> dict[str, int]:
        limit = limit or self.s.max_applications_per_run
        await self.preflight()
        async with StealthBrowser(self.s, self.gate) as browser:
            resolver = AnswerResolver(self.ai, self.s)
            self.filler = FormFiller(browser, resolver, self.s)
            sources = build_sources(self.s, browser, self.db, urls)
            if not sources:
                log.error("No job sources configured (SOURCES=%s)", self.s.sources)
                return self.stats
            processed = 0
            for source in sources:
                log.info("Discovering jobs from %s ...", source.name)
                try:
                    jobs = await source.discover(browser.page)
                except Exception as exc:
                    log.exception("Discovery failed for %s: %s", source.name, exc)
                    continue
                log.info("%s: %d new postings", source.name, len(jobs))
                for job in jobs:
                    if processed >= limit:
                        log.info("Reached MAX_APPLICATIONS_PER_RUN=%d", limit)
                        break
                    try:
                        await self.process(browser, source, job)
                    except KeyboardInterrupt:
                        raise
                    except ModelUnavailable:
                        # Every other posting would fail the same way, so stop cleanly
                        # rather than filling the database with identical failures.
                        log.error("Stopping the run: the model is unavailable")
                        raise
                    except Exception as exc:
                        log.exception("Unhandled error on %s", job.url)
                        self.db.record(job, STATUS_FAILED, notes=f"Unhandled: {exc}")
                        self._bump(STATUS_FAILED)
                    processed += 1
                    await browser.sleep(4.0, 1.5)
                if processed >= limit:
                    break
        return self.stats

    async def email_one(self, url: str, to: Optional[str] = None) -> dict[str, Any]:
        """Apply to one posting by writing to whoever it says to write to.

        The whole email route in one call, so it can be run against a link you found
        yourself rather than only reaching it by chance in the middle of a scan.
        """
        from email_apply import find_address

        async with StealthBrowser(self.s, self.gate) as browser:
            resolver = AnswerResolver(self.ai, self.s)
            self.filler = FormFiller(browser, resolver, self.s)
            job = JobPosting.from_url(url, source="urls")
            job.apply_url = ""
            try:
                source = build_sources(self.s, browser, self.db, [url])[0]
                job = await source.hydrate(browser.page, job)
            except Exception as exc:
                log.warning("Could not read %s: %s", url, exc)
            page_text = await self._page_text(browser)
            if not job.description or len(job.description) < 120:
                job.description = page_text
            address = (to or "").strip() or find_address(job, page_text)
            if not address:
                raise ValueError(
                    f"No application address on {url}. The posting may use a form "
                    f"instead, or give the address as an image. You can supply one "
                    f"yourself if you know it.")
            job.title = job.title or "Role"
            analysis = await self.ai.analyze_job(self.profile, job)
            job.title = job.title or analysis.job_title
            job.company = job.company or analysis.company_name
            resume = await self.resumes.build(self.profile, analysis, job)
            cover = self._cover_letter_factory(job)
            cover.cache["analysis"] = analysis
            status = await self._apply_by_email(job, analysis, resume, self.resumes.last_trim,
                                                address, cover, browser=browser)
        return {"status": status, "to": address, "job": job.to_dict(),
                "resume_path": str(resume)}

    async def check_email_account(self) -> dict[str, Any]:
        """Open Gmail and say whether the session is still good. Sends nothing."""
        from email_apply import EmailApplier, GmailTransport, NotConfigured

        mailer = EmailApplier(self.s, getattr(self, "gate", None))
        if mailer.transport != "gmail":
            missing = mailer.missing_settings()
            return {"transport": mailer.transport, "ready": not missing,
                    "detail": ("Ready to send through " + (self.s.smtp_host or "your mail server"))
                    if not missing else "Still needs " + ", ".join(missing) + " in .env"}
        async with StealthBrowser(self.s, self.gate) as browser:
            try:
                await GmailTransport(self.s, browser).open_mail()
            except NotConfigured as exc:
                return {"transport": "gmail", "ready": False, "detail": str(exc)}
            except Exception as exc:
                return {"transport": "gmail", "ready": False,
                        "detail": f"Could not open Gmail: {exc}"}
        return {"transport": "gmail", "ready": True,
                "detail": "Signed in to Gmail. Applications will be sent from this account."}

    async def analyze_only(self, url: str) -> dict[str, Any]:
        """Dry run: fetch a posting, score it, build the resume, apply nothing."""
        async with StealthBrowser(self.s, self.gate) as browser:
            resolver = AnswerResolver(self.ai, self.s)
            self.filler = FormFiller(browser, resolver, self.s)
            source = build_sources(self.s, browser, self.db, [url])[0]
            job = JobPosting.from_url(url, source=source.name)
            job = await source.hydrate(browser.page, job)
            analysis = await self.ai.analyze_job(self.profile, job)
            resume = await self.resumes.build(self.profile, analysis, job)
        return {"job": job.to_dict(), "analysis": analysis.model_dump(), "resume_path": str(resume)}


#: Sites worth signing in to once, and where their sign-in page lives.
SIGN_IN_PAGES: dict[str, str] = {
    "linkedin": "https://www.linkedin.com/login",
    "gmail": "https://mail.google.com/mail/u/0/",
    "google": "https://accounts.google.com/",
    "x": "https://x.com/login",
    "twitter": "https://x.com/login",
    "indeed": "https://secure.indeed.com/auth",
    "glassdoor": "https://www.glassdoor.com/profile/login_input.htm",
    "wellfound": "https://wellfound.com/login",
    "greenhouse": "https://my.greenhouse.io/applications",
}


async def login_flow(settings: Settings = default_settings, site: str = "linkedin") -> None:
    """Open the browser so you can sign in once. The profile keeps it after that.

    There is no password anywhere in this project. You sign in yourself, including
    whatever second factor the site asks for, and the browser profile directory holds
    the cookies afterwards, exactly as your everyday browser does. A session lasts until
    you sign out or delete the profile, so this is a thing you do once per site.
    """
    target = SIGN_IN_PAGES.get(site.strip().lower(), site)
    if not target.startswith(("http://", "https://")):
        raise ValueError(f"Unknown site {site!r}. Known: {', '.join(sorted(SIGN_IN_PAGES))}, "
                         f"or give a full URL.")
    gate = HumanGate(settings.human_gate_mode, settings.log_dir / "CONTINUE")
    # A sign-in needs a window whatever the usual setting says: it is the one thing
    # that cannot happen without you.
    settings.headless = False
    settings.hide_browser = False
    async with StealthBrowser(settings, gate) as browser:
        await browser.page.goto(target, wait_until="domcontentloaded")
        await gate.wait(f"Sign in at {target} in the browser window, then continue. "
                        f"The session is kept in {settings.user_data_dir.name} and reused "
                        f"on every run until you sign out.", allow_skip=False)
        log.info("Session for %s stored in %s", site, settings.user_data_dir)
