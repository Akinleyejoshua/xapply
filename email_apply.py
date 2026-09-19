"""Applying by email, for postings that ask for one instead of offering a form.

Plenty of roles never reach an applicant tracking system. The posting says "send your
CV to careers@example.com" and that is the whole process. This module does that: it
finds the address in the posting, attaches the tailored resume and cover letter, and
sends the message.

Sending email is the one thing this tool does that cannot be undone, so it is off by
default, it never invents a recipient, and it stops for you to read the draft before
anything leaves. Turning that confirmation off is a separate setting from turning the
feature on, because they are different decisions.
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid, parseaddr
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Iterable, Optional

from config import Settings
from models import JobPosting, ats_text

log = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
MAILTO_RE = re.compile(r"mailto:([^\"'>\s?]+)", re.I)

#: Local parts that mean "applications go here", best first. A posting often carries
#: several addresses and only one of them is the one being asked for.
APPLY_LOCALS = ("careers", "career", "jobs", "job", "recruiting", "recruitment", "recruit",
                "hiring", "hire", "talent", "apply", "applications", "application", "hr",
                "people", "cv", "resumes", "resume")
#: Local parts that are never where an application goes, however prominent they are.
NEVER_APPLY = ("noreply", "no-reply", "donotreply", "do-not-reply", "support", "help",
               "privacy", "legal", "press", "media", "abuse", "security", "billing",
               "sales", "marketing", "newsletter", "unsubscribe", "webmaster", "postmaster")
#: Domains that host something other than the employer.
NEVER_DOMAINS = ("example.com", "sentry.io", "wixpress.com", "squarespace.com", "google.com",
                 "gstatic.com", "schema.org", "w3.org")


def _local(address: str) -> str:
    return address.split("@", 1)[0].lower()


def _domain(address: str) -> str:
    return address.split("@", 1)[-1].lower()


def plausible(address: str) -> bool:
    """Whether an address could be where an application is meant to go."""
    if not address or address.count("@") != 1:
        return False
    local, domain = _local(address), _domain(address)
    if any(bad in local for bad in NEVER_APPLY):
        return False
    if any(domain == bad or domain.endswith("." + bad) for bad in NEVER_DOMAINS):
        return False
    return "." in domain and len(local) > 1


def rank(address: str) -> tuple[int, int]:
    """Sort key: an address named for applications beats a person's, which beats the rest."""
    local = _local(address)
    for i, wanted in enumerate(APPLY_LOCALS):
        if local == wanted:
            return (0, i)
        if wanted in local:
            return (1, i)
    return (2, 0)


def find_addresses(*texts: Optional[str]) -> list[str]:
    """Every address in the posting that could be an application address, best first."""
    found: list[str] = []
    for text in texts:
        if not text:
            continue
        for match in MAILTO_RE.findall(text):
            found.append(match.strip().lower())
        found.extend(m.lower() for m in EMAIL_RE.findall(text))
    seen: dict[str, None] = {}
    for address in found:
        address = address.strip(".,;:<>()[]'\"")
        if plausible(address) and address not in seen:
            seen[address] = None
    return sorted(seen, key=rank)


#: Sites that list other people's jobs. Their own addresses are never where an
#: application goes, and their pages are indexes rather than postings, so a scan that
#: keeps them returns a pile of results with nobody to write to.
JOB_BOARDS = (
    "indeed.com", "linkedin.com", "glassdoor.com", "ziprecruiter.com", "monster.com",
    "simplyhired.com", "careerbuilder.com", "totaljobs.com", "reed.co.uk", "seek.com",
    "jobberman.com", "myjobmag.com", "brightermonday.com", "jobsdb.com", "naukri.com",
    "wellfound.com", "angel.co", "otta.com", "welcometothejungle.com", "workable.com",
    "greenhouse.io", "lever.co", "ashbyhq.com", "smartrecruiters.com", "bamboohr.com",
    "jobvite.com", "icims.com", "taleo.net", "successfactors.com", "workday.com",
    "remoteok.com", "weworkremotely.com", "himalayas.app", "remotive.com", "flexjobs.com",
    "glassdoor.co.uk", "jooble.org", "adzuna.com", "talent.com", "jobstreet.com",
)

#: The wording that means "this is where to send it". An address has to sit near one of
#: these, or any address anywhere on the page counts, including a blog author's.
APPLY_NEAR_RE = re.compile(
    r"(send|email|e-mail|forward|submit|share|apply|applications?|cv|r[ée]sum[ée]|"
    r"interested|write)\b", re.I)
#: How far from that wording an address may sit and still be the one being offered.
NEAR_CHARS = 220


def is_job_board(url_or_address: str) -> bool:
    """Whether this is a site that lists other people's jobs."""
    text = (url_or_address or "").lower()
    host = text.split("@")[-1]
    if "//" in text:
        host = (urlparse(text).hostname or "").lower()
    return any(host == board or host.endswith("." + board) for board in JOB_BOARDS)


def application_address(text: str, url: str = "") -> Optional[str]:
    """The address a posting is asking you to write to, or None.

    Two things separate that from any other address on a page. It sits next to the
    wording that offers it, and it does not belong to a site that lists other people's
    jobs. Without those, a scan returns board index pages and blog posts, which have
    addresses and nobody to apply to.

    The page it appears on is deliberately not judged. A recruiter posting "send your CV
    to jane@company.com" on LinkedIn or X is exactly the kind of advert this exists to
    find, and rejecting the whole page for its domain threw those away. A board's index
    page is still refused, because the only addresses on it are its own.
    """
    body = text or ""
    near: list[str] = []
    for match in EMAIL_RE.finditer(body):
        address = match.group(0).strip(".,;:<>()[]'\"").lower()
        if not plausible(address) or is_job_board(address):
            continue
        window = body[max(0, match.start() - NEAR_CHARS): match.end() + NEAR_CHARS]
        if APPLY_NEAR_RE.search(window):
            near.append(address)
    if not near:
        return None
    return sorted(dict.fromkeys(near), key=rank)[0]


def find_address(job: JobPosting, page_text: str = "") -> Optional[str]:
    """The one address to apply to, or None when the posting names none.

    Checked against the posting's own words first, then the page it came from. A
    posting already known to be an application by email is trusted with a looser
    reading, because getting this far means something already decided it was one.
    """
    for body in (job.description, page_text):
        found = application_address(body or "", job.url)
        if found:
            return found
    loose = [a for a in find_addresses(job.description, page_text) if not is_job_board(a)]
    return loose[0] if loose else None


@dataclass
class Draft:
    """A message, before anything is sent."""

    to: str
    subject: str
    body: str
    attachments: list[Path]
    from_address: str = ""
    reply_to: str = ""

    def describe(self) -> str:
        names = ", ".join(p.name for p in self.attachments) or "nothing"
        return (f"To: {self.to}\nSubject: {self.subject}\nAttachments: {names}\n\n"
                f"{self.body}")

    def as_message(self) -> EmailMessage:
        message = EmailMessage()
        message["To"] = self.to
        message["From"] = self.from_address
        message["Subject"] = self.subject
        message["Date"] = formatdate(localtime=True)
        message["Message-ID"] = make_msgid()
        if self.reply_to:
            message["Reply-To"] = self.reply_to
        message.set_content(self.body)
        for path in self.attachments:
            kind, _ = mimetypes.guess_type(path.name)
            main, _, sub = (kind or "application/octet-stream").partition("/")
            try:
                message.add_attachment(path.read_bytes(), maintype=main, subtype=sub,
                                       filename=path.name)
            except OSError as exc:
                log.warning("Could not attach %s: %s", path, exc)
        return message


def _comparable(text: str) -> str:
    """Collapsed to letters and single spaces, so wrapping and ellipses do not matter."""
    return re.sub(r"\s+", " ", (text or "").replace("\u2026", " ")).strip().lower()


class NotConfigured(RuntimeError):
    """No mail server is set up, so nothing can be sent."""


class SendRefused(RuntimeError):
    """The message was composed but not sent, so nothing should be recorded as sent."""


class EmailApplier:
    """Compose and send one application email."""

    #: Subject lines recruiters expect. The role first, because that is what they filter on.
    SUBJECT = "Application for {title}{company} - {name}"

    def __init__(self, settings: Settings, gate: Any = None, browser: Any = None):
        self.s = settings
        self.gate = gate
        #: Needed only by the gmail transport, which drives a real signed-in session.
        self.browser = browser

    @property
    def transport(self) -> str:
        return (self.s.email_transport or "smtp").lower()

    # ---- configuration -------------------------------------------------
    @property
    def from_address(self) -> str:
        return (self.s.email_from or self.s.smtp_user or "").strip()

    def missing_settings(self) -> list[str]:
        """Which settings are still needed before anything can be sent.

        Nothing, for the gmail transport: the session is the credential, and whether it
        is still good is only knowable by opening Gmail, which `send` does.
        """
        if self.transport == "gmail":
            return []
        needed = {"SMTP_HOST": self.s.smtp_host, "SMTP_USER": self.s.smtp_user,
                  "SMTP_PASSWORD": self.s.smtp_password}
        missing = [name for name, value in needed.items() if not str(value or "").strip()]
        if not self.from_address:
            missing.append("EMAIL_FROM")
        return missing

    # ---- composing -----------------------------------------------------
    def compose(self, job: JobPosting, to: str, profile: dict[str, Any],
                letter_text: str, attachments: Iterable[Path]) -> Draft:
        """Build the message. Nothing here reaches the network."""
        name = (profile.get("name") or "").strip()
        company = f" at {job.company}" if job.company else ""
        subject = self.SUBJECT.format(title=job.title or "your open role",
                                      company=company, name=name or "application")
        body = letter_text.strip()
        contact = self.signature(profile, job)
        if contact:
            body = f"{body}\n\n{contact}"
        files = [Path(p) for p in attachments if p and Path(p).exists()]
        return Draft(to=to, subject=ats_text(subject), body=ats_text(body),
                     attachments=files, from_address=self.from_address,
                     reply_to=(self.s.email_reply_to or profile.get("email") or "").strip())

    @staticmethod
    def signature(profile: dict[str, Any], job: JobPosting) -> str:
        """Name and the links a recruiter will want, and nothing else."""
        lines = [(profile.get("name") or "").strip()]
        for key, label in (("email", ""), ("phone", ""), ("linkedin", ""),
                           ("github", ""), ("website", "")):
            value = str(profile.get(key) or "").strip()
            if value:
                lines.append(f"{label}{value}" if label else value)
        return "\n".join(line for line in lines if line)

    # ---- sending -------------------------------------------------------
    def _send_blocking(self, draft: Draft) -> None:
        """The actual SMTP conversation. Runs off the event loop."""
        message = draft.as_message()
        port = int(self.s.smtp_port)
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(self.s.smtp_host, port, context=context,
                                  timeout=self.s.smtp_timeout_s) as server:
                server.login(self.s.smtp_user, self.s.smtp_password)
                server.send_message(message)
            return
        with smtplib.SMTP(self.s.smtp_host, port, timeout=self.s.smtp_timeout_s) as server:
            server.ehlo()
            if self.s.smtp_starttls:
                server.starttls(context=context)
                server.ehlo()
            server.login(self.s.smtp_user, self.s.smtp_password)
            server.send_message(message)

    async def send(self, draft: Draft) -> None:
        """Send, after checking there is somewhere to send from and to."""
        if not parseaddr(draft.to)[1]:
            raise ValueError(f"{draft.to!r} is not an address to send to")
        if self.transport == "gmail":
            if self.browser is None:
                raise NotConfigured(
                    "The gmail transport sends through the browser, and there is none "
                    "open. Use the smtp transport, or run this from an apply run.")
            log.info("Sending the application to %s through your signed-in Gmail", draft.to)
            await GmailTransport(self.s, self.browser).send(draft)
            return
        missing = self.missing_settings()
        if missing:
            raise NotConfigured(
                "Email applying needs " + ", ".join(missing) + " in .env, or set "
                "EMAIL_TRANSPORT=gmail to send through your signed-in Gmail instead. "
                "For an SMTP password Gmail wants an app password, not your account one.")
        log.info("Sending the application to %s via %s", draft.to, self.s.smtp_host)
        await asyncio.to_thread(self._send_blocking, draft)

    def save_preview(self, draft: Draft, job: JobPosting) -> Path:
        """Write the message to disk so it can be read before or after it is sent."""
        folder = Path(self.s.log_dir) / "emails"
        folder.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{job.company or 'job'}_{job.job_id}")[:80]
        path = folder / f"{stem}.eml"
        try:
            path.write_bytes(bytes(draft.as_message()))
        except OSError as exc:
            log.warning("Could not save the email preview: %s", exc)
        return path


class GmailTransport:
    """Send through your signed-in Gmail, in the browser, rather than through SMTP.

    You sign in once with `make gmail-login`. The browser profile keeps that session,
    so there is no password anywhere in this project, two-factor authentication is
    something you did in your own browser, and the message lands in your real Sent
    folder where you can see it and reply from it.

    The cost is that this drives a web page, and web pages change. Every step here
    tries several ways of finding the same control, and anything it cannot find is
    reported plainly rather than guessed at, because a half-filled compose window that
    gets sent anyway is the worst outcome available.
    """

    #: Gmail's compose controls, most reliable selector first.
    COMPOSE = ('div[role="button"][gh="cm"]', 'div[gh="cm"]',
               'div[role="button"]:has-text("Compose")', 'text=Compose')
    TO = ('input[aria-label="To recipients"]', 'input[peoplekit-id][aria-label*="To"]',
          'textarea[name="to"]', 'input[name="to"]')
    SUBJECT = ('input[name="subjectbox"]', 'input[aria-label="Subject"]')
    BODY = ('div[aria-label="Message Body"]', 'div[role="textbox"][contenteditable="true"]')
    SEND = ('div[role="button"][aria-label^="Send"]', 'div[data-tooltip^="Send"]',
            'div[role="button"]:has-text("Send")')
    FILE_INPUT = 'input[type="file"]'
    #: Where Google sends you when the session has lapsed.
    SIGNED_OUT = ("accounts.google.com", "/ServiceLogin", "/signin")

    def __init__(self, settings: Settings, browser: Any):
        self.s = settings
        self.b = browser

    async def _first(self, page: Any, selectors: Iterable[str], what: str,
                     timeout: int = 15_000) -> Any:
        """The first of these controls that is actually on the page."""
        last: Optional[Exception] = None
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                await locator.wait_for(state="visible", timeout=timeout // len(tuple(selectors)) or 3000)
                return locator
            except Exception as exc:
                last = exc
        raise LookupError(f"Could not find the {what} in Gmail. The page may have changed, "
                          f"or you may be signed out. Last error: {last}")

    async def signed_in(self, page: Any) -> bool:
        return not any(mark in (page.url or "") for mark in self.SIGNED_OUT)

    async def open_mail(self) -> Any:
        """Open Gmail and confirm the session is still good."""
        page = self.b.page
        await self.b.goto(page, self.s.gmail_url)
        if not await self.signed_in(page):
            raise NotConfigured(
                "Not signed in to Gmail in this browser profile. Run `make gmail-login`, "
                "sign in once, and it will be remembered.")
        return page

    async def send(self, draft: Draft) -> None:
        """Compose and send, and do not return until Gmail says it went.

        The previous version clicked Send and logged success without looking. Anything
        that went wrong after the click, most often a recipient Gmail never accepted,
        left the message sitting in Drafts while the application was recorded as sent.
        A claim nobody checked is worse than a failure, because you stop watching.
        """
        page = await self.open_mail()
        log.info("Composing in Gmail")
        await (await self._first(page, self.COMPOSE, "Compose button")).click()
        await asyncio.sleep(1.2)

        to_field = await self._first(page, self.TO, "To field")
        await to_field.click()
        await to_field.fill(draft.to)
        # Gmail only accepts a recipient once it has been committed to a chip. Filling
        # the box leaves the text uncommitted, and Send then refuses the whole message.
        await page.keyboard.press("Tab")
        await asyncio.sleep(0.4)

        await (await self._first(page, self.SUBJECT, "Subject field")).fill(draft.subject)
        body = await self._first(page, self.BODY, "message body")
        await body.click()
        await body.type(draft.body, delay=0)

        if draft.attachments:
            # Gmail keeps a real file input in the compose window, so the attachment
            # goes straight to it and no operating-system dialog is involved.
            uploader = page.locator(self.FILE_INPUT).last
            await uploader.set_input_files([str(p) for p in draft.attachments])
            await self._await_attachments(page, len(draft.attachments))

        # Proof that the compose window really is open, before anything is claimed about
        # it closing. Without this, a compose that never opened reads as one that closed,
        # which is a message reported as sent that was never written.
        if not await self._present(page, self.BODY):
            raise SendRefused("The Gmail compose window did not open, so nothing was written.")

        await (await self._first(page, self.SEND, "Send button")).click()
        await self.confirm_sent(page, draft)
        await self.verify_in_sent(page, draft)

    #: Gmail says so in a small bar at the bottom. Wording varies by language, so the
    #: compose window closing is the signal that matters most.
    SENT_TOAST = ('text="Message sent"', 'text="Sending..."', '[role="alert"]:has-text("sent")')
    #: What it says when it will not send, usually because of the recipient.
    ERROR_DIALOG = ('[role="alertdialog"]', '.Kj-JD', 'text="Please specify at least one recipient"')
    #: How long to wait for Gmail to make up its mind.
    CONFIRM_TIMEOUT = 25.0

    async def confirm_sent(self, page: Any, draft: Draft) -> None:
        """Wait for evidence that Gmail sent it, and raise when there is none."""
        deadline = asyncio.get_running_loop().time() + self.CONFIRM_TIMEOUT
        while asyncio.get_running_loop().time() < deadline:
            complaint = await self._text_of(page, self.ERROR_DIALOG)
            if complaint:
                raise SendRefused(f"Gmail would not send it: {complaint[:160]}")
            if await self._gone(page, self.BODY):
                log.info("Gmail sent the message to %s", draft.to)
                return
            if await self._text_of(page, self.SENT_TOAST):
                log.info("Gmail sent the message to %s", draft.to)
                return
            await asyncio.sleep(0.8)
        raise SendRefused(
            f"Gmail did not confirm sending to {draft.to}. The message is most likely "
            f"still in your Drafts. Nothing has been recorded as sent.")

    async def _present(self, page: Any, selectors: Iterable[str]) -> bool:
        """Whether any of these is on the page."""
        for selector in selectors:
            try:
                if await page.locator(selector).count():
                    return True
            except Exception:
                continue
        return False

    async def _gone(self, page: Any, selectors: Iterable[str]) -> bool:
        """Whether none of these is on the page any more, which is how compose closes."""
        return not await self._present(page, selectors)

    #: Where Gmail lists what it has actually sent. Searched by subject, because that is
    #: the only evidence that does not depend on reading the compose window correctly.
    SENT_SEARCH = "https://mail.google.com/mail/u/0/#search/in%3Asent+subject%3A{q}"
    SENT_ROWS = "tr.zA"
    #: Gmail files a sent message within a second or two, but not instantly.
    SENT_TIMEOUT = 30.0

    async def verify_in_sent(self, page: Any, draft: Draft) -> None:
        """Look in Sent for the message, and refuse to claim anything until it is there.

        The reading of the compose window turned out not to be trustworthy: a window
        that never opened looks exactly like one that closed after sending. This asks
        Gmail what it actually sent, which is the only answer that cannot be faked by a
        selector matching the wrong thing.
        """
        from urllib.parse import quote

        needle = (draft.subject or "").strip()
        if not needle:
            return
        url = self.SENT_SEARCH.format(q=quote(needle))
        # Matched on what the rows say, not on how many there are. Gmail navigates by
        # the part of the address after the hash, which does not always re-run the
        # search, so a count can be of the previous list entirely.
        wanted = _comparable(needle)[:60]
        deadline = asyncio.get_running_loop().time() + self.SENT_TIMEOUT
        while asyncio.get_running_loop().time() < deadline:
            try:
                await page.goto(url, wait_until="domcontentloaded")
                await asyncio.sleep(2.0)
                await page.reload(wait_until="domcontentloaded")
                await asyncio.sleep(3.0)
                rows = await page.locator(self.SENT_ROWS).all_inner_texts()
            except Exception as exc:
                log.debug("could not read Sent: %s", exc)
                rows = []
            for row in rows:
                if wanted and wanted in _comparable(row):
                    log.info("Gmail has it in Sent: %r to %s", needle[:60], draft.to)
                    return
            await asyncio.sleep(2.0)
        raise SendRefused(
            f"Gmail has nothing in Sent matching {needle!r}, so the message to "
            f"{draft.to} was not sent. Nothing has been recorded as sent.")

    async def _text_of(self, page: Any, selectors: Iterable[str]) -> str:
        for selector in selectors:
            try:
                found = page.locator(selector).first
                if await found.count() and await found.is_visible():
                    return (await found.inner_text()).strip()
            except Exception:
                continue
        return ""

    async def _await_attachments(self, page: Any, expected: int, timeout: float = 90.0) -> None:
        """Wait for the uploads to finish, because Send discards one still in progress."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                done = await page.locator('div[aria-label*="Attachment"], .dL, .aZo').count()
            except Exception:
                done = 0
            if done >= expected:
                return
            await asyncio.sleep(1.0)
        log.warning("Gmail has not confirmed all %d attachment(s); sending anyway", expected)
