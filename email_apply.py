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


def find_address(job: JobPosting, page_text: str = "") -> Optional[str]:
    """The one address to apply to, or None when the posting names none."""
    found = find_addresses(job.description, page_text, job.url)
    return found[0] if found else None


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


class NotConfigured(RuntimeError):
    """No mail server is set up, so nothing can be sent."""


class EmailApplier:
    """Compose and send one application email."""

    #: Subject lines recruiters expect. The role first, because that is what they filter on.
    SUBJECT = "Application for {title}{company} - {name}"

    def __init__(self, settings: Settings, gate: Any = None):
        self.s = settings
        self.gate = gate

    # ---- configuration -------------------------------------------------
    @property
    def from_address(self) -> str:
        return (self.s.email_from or self.s.smtp_user or "").strip()

    def missing_settings(self) -> list[str]:
        """Which settings are still needed before anything can be sent."""
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
        missing = self.missing_settings()
        if missing:
            raise NotConfigured(
                "Email applying needs " + ", ".join(missing) + " in .env. "
                "For Gmail use an app password, not your account password.")
        if not parseaddr(draft.to)[1]:
            raise ValueError(f"{draft.to!r} is not an address to send to")
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
