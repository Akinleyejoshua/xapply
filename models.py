"""Shared data models (kept dependency-free so every module can import them)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlparse

# ATS identifiers
LINKEDIN = "linkedin"
GREENHOUSE = "greenhouse"
LEVER = "lever"
ASHBY = "ashby"
UNKNOWN = "unknown"

_LINKEDIN_JOB_RE = re.compile(r"linkedin\.com/jobs/view/(\d+)")

#: Typographic characters an LLM produces freely, and which applicant tracking systems
#: parse badly. A resume containing "Full\u2011stack" does not match a recruiter's search
#: for "Full-stack", so every string that reaches a PDF or a form field is flattened.
TYPOGRAPHIC = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2026": "...", "\u2022": "-", "\u00b7": "-", "\u2024": ".",
    "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2122": "(TM)", "\u00ae": "(R)", "\u00a9": "(c)",
}
_TYPO_RE = re.compile("|".join(map(re.escape, TYPOGRAPHIC)))


def ats_text(value: Any) -> Any:
    """Flatten typographic characters so a parser reads what a human reads.

    Applied to everything written into a resume or typed into an application form.
    Lists and dicts are walked, so a whole analysis can be passed through at once.
    """
    if isinstance(value, str):
        return _TYPO_RE.sub(lambda m: TYPOGRAPHIC[m.group(0)], value)
    if isinstance(value, list):
        return [ats_text(v) for v in value]
    if isinstance(value, dict):
        return {k: ats_text(v) for k, v in value.items()}
    return value


def detect_ats(url: str) -> str:
    """Map a URL to the ATS that hosts it."""
    if not url:
        return UNKNOWN
    host = (urlparse(url).hostname or "").lower()
    path = urlparse(url).path.lower()
    query = urlparse(url).query.lower()
    if host.endswith("linkedin.com") and "/jobs/view/" in path:
        return LINKEDIN
    if host.endswith("greenhouse.io") or "gh_jid=" in query:
        return GREENHOUSE
    if host.endswith("lever.co"):
        return LEVER
    if host.endswith("ashbyhq.com"):
        return ASHBY
    return UNKNOWN


#: Path segments that name a page, not a posting. Taking the last segment as the id
#: turned every Ashby link ending "/application" into the same posting, so the second
#: one you picked was refused as already applied, naming a job you had never seen.
GENERIC_SEGMENTS = {"application", "apply", "application_form", "job_app", "jobs", "job",
                    "embed", "index", "posting", "postings", "openings", "form"}
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_NUMERIC_ID_RE = re.compile(r"^\d{4,}$")
#: Query parameters that carry the posting id on an embedded board.
ID_PARAMS = ("gh_jid", "gh_src_token", "token")


def job_id_from_url(url: str) -> str:
    """Stable identifier for a posting.

    The identifier has to come from the posting itself, never from the page being
    viewed, because one posting is reachable at several addresses: the board page, the
    apply page and the embedded form all describe the same job.
    """
    m = _LINKEDIN_JOB_RE.search(url)
    if m:
        return m.group(1)
    clean = url.split("#")[0].rstrip("/")
    ats = detect_ats(url)
    if ats in (LEVER, ASHBY, GREENHOUSE):
        parsed = urlparse(clean)
        found = _UUID_RE.search(parsed.path)          # Lever and Ashby use a UUID
        if found:
            return f"{ats}-{found.group(0).lower()}"
        query = parse_qs(parsed.query)                # an embedded form carries it here
        for key in ID_PARAMS:
            value = (query.get(key) or [""])[0].strip()
            if _NUMERIC_ID_RE.match(value):
                return f"{ats}-{value}"
        for segment in reversed([p for p in parsed.path.split("/") if p]):
            if segment.lower() in GENERIC_SEGMENTS:
                continue                              # a page name, keep looking
            if _NUMERIC_ID_RE.match(segment) or len(segment) > 6:
                return f"{ats}-{segment}"
    # Nothing identifying in the address, so the address itself is the identity.
    return "url-" + hashlib.sha1(clean.encode()).hexdigest()[:16]


@dataclass
class JobPosting:
    job_id: str
    url: str
    title: str = ""
    company: str = ""
    location: str = ""
    description: str = ""
    source: str = LINKEDIN  # where we found it (linkedin | urls)
    ats: str = UNKNOWN  # where we apply (linkedin | greenhouse | lever | ashby | unknown)
    apply_url: str = ""  # external application URL when the posting hands off to another ATS
    easy_apply: bool = False
    relevance: float = 0.0   # 0-1, how well the title answered the search terms
    #: Where an application is sent when there is no form. Found on the posting itself,
    #: never guessed, and empty for everything that has a form to fill in.
    email_to: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_url(cls, url: str, source: str = "urls") -> "JobPosting":
        ats = detect_ats(url)
        return cls(job_id=job_id_from_url(url), url=url, source=source, ats=ats,
                   apply_url="" if ats == LINKEDIN else url)


@dataclass
class ApplyResult:
    status: str  # submitted | pending_human_review | skipped | failed
    note: str = ""
    answers: list[dict[str, Any]] = field(default_factory=list)  # every field the bot filled
    screenshot_path: str = ""
    apply_url: str = ""
