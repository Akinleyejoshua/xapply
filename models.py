"""Shared data models (kept dependency-free so every module can import them)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

# ATS identifiers
LINKEDIN = "linkedin"
GREENHOUSE = "greenhouse"
LEVER = "lever"
ASHBY = "ashby"
UNKNOWN = "unknown"

_LINKEDIN_JOB_RE = re.compile(r"linkedin\.com/jobs/view/(\d+)")


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


def job_id_from_url(url: str) -> str:
    """Stable identifier for a posting: LinkedIn numeric id, otherwise a URL hash."""
    m = _LINKEDIN_JOB_RE.search(url)
    if m:
        return m.group(1)
    clean = url.split("#")[0].rstrip("/")
    ats = detect_ats(url)
    tail = urlparse(clean).path.rstrip("/").split("/")[-1]
    if ats in (LEVER, ASHBY, GREENHOUSE) and tail:
        return f"{ats}-{tail}"
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
