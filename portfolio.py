"""Keeping your profile in step with your portfolio site.

Your portfolio already holds the projects, roles and skills you keep up to date. This
reads them from its public endpoints and merges them into `profile.json`, so the resume
builder and the cover letter writer work from the same material you publish.

It merges rather than replaces. Anything you have written by hand in `profile.json`,
the bullets you tuned for a particular kind of role, your screening defaults, the
wording of your summary, stays exactly as it is. A project already in the file keeps
its bullets and only gains what it was missing. The file is copied to a backup before
anything is written.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

log = logging.getLogger(__name__)

DEFAULT_SITE = "https://joshuapro.netlify.app"
#: What each endpoint is called, and which part of the profile it feeds.
#: Projects live under two names on the site, so both are read and merged.
PROJECT_ENDPOINTS = ("projects", "product-projects")
#: Where each blog post lives, built from its slug.
BLOG_URL = "{site}/blog/{slug}"
#: The social links worth keeping, and what the profile calls each one.
SOCIAL_FIELDS = {"github": "github", "linkedin": "linkedin", "twitter": "twitter",
                 "x": "twitter", "website": "website", "portfolio": "website"}
_TAGS = re.compile(r"<[^>]+>")
_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&quot;": '"', "&#39;": "'",
             "&lt;": "<", "&gt;": ">", "&apos;": "'"}


def plain(html: Any) -> str:
    """The words out of a rich-text field, without the markup it was written in.

    The site stores the bio as styled HTML. Dropped into a resume or a cover letter
    that markup would be read out literally, fonts and all.
    """
    text = str(html or "")
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<(br|/p|/div|/li)\s*/?>", "\n", text, flags=re.I)
    text = _TAGS.sub(" ", text)
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    text = re.sub(r"&#\d+;", " ", text)
    lines = [_TIDY.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def as_post(raw: dict[str, Any], site: str) -> Optional[dict[str, Any]]:
    """One blog post, as something worth pointing a recruiter at."""
    title = _TIDY.sub(" ", str(raw.get("title") or "")).strip()
    if not title or raw.get("isVisible") is False:
        return None
    slug = str(raw.get("slug") or "").strip()
    summary = plain(raw.get("excerpt") or "")
    if not summary:
        summary = " ".join(sentences(plain(raw.get("content") or ""))[:1])
    return {
        "title": title,
        "url": BLOG_URL.format(site=site.rstrip("/"), slug=slug) if slug else "",
        "summary": summary,
        "tags": [str(t).strip() for t in (raw.get("tags") or []) if str(t).strip()],
    }
#: Kept short: a portfolio is a static site and should answer at once.
TIMEOUT = 25.0
#: A sentence longer than this is a paragraph, and a bullet is not a paragraph.
MAX_BULLET = 220
#: How many bullets to take from a description.
MAX_BULLETS = 3

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_TIDY = re.compile(r"\s+")


def sentences(text: Any) -> list[str]:
    """A description turned into bullets, however the site happens to store it.

    Projects give one block of prose and roles give a list of lines already, so
    both are accepted rather than assuming whichever one was looked at first.
    """
    if isinstance(text, (list, tuple)):
        out: list[str] = []
        for line in text:
            out.extend(sentences(line))
        return out[:MAX_BULLETS]
    clean = _TIDY.sub(" ", str(text or "").replace("—", " - ")).strip()
    if not clean:
        return []
    out = []
    for part in _SENTENCE.split(clean):
        part = part.strip(" -–")
        if not part:
            continue
        if not part.endswith((".", "!", "?")):
            part += "."
        if len(part) <= MAX_BULLET:
            out.append(part)
    return out[:MAX_BULLETS]


def normalise(value: str) -> str:
    """For matching one entry against another despite punctuation and case."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


#: A name is the part before the colon. The site titles a project "xMachine:
#: Browser-Based Deep Learning", the profile calls it "xMachine", and they are the
#: same project. Matching the whole string would import it a second time.
def short_name(value: str) -> str:
    head = re.split(r"[:\u2013\u2014|(]", str(value or ""), 1)[0]
    return normalise(head) or normalise(value)


#: Below this a shared prefix means nothing: "AI" is the start of a great many things.
MIN_PREFIX = 5


def same_entry(one: str, other: str) -> bool:
    """Whether two names are the same thing written at different lengths."""
    a, b = short_name(one), short_name(other)
    if not a or not b:
        return False
    if a == b:
        return True
    long, short = (a, b) if len(a) >= len(b) else (b, a)
    return len(short) >= MIN_PREFIX and long.startswith(short)


@dataclass
class Changes:
    """What an import did, so it can be reported before anything is believed."""

    added: dict[str, list[str]] = field(default_factory=dict)
    updated: dict[str, list[str]] = field(default_factory=dict)
    unreachable: list[str] = field(default_factory=list)
    backup: Optional[Path] = None

    @property
    def touched(self) -> bool:
        return bool(self.added or self.updated)

    def summary(self) -> str:
        if self.unreachable and not self.touched:
            return "nothing imported: " + ", ".join(self.unreachable)
        parts = []
        for kind, names in self.added.items():
            parts.append(f"{len(names)} new {kind}")
        for kind, names in self.updated.items():
            parts.append(f"{len(names)} {kind} filled in")
        return "; ".join(parts) or "everything was already up to date"


async def fetch(site: str, name: str) -> Optional[Any]:
    """One endpoint, or None when it is not there. A missing one is not a failure."""
    url = f"{site.rstrip('/')}/api/{name}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url, headers={"accept": "application/json"})
    except httpx.HTTPError as exc:
        log.warning("Could not reach %s: %s", url, exc)
        return None
    if response.status_code != 200:
        log.info("%s answered %s, so there is nothing to import from it",
                 url, response.status_code)
        return None
    if "json" not in response.headers.get("content-type", ""):
        # A site that renders everything returns its 404 page with a 200, which parses
        # as HTML and never as the list this expects.
        log.info("%s did not return JSON, so it is not an endpoint", url)
        return None
    try:
        return response.json()
    except json.JSONDecodeError:
        log.info("%s returned something that is not JSON", url)
        return None


def as_project(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    """One portfolio project in the shape the profile and the resume builder expect."""
    name = _TIDY.sub(" ", str(raw.get("title") or "")).strip()
    if not name or raw.get("isVisible") is False:
        return None
    tech = [str(t).strip() for t in (raw.get("technologies") or []) if str(t).strip()]
    return {
        "name": name,
        "url": str(raw.get("liveUrl") or "").strip(),
        "github": str(raw.get("githubUrl") or "").strip(),
        "tech": tech,
        "bullets": sentences(raw.get("description") or ""),
    }


def as_role(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    """One portfolio role, dated the way the resume shows dates."""
    company = str(raw.get("company") or "").strip()
    title = str(raw.get("role") or "").strip()
    if not company and not title:
        return None

    def year(value: Any) -> str:
        text = str(value or "")
        found = re.match(r"(\d{4})", text)
        return found.group(1) if found else ""

    return {
        "company": company,
        "title": title,
        "location": str(raw.get("location") or "").strip(),
        "start": year(raw.get("startDate")),
        "end": "Present" if raw.get("isCurrent") else year(raw.get("endDate")),
        "bullets": sentences(raw.get("description") or ""),
    }


def merge_entries(existing: list[dict[str, Any]], incoming: list[dict[str, Any]],
                  key: str, protect: Iterable[str] = ("bullets",)) -> tuple[list, list, list]:
    """Merge by name, keeping what you wrote and adding only what was missing.

    Returns the merged list, the names added and the names filled in. Your own wording
    is never overwritten: a field you have already filled in is left exactly as it is,
    and the import only supplies what is empty.
    """
    merged = [dict(item) for item in existing]
    added: list[str] = []
    updated: list[str] = []

    def find(name: str) -> Optional[int]:
        for i, item in enumerate(merged):
            if same_entry(item.get(key, ""), name):
                return i
        return None

    for item in incoming:
        name = item.get(key, "")
        at = find(name)
        if at is None:
            merged.append(item)
            added.append(name)
            continue
        current = merged[at]
        changed = False
        for field_name, value in item.items():
            if not value:
                continue
            if field_name in protect and current.get(field_name):
                continue                      # your wording wins
            if not current.get(field_name):
                current[field_name] = value
                changed = True
        if changed:
            updated.append(name)
    return merged, added, updated


async def import_portfolio(profile_path: Path, site: str = DEFAULT_SITE,
                           what: Iterable[str] = ("projects", "experience", "skills",
                                                 "about", "blog"),
                           dry_run: bool = False) -> Changes:
    """Pull the portfolio into the profile, and say exactly what changed."""
    changes = Changes()
    profile_path = Path(profile_path)
    profile: dict[str, Any] = json.loads(profile_path.read_text(encoding="utf-8"))
    wanted = {w.strip().lower() for w in what}

    if "projects" in wanted:
        incoming: list[dict[str, Any]] = []
        reached = False
        for endpoint in PROJECT_ENDPOINTS:
            raw = await fetch(site, endpoint)
            if raw is None:
                continue
            reached = True
            incoming.extend(p for p in (as_project(r) for r in raw) if p)
        if not reached:
            changes.unreachable.append("projects")
        else:
            merged, added, updated = merge_entries(profile.get("projects") or [], incoming, "name")
            profile["projects"] = merged
            if added:
                changes.added["project(s)"] = added
            if updated:
                changes.updated["project(s)"] = updated

    if "experience" in wanted:
        raw = await fetch(site, "experience")
        if raw is None:
            changes.unreachable.append("experience")
        else:
            incoming = [r for r in (as_role(x) for x in raw) if r]
            merged, added, updated = merge_entries(profile.get("experience") or [],
                                                   incoming, "company")
            profile["experience"] = merged
            if added:
                changes.added["role(s)"] = added
            if updated:
                changes.updated["role(s)"] = updated

    if "skills" in wanted:
        raw = await fetch(site, "skills")
        if raw is None:
            changes.unreachable.append("skills")
        else:
            groups: dict[str, list[str]] = {k: list(v) for k, v in
                                            (profile.get("skills") or {}).items()}
            new_skills: list[str] = []
            for entry in raw:
                if entry.get("isVisible") is False:
                    continue
                name = str(entry.get("name") or "").strip()
                group = str(entry.get("category") or "other").strip().lower() or "other"
                if not name:
                    continue
                items = groups.setdefault(group, [])
                if not any(normalise(name) == normalise(s) for s in items):
                    items.append(name)
                    new_skills.append(f"{group}/{name}")
            profile["skills"] = groups
            if new_skills:
                changes.added["skill(s)"] = new_skills

    if "about" in wanted:
        raw = await fetch(site, "about")
        if raw is None:
            changes.unreachable.append("about")
        elif isinstance(raw, dict):
            filled: list[str] = []
            bio = plain(raw.get("bio"))
            # Only where the profile has nothing. A summary you wrote for applications
            # is aimed at a reader who is deciding about you, which a site bio is not.
            if bio and not str(profile.get("summary") or "").strip():
                profile["summary"] = bio
                filled.append("summary")
            for link in raw.get("socialLinks") or []:
                field_name = SOCIAL_FIELDS.get(str(link.get("platform") or "").lower())
                url = str(link.get("url") or "").strip()
                if field_name and url and not str(profile.get(field_name) or "").strip():
                    profile[field_name] = url
                    filled.append(field_name)
            if filled:
                changes.updated["about"] = filled

    if "blog" in wanted:
        raw = await fetch(site, "blog")
        if raw is None:
            changes.unreachable.append("blog")
        else:
            incoming = [p for p in (as_post(r, site) for r in raw) if p]
            merged, added, updated = merge_entries(profile.get("writing") or [], incoming,
                                                   "title", protect=("summary",))
            profile["writing"] = merged
            if added:
                changes.added["post(s)"] = added
            if updated:
                changes.updated["post(s)"] = updated

    if dry_run or not changes.touched:
        return changes

    # The file is somebody's careful work, so a copy of it survives a bad import.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = profile_path.with_name(f"{profile_path.stem}.{stamp}.backup.json")
    shutil.copy2(profile_path, backup)
    changes.backup = backup
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    log.info("Profile updated: %s (previous copy at %s)", changes.summary(), backup.name)
    return changes
