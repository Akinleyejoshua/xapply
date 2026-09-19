"""Job discovery that does not depend on LinkedIn.

Greenhouse, Lever and Ashby have no candidate-facing search, so the URLs have to
come from somewhere else. Five routes, in descending order of reliability:

  1. Public board APIs (greenhouse, lever, ashby)
     Company-scoped JSON endpoints, no key and no login. They return the full job
     description, so a posting is scored without opening a browser at all.
       https://boards-api.greenhouse.io/v1/boards/{token}/jobs
       https://api.lever.co/v0/postings/{token}?mode=json
       https://api.ashbyhq.com/posting-api/job-board/{token}

  2. Aggregators (remoteok, himalayas)
     Public feeds of remote roles. Their links point at the aggregator, so each
     one is resolved to the underlying Greenhouse/Lever/Ashby application URL.

  3. Google search (google)
     `site:job-boards.greenhouse.io "Full Stack Developer"` driven through Playwright.
     Google no longer puts result URLs in the page, so this source reads the company
     board named under each result and hands it to route 1, which returns the jobs in
     full. Boards that answer are written to companies.json, so what Google finds once
     keeps working without it. Google may refuse an automated search, and then this
     source pauses for a human rather than reporting an empty result.

Everything here is read-only HTTP except the Google source. None of it needs a
login, which is why these boards are far more stable to automate than LinkedIn.
"""
from __future__ import annotations

import asyncio
import html as html_lib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Optional
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

from config import Settings
from countries import ANYWHERE, COUNTRIES, country_matches
from matching import best_relevance
from database import Database
from models import ASHBY, GREENHOUSE, LEVER, UNKNOWN, JobPosting, detect_ats, job_id_from_url

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json, text/html;q=0.9"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "with", "in", "at", "to", "on",
    "senior", "junior", "staff", "lead", "principal", "mid", "level", "i", "ii", "iii",
}
ATS_URL_RE = re.compile(
    r"https?://(?:job-boards|boards)\.greenhouse\.io/[A-Za-z0-9_.-]+/jobs/\d+"
    r"|https?://jobs\.lever\.co/[A-Za-z0-9_.-]+/[0-9a-f-]{36}"
    r"|https?://jobs\.ashbyhq\.com/[A-Za-z0-9_.-]+/[0-9a-f-]{36}",
    re.I,
)
TAG_RE = re.compile(r"<[^>]+>")


def strip_html(raw: str) -> str:
    """Board APIs return HTML (sometimes double-escaped). Reduce it to plain text."""
    if not raw:
        return ""
    text = html_lib.unescape(raw)
    if "&lt;" in text or "&amp;" in text:
        text = html_lib.unescape(text)
    text = re.sub(r"<(br|/p|/div|/li|/h[1-6])\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<li[^>]*>", "- ", text, flags=re.I)
    text = TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


def query_tokens(queries: Iterable[str]) -> list[str]:
    """The raw search terms. Kept as a named step so sources read the same as before."""
    return [q.strip() for q in queries if q and q.strip()]


#: How much of a search term a title must cover to be worth scoring with the LLM.
#: Discovery is deliberately generous here: the model scores each posting properly
#: afterwards and skips anything below MATCH_THRESHOLD, so a near miss at this stage
#: costs one cheap API call, while a wrongly dropped posting is never seen again.
DEFAULT_TITLE_THRESHOLD = 0.45


def title_relevance(title: str, queries: Iterable[str]) -> float:
    """0 to 1: how well a posting title answers the user's search terms."""
    return best_relevance(title, queries)


def title_matches(title: str, queries: Iterable[str], threshold: Optional[float] = None) -> bool:
    """True when a title resembles at least one search term closely enough.

    'Data Analytics' finds 'Data Analyst' (0.97) and 'Analytics Engineer Intern' (0.50),
    but not 'Accounting Intern' (0.00). 'Backend Engineer' does not drag in
    'Sales Engineer' (0.29), because `engineer` alone barely narrows a search.
    """
    if not queries:
        return True
    limit = DEFAULT_TITLE_THRESHOLD if threshold is None else threshold
    return title_relevance(title, queries) >= limit


#: Ashby sets `isRemote: true` on hybrid roles too (505 of OpenAI's 537 "remote" jobs are
#: Hybrid), so that flag cannot be trusted on its own. `workplaceType` is authoritative
#: when present; otherwise fall back to what the location text says.
REMOTE_WORDS = ("remote", "anywhere", "distributed", "work from home", "wfh")
NOT_REMOTE_WORDS = ("hybrid", "on-site", "onsite", "in office", "in-office")


def looks_remote(location_text: str, workplace_type: Optional[str] = None) -> bool:
    """Whether a posting is genuinely remote, not merely hybrid."""
    wt = (workplace_type or "").strip().lower().replace("-", "").replace(" ", "")
    if wt:
        return wt == "remote"
    blob = (location_text or "").lower()
    if any(w in blob for w in NOT_REMOTE_WORDS):
        return False
    return any(w in blob for w in REMOTE_WORDS)


def location_matches(text: str, wanted: str, remote_only: bool,
                     workplace_type: Optional[str] = None,
                     countries: Optional[list[str]] = None) -> bool:
    """Gate a posting on where it is.

    Remote-only and the country list stack: with both set, a posting has to be
    genuinely remote *and* name one of the chosen countries. `search_location` is
    only consulted when no countries are chosen, so the two never fight.
    """
    if remote_only and not looks_remote(text, workplace_type):
        return False
    if countries:
        return country_matches(text, countries)
    if remote_only:
        return True
    w = (wanted or "").strip().lower()
    if not w or w in ("remote", "anywhere", "worldwide"):
        return True
    return w in (text or "").lower() or looks_remote(text, workplace_type)


# --------------------------------------------------------------------------
# Seniority
# --------------------------------------------------------------------------

SENIORITY_LEVELS = ("intern", "junior", "mid", "senior", "lead")

#: Checked in order; the first hit wins.
#: The junior *prefix* is checked before `lead` on purpose: "Associate Product Manager"
#: is a junior role, not a leadership one, even though it contains "manager".
SENIORITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("intern", re.compile(r"\b(intern|internship|co-?op|apprentice|trainee|placement)\b", re.I)),
    ("junior", re.compile(r"^\s*(associate|assistant|junior|jr\.?|entry[ -]?level|graduate|new ?grad)\b", re.I)),
    ("lead", re.compile(r"\b(lead|leader|staff|principal|distinguished|fellow|head of|"
                        r"director|vp|vice president|chief|architect)\b", re.I)),
    ("senior", re.compile(r"\b(senior|sr\.?|snr|experienced)\b", re.I)),
    ("junior", re.compile(r"\b(junior|jr\.?|entry[ -]?level|graduate|new ?grad|early career)\b", re.I)),
]


#: "Manager" only signals leadership when it is the role, not part of a product name.
#: "Engineering Manager" is a lead; "Software Engineer, Ads Manager" is not.
MANAGER_RE = re.compile(r"\bmanagers?\b", re.I)
IC_ROLE_RE = re.compile(r"\b(engineer|developer|scientist|analyst|designer|researcher|"
                        r"programmer|administrator|consultant)\b", re.I)


def _is_manager_role(title: str) -> bool:
    m = MANAGER_RE.search(title or "")
    if not m:
        return False
    ic = IC_ROLE_RE.search(title)
    # An individual-contributor noun before "manager" means the word belongs to a product.
    return not (ic and ic.start() < m.start())


def seniority_of(title: str) -> str:
    """Bucket a job title into intern / junior / mid / senior / lead.

    Titles carrying no level word at all are treated as `mid`, which is how most
    postings read ("Backend Engineer", "Full Stack Developer").
    """
    t = title or ""
    for level, pattern in SENIORITY_PATTERNS:
        if pattern.search(t):
            if level == "senior" and _is_manager_role(t):
                return "lead"          # "Senior Engineering Manager"
            return level
    return "lead" if _is_manager_role(t) else "mid"


def seniority_matches(title: str, wanted: Optional[list[str]]) -> bool:
    """True when the title's level is one the user asked for (empty list = any level)."""
    if not wanted:
        return True
    return seniority_of(title) in {w.strip().lower() for w in wanted}


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------


@dataclass
class ScanStats:
    """Why postings did not make it through, so an empty scan can explain itself."""

    seen: int = 0
    dropped_title: int = 0
    dropped_seniority: int = 0
    dropped_location: int = 0
    dropped_seen_before: int = 0
    dropped_no_description: int = 0
    dropped_unresolved: int = 0     # aggregators: no ATS link behind the listing
    kept: int = 0

    def __iadd__(self, other: "ScanStats") -> "ScanStats":
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    @property
    def dropped(self) -> int:
        return self.seen - self.kept

    def reasons(self) -> list[tuple[str, int]]:
        return [(label, value) for label, value in (
            ("search terms", self.dropped_title),
            ("seniority", self.dropped_seniority),
            ("location or country", self.dropped_location),
            ("already applied", self.dropped_seen_before),
            ("no description", self.dropped_no_description),
            ("no application link", self.dropped_unresolved),
        ) if value]

    def summary(self) -> str:
        if not self.seen:
            return "no postings returned by the board"
        parts = [f"{self.kept} kept of {self.seen}"]
        parts += [f"{n} dropped by {label}" for label, n in self.reasons()]
        return "; ".join(parts)


class ApiJobSource:
    """Discovery over plain HTTP. Postings arrive fully described, so hydrate() is a no-op."""

    name = "api"

    def __init__(self, settings: Settings, db: Database, browser: Any = None):
        self.s = settings
        self.db = db
        self.b = browser
        self.tokens = query_tokens(settings.search_queries)
        self.stats = ScanStats()
        #: Called with each batch of postings as they are found, so a long scan shows
        #: results while it runs and a stopped scan keeps what it already had.
        self.on_batch: Optional[Callable[[str, list[JobPosting]], None]] = None

    def _report(self, batch: list[JobPosting]) -> None:
        if batch and self.on_batch:
            try:
                self.on_batch(self.name, batch)
            except Exception as exc:          # a reporting failure must not stop a scan
                log.debug("progress callback failed: %s", exc)

    async def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=HEADERS, timeout=self.s.discovery_timeout_s,
                                 follow_redirects=True)

    async def _json(self, client: httpx.AsyncClient, url: str) -> Any:
        try:
            r = await client.get(url)
            if r.status_code != 200:
                log.debug("%s -> HTTP %s", url, r.status_code)
                return None
            return r.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log.debug("%s -> %s", url, exc)
            return None

    def _new(self, job: JobPosting) -> bool:
        """Skip anything already in the database, and anything with no usable description."""
        if self.db.has_job(self.name, job.job_id):
            self.stats.dropped_seen_before += 1
            return False
        if len(job.description) < 120:
            self.stats.dropped_no_description += 1
            return False
        self.stats.kept += 1
        return True

    async def discover(self, page: Any = None) -> list[JobPosting]:  # pragma: no cover
        raise NotImplementedError

    async def hydrate(self, page: Any, job: JobPosting) -> JobPosting:
        """The API already returned the description, so nothing to fetch."""
        return job


# --------------------------------------------------------------------------
# Public board APIs
# --------------------------------------------------------------------------


def load_company_tokens(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        log.warning("Company list %s not found; board API sources will find nothing", path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, list) and not k.startswith("_")}


class GreenhouseBoardSource(ApiJobSource):
    """https://boards-api.greenhouse.io/v1/boards/{token}/jobs"""

    name = GREENHOUSE
    LIST = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
    DETAIL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs/{job_id}"
    APPLY = "https://job-boards.greenhouse.io/{token}/jobs/{job_id}"

    def __init__(self, settings: Settings, db: Database, browser: Any = None,
                 tokens: Optional[list[str]] = None):
        super().__init__(settings, db, browser)
        self.companies = tokens if tokens is not None else load_company_tokens(settings.company_file).get(GREENHOUSE, [])

    async def discover(self, page: Any = None) -> list[JobPosting]:
        found: list[JobPosting] = []
        async with await self._client() as client:
            for token in self.companies:
                listing = await self._json(client, self.LIST.format(token=token))
                if not listing or "jobs" not in listing:
                    log.info("greenhouse/%s: no board", token)
                    continue
                candidates = []
                for j in listing["jobs"]:
                    self.stats.seen += 1
                    title = j.get("title", "")
                    if not title_matches(title, self.tokens, self.s.title_match_threshold):
                        self.stats.dropped_title += 1
                        continue
                    if not seniority_matches(title, self.s.seniority_levels):
                        self.stats.dropped_seniority += 1
                        continue
                    if not location_matches((j.get("location") or {}).get("name", ""),
                                            self.s.search_location, self.s.remote_only,
                                            countries=self.s.countries):
                        self.stats.dropped_location += 1
                        continue
                    candidates.append(j)
                candidates = candidates[: self.s.max_jobs_per_company]
                kept = 0
                for j in candidates:
                    job_id = str(j["id"])
                    if self.db.has_job(self.name, f"gh-{token}-{job_id}"):
                        # Checked before the detail fetch to save a request; still counted,
                        # so an empty scan can say "already applied" rather than nothing.
                        self.stats.dropped_seen_before += 1
                        continue
                    detail = await self._json(client, self.DETAIL.format(token=token, job_id=job_id))
                    if not detail:
                        continue
                    apply_url = j.get("absolute_url") or ""
                    if "greenhouse.io" not in apply_url:
                        # Company-hosted page that embeds the board; go straight to the real form.
                        apply_url = self.APPLY.format(token=token, job_id=job_id)
                    job = JobPosting(
                        job_id=f"gh-{token}-{job_id}",
                        url=j.get("absolute_url") or apply_url,
                        apply_url=apply_url,
                        title=(detail.get("title") or j.get("title") or "").strip(),
                        company=(detail.get("company_name") or token).strip(),
                        location=((detail.get("location") or {}).get("name") or "").strip(),
                        description=strip_html(detail.get("content", "")),
                        source=self.name, ats=GREENHOUSE,
                        relevance=title_relevance(detail.get("title") or j.get("title") or "",
                                                  self.tokens),
                    )
                    if self._new(job):
                        found.append(job)
                        kept += 1
                    await asyncio.sleep(self.s.discovery_delay_s)
                log.info("greenhouse/%s: %d kept of %d", token, kept, len(listing["jobs"]))
                self._report(found[-kept:] if kept else [])
        return found


class LeverBoardSource(ApiJobSource):
    """https://api.lever.co/v0/postings/{token}?mode=json"""

    name = LEVER
    LIST = "https://api.lever.co/v0/postings/{token}?mode=json"

    def __init__(self, settings: Settings, db: Database, browser: Any = None,
                 tokens: Optional[list[str]] = None):
        super().__init__(settings, db, browser)
        self.companies = tokens if tokens is not None else load_company_tokens(settings.company_file).get(LEVER, [])

    async def discover(self, page: Any = None) -> list[JobPosting]:
        found: list[JobPosting] = []
        async with await self._client() as client:
            for token in self.companies:
                postings = await self._json(client, self.LIST.format(token=token))
                if not isinstance(postings, list):
                    log.info("lever/%s: no board", token)
                    continue
                kept = 0
                for p in postings:
                    self.stats.seen += 1
                    cats = p.get("categories") or {}
                    title = (p.get("text") or "").strip()
                    loc = cats.get("location") or ""
                    if not title_matches(title, self.tokens, self.s.title_match_threshold):
                        self.stats.dropped_title += 1
                        continue
                    if not seniority_matches(title, self.s.seniority_levels):
                        self.stats.dropped_seniority += 1
                        continue
                    if not location_matches(loc, self.s.search_location, self.s.remote_only,
                                            workplace_type=p.get("workplaceType"),
                                            countries=self.s.countries):
                        self.stats.dropped_location += 1
                        continue
                    description = (p.get("descriptionPlain") or "") + "\n\n" + (p.get("additionalPlain") or "")
                    hosted = p.get("hostedUrl") or ""
                    job = JobPosting(
                        job_id=f"lever-{token}-{p.get('id')}",
                        url=hosted,
                        apply_url=p.get("applyUrl") or (hosted.rstrip("/") + "/apply" if hosted else ""),
                        title=title, company=token.replace("-", " ").title(), location=loc,
                        description=description.strip(), source=self.name, ats=LEVER,
                        relevance=title_relevance(title, self.tokens),
                    )
                    if self._new(job):
                        found.append(job)
                        kept += 1
                    if kept >= self.s.max_jobs_per_company:
                        break
                log.info("lever/%s: %d kept of %d", token, kept, len(postings))
                self._report(found[-kept:] if kept else [])
                await asyncio.sleep(self.s.discovery_delay_s)
        return found


class AshbyBoardSource(ApiJobSource):
    """https://api.ashbyhq.com/posting-api/job-board/{token}"""

    name = ASHBY
    LIST = "https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true"

    def __init__(self, settings: Settings, db: Database, browser: Any = None,
                 tokens: Optional[list[str]] = None):
        super().__init__(settings, db, browser)
        self.companies = tokens if tokens is not None else load_company_tokens(settings.company_file).get(ASHBY, [])

    async def discover(self, page: Any = None) -> list[JobPosting]:
        found: list[JobPosting] = []
        async with await self._client() as client:
            for token in self.companies:
                data = await self._json(client, self.LIST.format(token=token))
                if not data or "jobs" not in data:
                    log.info("ashby/%s: no board", token)
                    continue
                kept = 0
                for p in data["jobs"]:
                    if p.get("isListed") is False:
                        continue
                    self.stats.seen += 1
                    title = (p.get("title") or "").strip()
                    if not title_matches(title, self.tokens, self.s.title_match_threshold):
                        self.stats.dropped_title += 1
                        continue
                    if not seniority_matches(title, self.s.seniority_levels):
                        self.stats.dropped_seniority += 1
                        continue
                    locs = " ".join([p.get("location") or ""] +
                                    [s.get("location", "") for s in (p.get("secondaryLocations") or [])])
                    if not location_matches(locs, self.s.search_location, self.s.remote_only,
                                            workplace_type=p.get("workplaceType"),
                                            countries=self.s.countries):
                        self.stats.dropped_location += 1
                        continue
                    hosted = p.get("jobUrl") or ""
                    job = JobPosting(
                        job_id=f"ashby-{token}-{p.get('id')}",
                        url=hosted,
                        apply_url=p.get("applyUrl") or (hosted.rstrip("/") + "/application" if hosted else ""),
                        title=title, company=token.replace("-", " ").title(),
                        location=(p.get("location") or "").strip(),
                        description=(p.get("descriptionPlain") or strip_html(p.get("descriptionHtml", ""))).strip(),
                        source=self.name, ats=ASHBY,
                        relevance=title_relevance(title, self.tokens),
                    )
                    if self._new(job):
                        found.append(job)
                        kept += 1
                    if kept >= self.s.max_jobs_per_company:
                        break
                log.info("ashby/%s: %d kept of %d", token, kept, len(data["jobs"]))
                self._report(found[-kept:] if kept else [])
                await asyncio.sleep(self.s.discovery_delay_s)
        return found


# --------------------------------------------------------------------------
# Aggregators
# --------------------------------------------------------------------------


class AggregatorSource(ApiJobSource):
    """Shared logic: aggregator links have to be resolved to the real ATS application URL.

    Aggregator job pages are often behind Cloudflare and answer plain HTTP with 403
    (Himalayas does), so resolution falls back to the Playwright page when one is
    available. The browser is only used for the handful of listings that survive the
    title filter, not for the whole feed.
    """

    _browser_resolutions = 0

    async def resolve_ats_url(self, client: httpx.AsyncClient, link: str, description: str = "",
                              page: Any = None) -> str:
        """Find the Greenhouse/Lever/Ashby URL behind an aggregator listing."""
        direct = ATS_URL_RE.search(description or "")
        if direct:
            return direct.group(0)
        if not link:
            return ""
        if detect_ats(link) != UNKNOWN:
            return link
        blocked = False
        try:
            r = await client.get(link)
            if r.status_code in (401, 403, 429) or r.status_code >= 500:
                blocked = True
            else:
                if detect_ats(str(r.url)) != UNKNOWN:
                    return str(r.url)
                m = ATS_URL_RE.search(r.text or "")
                if m:
                    return m.group(0)
        except httpx.HTTPError as exc:
            log.debug("resolve %s -> %s", link, exc)
            blocked = True
        if not blocked or page is None or self.b is None:
            return ""
        if self._browser_resolutions >= self.s.max_browser_resolutions:
            log.debug("browser-resolution budget spent (%d); skipping %s",
                      self.s.max_browser_resolutions, link)
            return ""
        self._browser_resolutions += 1
        return await self._resolve_with_browser(page, link)

    async def _resolve_with_browser(self, page: Any, link: str) -> str:
        """Open the aggregator page in the real browser and read the outbound apply link."""
        try:
            await self.b.goto(page, link)
        except Exception as exc:
            log.debug("browser resolve %s -> %s", link, exc)
            return ""
        try:
            if detect_ats(page.url) != UNKNOWN:
                return page.url
            hrefs = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
            for h in hrefs:
                if detect_ats(h) != UNKNOWN:
                    return h
            body = await page.content()
            m = ATS_URL_RE.search(body or "")
            if m:
                return m.group(0)
            # Some aggregators only reveal the target after clicking Apply.
            apply_btn = page.get_by_role("link", name=re.compile(r"^\s*apply", re.I))
            if await apply_btn.count():
                href = await apply_btn.first.get_attribute("href")
                if href and detect_ats(href) != UNKNOWN:
                    return href
        except Exception as exc:
            log.debug("browser resolve parse %s -> %s", link, exc)
        return ""


class RemoteOKSource(AggregatorSource):
    """https://remoteok.com/api  (their terms ask for attribution when republishing)."""

    name = "remoteok"
    FEED = "https://remoteok.com/api"

    async def discover(self, page: Any = None) -> list[JobPosting]:
        found: list[JobPosting] = []
        async with await self._client() as client:
            data = await self._json(client, self.FEED)
            if not isinstance(data, list):
                log.warning("remoteok: unexpected feed shape")
                return found
            rows = [d for d in data if isinstance(d, dict) and d.get("position")]
            self.stats.seen += len(rows)
            matched = []
            for d in rows:
                if not title_matches(d.get("position", ""), self.tokens, self.s.title_match_threshold):
                    self.stats.dropped_title += 1
                elif not seniority_matches(d.get("position", ""), self.s.seniority_levels):
                    self.stats.dropped_seniority += 1
                else:
                    matched.append(d)
            log.info("remoteok: %d of %d listings match the title filter", len(matched), len(rows))
            for d in matched[: self.s.max_jobs_per_company]:
                link = d.get("apply_url") or d.get("url") or ""
                description = strip_html(d.get("description", ""))
                ats_url = await self.resolve_ats_url(client, link, description, page)
                if not ats_url:
                    self.stats.dropped_unresolved += 1
                    log.debug("remoteok: no ATS link behind %s", link)
                    continue
                loc = (d.get("location") or "Remote").strip()
                if not location_matches(loc, self.s.search_location, self.s.remote_only,
                                        countries=self.s.countries):
                    self.stats.dropped_location += 1
                    continue
                job = JobPosting(
                    job_id=job_id_from_url(ats_url), url=ats_url, apply_url=ats_url,
                    title=(d.get("position") or "").strip(), company=(d.get("company") or "").strip(),
                    location=loc,
                    description=description, source=self.name, ats=detect_ats(ats_url),
                    relevance=title_relevance(
                        d.get("position") or d.get("title") or "", self.tokens),
                )
                if self._new(job):
                    found.append(job)
                await asyncio.sleep(self.s.discovery_delay_s)
        return found


class HimalayasSource(AggregatorSource):
    """https://himalayas.app/jobs/api"""

    name = "himalayas"
    FEED = "https://himalayas.app/jobs/api?limit={limit}"

    async def discover(self, page: Any = None) -> list[JobPosting]:
        found: list[JobPosting] = []
        async with await self._client() as client:
            data = await self._json(client, self.FEED.format(limit=self.s.aggregator_page_size))
            jobs = (data or {}).get("jobs") or []
            self.stats.seen += len(jobs)
            matched = []
            for d in jobs:
                if not title_matches(d.get("title", ""), self.tokens, self.s.title_match_threshold):
                    self.stats.dropped_title += 1
                elif not seniority_matches(d.get("title", ""), self.s.seniority_levels):
                    self.stats.dropped_seniority += 1
                else:
                    matched.append(d)
            log.info("himalayas: %d of %d listings match the title filter", len(matched), len(jobs))
            for d in matched[: self.s.max_jobs_per_company]:
                link = d.get("applicationLink") or d.get("guid") or ""
                description = strip_html(d.get("description", ""))
                ats_url = await self.resolve_ats_url(client, link, description, page)
                if not ats_url:
                    self.stats.dropped_unresolved += 1
                    log.debug("himalayas: no ATS link behind %s", link)
                    continue
                loc = ", ".join(d.get("locationRestrictions") or []) or "Remote"
                if not location_matches(loc, self.s.search_location, self.s.remote_only,
                                        countries=self.s.countries):
                    self.stats.dropped_location += 1
                    continue
                job = JobPosting(
                    job_id=job_id_from_url(ats_url), url=ats_url, apply_url=ats_url,
                    title=(d.get("title") or "").strip(), company=(d.get("companyName") or "").strip(),
                    location=loc,
                    description=description, source=self.name, ats=detect_ats(ats_url),
                    relevance=title_relevance(
                        d.get("position") or d.get("title") or "", self.tokens),
                )
                if self._new(job):
                    found.append(job)
                await asyncio.sleep(self.s.discovery_delay_s)
        return found


# --------------------------------------------------------------------------
# Google search (browser driven)
# --------------------------------------------------------------------------


#: Google's bot wall. It never says "captcha", so the usual detection missed it and a
#: blocked search looked exactly like a search that simply found nothing.
SEARCH_BLOCKED_RE = re.compile(
    r"unusual traffic|detected unusual|automated queries|are you a robot|"
    r"before you continue to google|verify you.re human|/sorry/", re.I)

#: Hosts whose links are a real application form, and the ATS behind each one.
BOARD_HOSTS: dict[str, str] = {
    "job-boards.greenhouse.io": GREENHOUSE,
    "boards.greenhouse.io": GREENHOUSE,
    "jobs.lever.co": LEVER,
    "jobs.ashbyhq.com": ASHBY,
}

#: Google prints each result's address as "https://job-boards.greenhouse.io › acme › jobs".
#: The company slug in the middle is the board token, which is all we need.
CITE_RE = re.compile(r"https?://([\w.-]+)\s*(?:›|>|/)\s*([\w.-]+)")

#: Path segments that are part of the board's own URL shape, never a company.
NOT_A_TOKEN = {"jobs", "job", "embed", "boards", "www", "search"}


def unwrap_result_link(href: Optional[str]) -> Optional[str]:
    """Search engines wrap their results; return the destination URL where it is readable.

    Google's current results page is not readable this way: it points every link at
    `/goto?url=<opaque token>` and keeps the real address out of the DOM entirely.
    Those are resolved by following the redirect instead, in `resolve_wrapped`.
    """
    if not href:
        return None
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urlparse(href)
    except ValueError:
        return None
    host = parsed.hostname or ""
    query = parse_qs(parsed.query)
    if "google." in host and parsed.path == "/url":
        return query.get("q", [None])[0]
    if "duckduckgo.com" in host and parsed.path.startswith("/l/"):
        return query.get("uddg", [None])[0]
    if host.endswith("bing.com") and parsed.path.startswith("/ck/"):
        return query.get("u", [None])[0]
    return href


def board_reference(url: Optional[str]) -> Optional[tuple[str, str]]:
    """(ats, company token) for an application URL, or None if it is not one."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    ats = BOARD_HOSTS.get((parsed.hostname or "").lower())
    if not ats:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if not parts or parts[0].lower() in NOT_A_TOKEN:
        return None
    return ats, parts[0]


def tokens_from_cites(cites: Iterable[str]) -> dict[str, set[str]]:
    """Board tokens read off the printed addresses under each search result.

    This is the cheap half of the Google source. One results page names ten or twenty
    companies, and each one is a whole board the API can then read in full, with real
    descriptions, which a scraped link never has.
    """
    out: dict[str, set[str]] = {}
    for text in cites:
        clean = (text or "").strip()
        match = CITE_RE.match(clean)
        if not match:
            continue
        ats = BOARD_HOSTS.get(match.group(1).lower())
        token = match.group(2)
        if not ats or token.lower() in NOT_A_TOKEN or len(token) < 2:
            continue
        # Google shortens a long address with an ellipsis. An ellipsis is not a word
        # character, so it never reaches `token`; the only way to tell that the slug
        # was cut short is to look at what follows it.
        rest = clean[match.end(2):].lstrip()
        if rest.startswith("…") or rest.startswith("..."):
            continue
        out.setdefault(ats, set()).add(token)
    return out


class GoogleSearchSource(ApiJobSource):
    """`site:job-boards.greenhouse.io "Backend Engineer"` driven through the browser.

    Google stopped putting result URLs in the page. Every link now points at
    `/goto?url=<opaque token>`, so reading `href` returns nothing that looks like a
    job, which is why this source used to report "0 ATS links" on a page full of
    them. Two routes get the addresses back:

      1. the printed address under each result, which still names the company board.
         One page names ten or twenty boards, and the board API then returns every
         one of their jobs with a real description. This costs no extra requests.
      2. following `/goto?url=` to wherever it lands, which gives an exact job URL.
         Used only when the printed addresses yielded nothing, since it costs one
         navigation per result.

    So Google is used for what it is good at, finding *which companies are hiring*,
    and the board APIs do the rest. When Google serves its bot check instead, the run
    pauses so you can clear it, because only a person can.
    """

    name = "google"
    SITES = ("job-boards.greenhouse.io", "jobs.lever.co", "jobs.ashbyhq.com")
    #: `udm=14` asks for plain web results, without the AI panels that bury the list.
    SEARCH_URL = "https://www.google.com/search?q={q}&udm=14&num=30"
    #: Following an opaque link costs a page load, so only a few are ever resolved.
    MAX_LINK_RESOLVES = 6
    #: Google tolerates a handful of searches per minute from one browser.
    PAUSE_S = 4.0

    RESULTS_JS = """() => {
        const HOSTS = ['job-boards.greenhouse.io', 'boards.greenhouse.io',
                       'jobs.lever.co', 'jobs.ashbyhq.com'];
        const out = {direct: [], wrapped: [], cites: []};
        for (const a of document.querySelectorAll('a[href]')) {
            const raw = a.getAttribute('href') || '';
            let host = '';
            try { host = new URL(a.href).hostname; } catch (e) { host = ''; }
            if (HOSTS.includes(host)) out.direct.push(a.href);
            else if (raw.startsWith('/goto?') || raw.startsWith('/url?') ||
                     raw.startsWith('/l/?') || raw.startsWith('/ck/a?')) out.wrapped.push(a.href);
        }
        for (const c of document.querySelectorAll('cite')) out.cites.push(c.innerText || '');
        return out;
    }"""

    def __init__(self, settings: Settings, db: Database, browser: Any = None):
        super().__init__(settings, db, browser)
        #: Boards this run found, so a caller can offer to remember them.
        self.found_boards: dict[str, set[str]] = {}
        self.blocked_searches = 0

    # ---- one results page ---------------------------------------------
    async def is_blocked(self, page: Any) -> bool:
        try:
            if "/sorry/" in (page.url or ""):
                return True
            text = await page.evaluate(
                "() => document.body ? document.body.innerText.slice(0, 1200) : ''")
        except Exception:
            return False
        return bool(SEARCH_BLOCKED_RE.search(text or ""))

    async def resolve_wrapped(self, page: Any, links: list[str]) -> set[str]:
        """Follow Google's opaque links to the addresses they hide."""
        out: set[str] = set()
        if not links:
            return out
        try:
            context = page.context
        except Exception:
            return out
        for link in links[: self.MAX_LINK_RESOLVES]:
            tab = None
            try:
                tab = await context.new_page()
                await tab.goto(link, wait_until="commit",
                               timeout=self.s.navigation_timeout_ms)
                await tab.wait_for_timeout(800)
                if board_reference(tab.url):
                    out.add(tab.url.split("?")[0])
            except Exception as exc:
                log.debug("could not follow a result link: %s", exc)
            finally:
                if tab is not None:
                    try:
                        await tab.close()
                    except Exception:
                        pass
        return out

    async def harvest(self, page: Any) -> tuple[set[str], dict[str, set[str]]]:
        """Exact job URLs and company board tokens from the results page in view."""
        try:
            data = await page.evaluate(self.RESULTS_JS)
        except Exception as exc:
            log.warning("could not read the results page: %s", exc)
            return set(), {}

        urls: set[str] = set()
        for href in data.get("direct", []):
            target = unwrap_result_link(href)
            if board_reference(target):
                urls.add((target or "").split("?")[0])

        boards = tokens_from_cites(data.get("cites", []))
        if not boards and not urls:
            # The printed addresses gave nothing, so pay for the redirects instead.
            urls = await self.resolve_wrapped(page, data.get("wrapped", []))

        for url in urls:                       # an exact URL names its board too
            ref = board_reference(url)
            if ref:
                boards.setdefault(ref[0], set()).add(ref[1])
        return urls, boards

    # ---- the search itself ---------------------------------------------
    async def search(self, page: Any, terms: str) -> tuple[set[str], dict[str, set[str]]]:
        log.info("google: %s", terms)
        try:
            await self.b.goto(page, self.SEARCH_URL.format(q=quote_plus(terms)))
        except Exception as exc:
            log.warning("google search failed: %s", exc)
            return set(), {}

        if await self.is_blocked(page):
            log.warning("Google served its bot check instead of results")
            outcome = await self.b.gate.wait(
                "Google is showing its 'unusual traffic' check instead of search results. "
                "Clear it in the browser window, then continue. Skipping is fine: the "
                "Greenhouse, Lever and Ashby board APIs cover the same boards without a "
                "search engine.")
            if outcome == "skip" or await self.is_blocked(page):
                self.blocked_searches += 1
                return set(), {}

        urls, boards = await self.harvest(page)
        log.info("google: %d board(s), %d exact link(s)",
                 sum(len(v) for v in boards.values()), len(urls))
        return urls, boards

    async def discover(self, page: Any = None) -> list[JobPosting]:
        if page is None:
            log.warning("the google source needs a browser page; skipping it")
            return []

        urls: set[str] = set()
        boards: dict[str, set[str]] = {}
        searches = 0
        for site in self.SITES:
            for query in self.s.search_queries:
                terms = f'site:{site} "{query}"'
                where = (self.s.search_location or "").strip()
                if where and where.lower() not in ("", "anywhere", "worldwide"):
                    terms += f' "{where}"'
                searches += 1
                page_urls, page_boards = await self.search(page, terms)
                urls |= page_urls
                for ats, tokens in page_boards.items():
                    boards.setdefault(ats, set()).update(tokens)
                await self.b.sleep(self.PAUSE_S, 1.5)

        self.found_boards = boards
        if self.blocked_searches:
            log.warning(
                "Google blocked %d of %d searches. The Greenhouse, Lever and Ashby board "
                "APIs need no search engine and are not rate limited.",
                self.blocked_searches, searches)
        if not boards and not urls:
            return []

        log.info("google: reading %d board(s) through their APIs",
                 sum(len(v) for v in boards.values()))
        found = await self.read_boards(boards)
        found += self.remaining_urls(urls, found)
        return found

    # ---- turning boards into postings -----------------------------------
    async def read_boards(self, boards: dict[str, set[str]]) -> list[JobPosting]:
        """Hand each discovered board to the API source that understands it.

        The API returns the full description, which a search result never carries, so
        every posting arrives ready to score without opening another page. Boards are
        read one at a time so a board that does not answer can be told apart from one
        that answered with nothing you wanted, and only the former is forgotten.
        """
        found: list[JobPosting] = []
        confirmed: dict[str, set[str]] = {}
        for ats, tokens in boards.items():
            source_cls = SOURCE_REGISTRY.get(ats)
            if source_cls is None or not tokens:
                continue
            for token in sorted(tokens):
                source = source_cls(self.s, self.db, self.b, tokens=[token])
                source.on_batch = lambda _name, batch: self._report(batch)
                try:
                    found.extend(await source.discover())
                except Exception as exc:
                    log.warning("google: could not read %s/%s: %s", ats, token, exc)
                    continue
                finally:
                    self.stats += source.stats
                if source.stats.seen:
                    confirmed.setdefault(ats, set()).add(token)
        self.found_boards = confirmed
        self.remember_boards(confirmed)
        return found

    def remember_boards(self, boards: dict[str, set[str]]) -> dict[str, list[str]]:
        """Add confirmed boards to `companies.json` so the next scan needs no search.

        This is the lasting value of the Google source. A search costs a page load and
        can be refused; a company token costs nothing and keeps working, so a board is
        worth finding once and keeping. Only boards that answered are written, nothing
        is ever removed, and the file stays sorted so the diff is readable.
        """
        if not boards:
            return {}
        path = Path(self.s.company_file)
        try:
            stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("google: leaving %s alone, it is not readable: %s", path.name, exc)
            return {}

        added: dict[str, list[str]] = {}
        for ats, tokens in boards.items():
            current = stored.get(ats)
            if not isinstance(current, list):
                current = []
            fresh = sorted(t for t in tokens if t not in current)
            if not fresh:
                continue
            stored[ats] = sorted(current + fresh)
            added[ats] = fresh
        if not added:
            return {}
        try:
            path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:
            log.warning("google: could not save the new boards: %s", exc)
            return {}
        for ats, fresh in added.items():
            log.info("google: remembered %d new %s board(s): %s",
                     len(fresh), ats, ", ".join(fresh))
        return added

    def remaining_urls(self, urls: set[str], found: list[JobPosting]) -> list[JobPosting]:
        """Exact links the board APIs did not already cover, kept rather than discarded."""
        have = {j.url for j in found} | {j.apply_url for j in found if j.apply_url}
        extra: list[JobPosting] = []
        for url in sorted(urls - have):
            job = JobPosting.from_url(url, source=self.name)
            self.stats.seen += 1
            if self.db.has_job(self.name, job.job_id):
                self.stats.dropped_seen_before += 1
                continue
            self.stats.kept += 1
            extra.append(job)
        self._report(extra)
        return extra

    async def hydrate(self, page: Any, job: JobPosting) -> JobPosting:
        """Only the leftover links need this; board postings arrive already described."""
        if job.description:
            return job
        from job_search import extract_posting

        return await extract_posting(self.b, page, job)


#: Below this many results a scan is worth explaining, even though it is not empty.
FEW_RESULTS = 5


def explain_empty_scan(stats: "ScanStats", settings: Settings) -> list[str]:
    """Name the filter responsible when a scan returns nothing, or very little.

    Advice used to appear only for an empty result, which left the more common case
    unexplained: a scan that returns one posting out of two thousand looks like the
    search terms were too narrow, when it is usually the country filter.
    """
    if not stats.seen:
        return ["No board returned any postings. Check the company tokens with "
                "`python main.py companies --probe`."]
    if stats.kept > FEW_RESULTS:
        return []

    tips: list[str] = []
    if stats.kept:
        tips.append(f"Only {stats.kept} of {stats.seen} postings survived every filter. "
                    f"Here is where the rest went.")
    for label, n in sorted(stats.reasons(), key=lambda kv: -kv[1])[:3]:
        share = round(100 * n / stats.seen)
        if label == "search terms":
            tips.append(f"{n} ({share}%) did not resemble your search terms "
                        f"({', '.join(settings.search_queries)}). Most backend roles are titled "
                        f"'Software Engineer, <team>' rather than 'Backend Engineer', so lower "
                        f"Match sensitivity or add a broader term.")
        elif label == "seniority":
            tips.append(f"{n} ({share}%) were the wrong seniority. You have "
                        f"{', '.join(settings.seniority_levels)} selected; untick to allow any level.")
        elif label == "location or country":
            where = ", ".join(settings.countries) if settings.countries else settings.search_location
            detail = f"{n} ({share}%) were outside {where or 'your location filter'}"
            if settings.countries and settings.remote_only:
                detail += (". Remote-only and a country together are strict: a posting has to be "
                           "genuinely remote AND name that country. Most company boards are based "
                           "in the US and Europe, so try Anywhere / Worldwide, or clear the country "
                           "and keep remote-only")
            elif settings.countries:
                detail += (". These boards are mostly US and European, so a country outside that "
                           "returns very little. Anywhere / Worldwide keeps fully remote roles")
            elif settings.remote_only:
                detail += " because remote-only drops hybrid and on-site postings"
            tips.append(detail + ".")
        elif label == "already applied":
            tips.append(f"{n} are already in your database. "
                        "`python main.py delete --status failed` frees them up.")
        elif label == "no description":
            tips.append(f"{n} came back with no usable description.")
        elif label == "no application link":
            tips.append(f"{n} aggregator listings had no Greenhouse, Lever or Ashby link behind them.")
    return tips


SOURCE_REGISTRY: dict[str, type[ApiJobSource]] = {
    GREENHOUSE: GreenhouseBoardSource,
    LEVER: LeverBoardSource,
    ASHBY: AshbyBoardSource,
    "remoteok": RemoteOKSource,
    "himalayas": HimalayasSource,
    "google": GoogleSearchSource,
}
