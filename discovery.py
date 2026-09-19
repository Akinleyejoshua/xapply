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
     `site:job-boards.greenhouse.io "Full Stack Developer" "Remote"` driven through
     Playwright. Last resort: Google rate-limits and challenges automation, so this
     source pauses for a human when it is challenged.

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
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote_plus, urlparse

import httpx

from config import Settings
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


def query_tokens(queries: Iterable[str]) -> list[set[str]]:
    """['Python Developer'] -> [{'python', 'developer'}] for cheap title pre-filtering."""
    out = []
    for q in queries:
        toks = {t for t in re.split(r"[^a-z0-9+#.]+", q.lower()) if len(t) > 1 and t not in STOPWORDS}
        if toks:
            out.append(toks)
    return out


#: A single shared word is far too loose: "Machine Learning Engineer" would match every
#: posting containing "Engineer". Require a majority of a query's words instead.
TITLE_MATCH_RATIO = 0.6


def title_matches(title: str, token_sets: list[set[str]], ratio: float = TITLE_MATCH_RATIO) -> bool:
    """True when the title carries most of the words of at least one configured query.

    'Backend Engineer'         -> needs both words
    'Machine Learning Engineer'-> needs 2 of 3, so 'Machine Learning Scientist' still matches
                                  but a bare 'Sales Engineer' does not
    """
    if not token_sets:
        return True
    words = set(re.split(r"[^a-z0-9+#.]+", (title or "").lower()))
    for toks in token_sets:
        needed = max(1, math.ceil(len(toks) * ratio))
        if len(toks & words) >= needed:
            return True
    return False


def location_matches(text: str, wanted: str, remote_only: bool, is_remote: Optional[bool] = None) -> bool:
    blob = (text or "").lower()
    if remote_only:
        if is_remote is True:
            return True
        if is_remote is False:
            return False
        return "remote" in blob or "anywhere" in blob or "distributed" in blob
    w = (wanted or "").strip().lower()
    if not w or w in ("remote", "anywhere", "worldwide"):
        return True
    return w in blob or "remote" in blob


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------


class ApiJobSource:
    """Discovery over plain HTTP. Postings arrive fully described, so hydrate() is a no-op."""

    name = "api"

    def __init__(self, settings: Settings, db: Database, browser: Any = None):
        self.s = settings
        self.db = db
        self.b = browser
        self.tokens = query_tokens(settings.search_queries)

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
            return False
        return len(job.description) >= 120

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
                candidates = [
                    j for j in listing["jobs"]
                    if title_matches(j.get("title", ""), self.tokens)
                    and location_matches((j.get("location") or {}).get("name", ""),
                                         self.s.search_location, self.s.remote_only)
                ][: self.s.max_jobs_per_company]
                kept = 0
                for j in candidates:
                    job_id = str(j["id"])
                    if self.db.has_job(self.name, f"gh-{token}-{job_id}"):
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
                    )
                    if self._new(job):
                        found.append(job)
                        kept += 1
                    await asyncio.sleep(self.s.discovery_delay_s)
                log.info("greenhouse/%s: %d/%d postings match", token, kept, len(listing["jobs"]))
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
                    cats = p.get("categories") or {}
                    title = (p.get("text") or "").strip()
                    loc = cats.get("location") or ""
                    if not title_matches(title, self.tokens):
                        continue
                    if not location_matches(f"{loc} {cats.get('commitment','')} {p.get('workplaceType','')}",
                                            self.s.search_location, self.s.remote_only,
                                            is_remote=(p.get("workplaceType") == "remote") or None):
                        continue
                    description = (p.get("descriptionPlain") or "") + "\n\n" + (p.get("additionalPlain") or "")
                    hosted = p.get("hostedUrl") or ""
                    job = JobPosting(
                        job_id=f"lever-{token}-{p.get('id')}",
                        url=hosted,
                        apply_url=p.get("applyUrl") or (hosted.rstrip("/") + "/apply" if hosted else ""),
                        title=title, company=token.replace("-", " ").title(), location=loc,
                        description=description.strip(), source=self.name, ats=LEVER,
                    )
                    if self._new(job):
                        found.append(job)
                        kept += 1
                    if kept >= self.s.max_jobs_per_company:
                        break
                log.info("lever/%s: %d/%d postings match", token, kept, len(postings))
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
                    title = (p.get("title") or "").strip()
                    if not title_matches(title, self.tokens):
                        continue
                    locs = " ".join([p.get("location") or ""] +
                                    [s.get("location", "") for s in (p.get("secondaryLocations") or [])])
                    if not location_matches(f"{locs} {p.get('workplaceType','')}", self.s.search_location,
                                            self.s.remote_only, is_remote=p.get("isRemote")):
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
                    )
                    if self._new(job):
                        found.append(job)
                        kept += 1
                    if kept >= self.s.max_jobs_per_company:
                        break
                log.info("ashby/%s: %d/%d postings match", token, kept, len(data["jobs"]))
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
            matched = [d for d in rows if title_matches(d.get("position", ""), self.tokens)]
            log.info("remoteok: %d/%d listings match the title filter", len(matched), len(rows))
            for d in matched[: self.s.max_jobs_per_company]:
                link = d.get("apply_url") or d.get("url") or ""
                description = strip_html(d.get("description", ""))
                ats_url = await self.resolve_ats_url(client, link, description, page)
                if not ats_url:
                    log.debug("remoteok: no ATS link behind %s", link)
                    continue
                job = JobPosting(
                    job_id=job_id_from_url(ats_url), url=ats_url, apply_url=ats_url,
                    title=(d.get("position") or "").strip(), company=(d.get("company") or "").strip(),
                    location=(d.get("location") or "Remote").strip(),
                    description=description, source=self.name, ats=detect_ats(ats_url),
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
            matched = [d for d in jobs if title_matches(d.get("title", ""), self.tokens)]
            log.info("himalayas: %d/%d listings match the title filter", len(matched), len(jobs))
            for d in matched[: self.s.max_jobs_per_company]:
                link = d.get("applicationLink") or d.get("guid") or ""
                description = strip_html(d.get("description", ""))
                ats_url = await self.resolve_ats_url(client, link, description, page)
                if not ats_url:
                    log.debug("himalayas: no ATS link behind %s", link)
                    continue
                job = JobPosting(
                    job_id=job_id_from_url(ats_url), url=ats_url, apply_url=ats_url,
                    title=(d.get("title") or "").strip(), company=(d.get("companyName") or "").strip(),
                    location=", ".join(d.get("locationRestrictions") or []) or "Remote",
                    description=description, source=self.name, ats=detect_ats(ats_url),
                )
                if self._new(job):
                    found.append(job)
                await asyncio.sleep(self.s.discovery_delay_s)
        return found


# --------------------------------------------------------------------------
# Google search (browser driven)
# --------------------------------------------------------------------------


class GoogleSearchSource(ApiJobSource):
    """`site:job-boards.greenhouse.io "Full Stack Developer" "Remote"` through Playwright.

    Least reliable of the five: Google challenges automation aggressively. It runs
    through the same human gate as everything else, so a challenge pauses rather
    than crashes. Prefer the board APIs; use this to find companies you do not
    already have tokens for.
    """

    name = "google"
    SITES = ("job-boards.greenhouse.io", "jobs.lever.co", "jobs.ashbyhq.com")

    async def discover(self, page: Any = None) -> list[JobPosting]:
        if page is None:
            log.warning("google source needs a browser page; skipping")
            return []
        found: list[JobPosting] = []
        seen: set[str] = set()
        for site in self.SITES:
            for query in self.s.search_queries:
                terms = f'site:{site} "{query}"'
                if self.s.search_location and self.s.search_location.lower() not in ("", "anywhere"):
                    terms += f' "{self.s.search_location}"'
                url = f"https://www.google.com/search?q={quote_plus(terms)}&num=30"
                log.info("google: %s", terms)
                try:
                    await self.b.goto(page, url)
                except Exception as exc:
                    log.warning("google search failed: %s", exc)
                    continue
                await self.b.guard(page)  # consent wall / CAPTCHA -> pause for a human
                try:
                    hrefs = await page.eval_on_selector_all(
                        "a[href]", "els => els.map(e => e.href)"
                    )
                except Exception:
                    hrefs = []
                hits = {h for h in hrefs if ATS_URL_RE.fullmatch(h.split("?")[0])}
                log.info("google: %d ATS links on the results page", len(hits))
                for link in hits:
                    if link in seen:
                        continue
                    seen.add(link)
                    job = JobPosting.from_url(link, source=self.name)
                    if not self.db.has_job(self.name, job.job_id):
                        found.append(job)
                await self.b.sleep(4.0, 1.5)
        return found

    async def hydrate(self, page: Any, job: JobPosting) -> JobPosting:
        from job_search import extract_posting  # imported here to avoid a circular import

        return await extract_posting(self.b, page, job)


SOURCE_REGISTRY: dict[str, type[ApiJobSource]] = {
    GREENHOUSE: GreenhouseBoardSource,
    LEVER: LeverBoardSource,
    ASHBY: AshbyBoardSource,
    "remoteok": RemoteOKSource,
    "himalayas": HimalayasSource,
    "google": GoogleSearchSource,
}
