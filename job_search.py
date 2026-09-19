"""Job discovery + posting extraction.

Sources:
  LinkedInJobSource - LinkedIn job search (Easy Apply filter) + posting pages.
                      Postings that hand off to Greenhouse/Lever/Ashby are
                      detected and routed to the matching applier.
  UrlListSource     - explicit list of URLs (LinkedIn, Greenhouse, Lever, Ashby)
                      from `jobs.txt`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from browser_bot import StealthBrowser
from config import Settings
from database import Database
from models import ASHBY, GREENHOUSE, LEVER, LINKEDIN, UNKNOWN, JobPosting, detect_ats

log = logging.getLogger(__name__)

LINKEDIN_CARDS_JS = r"""
() => {
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const dedupe = s => { const h = Math.floor(s.length / 2); const a = s.slice(0, h).trim(), b = s.slice(h).trim();
                        return a && a === b ? a : s; };
  const out = []; const seen = new Set();
  document.querySelectorAll('li[data-occludable-job-id], div[data-job-id], a[href*="/jobs/view/"]').forEach(el => {
    let id = el.getAttribute('data-occludable-job-id') || el.getAttribute('data-job-id');
    if (!id) { const m = (el.getAttribute('href') || '').match(/\/jobs\/view\/(\d+)/); id = m && m[1]; }
    if (!id || seen.has(id)) return; seen.add(id);
    const card = el.closest('li') || el;
    const titleEl = card.querySelector('a[href*="/jobs/view/"] strong, .job-card-list__title strong, .job-card-list__title, a.job-card-container__link, a[href*="/jobs/view/"]');
    const companyEl = card.querySelector('.artdeco-entity-lockup__subtitle, .job-card-container__primary-description, .job-card-container__company-name, [class*="company-name" i], [class*="primary-description" i]');
    const locEl = card.querySelector('.artdeco-entity-lockup__caption, .job-card-container__metadata-item, [class*="metadata" i] li, [class*="caption" i]');
    const easy = /easy apply/i.test(card.innerText || '');
    out.push({ job_id: id, title: dedupe(clean(titleEl ? titleEl.innerText : '')),
               company: clean(companyEl ? companyEl.innerText : ''), location: clean(locEl ? locEl.innerText : ''),
               easy_apply: easy });
  });
  return out;
}
"""

DESCRIPTION_SELECTORS = {
    LINKEDIN: ("#job-details", ".jobs-description__content", ".jobs-box__html-content", ".jobs-description", "article"),
    GREENHOUSE: ("#content", "div.job__description", "#app_body", ".job-post", "main"),
    LEVER: (".posting-page", ".section-wrapper.page-full-width", ".posting", "main"),
    ASHBY: ("div[class*='jobPosting' i]", "div[class*='description' i]", "main"),
    UNKNOWN: ("main", "article", "body"),
}
COMPANY_SELECTORS = {
    LINKEDIN: (".job-details-jobs-unified-top-card__company-name a", ".job-details-jobs-unified-top-card__company-name",
               ".jobs-unified-top-card__company-name a", "a[href*='/company/']"),
    GREENHOUSE: (".company-name", "span.company-name", "div[class*='company' i]"),
    LEVER: (".main-header-logo img[alt]", ".posting-headline .sort-by-time", "a.main-header-logo"),
    ASHBY: ("div[class*='companyName' i]", "span[class*='companyName' i]", "img[alt*='logo' i]"),
    UNKNOWN: (),
}


def _pretty_slug(url: str, index: int = 1) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) > index - 1:
        return re.sub(r"[-_]+", " ", parts[index - 1]).title()
    return ""


async def _first_text(page: Page, selectors: tuple[str, ...], min_len: int = 0) -> str:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            n = min(await loc.count(), 4)
            for i in range(n):
                el = loc.nth(i)
                if not await el.is_visible():
                    continue
                if sel.endswith("[alt]") or "img" in sel:
                    text = await el.get_attribute("alt") or ""
                else:
                    text = await el.inner_text()
                text = re.sub(r"[ \t]+", " ", text).strip()
                if len(text) >= min_len and text:
                    return text
        except Exception:
            continue
    return ""


async def _meta(page: Page, prop: str) -> str:
    try:
        v = await page.locator(f'meta[property="{prop}"], meta[name="{prop}"]').first.get_attribute("content", timeout=1500)
        return (v or "").strip()
    except Exception:
        return ""


async def extract_posting(browser: StealthBrowser, page: Page, job: JobPosting) -> JobPosting:
    """Fill title / company / location / description from the posting page (ATS-aware)."""
    ats = job.ats if job.ats != UNKNOWN else detect_ats(job.url)
    if job.apply_url and ats != LINKEDIN:
        ats = detect_ats(job.apply_url)
    await browser.goto(page, job.url)
    await browser.human_scroll(page, 600)

    # Expand truncated descriptions ("See more")
    try:
        more = page.get_by_role("button", name=re.compile(r"see more|show more|read more", re.I))
        if await more.count() and await more.first.is_visible():
            await browser.human_click(more.first)
    except Exception:
        pass

    if not job.title:
        job.title = await _first_text(page, ("h1", "h2.t-24", "[class*='job-title' i]", "[class*='posting-headline' i] h2"))
        if not job.title:
            og = await _meta(page, "og:title") or (await page.title())
            job.title = re.split(r"\s+[|@-]\s+| at ", og)[0].strip() if og else ""
    if not job.company:
        job.company = await _first_text(page, COMPANY_SELECTORS.get(ats, ()))
        if not job.company:
            og_site = await _meta(page, "og:site_name")
            if og_site and "linkedin" not in og_site.lower():
                job.company = og_site
        if not job.company and ats in (GREENHOUSE, LEVER, ASHBY):
            job.company = _pretty_slug(job.apply_url or job.url, 1)
    if not job.location and ats == LINKEDIN:
        job.location = await _first_text(page, (
            ".job-details-jobs-unified-top-card__primary-description-container .tvm__text",
            ".jobs-unified-top-card__bullet", "[class*='primary-description' i]"))
        job.location = job.location.split("·")[0].strip() if job.location else ""

    job.description = await _first_text(page, DESCRIPTION_SELECTORS.get(ats, DESCRIPTION_SELECTORS[UNKNOWN]), min_len=200)
    if not job.description:
        job.description = await _first_text(page, DESCRIPTION_SELECTORS[UNKNOWN], min_len=200)
    job.description = job.description.strip()[:20_000]
    job.ats = ats
    return job


class LinkedInJobSource:
    name = LINKEDIN
    BASE = "https://www.linkedin.com/jobs/search/"

    def __init__(self, settings: Settings, browser: StealthBrowser, db: Database):
        self.s = settings
        self.b = browser
        self.db = db

    def search_url(self, query: str, start: int = 0) -> str:
        params: dict[str, str] = {"keywords": query, "location": self.s.search_location, "start": str(start)}
        if not self.s.follow_external_apply:
            params["f_AL"] = "true"  # Easy Apply only
        if self.s.posted_within_hours > 0:
            params["f_TPR"] = f"r{self.s.posted_within_hours * 3600}"
        return f"{self.BASE}?{urlencode(params)}"

    async def _scroll_results(self, page: Page) -> None:
        try:
            first = page.locator("li[data-occludable-job-id], div[data-job-id]").first
            await first.wait_for(state="visible", timeout=15_000)
            await first.hover()
        except PlaywrightTimeout:
            return
        for _ in range(6):
            await page.mouse.wheel(0, 900)
            await self.b.sleep(0.7, 0.2)

    async def discover(self, page: Page) -> list[JobPosting]:
        jobs: list[JobPosting] = []
        seen: set[str] = set()
        for query in self.s.search_queries:
            found = 0
            start = 0
            while found < self.s.max_jobs_per_query and start < 100:
                await self.b.goto(page, self.search_url(query, start))
                await self._scroll_results(page)
                cards = await page.evaluate(LINKEDIN_CARDS_JS)
                if not cards:
                    log.info("No job cards for %r at offset %d", query, start)
                    break
                new_on_page = 0
                for c in cards:
                    jid = c["job_id"]
                    if jid in seen:
                        continue
                    seen.add(jid)
                    new_on_page += 1
                    if self.db.has_job(LINKEDIN, jid):
                        continue
                    jobs.append(JobPosting(
                        job_id=jid, url=f"https://www.linkedin.com/jobs/view/{jid}/", title=c["title"],
                        company=c["company"], location=c["location"], source=LINKEDIN, ats=LINKEDIN,
                        easy_apply=bool(c.get("easy_apply")),
                    ))
                    found += 1
                    if found >= self.s.max_jobs_per_query:
                        break
                if new_on_page == 0:
                    break
                start += 25
            log.info("Query %r: %d new postings", query, found)
        return jobs

    async def hydrate(self, page: Page, job: JobPosting) -> JobPosting:
        await extract_posting(self.b, page, job)
        easy = page.get_by_role("button", name=re.compile(r"^\s*easy apply", re.I))
        job.easy_apply = bool(await easy.count()) and await easy.first.is_visible()
        job.ats = LINKEDIN
        if not job.easy_apply and self.s.follow_external_apply:
            external = await self._external_apply_url(page)
            if external:
                job.apply_url = external
                job.ats = detect_ats(external)
                log.info("External apply -> %s (%s)", external, job.ats)
        return job

    async def _external_apply_url(self, page: Page) -> str:
        """Click LinkedIn's 'Apply' (company site) button and capture where it leads."""
        btn = page.get_by_role("button", name=re.compile(r"^\s*apply\b(?! *filters)", re.I)).first
        try:
            await btn.wait_for(state="visible", timeout=4000)
        except PlaywrightTimeout:
            return ""
        context = page.context
        try:
            async with context.expect_page(timeout=12_000) as new_page_info:
                await self.b.human_click(btn)
                try:  # "You're leaving LinkedIn" interstitial
                    cont = page.get_by_role("button", name=re.compile(r"^continue", re.I))
                    if await cont.count() and await cont.first.is_visible():
                        await self.b.human_click(cont.first)
                except Exception:
                    pass
            new_page = await new_page_info.value
            try:
                await new_page.wait_for_load_state("domcontentloaded", timeout=20_000)
                await self.b.sleep(2.0, 0.4)
                url = new_page.url
            finally:
                await new_page.close()
            return url if detect_ats(url) != UNKNOWN else ""
        except PlaywrightTimeout:
            # Same-tab navigation
            if "linkedin.com" not in page.url:
                url = page.url
                await page.go_back()
                return url if detect_ats(url) != UNKNOWN else ""
            return ""
        except Exception as exc:
            log.debug("external apply detection failed: %s", exc)
            return ""


class UrlListSource:
    name = "urls"

    def __init__(self, settings: Settings, browser: StealthBrowser, db: Database, urls: Optional[list[str]] = None):
        self.s = settings
        self.b = browser
        self.db = db
        self.urls = urls
        self.linkedin = LinkedInJobSource(settings, browser, db)

    def _read_urls(self) -> list[str]:
        if self.urls is not None:
            return self.urls
        path = Path(self.s.url_list_file)
        if not path.exists():
            log.warning("URL list %s not found", path)
            return []
        return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]

    async def discover(self, page: Page) -> list[JobPosting]:
        jobs = []
        for url in self._read_urls():
            job = JobPosting.from_url(url, source=self.name)
            if self.db.has_job(self.name, job.job_id):
                continue
            jobs.append(job)
        return jobs

    async def hydrate(self, page: Page, job: JobPosting) -> JobPosting:
        if job.ats == LINKEDIN:
            return await self.linkedin.hydrate(page, job)
        return await extract_posting(self.b, page, job)


def build_sources(settings: Settings, browser: StealthBrowser, db: Database, urls: Optional[list[str]] = None):
    """Assemble the configured discovery sources.

    `linkedin` and `urls` drive a browser; the board APIs and aggregators in
    `discovery.py` are plain HTTP and return fully described postings.
    """
    from discovery import SOURCE_REGISTRY  # imported here to avoid a circular import

    sources = []
    if urls:
        return [UrlListSource(settings, browser, db, urls=urls)]
    for name in settings.sources:
        if name == LINKEDIN:
            sources.append(LinkedInJobSource(settings, browser, db))
        elif name == "urls":
            sources.append(UrlListSource(settings, browser, db))
        elif name in SOURCE_REGISTRY:
            sources.append(SOURCE_REGISTRY[name](settings, db, browser))
        else:
            log.warning("Unknown source %r ignored (known: linkedin, urls, %s)",
                        name, ", ".join(sorted(SOURCE_REGISTRY)))
    return sources


# --------------------------------------------------------------------------
# Lazy browser
# --------------------------------------------------------------------------

#: Sources that need a rendered page. Everything else is plain HTTP, so opening a
#: browser for them just leaves a blank window on screen while nothing happens.
BROWSER_SOURCES = {LINKEDIN, "urls", "google"}
#: These prefer HTTP but fall back to a page when a feed is behind Cloudflare.
BROWSER_FALLBACK_SOURCES = {"remoteok", "himalayas"}


class LazyBrowser:
    """Opens a real browser on first use and not before.

    It stands in for `StealthBrowser` and forwards every attribute, so sources and
    appliers cannot tell the difference. A scan that only touches the Greenhouse,
    Lever and Ashby APIs therefore never launches a window.
    """

    def __init__(self, settings: Settings, gate):
        self.s = settings
        self.gate = gate
        self._browser: Optional[StealthBrowser] = None

    async def __aenter__(self) -> "LazyBrowser":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def ensure(self) -> StealthBrowser:
        if self._browser is None:
            log.info("Launching browser (a source asked for a page)")
            self._browser = await StealthBrowser(self.s, self.gate).start()
        return self._browser

    async def page_for(self, source_name: str):
        """The page a source should use, or None when it does not need one."""
        if source_name in BROWSER_SOURCES:
            return (await self.ensure()).page
        if source_name in BROWSER_FALLBACK_SOURCES:
            return (await self.ensure()).page
        return None

    @property
    def started(self) -> bool:
        return self._browser is not None

    @property
    def page(self):
        if self._browser is None:
            raise RuntimeError("Browser has not been started; call ensure() or page_for() first")
        return self._browser.page

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None

    def __getattr__(self, item):
        """Forward sleep/goto/guard/human_click/... to the real browser once it exists."""
        browser = self.__dict__.get("_browser")
        if browser is None:
            raise AttributeError(
                f"LazyBrowser has no {item!r} yet: the browser has not been started. "
                "Call page_for()/ensure() first."
            )
        return getattr(browser, item)
