"""Turn a JobAnalysis + master profile into an ATS-friendly single-page PDF.

Rendering pipeline: Jinja2 HTML template -> headless Chromium (Playwright) -> PDF.
If the result spills onto a second page, the PDF is re-rendered at a slightly
smaller scale until it fits (max 5 attempts).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.async_api import async_playwright

from ai_agent import JobAnalysis
from config import Settings
from config import settings as default_settings
from models import JobPosting, ats_text

log = logging.getLogger(__name__)

SCALE_STEPS = (1.0, 0.95, 0.9, 0.85, 0.8)


def slugify(value: str, max_len: int = 48) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_")
    return (value or "Unknown")[:max_len].rstrip("_")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _count_pages(path: Path) -> Optional[int]:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        return None
    try:
        return len(PdfReader(str(path)).pages)
    except Exception as exc:  # pragma: no cover
        log.debug("Could not count pages of %s: %s", path, exc)
        return None


class ResumeBuilder:
    def __init__(self, settings: Settings = default_settings):
        self.settings = settings
        self.env = Environment(
            loader=FileSystemLoader(str(settings.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ---- view model -----------------------------------------------------
    def build_context(self, profile: dict, analysis: JobAnalysis) -> dict[str, Any]:
        """Merge the tailored content back into the profile with code-level guardrails."""
        groups = {(g.kind, _norm(g.name)): g for g in analysis.tailored_bullets}

        # The AI orders tailored_bullets most-relevant-first. Preserve that rank so that,
        # if the page overflows, the entries dropped are the ones it judged least relevant.
        rank = {(g.kind, _norm(g.name)): i for i, g in enumerate(analysis.tailored_bullets)}

        experience = []
        for pos, exp in enumerate(profile.get("experience", [])):
            key = ("experience", _norm(exp.get("company", "")))
            g = groups.get(key)
            bullets = [b.strip() for b in (g.bullets if g else exp.get("bullets", [])) if b.strip()]
            experience.append({**exp, "bullets": bullets or exp.get("bullets", []),
                               "_pos": pos, "_rank": rank.get(key, 999)})

        projects = []
        for pos, proj in enumerate(profile.get("projects", [])):
            key = ("project", _norm(proj.get("name", "")))
            g = groups.get(key)
            bullets = [b.strip() for b in (g.bullets if g else proj.get("bullets", [])) if b.strip()]
            projects.append({**proj, "bullets": bullets or proj.get("bullets", []),
                             "_pos": pos, "_rank": rank.get(key, 999)})
        # Experience stays in reverse-chronological order (profile order); projects follow AI relevance.
        projects.sort(key=lambda p: (p["_rank"], p["_pos"]))

        unmatched = [g.name for (kind, _), g in groups.items()
                     if not any(_norm(g.name) == _norm(e.get("company", "")) for e in profile.get("experience", []))
                     and not any(_norm(g.name) == _norm(p.get("name", "")) for p in profile.get("projects", []))]
        if unmatched:
            log.warning("Dropping tailored bullet groups that match no profile entry: %s", unmatched)

        # Skills: only keep highlighted skills that exist in the profile (hallucination guard).
        skill_groups_raw: dict[str, list[str]] = profile.get("skills", {}) or {}
        known = {_norm(s): s for items in skill_groups_raw.values() for s in items}
        highlighted = []
        for s in analysis.highlighted_skills:
            key = _norm(s)
            match = known.get(key) or next((v for k, v in known.items() if key and (key in k or k in key)), None)
            if match and match not in highlighted:
                highlighted.append(match)
        skill_groups = []
        used = set()
        if highlighted:
            skill_groups.append({"label": "Core", "items": highlighted})
            used.update(_norm(s) for s in highlighted)
        for label, items in skill_groups_raw.items():
            rest = []
            for item in items:  # a skill listed in two profile groups is printed once
                k = _norm(item)
                if k not in used:
                    used.add(k)
                    rest.append(item)
            if rest:
                skill_groups.append({"label": label.replace("_", " ").title(), "items": rest})

        links = []
        for key, label in (("linkedin", "LinkedIn"), ("github", "GitHub"), ("website", "Portfolio")):
            url = profile.get(key)
            if url:
                links.append({"label": label, "url": url, "display": re.sub(r"^https?://(www\.)?", "", url).rstrip("/")})

        return {
            "name": profile.get("name", ""),
            "headline": profile.get("headline", ""),
            "email": profile.get("email", ""),
            "phone": profile.get("phone", ""),
            "location": profile.get("location", ""),
            "links": links,
            "summary": analysis.tailored_summary.strip() or profile.get("summary", ""),
            "skill_groups": skill_groups,
            "experience": experience,
            "projects": projects,
            "education": profile.get("education", []),
            "certifications": profile.get("certifications", []),
        }

    def render_html(self, profile: dict, analysis: JobAnalysis) -> str:
        template = self.env.get_template("resume.html")
        # Flatten typographic characters so an applicant tracking system reads the same
        # words a human does: "Full\u2011stack" must not hide from a search for "Full-stack".
        return template.render(**ats_text(self.build_context(profile, analysis)))

    # ---- PDF ------------------------------------------------------------
    def output_path(self, job: JobPosting, analysis: JobAnalysis) -> Path:
        company = job.company or analysis.company_name
        role = job.title or analysis.job_title
        return self.settings.output_dir / f"{slugify(company)}_{slugify(role)}.pdf"

    def _variants(self, context: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """Progressively shorter versions of the same truthful content, longest first.

        Nothing is invented or reworded here; entries are only dropped, least relevant first,
        so a dense profile still fits on one ATS-friendly page.
        """
        out: list[tuple[str, dict[str, Any]]] = [("full", context)]
        n_proj, n_exp = len(context["projects"]), len(context["experience"])
        for keep_proj in (5, 4, 3, 2, 1):
            if keep_proj < n_proj:
                out.append((f"projects<={keep_proj}", {**context, "projects": context["projects"][:keep_proj]}))
        base = out[-1][1]
        for keep_exp in (4, 3):
            if keep_exp < n_exp:
                out.append((f"experience<={keep_exp}",
                            {**base, "experience": base["experience"][:keep_exp]}))
        trimmed = out[-1][1]
        out.append(("no-projects", {**trimmed, "projects": []}))
        return out

    async def build(self, profile: dict, analysis: JobAnalysis, job: JobPosting) -> Path:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_path(job, analysis)
        template = self.env.get_template("resume.html")
        context = ats_text(self.build_context(profile, analysis))
        variants = [(name, template.render(**ctx)) for name, ctx in self._variants(context)]
        used = await render_first_that_fits(variants, path)
        log.info("Resume written to %s (%s)", path, used)
        return path


async def render_first_that_fits(variants: list[tuple[str, str]], path: Path) -> str:
    """Render each HTML variant until one lands on a single page; shrink the last one if needed.

    Strategy, in order: full content at 100% scale, then progressively trimmed content, then
    scale reduction on the shortest variant. `page.pdf` only works in headless Chromium, so a
    dedicated headless browser is used even when the applier runs headed.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.emulate_media(media="print")
            for name, html in variants:
                await page.set_content(html, wait_until="load")
                await page.pdf(path=str(path), format="Letter", print_background=True,
                               prefer_css_page_size=True, scale=1.0)
                pages = _count_pages(path)
                if pages is None or pages <= 1:
                    return name
                log.debug("Variant %r is %d pages, trimming further", name, pages)
            for scale in SCALE_STEPS[1:]:  # shortest variant is still loaded
                await page.pdf(path=str(path), format="Letter", print_background=True,
                               prefer_css_page_size=True, scale=scale)
                pages = _count_pages(path)
                if pages is None or pages <= 1:
                    return f"{variants[-1][0]} @ {scale:.2f}"
                log.info("Resume is %d pages at scale %.2f, shrinking", pages, scale)
        finally:
            await browser.close()
    log.warning("Resume still exceeds one page at minimum scale: %s", path)
    return "overflow"


async def html_to_pdf(html: str, path: Path, single_page: bool = True) -> Path:
    """Render a single HTML string to PDF (kept for ad-hoc use and tests)."""
    await render_first_that_fits([("single", html)], path)
    return path
