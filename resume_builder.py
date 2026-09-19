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
from models import JobPosting

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

        experience = []
        for exp in profile.get("experience", []):
            g = groups.get(("experience", _norm(exp.get("company", ""))))
            bullets = [b.strip() for b in (g.bullets if g else exp.get("bullets", [])) if b.strip()]
            experience.append({**exp, "bullets": bullets or exp.get("bullets", [])})

        projects = []
        for proj in profile.get("projects", []):
            g = groups.get(("project", _norm(proj.get("name", ""))))
            bullets = [b.strip() for b in (g.bullets if g else proj.get("bullets", [])) if b.strip()]
            projects.append({**proj, "bullets": bullets or proj.get("bullets", [])})

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
        if highlighted:
            skill_groups.append({"label": "Core", "items": highlighted})
        for label, items in skill_groups_raw.items():
            rest = [s for s in items if s not in highlighted]
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
        return template.render(**self.build_context(profile, analysis))

    # ---- PDF ------------------------------------------------------------
    def output_path(self, job: JobPosting, analysis: JobAnalysis) -> Path:
        company = job.company or analysis.company_name
        role = job.title or analysis.job_title
        return self.settings.output_dir / f"{slugify(company)}_{slugify(role)}.pdf"

    async def build(self, profile: dict, analysis: JobAnalysis, job: JobPosting) -> Path:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        html = self.render_html(profile, analysis)
        path = self.output_path(job, analysis)
        await html_to_pdf(html, path, single_page=True)
        log.info("Resume written to %s", path)
        return path


async def html_to_pdf(html: str, path: Path, single_page: bool = True) -> Path:
    """Render HTML to PDF with headless Chromium (page.pdf only works headless)."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(html, wait_until="load")
            await page.emulate_media(media="print")
            for scale in SCALE_STEPS:
                await page.pdf(
                    path=str(path),
                    format="Letter",
                    print_background=True,
                    prefer_css_page_size=True,
                    scale=scale,
                )
                if not single_page:
                    break
                pages = _count_pages(path)
                if pages is None or pages <= 1:
                    break
                log.info("Resume is %d pages at scale %.2f, shrinking", pages, scale)
        finally:
            await browser.close()
    return path
