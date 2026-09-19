"""Turn a JobAnalysis + master profile into an ATS-friendly single-page PDF.

Rendering pipeline: Jinja2 HTML template -> headless Chromium (Playwright) -> PDF.
If the result spills onto a second page, the PDF is re-rendered at a slightly
smaller scale until it fits (max 5 attempts).
"""
from __future__ import annotations

import base64
import difflib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.async_api import async_playwright

from ai_agent import CoverLetter, JobAnalysis
from config import Settings
from config import settings as default_settings
from models import JobPosting, ats_text

log = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"

SCALE_STEPS = (1.0, 0.95, 0.9, 0.85, 0.8)


#: Profile skill groups are keys like `ai_ml`, and a naive title-case renders them
#: "Ai Ml". Anything not listed falls back to title case with underscores removed.
SKILL_GROUP_LABELS = {
    "ai_ml": "AI/ML", "ai": "AI", "ml": "ML", "nlp": "NLP", "ui_ux": "UI/UX",
    "data_analytics": "Data & Analytics", "languages_tools": "Languages & Tools",
    "devops": "DevOps", "cloud_devops": "Cloud & DevOps", "web3": "Web3",
    "frontend": "Frontend", "backend": "Backend", "database": "Databases",
    "databases": "Databases", "mobile": "Mobile", "tools": "Tools",
    "practices": "Practices", "languages": "Languages", "frameworks": "Frameworks",
    "data": "Data", "qa": "QA", "apis": "APIs",
}


def skill_group_label(key: str) -> str:
    """Human-readable heading for a skills group key."""
    flat = (key or "").strip().lower()
    if flat in SKILL_GROUP_LABELS:
        return SKILL_GROUP_LABELS[flat]
    words = re.split(r"[_\s]+", flat)
    return " ".join(SKILL_GROUP_LABELS.get(w, w.title()) for w in words if w)


def slugify(value: str, max_len: int = 48) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_")
    return (value or "Unknown")[:max_len].rstrip("_")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _match_entry(name: str, candidates: list[str]) -> Optional[str]:
    """Find the profile entry a tailored bullet group refers to.

    The model is asked to copy the company name verbatim and usually does, but it
    shortens "BLNR (Open Source)" to "BLNR" often enough to matter. An exact lookup
    silently dropped those bullets, so the resume quietly lost a whole role's tailoring.
    Matching is therefore exact first, then containment, then close similarity.
    """
    target = _norm(name)
    if not target:
        return None
    flat = {c: _norm(c) for c in candidates if c}
    for candidate, value in flat.items():
        if value == target:
            return candidate
    contained = [c for c, value in flat.items()
                 if value and (value.startswith(target) or target.startswith(value))]
    if len(contained) == 1:
        return contained[0]
    close = difflib.get_close_matches(target, list(flat.values()), n=1, cutoff=0.8)
    if close:
        return next(c for c, value in flat.items() if value == close[0])
    return None


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
        self.last_trim = "nothing trimmed"   # what the most recent render gave up
        self.env = Environment(
            loader=FileSystemLoader(str(settings.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ---- view model -----------------------------------------------------
    def build_context(self, profile: dict, analysis: JobAnalysis) -> dict[str, Any]:
        """Merge the tailored content back into the profile with code-level guardrails."""
        # Attach each tailored group to the profile entry it names, tolerating a shortened
        # or lengthened name. The AI orders groups most-relevant-first, and that rank is
        # kept so an overflowing page drops what it judged least relevant.
        exp_names = [e.get("company", "") for e in profile.get("experience", [])]
        proj_names = [p.get("name", "") for p in profile.get("projects", [])]
        by_entry: dict[tuple[str, str], Any] = {}
        rank: dict[tuple[str, str], int] = {}
        unmatched: list[str] = []
        for i, g in enumerate(analysis.tailored_bullets):
            kind = g.kind
            hit = _match_entry(g.name, exp_names if kind == "experience" else proj_names)
            if hit is None:
                # The model sometimes files a project under experience, or the reverse.
                kind = "project" if kind == "experience" else "experience"
                hit = _match_entry(g.name, proj_names if kind == "project" else exp_names)
            if hit is None:
                unmatched.append(g.name)
                continue
            key = (kind, hit)
            by_entry.setdefault(key, g)
            rank.setdefault(key, i)

        experience = []
        for pos, exp in enumerate(profile.get("experience", [])):
            key = ("experience", exp.get("company", ""))
            g = by_entry.get(key)
            bullets = [b.strip() for b in (g.bullets if g else exp.get("bullets", [])) if b.strip()]
            experience.append({**exp, "bullets": bullets or exp.get("bullets", []),
                               "_pos": pos, "_rank": rank.get(key, 999)})

        projects = []
        for pos, proj in enumerate(profile.get("projects", [])):
            key = ("project", proj.get("name", ""))
            g = by_entry.get(key)
            bullets = [b.strip() for b in (g.bullets if g else proj.get("bullets", [])) if b.strip()]
            projects.append({**proj, "bullets": bullets or proj.get("bullets", []),
                             "_pos": pos, "_rank": rank.get(key, 999)})
        # Experience stays in reverse-chronological order; projects follow AI relevance.
        projects.sort(key=lambda p: (p["_rank"], p["_pos"]))

        if unmatched:
            log.warning("Dropping tailored bullet groups that name no profile entry: %s", unmatched)

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
                skill_groups.append({"label": skill_group_label(label), "items": rest})

        links = self._links(profile)

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

    #: One static instance per weight the resume uses. A variable font renders at its
    #: lightest instance in a PDF, which left the whole document in ExtraLight, so the
    #: weights are shipped separately and declared individually.
    FONT_WEIGHTS = (("BricolageGrotesque-Regular.ttf", 400),
                    ("BricolageGrotesque-SemiBold.ttf", 600),
                    ("BricolageGrotesque-Bold.ttf", 700))

    @staticmethod
    @lru_cache(maxsize=1)
    def font_face() -> str:
        """The project typeface as inline @font-face rules.

        The PDF is rendered from a string with no base URL, and a resume should not
        depend on a network fetch, so each weight is embedded outright. Returns an
        empty string when the files are missing, and the template falls back to
        Helvetica rather than failing.
        """
        rules = []
        for name, weight in ResumeBuilder.FONT_WEIGHTS:
            path = FONT_DIR / name
            if not path.exists():
                continue
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            rules.append("@font-face{font-family:'Bricolage Grotesque';"
                         f"src:url(data:font/ttf;base64,{data}) format('truetype');"
                         f"font-weight:{weight};font-style:normal;font-display:block;}}")
        if not rules:
            log.warning("No font files in %s; the resume will use the fallback face", FONT_DIR)
        return "".join(rules)

    @staticmethod
    def _links(profile: dict) -> list[dict[str, str]]:
        """Contact links, shown the same way on the resume and the cover letter."""
        links = []
        for key, label in (("linkedin", "LinkedIn"), ("github", "GitHub"), ("website", "Portfolio")):
            url = profile.get(key)
            if url:
                links.append({"label": label, "url": url,
                              "display": re.sub(r"^https?://(www\.)?", "", url).rstrip("/")})
        return links

    def render_html(self, profile: dict, analysis: JobAnalysis) -> str:
        template = self.env.get_template("resume.html")
        # Flatten typographic characters so an applicant tracking system reads the same
        # words a human does: "Full\u2011stack" must not hide from a search for "Full-stack".
        return template.render(font_face=self.font_face(),
                               **ats_text(self.build_context(profile, analysis)))

    # ---- PDF ------------------------------------------------------------
    def output_path(self, job: JobPosting, analysis: JobAnalysis) -> Path:
        company = job.company or analysis.company_name
        role = job.title or analysis.job_title
        return self.settings.output_dir / f"{slugify(company)}_{slugify(role)}.pdf"

    # ---- cover letter ---------------------------------------------------
    def cover_letter_path(self, job: JobPosting, analysis: Optional[JobAnalysis] = None) -> Path:
        company = job.company or (analysis.company_name if analysis else "")
        role = job.title or (analysis.job_title if analysis else "")
        return self.settings.output_dir / f"{slugify(company)}_{slugify(role)}_cover_letter.pdf"

    def render_cover_letter(self, profile: dict, letter: CoverLetter, job: JobPosting,
                            analysis: Optional[JobAnalysis] = None) -> str:
        template = self.env.get_template("cover_letter.html")
        context = {
            "name": profile.get("name", ""),
            "headline": profile.get("headline", ""),
            "email": profile.get("email", ""),
            "phone": profile.get("phone", ""),
            "location": profile.get("location", ""),
            "links": self._links(profile),
            "role": job.title or (analysis.job_title if analysis else ""),
            "company": job.company or (analysis.company_name if analysis else ""),
            "greeting": letter.greeting,
            "paragraphs": letter.paragraphs,
            "signature": letter.signature,
        }
        return template.render(font_face=self.font_face(), **ats_text(context))

    async def build_cover_letter(self, profile: dict, letter: CoverLetter, job: JobPosting,
                                 analysis: Optional[JobAnalysis] = None) -> Path:
        """Render a cover letter to its own single-page PDF."""
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.cover_letter_path(job, analysis)
        html = self.render_cover_letter(profile, letter, job, analysis)
        await render_first_that_fits([("cover", html)], path, max_pages=1)
        log.info("Cover letter written to %s", path)
        return path

    # ---- fitting the resume to the page budget --------------------------
    def _variants(self, context: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """Progressively shorter versions of the same truthful content, longest first.

        Nothing is invented or reworded; content is only reduced. The order matters:
        shortening bullets costs the least, dropping projects costs more, and dropping a
        job costs most of all, so work history is touched last and only when allowed.
        Each variant carries a plain-English note of what it gave up.
        """
        out: list[tuple[str, dict[str, Any]]] = [("nothing trimmed", context)]
        experience, projects = context["experience"], context["projects"]

        def with_bullets(ctx: dict[str, Any], cap: int) -> dict[str, Any]:
            return {**ctx,
                    "experience": [{**e, "bullets": e["bullets"][:cap]} for e in ctx["experience"]],
                    "projects": [{**p, "bullets": p["bullets"][:cap]} for p in ctx["projects"]]}

        # 1. fewer bullets per entry: every role and project still appears
        for cap in (4, 3):
            if any(len(e["bullets"]) > cap for e in experience + projects):
                out.append((f"at most {cap} bullets per entry", with_bullets(context, cap)))

        base = out[-1][1]
        # 2. fewer projects, least relevant first
        for keep in (5, 4, 3, 2, 1, 0):
            if keep < len(base["projects"]):
                label = "projects removed" if keep == 0 else f"only the top {keep} project(s)"
                out.append((label, {**base, "projects": base["projects"][:keep]}))

        base = out[-1][1]
        if any(len(e["bullets"]) > 2 for e in base["experience"]):
            out.append(("at most 2 bullets per role", with_bullets(base, 2)))

        # 3. last resort, and only if allowed: drop the oldest roles
        if self.settings.resume_may_drop_experience:
            base = out[-1][1]
            for keep in (4, 3, 2, 1):
                if keep < len(base["experience"]):
                    dropped = [e["company"] for e in base["experience"][keep:]]
                    out.append((f"dropped {', '.join(dropped)}",
                                {**base, "experience": base["experience"][:keep]}))
        return out

    async def build(self, profile: dict, analysis: JobAnalysis, job: JobPosting) -> Path:
        """Render the tailored resume, and record anything it had to leave out."""
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_path(job, analysis)
        template = self.env.get_template("resume.html")
        context = ats_text(self.build_context(profile, analysis))
        face = self.font_face()
        variants = [(name, template.render(font_face=face, **ctx))
                    for name, ctx in self._variants(context)]
        used = await render_first_that_fits(variants, path, self.settings.resume_max_pages)
        self.last_trim = used
        if used == "nothing trimmed":
            log.info("Resume written to %s (everything fits in %d page(s))",
                     path, self.settings.resume_max_pages)
        elif used == "overflow":
            log.warning("Resume %s still exceeds %d page(s) even after trimming",
                        path, self.settings.resume_max_pages)
        else:
            log.warning("Resume %s had to be shortened to fit %d page(s): %s. "
                        "Raise RESUME_MAX_PAGES to keep more.",
                        path.name, self.settings.resume_max_pages, used)
        return path


async def render_first_that_fits(variants: list[tuple[str, str]], path: Path,
                                 max_pages: int = 2) -> str:
    """Render each variant until one fits, and say which one was used.

    Order: the full content at 100% scale, then progressively trimmed content, then a
    scale reduction on the shortest variant. `page.pdf` only works in headless Chromium,
    so a dedicated headless browser is used even when the applier runs headed.
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
                if pages is None or pages <= max_pages:
                    return name
                log.debug("Variant %r is %d pages, trimming further", name, pages)
            for scale in SCALE_STEPS[1:]:  # shortest variant is still loaded
                await page.pdf(path=str(path), format="Letter", print_background=True,
                               prefer_css_page_size=True, scale=scale)
                pages = _count_pages(path)
                if pages is None or pages <= max_pages:
                    return f"{variants[-1][0]} @ {scale:.2f}"
                log.info("Resume is %d pages at scale %.2f, shrinking", pages, scale)
        finally:
            await browser.close()
    log.warning("Resume still exceeds %d page(s) at minimum scale: %s", max_pages, path)
    return "overflow"


async def html_to_pdf(html: str, path: Path, max_pages: int = 1) -> Path:
    """Render one HTML string to PDF (kept for ad-hoc use and tests)."""
    await render_first_that_fits([("single", html)], path, max_pages)
    return path
