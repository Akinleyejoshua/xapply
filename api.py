"""FastAPI admin API + web dashboard.

Everything the CLI can do is reachable here, so the terminal is optional:

  overview   stats, application list, per-application detail, resume PDF,
             review screenshot, CSV export, audit files
  discover   scan the configured sources and review the results before applying
  apply      start a full run, or apply to a hand-picked set of postings
  control    watch the live run log, release the human-in-the-loop pause, stop a run
  settings   switch LLM provider/model, sources, queries, threshold, auto-submit
  data       read and edit profile.json and companies.json

Auth: send `X-Admin-Token: <ADMIN_TOKEN>` (or `?token=`). When ADMIN_TOKEN is
left at its default the API only listens on localhost and auth is skipped.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from config import PERSISTED_KEYS, Settings
from config import settings as default_settings
from database import STATUSES, Database
from countries import COUNTRIES
from discovery import SENIORITY_LEVELS
from models import JobPosting

log = logging.getLogger(__name__)

DEFAULT_TOKEN = "change-me"


# ---- schemas --------------------------------------------------------------


class ApplicationSummary(BaseModel):
    id: int
    job_id: str
    source: str
    ats: str
    company: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    apply_url: Optional[str] = None
    match_score: Optional[int] = None
    status: str
    notes: Optional[str] = None
    resume_path: Optional[str] = None
    screenshot_path: Optional[str] = None
    created_at: str
    updated_at: str


class ApplicationDetail(ApplicationSummary):
    description: Optional[str] = None
    analysis: Optional[dict[str, Any]] = None
    answers: Optional[list[dict[str, Any]]] = None


class StatsResponse(BaseModel):
    total: int
    avg_match_score: Optional[float] = None
    last_activity: Optional[str] = None
    by_status: dict[str, int]
    submitted_by_ats: dict[str, int] = Field(default_factory=dict)


class StatusUpdate(BaseModel):
    status: str
    notes: Optional[str] = None


class ConfigPatch(BaseModel):
    """Runtime settings. Everything is optional; only supplied fields change."""

    llm_provider: Optional[Literal["gemini", "nvidia"]] = None
    gemini_model: Optional[str] = None
    nvidia_model: Optional[str] = None
    sources: Optional[list[str]] = None
    search_queries: Optional[list[str]] = None
    search_location: Optional[str] = None
    remote_only: Optional[bool] = None
    seniority_levels: Optional[list[Literal["intern", "junior", "mid", "senior", "lead"]]] = None
    countries: Optional[list[str]] = None
    match_threshold: Optional[int] = Field(None, ge=0, le=100)
    auto_submit: Optional[bool] = None
    headless: Optional[bool] = None
    max_applications_per_run: Optional[int] = Field(None, ge=1, le=200)
    max_jobs_per_company: Optional[int] = Field(None, ge=1, le=200)
    follow_companies: Optional[bool] = None


class DiscoverRequest(BaseModel):
    sources: Optional[list[str]] = None
    queries: Optional[list[str]] = None
    location: Optional[str] = None
    remote_only: Optional[bool] = None
    seniority_levels: Optional[list[str]] = None
    countries: Optional[list[str]] = None


class RunRequest(BaseModel):
    urls: Optional[list[str]] = None
    limit: Optional[int] = None
    auto_submit: Optional[bool] = None


class CompanyAdd(BaseModel):
    ats: Literal["greenhouse", "lever", "ashby"]
    token: str


class UrlCheck(BaseModel):
    """Job URLs pasted by hand, to be inspected before applying."""

    urls: list[str]


class DeleteRequest(BaseModel):
    """Exactly one of ids / status / all selects what to remove."""

    ids: Optional[list[int]] = None
    status: Optional[str] = None
    all: bool = False
    files: bool = False          # also delete the generated PDF and the screenshot


def _remove_artifacts(row: dict[str, Any]) -> None:
    """Delete the generated resume and review screenshot belonging to one application."""
    for key in ("resume_path", "screenshot_path"):
        raw = row.get(key)
        if not raw:
            continue
        try:
            Path(raw).unlink(missing_ok=True)
        except OSError as exc:
            log.debug("could not remove %s: %s", raw, exc)


# ---- app ------------------------------------------------------------------


def create_app(settings: Settings = default_settings, db: Optional[Database] = None,
               gate: Optional[Any] = None) -> FastAPI:
    settings.ensure_dirs()
    database = db or Database(settings.db_path)
    database.init()
    app = FastAPI(
        title="XApply Admin",
        version="2.0.0",
        description="Scan job boards, apply, and review every application the bot made.",
    )
    app.state.settings = settings
    app.state.db = database
    app.state.gate = gate
    app.state.task = None            # the one running discover/apply task
    app.state.task_kind = ""         # "discover" | "run" | ""
    app.state.log = []               # live activity log shown in the UI
    app.state.discovered = []        # JobPosting dicts from the last scan
    app.state.scan_stats = {}        # why the last scan kept or dropped what it did

    def note(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        app.state.log.append(f"{stamp}  {message}")
        del app.state.log[:-400]
        log.info(message)

    app.state.note = note

    def auth(request: Request, token: Optional[str] = Query(None)) -> None:
        expected = settings.admin_token
        if not expected or expected == DEFAULT_TOKEN:
            return  # local-only development mode
        supplied = request.headers.get("x-admin-token") or token
        if supplied != expected:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing admin token")

    def busy() -> bool:
        return bool(app.state.task and not app.state.task.done())

    def start(kind: str, coro_factory: Any) -> None:
        if busy():
            raise HTTPException(409, f"A {app.state.task_kind} is already running")
        app.state.task_kind = kind

        async def wrapper() -> None:
            try:
                await coro_factory()
            except asyncio.CancelledError:
                note(f"{kind} cancelled")
                raise
            except Exception as exc:
                log.exception("%s failed", kind)
                note(f"{kind} failed: {exc}")
            finally:
                app.state.task_kind = ""

        app.state.task = asyncio.create_task(wrapper())

    # ---- overview ---------------------------------------------------------
    @app.get("/api/stats", response_model=StatsResponse, dependencies=[Depends(auth)], tags=["overview"])
    def get_stats() -> dict[str, Any]:
        return database.stats()

    @app.get("/api/applications", response_model=list[ApplicationSummary],
             dependencies=[Depends(auth)], tags=["overview"])
    def list_applications(
        status_filter: Optional[str] = Query(None, alias="status", description="|".join(STATUSES)),
        search: Optional[str] = Query(None, description="Substring of company, title or URL"),
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> list[dict[str, Any]]:
        if status_filter and status_filter not in STATUSES:
            raise HTTPException(400, f"status must be one of {STATUSES}")
        return database.list(status=status_filter, search=search, limit=limit, offset=offset)

    @app.get("/api/applications/{app_id}", response_model=ApplicationDetail,
             dependencies=[Depends(auth)], tags=["overview"])
    def get_application(app_id: int) -> dict[str, Any]:
        row = database.get(app_id)
        if not row:
            raise HTTPException(404, "Application not found")
        return row

    @app.patch("/api/applications/{app_id}", dependencies=[Depends(auth)], tags=["overview"])
    def patch_application(app_id: int, body: StatusUpdate) -> dict[str, Any]:
        if body.status not in STATUSES:
            raise HTTPException(400, f"status must be one of {STATUSES}")
        if not database.update_status(app_id, body.status, body.notes):
            raise HTTPException(404, "Application not found")
        return database.get(app_id) or {}

    @app.delete("/api/applications/{app_id}", dependencies=[Depends(auth)], tags=["overview"])
    def delete_application(app_id: int, files: bool = Query(False, description="also delete the PDF and screenshot")) -> dict[str, Any]:
        """Remove one application. The posting becomes eligible for discovery again."""
        row = database.get(app_id)
        if not row:
            raise HTTPException(404, "Application not found")
        if files:
            _remove_artifacts(row)
        database.delete(app_id)
        note(f"deleted application #{app_id} ({row.get('company')} / {row.get('title')})")
        return {"deleted": 1, "id": app_id}

    @app.post("/api/applications/delete", dependencies=[Depends(auth)], tags=["overview"])
    def delete_applications(body: DeleteRequest) -> dict[str, Any]:
        """Bulk delete: a list of ids, everything with one status, or the whole table."""
        rows: list[dict[str, Any]] = []
        if body.ids:
            rows = [r for r in (database.get(i) for i in body.ids) if r]
        elif body.status:
            rows = database.list(status=body.status, limit=100_000)
        elif body.all:
            rows = database.list(limit=100_000)
        else:
            raise HTTPException(400, "Provide ids, a status, or all=true")
        if body.files:
            for r in rows:
                _remove_artifacts(r)
        if body.ids:
            n = database.delete_many(body.ids)
        elif body.status:
            n = database.delete_by_status(body.status)
        else:
            n = database.delete_all()
        note(f"deleted {n} application(s)")
        return {"deleted": n}

    @app.get("/api/applications/{app_id}/resume", dependencies=[Depends(auth)], tags=["files"])
    def get_resume(app_id: int) -> FileResponse:
        row = database.get(app_id)
        if not row or not row.get("resume_path"):
            raise HTTPException(404, "No resume stored for this application")
        path = Path(row["resume_path"])
        if not path.exists():
            raise HTTPException(404, f"Resume file missing: {path}")
        return FileResponse(path, media_type="application/pdf", filename=path.name)

    @app.get("/api/applications/{app_id}/screenshot", dependencies=[Depends(auth)], tags=["files"])
    def get_screenshot(app_id: int) -> FileResponse:
        row = database.get(app_id)
        if not row or not row.get("screenshot_path"):
            raise HTTPException(404, "No screenshot stored for this application")
        path = Path(row["screenshot_path"])
        if not path.exists():
            raise HTTPException(404, f"Screenshot file missing: {path}")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/export.csv", dependencies=[Depends(auth)], tags=["overview"])
    def export_csv(status_filter: Optional[str] = Query(None, alias="status")) -> StreamingResponse:
        import csv
        import io

        rows = database.list(status=status_filter, limit=100_000)
        cols = ["id", "status", "match_score", "company", "title", "location", "ats", "source",
                "url", "apply_url", "resume_path", "notes", "created_at", "updated_at"]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        buf.seek(0)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="xapply_{stamp}.csv"'})

    @app.get("/api/audits", dependencies=[Depends(auth)], tags=["overview"])
    def list_audits(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        files = sorted(settings.audit_dir.glob("*.json"), reverse=True)[:limit]
        out = []
        for f in files:
            try:
                d = json.loads(f.read_text())
                out.append({"file": f.name, "status": d.get("status"), "recorded_at": d.get("recorded_at"),
                            "company": d.get("job", {}).get("company"), "title": d.get("job", {}).get("title"),
                            "answers": len(d.get("answers_submitted") or [])})
            except Exception as exc:
                out.append({"file": f.name, "error": str(exc)})
        return out

    @app.get("/api/audits/{filename}", dependencies=[Depends(auth)], tags=["overview"])
    def get_audit(filename: str) -> dict[str, Any]:
        path = (settings.audit_dir / filename).resolve()
        if not str(path).startswith(str(settings.audit_dir.resolve())) or not path.exists():
            raise HTTPException(404, "Audit file not found")
        return json.loads(path.read_text())

    # ---- settings ---------------------------------------------------------
    @app.get("/api/config", dependencies=[Depends(auth)], tags=["settings"])
    def get_config() -> dict[str, Any]:
        return {
            "llm_provider": settings.llm_provider,
            "gemini_model": settings.gemini_model,
            "nvidia_model": settings.nvidia_model,
            "active_model": settings.active_model,
            "has_gemini_key": bool(settings.gemini_api_key),
            "has_nvidia_key": bool(settings.nvidia_api_key),
            "sources": settings.sources,
            "search_queries": settings.search_queries,
            "search_location": settings.search_location,
            "remote_only": settings.remote_only,
            "seniority_levels": settings.seniority_levels,
            "known_seniority": list(SENIORITY_LEVELS),
            "countries": settings.countries,
            "known_countries": COUNTRIES,
            "match_threshold": settings.match_threshold,
            "auto_submit": settings.auto_submit,
            "headless": settings.headless,
            "max_applications_per_run": settings.max_applications_per_run,
            "max_jobs_per_company": settings.max_jobs_per_company,
            "follow_companies": settings.follow_companies,
            "known_sources": ["greenhouse", "ashby", "lever", "remoteok", "himalayas",
                              "google", "linkedin", "urls"],
            # Which of the values above came from settings.local.json rather than .env,
            # so the UI can show that a choice is remembered.
            "saved": sorted(settings.saved_overrides()),
        }

    @app.patch("/api/config", dependencies=[Depends(auth)], tags=["settings"])
    def patch_config(body: ConfigPatch) -> dict[str, Any]:
        """Change settings and remember them, so every page and a later restart agree."""
        if body.countries is not None:
            unknown = [c for c in body.countries if c not in COUNTRIES]
            if unknown:
                raise HTTPException(422, f"Unknown country/countries: {', '.join(unknown)}")
        changed = []
        for key, value in body.model_dump(exclude_none=True).items():
            try:
                setattr(settings, key, value)
            except ValidationError as exc:
                raise HTTPException(422, f"{key}: {exc.errors()[0]['msg']}") from exc
            changed.append(key)
        if changed:
            settings.save_overrides(changed)
            note(f"settings saved: {', '.join(changed)}")
        return get_config()

    @app.post("/api/config/reset", dependencies=[Depends(auth)], tags=["settings"])
    def reset_config() -> dict[str, Any]:
        """Forget every saved UI choice and go back to what `.env` says."""
        settings.clear_overrides()
        fresh = Settings()
        for key in PERSISTED_KEYS:
            setattr(settings, key, getattr(fresh, key))
        note("settings reset to .env")
        return get_config()

    GEMINI_FALLBACK = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

    @app.get("/api/models", dependencies=[Depends(auth)], tags=["settings"])
    async def list_models(provider: Optional[str] = None) -> dict[str, Any]:
        """Live model list from whichever provider is asked for.

        NVIDIA's /models is public. Gemini's ListModels needs the key but works even
        when generateContent quota is exhausted, so it stays useful for diagnosis.
        """
        which = (provider or settings.llm_provider).lower()
        if which == "nvidia":
            url = settings.nvidia_base_url.rstrip("/") + "/models"
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.get(url)
                    r.raise_for_status()
                    ids = sorted(m["id"] for m in r.json().get("data", []))
            except Exception as exc:
                raise HTTPException(502, f"Could not reach {url}: {exc}")
            return {"provider": "nvidia", "models": ids}

        if not settings.gemini_api_key:
            return {"provider": "gemini", "models": GEMINI_FALLBACK, "note": "GEMINI_API_KEY not set"}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get("https://generativelanguage.googleapis.com/v1beta/models",
                                     params={"key": settings.gemini_api_key, "pageSize": 200})
                r.raise_for_status()
                ids = sorted(
                    m["name"].split("/")[-1] for m in r.json().get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                    and not any(x in m["name"] for x in ("-tts", "-image", "embed", "aqa"))
                )
        except Exception as exc:
            log.warning("Gemini ListModels failed: %s", exc)
            return {"provider": "gemini", "models": GEMINI_FALLBACK, "note": str(exc)[:160]}
        return {"provider": "gemini", "models": ids or GEMINI_FALLBACK}

    @app.get("/api/profile", dependencies=[Depends(auth)], tags=["data"])
    def get_profile() -> dict[str, Any]:
        if not settings.profile_path.exists():
            raise HTTPException(404, f"{settings.profile_path} not found")
        return json.loads(settings.profile_path.read_text(encoding="utf-8"))

    @app.put("/api/profile", dependencies=[Depends(auth)], tags=["data"])
    def put_profile(body: dict[str, Any]) -> dict[str, Any]:
        for key in ("name", "email"):
            if not body.get(key):
                raise HTTPException(400, f"profile.json needs a non-empty {key!r}")
        backup = settings.profile_path.with_suffix(".json.bak")
        if settings.profile_path.exists():
            backup.write_text(settings.profile_path.read_text(encoding="utf-8"), encoding="utf-8")
        settings.profile_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        note(f"profile.json saved (previous version kept as {backup.name})")
        return {"saved": True, "backup": str(backup)}

    @app.get("/api/companies", dependencies=[Depends(auth)], tags=["data"])
    def get_companies() -> dict[str, Any]:
        from discovery import load_company_tokens

        return load_company_tokens(settings.company_file)

    @app.post("/api/companies", dependencies=[Depends(auth)], tags=["data"])
    def add_company(body: CompanyAdd) -> dict[str, Any]:
        path = settings.company_file
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.setdefault(body.ats, [])
        token = body.token.strip()
        if token and token not in data[body.ats]:
            data[body.ats].append(token)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            note(f"added company {body.ats}:{token}")
        return {k: v for k, v in data.items() if not k.startswith("_")}

    # NOTE: the path param is not called `token`; that name is taken by the auth query param.
    @app.delete("/api/companies/{ats}/{board_token}", dependencies=[Depends(auth)], tags=["data"])
    def remove_company(ats: str, board_token: str) -> dict[str, Any]:
        path = settings.company_file
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if board_token in data.get(ats, []):
            data[ats].remove(board_token)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            note(f"removed company {ats}:{board_token}")
        return {k: v for k, v in data.items() if not k.startswith("_")}

    # ---- discover ---------------------------------------------------------
    @app.get("/api/discovered", dependencies=[Depends(auth)], tags=["discover"])
    def get_discovered() -> list[dict[str, Any]]:
        return app.state.discovered

    @app.get("/api/scan-stats", dependencies=[Depends(auth)], tags=["discover"])
    def get_scan_stats() -> dict[str, Any]:
        """How many postings the last scan examined, and what dropped them."""
        return app.state.scan_stats or {"seen": 0, "kept": 0, "reasons": {}, "tips": []}

    @app.post("/admin/discover", dependencies=[Depends(auth)], tags=["discover"])
    async def trigger_discover(body: DiscoverRequest) -> dict[str, Any]:
        if body.sources:
            settings.sources = body.sources
        if body.queries:
            settings.search_queries = body.queries
        if body.location is not None:
            settings.search_location = body.location
        if body.remote_only is not None:
            settings.remote_only = body.remote_only
        if body.seniority_levels is not None:
            settings.seniority_levels = body.seniority_levels
        if body.countries is not None:
            settings.countries = body.countries

        async def _go() -> None:
            from browser_bot import HumanGate
            from job_search import LazyBrowser, build_sources

            where = ", ".join(settings.countries) if settings.countries else settings.search_location
            scope = ("remote only, " if settings.remote_only else "") + (where or "anywhere")
            levels = ", ".join(settings.seniority_levels) if settings.seniority_levels else "any level"
            note(f"scanning {', '.join(settings.sources)} for "
                 f"{', '.join(settings.search_queries)} ({scope}, {levels})")
            gate = app.state.gate or HumanGate(settings.human_gate_mode, settings.log_dir / "CONTINUE")
            app.state.gate = gate
            # LazyBrowser only opens a real window if a source actually asks for a page,
            # so a pure board-API scan never pops a blank browser.
            from discovery import ScanStats, explain_empty_scan

            found: list[Any] = []
            totals = ScanStats()
            async with LazyBrowser(settings, gate) as lazy:
                for src in build_sources(settings, lazy, database):
                    page = await lazy.page_for(src.name)
                    got = await src.discover(page)
                    stats = getattr(src, "stats", None)
                    note(f"{src.name}: {len(got)} posting(s)"
                         + (f" ({stats.summary()})" if stats else ""))
                    if stats:
                        totals += stats
                    found.extend(got)
            app.state.discovered = [j.to_dict() for j in found]
            app.state.scan_stats = {
                "seen": totals.seen, "kept": totals.kept,
                "reasons": dict(totals.reasons()),
                "tips": explain_empty_scan(totals, settings),
            }
            note(f"scan finished: {len(found)} posting(s) ready to review "
                 f"({totals.seen} postings examined)")
            for tip in app.state.scan_stats["tips"]:
                note("why nothing matched: " + tip)

        start("discover", _go)
        return {"started": True, "sources": settings.sources}

    @app.post("/api/detect", dependencies=[Depends(auth)], tags=["discover"])
    def detect_urls(body: UrlCheck) -> list[dict[str, Any]]:
        """Tell the UI which ATS each pasted URL maps to, and whether it is already applied to."""
        from models import UNKNOWN, detect_ats, job_id_from_url

        out = []
        for raw in body.urls:
            url = raw.strip()
            if not url:
                continue
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            ats = detect_ats(url)
            job_id = job_id_from_url(url)
            seen = database.has_job("urls", job_id)
            out.append({
                "url": url, "ats": ats, "job_id": job_id, "already_seen": seen,
                "supported": ats != UNKNOWN,
                "note": ("Already in the database; it will be skipped" if seen else
                         "Ready" if ats != UNKNOWN else
                         "Not a Greenhouse, Lever, Ashby or LinkedIn URL. "
                         "The bot will open it but may not find a form it understands."),
            })
        return out

    # ---- apply ------------------------------------------------------------
    @app.post("/admin/run", dependencies=[Depends(auth)], tags=["apply"])
    async def trigger_run(body: RunRequest) -> dict[str, Any]:
        from pipeline import Pipeline  # lazy: needs an LLM key

        if body.auto_submit is not None:
            settings.auto_submit = body.auto_submit
        pipeline = Pipeline(settings, database, app.state.gate)
        app.state.gate = pipeline.gate
        urls, limit = body.urls, body.limit

        async def _go() -> None:
            mode = "AUTO-SUBMIT" if settings.auto_submit else "assisted"
            note(f"run started ({mode}, {settings.llm_provider}/{settings.active_model})")
            stats = await pipeline.run(urls=urls, limit=limit)
            note(f"run finished: {stats}")

        start("run", _go)
        return {"started": True, "auto_submit": settings.auto_submit}

    @app.post("/admin/apply-selected", dependencies=[Depends(auth)], tags=["apply"])
    async def apply_selected(body: RunRequest) -> dict[str, Any]:
        """Apply to an explicit list of URLs: scan results, or ones you pasted in yourself."""
        urls = []
        for raw in body.urls or []:
            url = raw.strip()
            if not url:
                continue
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            urls.append(url)
        if not urls:
            raise HTTPException(400, "Provide the URLs to apply to")
        return await trigger_run(RunRequest(urls=urls, limit=body.limit or len(urls),
                                            auto_submit=body.auto_submit))

    @app.post("/admin/stop", dependencies=[Depends(auth)], tags=["apply"])
    def stop_task() -> dict[str, Any]:
        if not busy():
            return {"stopped": False, "detail": "Nothing is running"}
        app.state.task.cancel()
        note("stop requested")
        return {"stopped": True}

    # ---- control ----------------------------------------------------------
    @app.get("/api/gate", dependencies=[Depends(auth)], tags=["control"])
    def gate_status() -> dict[str, Any]:
        g = app.state.gate
        return g.status() if g else {"paused": False, "reason": "", "paused_since": None}

    @app.post("/admin/continue", dependencies=[Depends(auth)], tags=["control"])
    def release_gate() -> dict[str, Any]:
        g = app.state.gate
        if not g:
            raise HTTPException(409, "No run is attached to this API process")
        if not g.paused:
            return {"released": False, "detail": "Bot is not waiting for a human"}
        g.release()
        note("human released the pause")
        return {"released": True}

    @app.get("/api/run", dependencies=[Depends(auth)], tags=["control"])
    def run_status() -> dict[str, Any]:
        g = app.state.gate
        return {
            "running": busy(),
            "kind": app.state.task_kind,
            "auto_submit": settings.auto_submit,
            "provider": settings.llm_provider,
            "model": settings.active_model,
            "gate": g.status() if g else {"paused": False, "reason": "", "paused_since": None},
            "log": app.state.log[-120:],
            "discovered": len(app.state.discovered),
        }

    @app.get("/health", tags=["control"])
    def health() -> dict[str, Any]:
        return {"ok": True, "db": str(settings.db_path), "auto_submit": settings.auto_submit,
                "provider": settings.llm_provider, "model": settings.active_model}

    # ---- dashboard --------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(500, f"Dashboard file missing: {index}")
        return index.read_text(encoding="utf-8")

    return app


STATIC_DIR = BASE_STATIC = Path(__file__).resolve().parent / "static"

app = None  # lazily created by main.py / uvicorn factory


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
