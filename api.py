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
from email_apply import EmailApplier
from integrations import BY_NAME as INTEGRATIONS_BY_NAME
from integrations import forget as integration_forget
from integrations import signed_in as integration_signed_in
from integrations import status as integration_status
from countries import COUNTRIES
from llm import verified_table
from discovery import SENIORITY_LEVELS
from models import JobPosting

log = logging.getLogger(__name__)

#: How many lines of activity to keep. A single application produces dozens.
LOG_LINES = 900
#: Modules whose logs belong in the activity feed. Playwright and httpx chatter does not.
FEED_LOGGERS = frozenset({"pipeline", "appliers", "browser_bot", "discovery", "ai_agent",
                          "resume_builder", "email_apply", "job_search", "llm"})
#: Log level to the colour the dashboard gives it.
LEVEL_KIND = {logging.DEBUG: "dim", logging.INFO: "info",
              logging.WARNING: "warn", logging.ERROR: "error",
              logging.CRITICAL: "error"}

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


#: Changing any of these means a different model will be used, so it is worth checking.
MODEL_KEYS = {"nvidia_model", "gemini_model", "opencode_model", "llm_provider"}
#: Of those, the ones that represent a deliberate choice of model.
MODEL_FIELDS = {"nvidia_model", "gemini_model", "opencode_model"}


class ConfigPatch(BaseModel):
    """Runtime settings. Everything is optional; only supplied fields change."""

    llm_provider: Optional[Literal["gemini", "nvidia", "opencode"]] = None
    gemini_model: Optional[str] = None
    nvidia_model: Optional[str] = None
    opencode_model: Optional[str] = None
    #: Extra HTTP headers for the LLM request, e.g. a User-Agent a gateway expects.
    llm_extra_headers: Optional[dict[str, str]] = None
    #: Save a model even though it did not answer, for one you know is coming back.
    force: Optional[bool] = None
    sources: Optional[list[str]] = None
    search_queries: Optional[list[str]] = None
    search_location: Optional[str] = None
    remote_only: Optional[bool] = None
    seniority_levels: Optional[list[Literal["intern", "junior", "mid", "senior", "lead"]]] = None
    countries: Optional[list[str]] = None
    title_match_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    match_threshold: Optional[int] = Field(None, ge=0, le=100)
    fill_mode: Optional[Literal["documents", "assisted", "auto"]] = None
    auto_submit: Optional[bool] = None      # legacy alias for fill_mode="auto"
    fill_all_at_once: Optional[bool] = None
    verify_after_fill: Optional[bool] = None
    email_apply: Optional[bool] = None
    email_transport: Optional[Literal["smtp", "gmail"]] = None
    email_auto_send: Optional[bool] = None
    search_x: Optional[bool] = None
    hide_browser: Optional[bool] = None
    open_window_when_needed: Optional[bool] = None
    challenge_action: Optional[Literal["wait", "show", "skip"]] = None
    headless: Optional[bool] = None
    max_applications_per_run: Optional[int] = Field(None, ge=1, le=200)
    max_jobs_per_company: Optional[int] = Field(None, ge=1, le=200)
    follow_companies: Optional[bool] = None
    resume_max_pages: Optional[int] = Field(None, ge=1, le=3)
    resume_may_drop_experience: Optional[bool] = None


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


class ModelTest(BaseModel):
    """A model id to verify against the provider."""

    model: str
    provider: Optional[Literal["gemini", "nvidia", "opencode"]] = None


class BoardLookup(BaseModel):
    """Anything that might name a company board: a job link, a board link, or a token."""

    text: str


class DiscoveredDelete(BaseModel):
    """Exactly one of job_ids / all selects what to remove from the scan results."""

    job_ids: Optional[list[str]] = None
    all: bool = False


class EmailApplyRequest(BaseModel):
    """One posting, applied to by writing to whoever it says to write to."""

    url: str
    #: Only when the posting hides the address, or names several and you know which.
    to: Optional[str] = None


class RetryRequest(BaseModel):
    """Which applications to try again: specific ids, or everything with one status."""

    ids: Optional[list[int]] = None
    status: Optional[Literal["failed", "pending_human_review"]] = None
    limit: Optional[int] = None


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
    app.state.blocker = None         # a configuration problem that stopped the last run

    def note(message: str, kind: str = "info") -> None:
        """Add a line to the activity feed. `kind` decides how it is coloured."""
        app.state.log.append({
            "at": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "kind": kind,
            "text": str(message),
        })
        del app.state.log[:-LOG_LINES]
        log.info(message)

    app.state.note = note

    # Everything the run does is already logged by the module doing it, so the feed is
    # fed from the logs rather than from calls sprinkled through the pipeline. That is
    # what makes it show the whole flow, including each field as it is filled, instead
    # of only the handful of moments the API happens to know about.
    class FeedHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.name.split(".")[0] not in FEED_LOGGERS:
                return
            try:
                text = record.getMessage()
            except Exception:
                return
            app.state.log.append({
                "at": datetime.fromtimestamp(record.created, timezone.utc).strftime("%H:%M:%S"),
                "kind": LEVEL_KIND.get(record.levelno, "info"),
                "text": text[:500],
                "from": record.name.split(".")[0],
            })
            del app.state.log[:-LOG_LINES]

    handler = FeedHandler(level=logging.INFO)
    for name in FEED_LOGGERS:
        module_log = logging.getLogger(name)
        module_log.addHandler(handler)
        # A logger filters by level before any handler is consulted, so without this
        # every INFO line is dropped and the feed shows only warnings and failures,
        # which is the opposite of showing the whole flow. An explicit DEBUG set
        # elsewhere is left alone.
        if module_log.getEffectiveLevel() > logging.INFO:
            module_log.setLevel(logging.INFO)
    app.state.feed_handler = handler

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

    @app.post("/api/applications/retry", dependencies=[Depends(auth)], tags=["apply"])
    async def retry_applications(body: RetryRequest) -> dict[str, Any]:
        """Try again on applications that failed, or that are still waiting on you.

        The old rows are removed first. Discovery skips anything already in the table,
        so leaving them would mean the retry quietly did nothing, and their outcome is
        about to be replaced anyway. The audit files under logs/applications keep the
        history of what happened the first time.
        """
        if body.ids:
            rows = [r for r in (database.get(i) for i in body.ids) if r]
        elif body.status:
            rows = database.list(status=body.status, limit=body.limit or 100)
        else:
            raise HTTPException(400, "Pass ids, or a status of failed or pending_human_review")
        if not rows:
            raise HTTPException(404, "Nothing matched, so there is nothing to retry")

        urls, skipped = [], []
        for r in rows:
            link = (r.get("apply_url") or r.get("url") or "").strip()
            if link:
                urls.append(link)
            else:
                skipped.append(r.get("id"))
        if not urls:
            raise HTTPException(400, "None of those applications recorded a link to retry")

        database.delete_many([r["id"] for r in rows if r.get("id")])
        database.delete_by_urls(urls)
        note(f"retrying {len(urls)} application(s)"
             + (f"; {len(skipped)} had no link and were left alone" if skipped else ""))
        result = await trigger_run(RunRequest(urls=urls, limit=body.limit or len(urls)))
        return {"retrying": len(urls), "skipped": len(skipped), **result}

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
            "opencode_model": settings.opencode_model,
            "has_opencode_key": bool(settings.opencode_api_key),
            "llm_extra_headers": settings.llm_extra_headers,
            "known_providers": ["gemini", "nvidia", "opencode"],
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
            "title_match_threshold": settings.title_match_threshold,
            "match_threshold": settings.match_threshold,
            "fill_mode": settings.fill_mode,
            "fill_all_at_once": settings.fill_all_at_once,
            "verify_after_fill": settings.verify_after_fill,
            "email_apply": settings.email_apply,
            "email_transport": settings.email_transport,
            "email_auto_send": settings.email_auto_send,
            "search_x": settings.search_x,
            "hide_browser": settings.hide_browser,
            "open_window_when_needed": settings.open_window_when_needed,
            "challenge_action": settings.challenge_action,
            # Whether a mail server is actually set up. The credentials themselves stay
            # in .env and are never sent to the browser.
            "email_ready": not EmailApplier(settings).missing_settings(),
            "email_missing": EmailApplier(settings).missing_settings(),
            "auto_submit": settings.auto_submit,
            "headless": settings.headless,
            "max_applications_per_run": settings.max_applications_per_run,
            "max_jobs_per_company": settings.max_jobs_per_company,
            "follow_companies": settings.follow_companies,
            "resume_max_pages": settings.resume_max_pages,
            "resume_may_drop_experience": settings.resume_may_drop_experience,
            "known_sources": ["greenhouse", "ashby", "lever", "remoteok", "himalayas", "emails",
                              "google", "linkedin", "urls"],
            # Which of the values above came from settings.local.json rather than .env,
            # so the UI can show that a choice is remembered.
            "saved": sorted(settings.saved_overrides()),
        }

    @app.patch("/api/config", dependencies=[Depends(auth)], tags=["settings"])
    async def patch_config(body: ConfigPatch) -> dict[str, Any]:
        """Change settings and remember them, so every page and a later restart agree."""
        if body.countries is not None:
            unknown = [c for c in body.countries if c not in COUNTRIES]
            if unknown:
                raise HTTPException(422, f"Unknown country/countries: {', '.join(unknown)}")
        fields = body.model_dump(exclude_none=True)
        force = bool(fields.pop("force", False))
        # auto_submit is the old name for fill_mode="auto"; translate rather than store both.
        if "auto_submit" in fields and "fill_mode" not in fields:
            fields["fill_mode"] = "auto" if fields.pop("auto_submit") else (
                settings.fill_mode if settings.fill_mode != "auto" else "assisted")
        fields.pop("auto_submit", None)
        before = {k: getattr(settings, k) for k in fields}
        changed = []
        for key, value in fields.items():
            try:
                setattr(settings, key, value)
            except ValidationError as exc:
                raise HTTPException(422, f"{key}: {exc.errors()[0]['msg']}") from exc
            changed.append(key)

        check = None
        if MODEL_KEYS & set(changed):
            check = await _check_active_model()
            # Choosing a model that cannot answer is refused, because it breaks every run
            # and can arrive from a stale browser tab as easily as from a typo. Merely
            # switching provider is not: you have to get there before you can pick a model.
            picked_a_model = bool(MODEL_FIELDS & set(changed))
            if picked_a_model and not check["ok"] and not force:
                for key, value in before.items():
                    setattr(settings, key, value)
                raise HTTPException(422, {
                    "message": f"{check['model']} does not answer, so it was not saved.",
                    "detail": check["detail"], "model": check["model"],
                    "hint": "Press 'Check which models work' to see the ones that do.",
                })
        if changed:
            settings.save_overrides(changed)
            note(f"settings saved: {', '.join(changed)}")
        out = get_config()
        if check:
            out["model_check"] = check
        return out

    async def _check_active_model() -> dict[str, Any]:
        """Tell the caller straight away whether the model it just chose actually answers.

        A provider can list a model it has never deployed, so a choice that looks valid
        can silently break the next run. Checking here means the mistake is visible at
        the moment it is made, including when an old browser tab sends a stale value.
        """
        from llm import check_model, note_model_result

        model = settings.active_model
        known = verified_table(settings.llm_provider).get(model)
        if known is True:
            return {"model": model, "ok": True, "detail": "Verified earlier"}
        result = await check_model(settings, settings.llm_provider, model)
        ok = bool(result.get("ok") or result.get("status") == 503)
        note_model_result(settings.llm_provider, model, ok)
        if not ok:
            note(f"WARNING: {model} does not answer. {result.get('detail', '')[:120]}")
        return {"model": model, "ok": ok, "detail": result.get("detail", "")}

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
        if which == "opencode":
            from llm import is_chat_model, llm_headers, verified_table

            if not settings.opencode_api_key:
                return {"provider": "opencode", "models": [], "chat_models": [],
                        "note": "OPENCODE_API_KEY is not set in .env"}
            url = settings.opencode_base_url.rstrip("/") + "/models"
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.get(url, headers=llm_headers(settings, settings.opencode_api_key))
                    r.raise_for_status()
                    ids = sorted(m["id"] for m in r.json().get("data", []))
            except Exception as exc:
                raise HTTPException(502, f"Could not reach {url}: {exc}")
            known = verified_table("opencode")
            return {"provider": "opencode", "models": ids,
                    "chat_models": [i for i in ids if is_chat_model(i)],
                    "verified": sorted(m for m, ok in known.items() if ok),
                    "broken": sorted(m for m, ok in known.items() if not ok),
                    "note": "Models ending in -free only work inside OpenCode's own client. "
                            "The rest need a payment method on your workspace."}
        if which == "nvidia":
            url = settings.nvidia_base_url.rstrip("/") + "/models"
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.get(url)
                    r.raise_for_status()
                    ids = sorted(m["id"] for m in r.json().get("data", []))
            except Exception as exc:
                raise HTTPException(502, f"Could not reach {url}: {exc}")
            from llm import is_chat_model

            from llm import verified_table

            known = verified_table("nvidia")
            return {"provider": "nvidia", "models": ids,
                    "chat_models": [i for i in ids if is_chat_model(i)],
                    "verified": sorted(m for m, ok in known.items() if ok),
                    "broken": sorted(m for m, ok in known.items() if not ok),
                    "note": "NVIDIA lists models it has not deployed, so a listed model can "
                            "still fail. Press 'Check which models work' to find out, or type "
                            "an id that is not listed at all."}

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
        from llm import verified_table

        known = verified_table("gemini")
        return {"provider": "gemini", "models": ids or GEMINI_FALLBACK,
                "chat_models": ids or GEMINI_FALLBACK,
                "verified": sorted(m for m, ok in known.items() if ok),
                "broken": sorted(m for m, ok in known.items() if not ok)}

    @app.post("/api/models/verify", dependencies=[Depends(auth)], tags=["settings"])
    async def verify_all_models(provider: Optional[str] = None) -> dict[str, Any]:
        """Probe every listed chat model and report which ones actually answer.

        NVIDIA's catalogue lists models it has not deployed: of 61 chat-capable entries,
        only a handful respond. Guessing from the list wastes a whole run, so this
        settles it once and the picker remembers.
        """
        from llm import is_chat_model, verify_models

        which = (provider or settings.llm_provider).lower()
        listing = await list_models(which)
        candidates = listing.get("chat_models") or [m for m in listing["models"] if is_chat_model(m)]
        note(f"checking {len(candidates)} {which} models…")
        table = await verify_models(settings, which, candidates)
        working = sorted(m for m, ok in table.items() if ok)
        note(f"{len(working)} of {len(candidates)} {which} models answered")
        return {"provider": which, "checked": len(candidates), "working": working,
                "failed": sorted(m for m, ok in table.items() if not ok)}

    @app.post("/api/models/test", dependencies=[Depends(auth)], tags=["settings"])
    async def test_model(body: ModelTest) -> dict[str, Any]:
        """Send one tiny prompt so a model choice can be confirmed before a run."""
        from llm import check_model

        from llm import note_model_result

        provider = body.provider or settings.llm_provider
        result = await check_model(settings, provider, body.model)
        note_model_result(provider, body.model,
                          bool(result.get("ok") or result.get("status") == 503))
        note(f"model check {body.model}: {'ok' if result.get('ok') else result.get('detail', 'failed')[:80]}")
        return result

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

    def _write_companies(data: dict[str, Any]) -> dict[str, Any]:
        settings.company_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def _read_companies() -> dict[str, Any]:
        path = settings.company_file
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @app.post("/api/companies", dependencies=[Depends(auth)], tags=["data"])
    def add_company(body: CompanyAdd) -> dict[str, Any]:
        data = _read_companies()
        data.setdefault(body.ats, [])
        token = body.token.strip()
        if token and token not in data[body.ats]:
            data[body.ats].append(token)
            note(f"added company {body.ats}:{token}")
            return _write_companies(data)
        return {k: v for k, v in data.items() if not k.startswith("_")}

    @app.post("/api/companies/resolve", dependencies=[Depends(auth)], tags=["data"])
    async def resolve_company(body: BoardLookup) -> dict[str, Any]:
        """Identify the board behind a pasted job or board URL, and verify it is live."""
        from discovery import resolve_board

        found = await resolve_board(body.text)
        if found.get("ok"):
            existing = _read_companies().get(found["ats"], [])
            found["already_added"] = found["token"] in existing
        return found

    @app.post("/api/companies/add-from-url", dependencies=[Depends(auth)], tags=["data"])
    async def add_company_from_url(body: BoardLookup) -> dict[str, Any]:
        """Add a board from anything the user pastes: a job link, a board link or a token."""
        from discovery import resolve_board

        found = await resolve_board(body.text)
        if not found.get("ok"):
            raise HTTPException(422, found.get("detail", "Could not identify that board"))
        data = _read_companies()
        data.setdefault(found["ats"], [])
        if found["token"] not in data[found["ats"]]:
            data[found["ats"]].append(found["token"])
            _write_companies(data)
            note(f"added {found['ats']}:{found['token']} ({found['open_roles']} open roles)")
        return {"added": found, "companies": {k: v for k, v in data.items() if not k.startswith("_")}}

    @app.get("/api/companies/probe", dependencies=[Depends(auth)], tags=["data"])
    async def probe_companies() -> list[dict[str, Any]]:
        """Open-role count for every configured board, so dead tokens are visible."""
        import asyncio as _asyncio

        import httpx as _httpx

        from discovery import HEADERS, probe_board

        tokens = [(ats, t) for ats, group in _read_companies().items() for t in group]
        async with _httpx.AsyncClient(headers=HEADERS, timeout=25, follow_redirects=True) as client:
            results = await _asyncio.gather(
                *(probe_board(client, ats, t) for ats, t in tokens), return_exceptions=True)
        out = []
        for (ats, token), res in zip(tokens, results):
            if isinstance(res, dict):
                out.append({**res, "ok": True})
            else:
                out.append({"ats": ats, "token": token, "ok": False, "open_roles": 0,
                            "company": token})
        return out

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

    # ---- applying by email ------------------------------------------------
    @app.post("/api/email/check", dependencies=[Depends(auth)], tags=["apply"])
    async def check_email() -> dict[str, Any]:
        """Open Gmail and say whether applications can be sent from it. Sends nothing."""
        from llm import LLMError
        from pipeline import Pipeline

        try:
            pipeline = Pipeline(settings, database, app.state.gate)
        except LLMError as exc:
            raise HTTPException(400, str(exc)) from exc
        app.state.gate = pipeline.gate
        out = await pipeline.check_email_account()
        note(f"email check: {out['detail']}", "info" if out["ready"] else "warn")
        return out

    @app.post("/api/email/apply", dependencies=[Depends(auth)], tags=["apply"])
    async def apply_by_email(body: EmailApplyRequest) -> dict[str, Any]:
        """Apply to one posting by writing to the address it gives.

        The same route a scan takes when it meets a posting with no form, reachable
        directly so it can be used on a link you found yourself.
        """
        from llm import LLMError
        from pipeline import Pipeline

        if not settings.email_apply:
            raise HTTPException(400, "Turn on 'Apply by email' in Settings first.")
        url = (body.url or "").strip()
        if not url:
            raise HTTPException(400, "Give the link to the posting.")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            pipeline = Pipeline(settings, database, app.state.gate)
        except LLMError as exc:
            raise HTTPException(400, str(exc)) from exc
        app.state.gate = pipeline.gate

        async def _go() -> None:
            note(f"applying by email to {url}", "info")
            try:
                out = await pipeline.email_one(url, body.to)
            except ValueError as exc:
                note(str(exc), "warn")
                return
            note(f"{out['status']}: {out['to']}",
                 "info" if out["status"] == "submitted" else "warn")

        start("email", _go)
        return {"started": True, "url": url}

    # ---- accounts ---------------------------------------------------------
    @app.get("/api/integrations", dependencies=[Depends(auth)], tags=["accounts"])
    def get_integrations() -> list[dict[str, Any]]:
        """Which accounts this browser profile is still signed in to.

        Read from the profile's own cookie store rather than by opening each site, so
        the whole list is answered at once. Without it, whether a run can read X or send
        from Gmail is only discoverable by starting one and watching it fail.
        """
        return integration_status(settings.user_data_dir, settings)

    @app.post("/api/integrations/{name}/login", dependencies=[Depends(auth)], tags=["accounts"])
    async def sign_in(name: str) -> dict[str, Any]:
        """Open a window at the sign-in page. You do the signing in."""
        from pipeline import login_flow

        item = INTEGRATIONS_BY_NAME.get(name)
        if item is None:
            raise HTTPException(404, f"No account called {name!r}")

        async def _go() -> None:
            note(f"opening {item.label} to sign in", "info")
            await login_flow(settings, name)
            note(f"{item.label}: signed in" if integration_signed_in(
                settings.user_data_dir, item) else
                f"{item.label}: still signed out", "info")

        start(f"signin:{name}", _go)
        return {"opening": item.sign_in_url, "label": item.label}

    @app.post("/api/integrations/{name}/logout", dependencies=[Depends(auth)], tags=["accounts"])
    def sign_out(name: str) -> dict[str, Any]:
        """Forget this account by removing its cookies from the profile."""
        if busy():
            raise HTTPException(409, "Something is running. Stop it first, so the "
                                     "browser is not holding the cookie store open.")
        try:
            removed = integration_forget(settings.user_data_dir, name)
        except KeyError:
            raise HTTPException(404, f"No account called {name!r}") from None
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        note(f"signed out of {name}", "info")
        return {"forgotten": removed, "name": name}

    # ---- discover ---------------------------------------------------------
    @app.get("/api/discovered", dependencies=[Depends(auth)], tags=["discover"])
    def get_discovered(limit: int = Query(500, ge=1, le=5000),
                       search: Optional[str] = None) -> list[dict[str, Any]]:
        """Everything scans have turned up, newest and most relevant first.

        Read from the database rather than from memory, so a ten-minute scan is not
        lost to a restart and the terminal sees the same list as the dashboard.
        """
        return database.list_discovered(limit=limit, search=search)

    @app.delete("/api/discovered/{job_id}", dependencies=[Depends(auth)], tags=["discover"])
    def delete_discovered_one(job_id: str) -> dict[str, Any]:
        n = database.delete_discovered([job_id])
        if not n:
            raise HTTPException(404, f"No scan result with job_id {job_id!r}")
        app.state.discovered = [j for j in app.state.discovered if j.get("job_id") != job_id]
        note(f"removed {job_id} from the scan results")
        return {"deleted": n, "job_id": job_id}

    @app.post("/api/discovered/delete", dependencies=[Depends(auth)], tags=["discover"])
    def delete_discovered_many(body: DiscoveredDelete) -> dict[str, Any]:
        """Bulk removal: a list of job ids, or the whole list."""
        if body.all:
            n = database.delete_discovered_all()
            app.state.discovered = []
        elif body.job_ids:
            n = database.delete_discovered(body.job_ids)
            gone = set(body.job_ids)
            app.state.discovered = [j for j in app.state.discovered
                                    if j.get("job_id") not in gone]
        else:
            raise HTTPException(400, "Pass job_ids, or all=true to clear the list")
        note(f"removed {n} scan result(s)")
        return {"deleted": n}

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
            current: list[Any] = []       # the source being scanned, for its live counters

            def live_totals() -> ScanStats:
                """Finished sources plus however far the current one has got."""
                snapshot = ScanStats()
                snapshot += totals
                for src in current:
                    stats = getattr(src, "stats", None)
                    if stats:
                        snapshot += stats
                snapshot.kept = len(found)
                return snapshot

            def publish() -> None:
                """Make what has been found so far visible straight away.

                A scan over thirty company boards takes minutes, and stopping it used to
                throw away everything it had gathered. Results are published as each
                board finishes, so they survive a stop and appear while the scan runs.
                """
                snapshot = live_totals()
                app.state.discovered = [j.to_dict() for j in found]
                app.state.scan_stats = {
                    "seen": snapshot.seen, "kept": len(found),
                    "reasons": dict(snapshot.reasons()),
                    "tips": [], "partial": True,
                }

            def on_batch(source_name: str, batch: list[Any]) -> None:
                found.extend(batch)
                # Saved per batch, so a scan you stop halfway still leaves its results
                # behind, and a restart does not throw away ten minutes of scanning.
                try:
                    database.save_discovered(batch)
                except Exception as exc:                 # never let storage stop a scan
                    log.warning("could not save scan results: %s", exc)
                publish()

            try:
                async with LazyBrowser(settings, gate) as lazy:
                    for src in build_sources(settings, lazy, database):
                        src.on_batch = on_batch
                        current[:] = [src]
                        page = await lazy.page_for(src.name)
                        before = len(found)
                        got = await src.discover(page)
                        current.clear()
                        # Sources that do not report incrementally still contribute here.
                        for job in got:
                            if job not in found:
                                found.append(job)
                        stats = getattr(src, "stats", None)
                        if stats:
                            totals += stats
                            totals.kept = len(found)
                        note(f"{src.name}: {len(found) - before} posting(s)"
                             + (f" ({stats.summary()})" if stats else ""))
                        # The Google source has two outcomes worth saying out loud: the
                        # boards it added, which make the next scan faster, and the
                        # searches Google refused, which otherwise look like "found none".
                        remembered = getattr(src, "remembered", None)
                        if remembered:
                            for ats, fresh in remembered.items():
                                note(f"added {len(fresh)} new {ats} board(s) to your company "
                                     f"list: {', '.join(fresh[:8])}"
                                     + (" and more" if len(fresh) > 8 else ""))
                        slow = getattr(src, "slow_boards", None)
                        if slow:
                            note(f"skipped {len(slow)} board(s) that were too slow to "
                                 f"download. Raise BOARD_FETCH_TIMEOUT_S in .env to wait "
                                 f"longer for them.")
                        refused = getattr(src, "blocked_searches", 0)
                        if refused:
                            note(f"Google refused {refused} search(es) with its bot check. "
                                 f"The board APIs need no search engine.")
                        publish()
            except asyncio.CancelledError:
                database.save_discovered(found)
                publish()
                snapshot = live_totals()
                note(f"scan stopped early; keeping the {len(found)} posting(s) found so far "
                     f"({snapshot.seen} examined before the stop)")
                raise

            database.save_discovered(found)
            app.state.discovered = [j.to_dict() for j in found]
            app.state.scan_stats = {
                "seen": totals.seen, "kept": len(found),
                "reasons": dict(totals.reasons()),
                "tips": explain_empty_scan(totals, settings),
                "partial": False,
            }
            note(f"scan finished: {len(found)} posting(s) ready to review "
                 f"({totals.seen} postings examined)")
            for tip in app.state.scan_stats["tips"]:
                note(tip)

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
        from llm import LLMError
        from pipeline import Pipeline  # lazy: needs an LLM key

        if body.auto_submit is not None:
            settings.fill_mode = "auto" if body.auto_submit else (
                settings.fill_mode if settings.fill_mode != "auto" else "assisted")
        try:
            pipeline = Pipeline(settings, database, app.state.gate)
        except LLMError as exc:
            # A missing or rejected key is a setup problem with a clear remedy, not a
            # crash. Returning 500 here showed the browser nothing it could act on.
            note(str(exc))
            raise HTTPException(400, str(exc)) from exc
        app.state.gate = pipeline.gate
        urls, limit = body.urls, body.limit

        async def _go() -> None:
            from llm import ModelUnavailable

            mode = settings.fill_mode
            note(f"run started ({mode}, {settings.llm_provider}/{settings.active_model})")
            try:
                stats = await pipeline.run(urls=urls, limit=limit)
            except ModelUnavailable as exc:
                app.state.blocker = {"kind": "model", "detail": str(exc),
                                     "model": settings.active_model,
                                     "provider": settings.llm_provider}
                for line in str(exc).splitlines():
                    note(line.strip())
                return
            app.state.blocker = None
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

        # Applying twice to the same posting is not something the bot will do, so a link
        # already in the database is dropped during discovery. Checking here means you
        # are told why, instead of watching a browser open and close for no visible reason.
        from models import job_id_from_url

        fresh: list[str] = []
        already: list[dict[str, Any]] = []
        for url in urls:
            row = database.find_job("urls", job_id_from_url(url))
            if row:
                already.append(row)
            else:
                fresh.append(url)
        if not fresh:
            # Naming them matters: "already applied" without saying which one sends you
            # hunting through a filtered list for a row you cannot identify.
            which = "; ".join(
                f"#{r['id']} {r.get('company') or '?'} / {r.get('title') or '?'} ({r['status']})"
                for r in already[:5])
            raise HTTPException(
                400,
                f"Already applied: {which}"
                + (f" and {len(already) - 5} more" if len(already) > 5 else "")
                + ". Open the Applications tab, clear the status filter to All, and use "
                  "Retry on that row to run it again.")
        if already:
            note(f"{len(already)} of {len(urls)} selected posting(s) are already applied to "
                 f"and were left out; applying to the other {len(fresh)}")
        urls = fresh
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

    @app.post("/admin/skip", dependencies=[Depends(auth)], tags=["control"])
    def skip_job() -> dict[str, Any]:
        """Abandon the posting the agent is paused on and move to the next one."""
        g = app.state.gate
        if not g:
            raise HTTPException(409, "No run is attached to this API process")
        if not g.paused:
            return {"skipped": False, "detail": "Bot is not waiting for a human"}
        g.skip()
        note("human skipped this job")
        return {"skipped": True}

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

    @app.post("/admin/clear-blocker", dependencies=[Depends(auth)], tags=["control"])
    def clear_blocker() -> dict[str, Any]:
        """Dismiss the configuration problem reported by the last run."""
        app.state.blocker = None
        return {"cleared": True}

    @app.get("/api/run", dependencies=[Depends(auth)], tags=["control"])
    def run_status() -> dict[str, Any]:
        g = app.state.gate
        return {
            "running": busy(),
            "kind": app.state.task_kind,
            "auto_submit": settings.auto_submit,
            "fill_mode": settings.fill_mode,
            "provider": settings.llm_provider,
            "model": settings.active_model,
            "gate": g.status() if g else {"paused": False, "reason": "", "paused_since": None},
            # Two shapes on purpose. `activity` is the structured feed the dashboard
            # colours. `log` stays the plain list of strings it has always been, so a
            # tab still running the previous script shows text rather than a column of
            # [object Object].
            "activity": app.state.log[-240:],
            "log": [f"{row['at']}  {row['text']}" for row in app.state.log[-240:]],
            "discovered": len(app.state.discovered),
            "blocker": app.state.blocker,
        }

    @app.get("/health", tags=["control"])
    def health() -> dict[str, Any]:
        return {"ok": True, "db": str(settings.db_path), "auto_submit": settings.auto_submit,
                "provider": settings.llm_provider, "model": settings.active_model}

    # ---- static assets ----------------------------------------------------
    @app.get("/assets/fonts/{filename}", include_in_schema=False)
    def font_file(filename: str) -> FileResponse:
        """Serve the project typeface locally, so the dashboard needs no CDN."""
        path = (ASSET_DIR / "fonts" / filename).resolve()
        if not str(path).startswith(str((ASSET_DIR / "fonts").resolve())) or not path.exists():
            raise HTTPException(404, "Font not found")
        media = {"woff2": "font/woff2", "woff": "font/woff",
                 "ttf": "font/ttf"}.get(path.suffix.lstrip("."), "application/octet-stream")
        return FileResponse(path, media_type=media,
                            headers={"Cache-Control": "public, max-age=604800"})

    # ---- dashboard --------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(500, f"Dashboard file missing: {index}")
        return index.read_text(encoding="utf-8")

    return app


STATIC_DIR = BASE_STATIC = Path(__file__).resolve().parent / "static"
ASSET_DIR = Path(__file__).resolve().parent / "assets"

app = None  # lazily created by main.py / uvicorn factory


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
