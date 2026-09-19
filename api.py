"""FastAPI admin API + HTML dashboard.

Read-only overview of every application the bot has looked at, plus two control
endpoints: release the human-in-the-loop gate, and trigger a run.

Auth: send `X-Admin-Token: <ADMIN_TOKEN>` (or `?token=`). When ADMIN_TOKEN is
left at its default the API only listens on localhost and auth is skipped with
a warning.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import Settings
from config import settings as default_settings
from database import STATUSES, Database

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


class RunRequest(BaseModel):
    urls: Optional[list[str]] = None
    limit: Optional[int] = None
    auto_submit: Optional[bool] = None


# ---- app ------------------------------------------------------------------


def create_app(settings: Settings = default_settings, db: Optional[Database] = None,
               gate: Optional[Any] = None) -> FastAPI:
    settings.ensure_dirs()
    database = db or Database(settings.db_path)
    database.init()
    app = FastAPI(
        title="XApply Admin",
        version="1.0.0",
        description="Overview of every job the bot analyzed, applied to, skipped or failed on.",
    )
    app.state.settings = settings
    app.state.db = database
    app.state.gate = gate
    app.state.run_task: Optional[asyncio.Task] = None
    app.state.run_log: list[str] = []

    def auth(request: Request, token: Optional[str] = Query(None)) -> None:
        expected = settings.admin_token
        if not expected or expected == DEFAULT_TOKEN:
            return  # local-only development mode
        supplied = request.headers.get("x-admin-token") or token
        if supplied != expected:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing admin token")

    # ---- data endpoints ---------------------------------------------------
    @app.get("/api/stats", response_model=StatsResponse, dependencies=[Depends(auth)], tags=["overview"])
    def get_stats() -> dict[str, Any]:
        return database.stats()

    @app.get("/api/applications", response_model=list[ApplicationSummary],
             dependencies=[Depends(auth)], tags=["overview"])
    def list_applications(
        status_filter: Optional[str] = Query(None, alias="status", description="|".join(STATUSES)),
        search: Optional[str] = Query(None, description="Substring of company, title or URL"),
        limit: int = Query(100, ge=1, le=1000),
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

    # ---- control endpoints ------------------------------------------------
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
        return {"released": True}

    @app.get("/api/run", dependencies=[Depends(auth)], tags=["control"])
    def run_status() -> dict[str, Any]:
        task = app.state.run_task
        return {
            "running": bool(task and not task.done()),
            "auto_submit": settings.auto_submit,
            "log_tail": app.state.run_log[-30:],
        }

    @app.post("/admin/run", dependencies=[Depends(auth)], tags=["control"])
    async def trigger_run(body: RunRequest) -> dict[str, Any]:
        task = app.state.run_task
        if task and not task.done():
            raise HTTPException(409, "A run is already in progress")
        from pipeline import Pipeline  # imported lazily: needs GEMINI_API_KEY

        if body.auto_submit is not None:
            settings.auto_submit = body.auto_submit
        gate = app.state.gate
        pipeline = Pipeline(settings, database, gate)
        if gate is None:
            app.state.gate = pipeline.gate

        async def _run() -> None:
            app.state.run_log.append(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} run started")
            try:
                stats = await pipeline.run(urls=body.urls, limit=body.limit)
                app.state.run_log.append(f"run finished: {stats}")
            except Exception as exc:
                log.exception("Triggered run failed")
                app.state.run_log.append(f"run failed: {exc}")

        app.state.run_task = asyncio.create_task(_run())
        return {"started": True, "auto_submit": settings.auto_submit}

    @app.get("/health", tags=["control"])
    def health() -> dict[str, Any]:
        return {"ok": True, "db": str(settings.db_path), "auto_submit": settings.auto_submit}

    # ---- dashboard --------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        return DASHBOARD_HTML

    return app


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XApply Admin</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a7;--ok:#3fb950;--warn:#d29922;--err:#f85149;--skip:#6e7681;--acc:#58a6ff}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;--fg:#11151c;--dim:#5c6570}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:600}.sp{flex:1}
.wrap{padding:20px 24px;max-width:1500px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .n{font-size:26px;font-weight:600}.card .l{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.5px}
input,select,button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:8px 11px;font:inherit}
button{cursor:pointer}button:hover{border-color:var(--acc)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th{text-align:left;padding:10px 12px;font-size:11px;letter-spacing:.6px;text-transform:uppercase;color:var(--dim);border-bottom:1px solid var(--line)}
td{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}tr.row{cursor:pointer}tr.row:hover td{background:rgba(88,166,255,.07)}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:600}
.submitted{background:rgba(63,185,80,.16);color:var(--ok)}.pending_human_review{background:rgba(210,153,34,.16);color:var(--warn)}
.skipped{background:rgba(110,118,129,.18);color:var(--skip)}.failed{background:rgba(248,81,73,.16);color:var(--err)}
.in_progress{background:rgba(88,166,255,.16);color:var(--acc)}
.score{font-variant-numeric:tabular-nums;font-weight:600}.dim{color:var(--dim)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
dialog{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:12px;max-width:900px;width:92%;padding:0}
dialog::backdrop{background:rgba(0,0,0,.6)}.dh{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
.db{padding:16px 20px;max-height:70vh;overflow:auto}.sec{margin:14px 0 6px;font-size:11px;letter-spacing:.6px;text-transform:uppercase;color:var(--dim)}
.kv{display:grid;grid-template-columns:150px 1fr;gap:4px 12px}.kv div:nth-child(odd){color:var(--dim)}
ul{margin:4px 0;padding-left:18px}a{color:var(--acc)}
.banner{background:rgba(210,153,34,.16);border:1px solid var(--warn);color:var(--warn);padding:10px 14px;border-radius:8px;margin-bottom:14px;display:none}
</style></head><body>
<header><h1>XApply</h1><span class="dim" id="mode"></span><span class="sp"></span>
<input id="q" placeholder="Search company / role / URL" style="width:240px">
<select id="st"><option value="">All statuses</option><option value="submitted">Submitted</option>
<option value="pending_human_review">Awaiting human</option><option value="skipped">Skipped</option>
<option value="failed">Failed</option><option value="in_progress">In progress</option></select>
<button onclick="load()">Refresh</button><button onclick="location.href=api('/api/export.csv')">Export CSV</button></header>
<div class="wrap">
<div class="banner" id="gate"></div>
<div class="cards" id="cards"></div>
<table><thead><tr><th>ID</th><th>Status</th><th>Score</th><th>Company</th><th>Role</th><th>ATS</th><th>Updated</th></tr></thead>
<tbody id="rows"></tbody></table>
<p class="dim" style="margin-top:14px">Click a row for the full job description, AI analysis and every answer submitted. API docs at <a href="/docs">/docs</a>.</p>
</div>
<dialog id="dlg"><div class="dh"><b id="dt"></b><span class="sp"></span><button onclick="dlg.close()">Close</button></div><div class="db" id="dbody"></div></dialog>
<script>
const tok = new URLSearchParams(location.search).get('token') || '';
const api = p => p + (tok ? (p.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(tok) : '');
const esc = s => (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ago = s => { if(!s) return '-'; const d=(Date.now()-new Date(s).getTime())/1000;
  return d<60?`${d|0}s ago`:d<3600?`${d/60|0}m ago`:d<86400?`${d/3600|0}h ago`:`${d/86400|0}d ago`; };
const LBL={submitted:'Submitted',pending_human_review:'Awaiting human',skipped:'Skipped',failed:'Failed',in_progress:'In progress'};
async function load(){
  const [stats, rows, health, gate] = await Promise.all([
    fetch(api('/api/stats')).then(r=>r.json()),
    fetch(api('/api/applications?limit=500&status='+encodeURIComponent(st.value)+'&search='+encodeURIComponent(q.value))).then(r=>r.json()),
    fetch(api('/health')).then(r=>r.json()).catch(()=>({})),
    fetch(api('/api/gate')).then(r=>r.json()).catch(()=>({paused:false}))]);
  mode.textContent = health.auto_submit ? 'AUTO-SUBMIT ON' : 'assisted mode';
  const g = document.getElementById('gate');
  if(gate.paused){ g.style.display='block'; g.innerHTML = 'Bot is waiting for you: ' + esc(gate.reason) +
    ' <button onclick="cont()" style="margin-left:10px">Continue</button>'; } else { g.style.display='none'; }
  cards.innerHTML = [['Total', stats.total], ['Submitted', stats.by_status.submitted],
    ['Awaiting human', stats.by_status.pending_human_review], ['Skipped', stats.by_status.skipped],
    ['Failed', stats.by_status.failed], ['Avg match', stats.avg_match_score ?? '-']]
    .map(([l,n])=>`<div class="card"><div class="n">${esc(n??0)}</div><div class="l">${l}</div></div>`).join('');
  document.getElementById('rows').innerHTML = rows.map(r=>`<tr class="row" onclick="detail(${r.id})">
    <td class="dim mono">${r.id}</td><td><span class="pill ${r.status}">${LBL[r.status]||r.status}</span></td>
    <td class="score">${r.match_score ?? '-'}</td><td>${esc(r.company)}</td><td>${esc(r.title)}</td>
    <td class="dim">${esc(r.ats)}</td><td class="dim">${ago(r.updated_at)}</td></tr>`).join('')
    || '<tr><td colspan=7 class="dim" style="padding:20px">No applications yet. Run <span class="mono">make run</span>.</td></tr>';
}
async function cont(){ await fetch(api('/admin/continue'), {method:'POST'}); load(); }
async function detail(id){
  const d = await fetch(api('/api/applications/'+id)).then(r=>r.json());
  dt.textContent = `#${d.id} ${d.title||''} @ ${d.company||''}`;
  const a = d.analysis || {}; const ans = d.answers || [];
  dbody.innerHTML = `
  <div class="kv"><div>Status</div><div><span class="pill ${d.status}">${LBL[d.status]||d.status}</span> ${esc(d.notes||'')}</div>
  <div>Match score</div><div class="score">${d.match_score ?? '-'}</div>
  <div>ATS / source</div><div>${esc(d.ats)} (found via ${esc(d.source)})</div>
  <div>Posting</div><div><a href="${esc(d.url)}" target="_blank">${esc(d.url)}</a></div>
  ${d.apply_url && d.apply_url!==d.url ? `<div>Applied at</div><div><a href="${esc(d.apply_url)}" target="_blank">${esc(d.apply_url)}</a></div>`:''}
  <div>Resume</div><div>${d.resume_path?`<a href="${api('/api/applications/'+d.id+'/resume')}" target="_blank">${esc(d.resume_path.split('/').pop())}</a>`:'-'}</div>
  <div>Screenshot</div><div>${d.screenshot_path?`<a href="${api('/api/applications/'+d.id+'/screenshot')}" target="_blank">view</a>`:'-'}</div>
  <div>Updated</div><div>${esc(d.updated_at)}</div></div>
  ${a.match_rationale?`<div class="sec">AI rationale</div><div>${esc(a.match_rationale)}</div>`:''}
  ${(a.missing_requirements||[]).length?`<div class="sec">Gaps</div><ul>${a.missing_requirements.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''}
  ${a.tailored_summary?`<div class="sec">Tailored summary</div><div>${esc(a.tailored_summary)}</div>`:''}
  ${(a.tailored_bullets||[]).map(g=>`<div class="sec">${esc(g.name)} (${esc(g.kind)})</div><ul>${(g.bullets||[]).map(b=>`<li>${esc(b)}</li>`).join('')}</ul>`).join('')}
  ${(a.answers||[]).length?`<div class="sec">Predicted screening answers</div><table>${a.answers.map(x=>`<tr><td>${esc(x.question)}</td><td>${esc(x.answer)}</td></tr>`).join('')}</table>`:''}
  <div class="sec">Form fields submitted (${ans.length})</div>
  ${ans.length?`<table><thead><tr><th>Field</th><th>Value</th><th>Source</th><th>OK</th></tr></thead><tbody>
   ${ans.map(x=>`<tr><td>${esc(x.label)}</td><td>${esc(x.value)}</td><td class="dim mono">${esc(x.source)} ${x.confidence??''}</td><td>${x.ok?'✓':'✗'}</td></tr>`).join('')}
   </tbody></table>`:'<div class="dim">None recorded</div>'}
  <div class="sec">Job description</div><div class="mono" style="white-space:pre-wrap;max-height:300px;overflow:auto">${esc(d.description||'(not stored)')}</div>`;
  dlg.showModal();
}
q.addEventListener('keyup', e => e.key === 'Enter' && load()); st.addEventListener('change', load);
load(); setInterval(load, 20000);
</script></body></html>"""


app = None  # lazily created by main.py / uvicorn factory


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
