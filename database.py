"""SQLite storage + audit trail.

One row per job the bot has looked at. The row keeps everything a human needs
to review the application afterwards: the job description, the AI analysis
(match score, tailored summary/bullets, predicted screening answers), the
answers actually typed into the form, the tailored resume path and a
screenshot of the final review page.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from models import JobPosting

STATUS_SKIPPED = "skipped"
STATUS_PENDING = "pending_human_review"
STATUS_SUBMITTED = "submitted"
STATUS_FAILED = "failed"
STATUS_IN_PROGRESS = "in_progress"
STATUSES = (STATUS_SKIPPED, STATUS_PENDING, STATUS_SUBMITTED, STATUS_FAILED, STATUS_IN_PROGRESS)

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'linkedin',
    ats             TEXT NOT NULL DEFAULT 'unknown',
    company         TEXT,
    title           TEXT,
    location        TEXT,
    url             TEXT,
    apply_url       TEXT,
    description     TEXT,
    match_score     INTEGER,
    status          TEXT NOT NULL,
    notes           TEXT,
    resume_path     TEXT,
    screenshot_path TEXT,
    analysis_json   TEXT,
    answers_json    TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(source, job_id)
);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_created ON applications(created_at);
"""

JSON_COLUMNS = ("analysis_json", "answers_json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row, include_text: bool = True) -> dict[str, Any]:
    d = dict(row)
    for col in JSON_COLUMNS:
        raw = d.pop(col, None)
        key = col.replace("_json", "")
        try:
            d[key] = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            d[key] = raw
    if not include_text:
        d.pop("description", None)
        d.pop("analysis", None)
        d.pop("answers", None)
    return d


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ---- writes -------------------------------------------------------
    def has_job(self, source: str, job_id: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM applications WHERE source=? AND job_id=?", (source, job_id)
            ).fetchone()
        return row is not None

    def record(
        self,
        job: JobPosting,
        status: str,
        *,
        match_score: Optional[int] = None,
        notes: Optional[str] = None,
        resume_path: Optional[str] = None,
        screenshot_path: Optional[str] = None,
        analysis: Optional[dict[str, Any]] = None,
        answers: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        """Insert or update the row for `job`. Only non-None fields overwrite."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        now = _now()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO applications
                    (job_id, source, ats, company, title, location, url, apply_url, description,
                     match_score, status, notes, resume_path, screenshot_path,
                     analysis_json, answers_json, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source, job_id) DO UPDATE SET
                    ats             = excluded.ats,
                    company         = COALESCE(NULLIF(excluded.company, ''), applications.company),
                    title           = COALESCE(NULLIF(excluded.title, ''), applications.title),
                    location        = COALESCE(NULLIF(excluded.location, ''), applications.location),
                    url             = excluded.url,
                    apply_url       = COALESCE(NULLIF(excluded.apply_url, ''), applications.apply_url),
                    description     = COALESCE(NULLIF(excluded.description, ''), applications.description),
                    match_score     = COALESCE(excluded.match_score, applications.match_score),
                    status          = excluded.status,
                    notes           = COALESCE(excluded.notes, applications.notes),
                    resume_path     = COALESCE(excluded.resume_path, applications.resume_path),
                    screenshot_path = COALESCE(excluded.screenshot_path, applications.screenshot_path),
                    analysis_json   = COALESCE(excluded.analysis_json, applications.analysis_json),
                    answers_json    = COALESCE(excluded.answers_json, applications.answers_json),
                    updated_at      = excluded.updated_at
                """,
                (
                    job.job_id, job.source, job.ats, job.company, job.title, job.location,
                    job.url, job.apply_url, job.description, match_score, status, notes,
                    resume_path, screenshot_path,
                    json.dumps(analysis, ensure_ascii=False) if analysis is not None else None,
                    json.dumps(answers, ensure_ascii=False) if answers is not None else None,
                    now, now,
                ),
            )
            row = c.execute(
                "SELECT id FROM applications WHERE source=? AND job_id=?", (job.source, job.job_id)
            ).fetchone()
        return int(row["id"])

    def update_status(self, app_id: int, status: str, notes: Optional[str] = None) -> bool:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._conn() as c:
            cur = c.execute(
                "UPDATE applications SET status=?, notes=COALESCE(?, notes), updated_at=? WHERE id=?",
                (status, notes, _now(), app_id),
            )
        return cur.rowcount > 0

    def delete(self, app_id: int) -> bool:
        """Remove one application. The job becomes eligible for discovery again."""
        with self._conn() as c:
            return c.execute("DELETE FROM applications WHERE id=?", (app_id,)).rowcount > 0

    def delete_many(self, ids: list[int]) -> int:
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self._conn() as c:
            return c.execute(f"DELETE FROM applications WHERE id IN ({marks})", ids).rowcount

    def delete_by_status(self, status: str) -> int:
        """Clear out everything with one status, e.g. every failed attempt."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._conn() as c:
            return c.execute("DELETE FROM applications WHERE status=?", (status,)).rowcount

    def delete_all(self) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM applications").rowcount

    # ---- reads --------------------------------------------------------
    def get(self, app_id: int) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM applications WHERE id=?", (app_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        include_text: bool = False,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM applications"
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if search:
            clauses.append("(company LIKE ? OR title LIKE ? OR url LIKE ?)")
            params.extend([f"%{search}%"] * 3)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [_row_to_dict(r, include_text=include_text) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._conn() as c:
            by_status = {
                r["status"]: r["n"]
                for r in c.execute("SELECT status, COUNT(*) AS n FROM applications GROUP BY status")
            }
            agg = c.execute(
                "SELECT COUNT(*) AS total, AVG(match_score) AS avg_score, MAX(updated_at) AS last_activity "
                "FROM applications"
            ).fetchone()
            by_ats = {
                r["ats"]: r["n"]
                for r in c.execute(
                    "SELECT ats, COUNT(*) AS n FROM applications WHERE status='submitted' GROUP BY ats"
                )
            }
        return {
            "total": agg["total"],
            "avg_match_score": round(agg["avg_score"], 1) if agg["avg_score"] is not None else None,
            "last_activity": agg["last_activity"],
            "by_status": {s: by_status.get(s, 0) for s in STATUSES},
            "submitted_by_ats": by_ats,
        }
