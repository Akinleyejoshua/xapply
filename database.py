"""SQLite storage + audit trail.

One row per job the bot has looked at. The row keeps everything a human needs
to review the application afterwards: the job description, the AI analysis
(match score, tailored summary/bullets, predicted screening answers), the
answers actually typed into the form, the tailored resume path and a
screenshot of the final review page.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from models import JobPosting

log = logging.getLogger(__name__)

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

-- Postings a scan turned up, kept separately from applications because finding a job
-- is not the same as having applied to it. They live here rather than in the browser
-- so that a restart does not lose a ten-minute scan, and so the terminal and the
-- dashboard see the same list.
CREATE TABLE IF NOT EXISTS discovered (
    job_id      TEXT NOT NULL,
    source      TEXT NOT NULL,
    ats         TEXT NOT NULL DEFAULT 'unknown',
    company     TEXT,
    title       TEXT,
    location    TEXT,
    url         TEXT,
    apply_url   TEXT,
    description TEXT,
    relevance   REAL NOT NULL DEFAULT 0,
    found_at    TEXT NOT NULL,
    PRIMARY KEY (source, job_id)
);
CREATE INDEX IF NOT EXISTS idx_discovered_found ON discovered(found_at);
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
            self._repair_page_named_ids(c)

    def _repair_page_named_ids(self, c: sqlite3.Connection) -> int:
        """Re-key rows that were identified by the page rather than by the posting.

        An Ashby link ending "/application" used to reduce to the id "ashby-application",
        so every Ashby posting looked like the same one: the first was recorded, and the
        next one you picked was refused as already applied, naming a job you had never
        seen. Those rows are re-keyed here so the refusal stops.
        """
        from models import GENERIC_SEGMENTS, job_id_from_url

        suspect = [f"%-{seg}" for seg in sorted(GENERIC_SEGMENTS)]
        where = " OR ".join("job_id LIKE ?" for _ in suspect)
        rows = c.execute(f"SELECT id, job_id, source, url FROM applications WHERE {where}",
                         suspect).fetchall()
        fixed = 0
        for row in rows:
            correct = job_id_from_url(row["url"] or "")
            if not correct or correct == row["job_id"]:
                continue
            clash = c.execute(
                "SELECT id FROM applications WHERE source=? AND job_id=? AND id<>?",
                (row["source"], correct, row["id"])).fetchone()
            if clash:
                # Two rows for one posting. The older duplicate goes; the newest record
                # is the one that describes what actually happened.
                c.execute("DELETE FROM applications WHERE id=?", (min(clash["id"], row["id"]),))
            c.execute("UPDATE applications SET job_id=? WHERE id=?", (correct, row["id"]))
            fixed += 1
        if fixed:
            log.info("Re-keyed %d application(s) that were identified by the page rather "
                     "than the posting", fixed)
        return fixed

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

    def find_job(self, source: str, job_id: str) -> Optional[dict[str, Any]]:
        """The application already recorded for this posting, if there is one."""
        with self._conn() as c:
            row = c.execute("SELECT * FROM applications WHERE source=? AND job_id=?",
                            (source, job_id)).fetchone()
        return _row_to_dict(row, include_text=False) if row else None

    def delete_by_urls(self, urls: list[str]) -> int:
        """Forget every application for these links, whatever source found them.

        Used before a retry. Discovery skips anything already in the table, and the
        board that first found a job is not always the source a retry goes through,
        so matching on the link is the only reliable way to clear the path.
        """
        wanted = [u for u in (urls or []) if u]
        if not wanted:
            return 0
        marks = ",".join("?" * len(wanted))
        with self._conn() as c:
            return c.execute(
                f"DELETE FROM applications WHERE url IN ({marks}) OR apply_url IN ({marks})",
                wanted + wanted,
            ).rowcount

    # ---- discovered postings ------------------------------------------
    def save_discovered(self, jobs: list[Any]) -> int:
        """Remember what a scan found. Re-finding a posting refreshes it rather than
        duplicating it, so scanning twice does not double the list."""
        if not jobs:
            return 0
        now = _now()
        rows = []
        for j in jobs:
            d = j.to_dict() if hasattr(j, "to_dict") else dict(j)
            rows.append((
                d.get("job_id") or "", d.get("source") or "", d.get("ats") or "unknown",
                d.get("company") or "", d.get("title") or "", d.get("location") or "",
                d.get("url") or "", d.get("apply_url") or "", d.get("description") or "",
                float(d.get("relevance") or 0), now,
            ))
        with self._conn() as c:
            c.executemany(
                """INSERT INTO discovered
                       (job_id, source, ats, company, title, location, url, apply_url,
                        description, relevance, found_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source, job_id) DO UPDATE SET
                       ats=excluded.ats, company=excluded.company, title=excluded.title,
                       location=excluded.location, url=excluded.url,
                       apply_url=excluded.apply_url,
                       description=COALESCE(NULLIF(excluded.description, ''), discovered.description),
                       relevance=excluded.relevance, found_at=excluded.found_at""",
                rows,
            )
        return len(rows)

    #: A scan result and an application are the same posting when either link matches.
    #: A subquery rather than a join, so two applications for one posting cannot turn
    #: a single scan result into two rows.
    _APPLIED = """(
        SELECT a.{col} FROM applications a
        WHERE (NULLIF(a.url,'')       = NULLIF(d.url,''))
           OR (NULLIF(a.apply_url,'') = NULLIF(d.apply_url,''))
           OR (NULLIF(a.url,'')       = NULLIF(d.apply_url,''))
           OR (NULLIF(a.apply_url,'') = NULLIF(d.url,''))
        ORDER BY a.updated_at DESC LIMIT 1
    ) AS applied_{col}"""

    def list_discovered(self, limit: int = 500, offset: int = 0,
                        search: Optional[str] = None,
                        include_text: bool = False) -> list[dict[str, Any]]:
        """Scan results, each carrying the state of the application made from it.

        Without this you cannot tell by looking which postings you have already applied
        to, so you select one, the run finds nothing new to do, and the browser opens
        and closes for no visible reason.
        """
        sql = ("SELECT d.*, " + self._APPLIED.format(col="status") + ", "
               + self._APPLIED.format(col="id") + " FROM discovered d")
        params: list[Any] = []
        if search:
            sql += " WHERE (d.company LIKE ? OR d.title LIKE ? OR d.url LIKE ?)"
            params.extend([f"%{search}%"] * 3)
        sql += " ORDER BY d.relevance DESC, d.found_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if not include_text:
                d.pop("description", None)
            out.append(d)
        return out

    def count_discovered(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) AS n FROM discovered").fetchone()["n"])

    def delete_discovered(self, job_ids: list[str]) -> int:
        if not job_ids:
            return 0
        marks = ",".join("?" * len(job_ids))
        with self._conn() as c:
            return c.execute(f"DELETE FROM discovered WHERE job_id IN ({marks})", job_ids).rowcount

    def delete_discovered_all(self) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM discovered").rowcount

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
