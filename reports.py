"""Terminal reporting: browse everything the bot did without leaving the shell.

Rendered with plain ANSI + str.format so it works in any terminal and needs no
extra dependency.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from database import (
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    STATUS_SKIPPED,
    STATUS_SUBMITTED,
    Database,
)

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


BOLD, DIM = "1", "2"
STATUS_STYLE = {
    STATUS_SUBMITTED: ("32", "OK "),
    STATUS_PENDING: ("33", "HUM"),
    STATUS_SKIPPED: ("90", "SKP"),
    STATUS_FAILED: ("31", "ERR"),
    STATUS_IN_PROGRESS: ("36", "RUN"),
}


def _term_width(default: int = 100) -> int:
    return max(70, shutil.get_terminal_size((default, 24)).columns)


def _trunc(s: Any, n: int) -> str:
    s = "" if s is None else str(s).replace("\n", " ").strip()
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


def _ago(iso: Optional[str]) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return str(iso)[:16]
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    for limit, div, unit in ((60, 1, "s"), (3600, 60, "m"), (86400, 3600, "h"), (2592000, 86400, "d")):
        if secs < limit:
            return f"{int(secs / div)}{unit} ago"
    return dt.strftime("%Y-%m-%d")


def _status(status: str) -> str:
    color, short = STATUS_STYLE.get(status, ("0", status[:3].upper()))
    return c(short, color)


def _bar(value: int, total: int, width: int = 24) -> str:
    filled = 0 if not total else round(width * value / total)
    return "█" * filled + c("─" * (width - filled), DIM)


# ---------------------------------------------------------------------------


def print_summary(db: Database) -> None:
    s = db.stats()
    width = _term_width()
    print()
    print(c("  XAPPLY - APPLICATION OVERVIEW".ljust(width - 2), BOLD))
    print(c("  " + "─" * (width - 4), DIM))
    total = s["total"] or 0
    if not total:
        print("  No applications recorded yet. Run `make run` to start.\n")
        return
    order = [STATUS_SUBMITTED, STATUS_PENDING, STATUS_SKIPPED, STATUS_FAILED, STATUS_IN_PROGRESS]
    labels = {STATUS_SUBMITTED: "Submitted", STATUS_PENDING: "Awaiting human", STATUS_SKIPPED: "Skipped",
              STATUS_FAILED: "Failed", STATUS_IN_PROGRESS: "In progress"}
    for st in order:
        n = s["by_status"].get(st, 0)
        color = STATUS_STYLE[st][0]
        print(f"  {c(labels[st].ljust(15), color)} {str(n).rjust(4)}  {_bar(n, total)}  {n * 100 // total if total else 0}%")
    print(c("  " + "─" * (width - 4), DIM))
    avg = s["avg_match_score"]
    print(f"  {'Total jobs seen'.ljust(15)} {str(total).rjust(4)}"
          f"   avg match {avg if avg is not None else '-'}"
          f"   last activity {_ago(s['last_activity'])}")
    if s["submitted_by_ats"]:
        by = "  ".join(f"{k}: {v}" for k, v in sorted(s["submitted_by_ats"].items()))
        print(f"  {'Submitted via'.ljust(15)}      {by}")
    print()


def print_table(rows: list[dict[str, Any]], title: str = "") -> None:
    width = _term_width()
    if title:
        print(c(f"\n  {title}", BOLD))
    if not rows:
        print(c("  (no matching applications)\n", DIM))
        return
    # id(5) st(4) score(6) company(22) title(?) ats(11) when(9)
    fixed = 5 + 4 + 6 + 22 + 11 + 9 + 12
    title_w = max(18, width - fixed)
    head = (f"  {'ID'.ljust(4)} {'ST'.ljust(3)} {'SCORE'.rjust(5)}  {'COMPANY'.ljust(22)} "
            f"{'ROLE'.ljust(title_w)} {'ATS'.ljust(10)} {'WHEN'.ljust(8)}")
    print(c(head, DIM))
    for r in rows:
        score = r.get("match_score")
        score_s = f"{score:>3}" if score is not None else "  -"
        if score is not None:
            score_s = c(score_s, "32" if score >= 80 else "33" if score >= 65 else "90")
        print(f"  {str(r['id']).ljust(4)} {_status(r['status'])} {score_s}    "
              f"{_trunc(r.get('company'), 22).ljust(22)} {_trunc(r.get('title'), title_w).ljust(title_w)} "
              f"{_trunc(r.get('ats'), 10).ljust(10)} {_ago(r.get('updated_at')).ljust(8)}")
    print()


def print_detail(db: Database, app_id: int) -> int:
    row = db.get(app_id)
    if not row:
        print(c(f"  No application with id {app_id}\n", "31"))
        return 1
    width = _term_width()
    line = c("  " + "─" * (width - 4), DIM)
    print()
    print(c(f"  #{row['id']}  {row.get('title') or '?'}  @  {row.get('company') or '?'}", BOLD))
    print(line)
    print(f"  Status       {_status(row['status'])} {row['status']}")
    print(f"  Match score  {row.get('match_score') if row.get('match_score') is not None else '-'}")
    print(f"  ATS/source   {row.get('ats')} (found via {row.get('source')})")
    print(f"  Location     {row.get('location') or '-'}")
    print(f"  Posting      {row.get('url')}")
    if row.get("apply_url") and row["apply_url"] != row.get("url"):
        print(f"  Applied at   {row['apply_url']}")
    print(f"  Resume       {row.get('resume_path') or '-'}")
    print(f"  Screenshot   {row.get('screenshot_path') or '-'}")
    print(f"  First seen   {row.get('created_at')}   updated {row.get('updated_at')} ({_ago(row.get('updated_at'))})")
    if row.get("notes"):
        print(f"  Note         {_trunc(row['notes'], width - 20)}")

    analysis = row.get("analysis") or {}
    if analysis:
        print(line)
        print(c("  AI ANALYSIS", BOLD))
        print(f"  Rationale    {_trunc(analysis.get('match_rationale'), width - 18)}")
        if analysis.get("missing_requirements"):
            print(f"  Gaps         {_trunc('; '.join(analysis['missing_requirements']), width - 18)}")
        print(f"  Summary      {_trunc(analysis.get('tailored_summary'), width - 18)}")
        if analysis.get("highlighted_skills"):
            print(f"  Skills       {_trunc(', '.join(analysis['highlighted_skills']), width - 18)}")
        for group in analysis.get("tailored_bullets", []):
            print(c(f"  {group.get('name')} ({group.get('kind')})", DIM))
            for b in group.get("bullets", []):
                print(f"    • {_trunc(b, width - 8)}")
        if analysis.get("answers"):
            print(c("  Predicted screening answers", DIM))
            for qa in analysis["answers"]:
                print(f"    {_trunc(qa.get('question'), 52).ljust(52)} -> {_trunc(qa.get('answer'), width - 62)}")

    answers = row.get("answers") or []
    print(line)
    print(c(f"  FORM FIELDS THE BOT FILLED ({len(answers)})", BOLD))
    if not answers:
        print(c("  (none recorded)", DIM))
    for a in answers:
        mark = c("✓", "32") if a.get("ok") else c("✗", "31")
        src = a.get("source", "?")
        conf = a.get("confidence")
        conf_s = f" {conf:.2f}" if isinstance(conf, (int, float)) else ""
        print(f"  {mark} {_trunc(a.get('label'), 44).ljust(44)} = {_trunc(a.get('value'), width - 78).ljust(max(0, width - 78))}"
              f" {c(f'[{src}{conf_s}]', DIM)}")
    print()
    return 0


def print_description(db: Database, app_id: int) -> int:
    row = db.get(app_id)
    if not row:
        print(c(f"  No application with id {app_id}\n", "31"))
        return 1
    print(c(f"\n  JOB DESCRIPTION - #{row['id']} {row.get('title')} @ {row.get('company')}\n", BOLD))
    print(row.get("description") or "(no description stored)")
    print()
    return 0


def print_audit_files(audit_dir: Path, limit: int = 20) -> None:
    files = sorted(audit_dir.glob("*.json"), reverse=True)[:limit]
    print(c(f"\n  AUDIT FILES in {audit_dir} ({len(files)} shown)", BOLD))
    if not files:
        print(c("  (none yet)\n", DIM))
        return
    for f in files:
        try:
            d = json.loads(f.read_text())
            job = d.get("job", {})
            print(f"  {_status(d.get('status', ''))} {f.name.ljust(46)} "
                  f"{_trunc(job.get('company'), 20).ljust(20)} {_trunc(job.get('title'), 34)}")
        except Exception:
            print(f"      {f.name}")
    print()


def export_csv(db: Database, path: Path, status: Optional[str] = None) -> Path:
    import csv

    rows = db.list(status=status, limit=100_000, include_text=False)
    cols = ["id", "status", "match_score", "company", "title", "location", "ats", "source",
            "url", "apply_url", "resume_path", "screenshot_path", "notes", "created_at", "updated_at"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def print_discovered(jobs: list[Any], sources: list[str]) -> None:
    """Preview of `python main.py discover`: what would be applied to, grouped by ATS."""
    width = _term_width()
    print()
    print(c(f"  DISCOVERED {len(jobs)} POSTING(S) from {', '.join(sources)}".ljust(width - 2), BOLD))
    print(c("  " + "─" * (width - 4), DIM))
    if not jobs:
        print("  Nothing matched. Widen SEARCH_QUERIES, set REMOTE_ONLY=false, "
              "or add companies with `python main.py companies --add ashby:<token>`.\n")
        return
    by_ats: dict[str, list[Any]] = {}
    for j in jobs:
        by_ats.setdefault(j.ats, []).append(j)
    title_w = max(24, width - 64)
    for ats, items in sorted(by_ats.items(), key=lambda kv: -len(kv[1])):
        print(c(f"\n  {ats.upper()} ({len(items)})", BOLD))
        for j in items:
            desc = f"{len(j.description)}c" if j.description else c("no desc", "33")
            print(f"    {_trunc(j.company, 20).ljust(20)} {_trunc(j.title, title_w).ljust(title_w)} "
                  f"{_trunc(j.location, 18).ljust(18)} {desc}")
            print(c(f"      {j.apply_url or j.url}", DIM))
    print(c("\n  " + "─" * (width - 4), DIM))
    ready = sum(1 for j in jobs if j.description)
    print(f"  {ready}/{len(jobs)} arrived with a full description, so they are scored without opening a browser.")
    print("  Apply with: python main.py run\n")
