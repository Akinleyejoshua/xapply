#!/usr/bin/env python3
"""XApply CLI.

  python main.py login                    sign in once; the session is persisted
  python main.py run [--auto-submit]      discover -> analyze -> tailor -> apply
  python main.py analyze --url URL        dry run on one posting (no application)
  python main.py serve                    FastAPI admin dashboard
  python main.py list [--status ...]      terminal overview of all applications
  python main.py show ID                  everything recorded for one application
  python main.py stats                    summary counters
  python main.py watch                    live-refreshing terminal dashboard
  python main.py export --out file.csv    dump the database to CSV
  python main.py init-db                  create the SQLite schema
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from config import settings

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-14s %(message)s"


def setup_logging(verbose: bool = False) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(settings.log_dir / "xapply.log", encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=LOG_FORMAT,
                        handlers=handlers, force=True)
    for noisy in ("httpx", "httpcore", "google_genai.models", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _db():
    from database import Database

    db = Database(settings.db_path)
    db.init()
    return db


# ---- commands --------------------------------------------------------------


def cmd_init_db(args: argparse.Namespace) -> int:
    _db()
    print(f"Database ready at {settings.db_path}")
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    from pipeline import login_flow

    asyncio.run(login_flow(settings))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from pipeline import Pipeline

    if args.auto_submit:
        settings.auto_submit = True
    if args.assisted:
        settings.auto_submit = False
    if args.threshold is not None:
        settings.match_threshold = args.threshold
    if args.headless:
        settings.headless = True
    urls = None
    if args.url:
        urls = list(args.url)
    elif args.urls_file:
        urls = [l.strip() for l in Path(args.urls_file).read_text().splitlines()
                if l.strip() and not l.startswith("#")]

    mode = "AUTO-SUBMIT (the bot clicks Submit)" if settings.auto_submit else "ASSISTED (you click Submit)"
    print("\n" + "=" * 72)
    print(f"  XApply | mode: {mode}")
    print(f"  Threshold: {settings.match_threshold}   Max applications: {args.limit or settings.max_applications_per_run}")
    print(f"  Sources: {', '.join(settings.sources)}   Model: {settings.gemini_model}")
    print("=" * 72 + "\n")

    pipeline = Pipeline(settings)
    try:
        stats = asyncio.run(pipeline.run(urls=urls, limit=args.limit))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    print("\nRun finished:", json.dumps(stats, indent=2) if stats else "no jobs processed")
    cmd_stats(args)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from pipeline import Pipeline

    pipeline = Pipeline(settings)
    out = asyncio.run(pipeline.analyze_only(args.url))
    a = out["analysis"]
    print(json.dumps({"job": {k: v for k, v in out["job"].items() if k != "description"},
                      "match_score": a["match_score"], "rationale": a["match_rationale"],
                      "missing": a["missing_requirements"], "summary": a["tailored_summary"],
                      "resume": out["resume_path"]}, indent=2))
    print(f"\nResume written to {out['resume_path']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from api import create_app

    if settings.admin_token == "change-me" and args.host not in ("127.0.0.1", "localhost"):
        print("Refusing to bind a non-local host with the default ADMIN_TOKEN. Set ADMIN_TOKEN in .env.")
        return 2
    app = create_app(settings)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  XApply admin dashboard: {url}\n  API docs:               {url}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    import reports

    db = _db()
    rows = db.list(status=args.status, search=args.search, limit=args.limit)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    title = "APPLICATIONS" + (f" [{args.status}]" if args.status else "") + (f" matching {args.search!r}" if args.search else "")
    reports.print_table(rows, title)
    print(f"  Showing {len(rows)} row(s). `python main.py show <ID>` for full detail.\n")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    import reports

    db = _db()
    if args.json:
        row = db.get(args.id)
        if not row:
            print(f"No application with id {args.id}")
            return 1
        print(json.dumps(row, indent=2))
        return 0
    if args.description:
        return reports.print_description(db, args.id)
    return reports.print_detail(db, args.id)


def cmd_stats(args: argparse.Namespace) -> int:
    import reports

    db = _db()
    if getattr(args, "json", False):
        print(json.dumps(db.stats(), indent=2))
        return 0
    reports.print_summary(db)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    import reports

    db = _db()
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            reports.print_summary(db)
            reports.print_table(db.list(status=args.status, limit=args.limit), "RECENT APPLICATIONS")
            print(f"  Refreshing every {args.interval}s. Ctrl-C to stop.")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


def cmd_audits(args: argparse.Namespace) -> int:
    import reports

    reports.print_audit_files(settings.audit_dir, args.limit)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import reports

    path = reports.export_csv(_db(), Path(args.out), args.status)
    print(f"Exported to {path}")
    return 0


# ---- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xapply", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the SQLite schema").set_defaults(func=cmd_init_db)
    sub.add_parser("login", help="open the browser to sign in once").set_defaults(func=cmd_login)

    r = sub.add_parser("run", help="discover, analyze, tailor and apply")
    r.add_argument("--auto-submit", action="store_true", help="bot clicks Submit itself")
    r.add_argument("--assisted", action="store_true", help="force assisted mode (you click Submit)")
    r.add_argument("--limit", type=int, help="max applications this run")
    r.add_argument("--threshold", type=int, help="override MATCH_THRESHOLD")
    r.add_argument("--headless", action="store_true", help="run headless (not recommended)")
    r.add_argument("--url", action="append", help="apply to this URL only (repeatable)")
    r.add_argument("--urls-file", help="file of URLs, one per line")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyze", help="score one posting and build the resume, apply nothing")
    a.add_argument("--url", required=True)
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("serve", help="FastAPI admin dashboard")
    s.add_argument("--host", default=settings.api_host)
    s.add_argument("--port", type=int, default=settings.api_port)
    s.set_defaults(func=cmd_serve)

    l = sub.add_parser("list", help="terminal overview of all applications")
    l.add_argument("--status", choices=["submitted", "pending_human_review", "skipped", "failed", "in_progress"])
    l.add_argument("--search", help="substring of company, title or URL")
    l.add_argument("--limit", type=int, default=100)
    l.add_argument("--json", action="store_true")
    l.set_defaults(func=cmd_list)

    sh = sub.add_parser("show", help="full record for one application")
    sh.add_argument("id", type=int)
    sh.add_argument("--description", action="store_true", help="print the stored job description")
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(func=cmd_show)

    st = sub.add_parser("stats", help="summary counters")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_stats)

    w = sub.add_parser("watch", help="live-refreshing terminal dashboard")
    w.add_argument("--interval", type=int, default=10)
    w.add_argument("--limit", type=int, default=25)
    w.add_argument("--status", choices=["submitted", "pending_human_review", "skipped", "failed", "in_progress"])
    w.set_defaults(func=cmd_watch)

    au = sub.add_parser("audits", help="list the per-application audit JSON files")
    au.add_argument("--limit", type=int, default=20)
    au.set_defaults(func=cmd_audits)

    e = sub.add_parser("export", help="dump the database to CSV")
    e.add_argument("--out", default="applications.csv")
    e.add_argument("--status")
    e.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"\nError: {exc}\n")
        return 2
    except RuntimeError as exc:
        print(f"\nError: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
