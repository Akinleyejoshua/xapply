"""Accounts you are signed in to, and how long that lasts.

Signing in is something only you can do, so this project never handles a password. You
sign in once in the browser window and the profile directory keeps the cookies, exactly
as your everyday browser does. The session then lasts until you sign out or the site
expires it.

What this module adds is the ability to *see* that. Without it, whether a run will be
able to read X or send from Gmail is only discoverable by starting a run and watching it
fail. The check reads the profile's own cookie database rather than opening a browser,
so the dashboard can show the state of every account at once and in no time.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

#: Chrome keeps its cookies here inside a profile directory. The newer path first.
COOKIE_PATHS = ("Default/Network/Cookies", "Default/Cookies", "Network/Cookies", "Cookies")


@dataclass(frozen=True)
class Integration:
    """One account the tool can make use of, if you are signed in to it."""

    name: str
    label: str
    #: Where to send you to sign in.
    sign_in_url: str
    #: Cookie domains that mean a session exists.
    domains: tuple[str, ...]
    #: Cookie names that only exist once signed in. Any one of them is enough.
    session_cookies: tuple[str, ...]
    #: What stops working without it, in a sentence you can act on.
    used_for: str = ""
    #: Whether anything is currently configured to use it.
    requires: tuple[str, ...] = field(default_factory=tuple)


INTEGRATIONS: tuple[Integration, ...] = (
    Integration(
        name="gmail", label="Gmail", sign_in_url="https://mail.google.com/mail/u/0/",
        domains=("google.com", "mail.google.com", "accounts.google.com"),
        session_cookies=("SID", "__Secure-1PSID", "__Secure-3PSID", "OSID"),
        used_for="Sending applications by email, from your own account.",
        requires=("email_apply", "email_transport=gmail")),
    Integration(
        name="x", label="X", sign_in_url="https://x.com/login",
        domains=("x.com", "twitter.com"),
        session_cookies=("auth_token", "ct0"),
        used_for="Searching X for roles advertised with an address.",
        requires=("search_x",)),
    Integration(
        name="linkedin", label="LinkedIn", sign_in_url="https://www.linkedin.com/login",
        domains=("linkedin.com", "www.linkedin.com"),
        session_cookies=("li_at",),
        used_for="The linkedin source and Easy Apply.",
        requires=("sources contains linkedin",)),
    Integration(
        name="indeed", label="Indeed", sign_in_url="https://secure.indeed.com/auth",
        domains=("indeed.com", "www.indeed.com"),
        session_cookies=("SHOE", "CTK", "INDEED_CSRF_TOKEN"),
        used_for="Opening Indeed postings without being asked to sign in."),
    Integration(
        name="glassdoor", label="Glassdoor",
        sign_in_url="https://www.glassdoor.com/profile/login_input.htm",
        domains=("glassdoor.com", "www.glassdoor.com"),
        session_cookies=("GSESSIONID", "gdId"),
        used_for="Reading Glassdoor postings past the sign-in wall."),
    Integration(
        name="wellfound", label="Wellfound", sign_in_url="https://wellfound.com/login",
        domains=("wellfound.com", "angel.co"),
        session_cookies=("_wellfound", "_angellist"),
        used_for="Reading Wellfound postings."),
    Integration(
        name="greenhouse", label="Greenhouse", sign_in_url="https://my.greenhouse.io/applications",
        domains=("greenhouse.io", "my.greenhouse.io"),
        session_cookies=("_session_id", "sessionid"),
        used_for="Seeing your own Greenhouse applications."),
)

BY_NAME = {i.name: i for i in INTEGRATIONS}


def cookie_db(profile_dir: Path) -> Optional[Path]:
    """Where this profile keeps its cookies, or None if it has none yet."""
    for relative in COOKIE_PATHS:
        candidate = Path(profile_dir) / relative
        if candidate.exists():
            return candidate
    return None


def _read_cookies(path: Path) -> list[tuple[str, str, int]]:
    """(host, name, expiry) for every cookie, read without disturbing the browser.

    The file is copied first. Chrome holds it open and locked while running, and a
    read-only query against a live SQLite file can still fail or block.
    """
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "cookies.db"
        try:
            shutil.copy2(path, copy)
            conn = sqlite3.connect(f"file:{copy}?mode=ro", uri=True, timeout=5)
        except (OSError, sqlite3.Error) as exc:
            log.debug("could not read %s: %s", path, exc)
            return []
        try:
            rows = conn.execute(
                "SELECT host_key, name, expires_utc FROM cookies").fetchall()
        except sqlite3.Error as exc:
            log.debug("unexpected cookie table in %s: %s", path, exc)
            return []
        finally:
            conn.close()
    return [(str(h or ""), str(n or ""), int(e or 0)) for h, n, e in rows]


def _live(expires_utc: int) -> bool:
    """Whether a cookie has not expired. 0 means it dies with the browser session."""
    if expires_utc <= 0:
        return True
    # Chrome counts microseconds from 1601, which is 11644473600 seconds before 1970.
    return (expires_utc / 1_000_000) - 11_644_473_600 > time.time()


def signed_in(profile_dir: Path, integration: Integration,
              cookies: Optional[Iterable[tuple[str, str, int]]] = None) -> bool:
    """Whether this profile still holds a session for the account."""
    if cookies is None:
        path = cookie_db(profile_dir)
        cookies = _read_cookies(path) if path else []
    for host, name, expires in cookies:
        host = host.lstrip(".")
        if name not in integration.session_cookies:
            continue
        if any(host == d or host.endswith("." + d) for d in integration.domains):
            if _live(expires):
                return True
    return False


def status(profile_dir: Path, settings: Any = None) -> list[dict[str, Any]]:
    """Every account, whether it is signed in, and whether anything needs it."""
    path = cookie_db(Path(profile_dir))
    cookies = _read_cookies(path) if path else []
    out = []
    for item in INTEGRATIONS:
        out.append({
            "name": item.name,
            "label": item.label,
            "signed_in": signed_in(Path(profile_dir), item, cookies),
            "sign_in_url": item.sign_in_url,
            "used_for": item.used_for,
            "needed_now": _needed(item, settings),
        })
    return out


def _needed(item: Integration, settings: Any) -> bool:
    """Whether the current settings actually depend on this account right now."""
    if settings is None:
        return False
    if item.name == "gmail":
        return bool(getattr(settings, "email_apply", False)
                    and getattr(settings, "email_transport", "") == "gmail")
    if item.name == "x":
        return bool(getattr(settings, "search_x", False))
    if item.name == "linkedin":
        return "linkedin" in (getattr(settings, "sources", None) or [])
    return False


def forget(profile_dir: Path, name: str) -> int:
    """Remove this account's cookies from the profile, which signs it out.

    Only while the browser is closed: Chrome holds the file open, and writing to it
    underneath a running browser corrupts it. Signing out in the window is always safe,
    and this exists for when the window is not the convenient place to do it.
    """
    item = BY_NAME.get(name)
    if item is None:
        raise KeyError(f"No integration named {name!r}")
    path = cookie_db(Path(profile_dir))
    if path is None:
        return 0
    try:
        conn = sqlite3.connect(path, timeout=5)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not open the cookie store: {exc}") from exc
    try:
        removed = 0
        for domain in item.domains:
            cur = conn.execute(
                "DELETE FROM cookies WHERE host_key = ? OR host_key LIKE ?",
                (domain, f"%.{domain}"))
            removed += cur.rowcount or 0
        conn.commit()
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Could not sign out of {item.label}: {exc}. Close the browser first, or "
            f"sign out in the window instead.") from exc
    finally:
        conn.close()
    log.info("Removed %d %s cookie(s) from the profile", removed, item.label)
    return removed
