"""Playwright runner.

Pieces:
  HumanGate       - pauses the bot and waits for a human (terminal Enter, API call,
                    or a marker file) whenever a CAPTCHA / login wall / final review
                    page / unanswerable required field is hit.
  StealthBrowser  - persistent, non-headless context with webdriver masking, a
                    realistic UA/viewport, Gaussian delays, natural scrolling and
                    CAPTCHA detection.
  FormFiller      - ATS-agnostic form discovery + filling (text, textarea, number,
                    select, custom combobox, radio groups, checkboxes) using
                    label-first resilient selectors.
  AnswerResolver  - decides what to type: profile rules -> predicted screening
                    answers -> screening defaults -> live Gemini call -> human.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import (
    BrowserContext,
    Locator,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeout,
    async_playwright,
)

from ai_agent import AIAgent, JobAnalysis
from config import Settings
from matching import stem_equal, word_similarity
from models import JobPosting, ats_text

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Stealth + CAPTCHA heuristics
# --------------------------------------------------------------------------

#: Enough visible inputs to be worth filling. Used to decide whether a CAPTCHA on the
#: page is an inline form widget (fill first) or a wall standing in front of content.
FILLABLE_FORM_JS = r"""
() => {
  const sel = 'input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=search]), textarea, select';
  const visible = n => {
    const st = getComputedStyle(n);
    if (st.visibility === 'hidden' || st.display === 'none') return false;
    const r = n.getBoundingClientRect();
    return r.width > 0 || r.height > 0 || n.type === 'file';
  };
  const nodes = [...document.querySelectorAll(sel)];
  const shown = nodes.filter(visible);
  const hasFile = nodes.some(n => (n.getAttribute('type') || '').toLowerCase() === 'file');
  const hasPassword = shown.some(n => (n.getAttribute('type') || '').toLowerCase() === 'password');
  if (hasPassword) return false;          // a sign-in page is a wall, not a form to fill
  return hasFile || shown.length >= 4;
}
"""

STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
if (!window.chrome) { window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){} }; }
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
try {
  const originalQuery = window.navigator.permissions.query.bind(window.navigator.permissions);
  window.navigator.permissions.query = (parameters) =>
    parameters && parameters.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : originalQuery(parameters);
} catch (e) {}
"""

CAPTCHA_SELECTORS = [
    'iframe[src*="recaptcha"]',
    'iframe[src*="hcaptcha"]',
    'iframe[src*="arkoselabs"]',
    'iframe[src*="funcaptcha"]',
    'iframe[src*="turnstile"]',
    'iframe[title*="challenge" i]',
    'iframe[title*="captcha" i]',
    "#captcha",
    ".g-recaptcha",
    ".h-captcha",
    ".cf-turnstile",
    "#challenge-form",
    "#cf-challenge-running",
    'div[id^="captcha"]',
    'input[name="captcha"]',
    "#captcha-internal",
]
CAPTCHA_TEXT_RE = re.compile(
    r"(let'?s do a quick security check|security (check|verification)|verify (that )?you('?re| are) (a )?human"
    r"|are you a robot|unusual activity|complete the puzzle|prove you are human|checking your browser)",
    re.I,
)
LOGIN_URL_HINTS = ("/login", "/checkpoint/", "/authwall", "/uas/login", "/signin", "/sign-in")

PLACEHOLDER_OPTION_RE = re.compile(
    r"^(please\s+)?(select|choose|pick)(\s+(an?|your|one|the))?(\s+(option|answer|value|item|choice))?$", re.I
)
PLACEHOLDER_LITERALS = {"", "-", "--", "---", "n/a", "na", "none", "null", "\u2014", "\u2013"}
AGREE_RE = re.compile(r"(agree|consent|acknowledge|certify|confirm|accept|i have read|authorize)", re.I)
FOLLOW_RE = re.compile(r"follow", re.I)
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|\u2013|\u2014|to)\s*(\d+(?:\.\d+)?)")
AT_LEAST_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+|(?:more than|over|at least|above|greater than|\u2265|>=?)\s*(\d+(?:\.\d+)?)", re.I
)
AT_MOST_RE = re.compile(r"(?:less than|under|below|fewer than|at most|up to|\u2264|<=?)\s*(\d+(?:\.\d+)?)", re.I)


def is_placeholder_option(option: str) -> bool:
    """True for 'Select an option', '---', 'N/A' style entries that are not real answers."""
    o = (option or "").strip().strip(".:\u2026 ").lower()
    return o in PLACEHOLDER_LITERALS or bool(PLACEHOLDER_OPTION_RE.fullmatch(o))


def extract_number(text: str) -> Optional[float]:
    m = NUMBER_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _numeric_option_match(n: float, options: list[str]) -> Optional[str]:
    """Map a number onto options that may be ranges ('3-5 years', '10+', 'under 2')."""
    scored: list[tuple[str, float]] = []
    for o in options:
        r = RANGE_RE.search(o)
        if r:
            lo, hi = float(r.group(1)), float(r.group(2))
            if lo <= n <= hi:
                return o
            scored.append((o, min(abs(n - lo), abs(n - hi))))
            continue
        at_least = AT_LEAST_RE.search(o)
        if at_least:
            bound = float(at_least.group(1) or at_least.group(2))
            if n >= bound:
                return o
            scored.append((o, bound - n))
            continue
        at_most = AT_MOST_RE.search(o)
        if at_most:
            bound = float(at_most.group(1))
            if n < bound:
                return o
            scored.append((o, n - bound))
            continue
        v = extract_number(o)
        if v is not None:
            scored.append((o, abs(v - n)))
    return min(scored, key=lambda t: t[1])[0] if scored else None


AFFIRMATIVE_RE = re.compile(
    r"^\s*(yes|y|true|1|i (am|do|have|will|agree|consent|certify)|agree|accept|confirm|checked|"
    r"fluent|native|proficient|advanced|authorized)\b", re.I
)
NEGATIVE_RE = re.compile(
    r"^\s*(no|n|false|0|none|never|not? |i (am|do|have|will) ?n[o']t|decline|prefer not|"
    r"unknown|n/?a)\b", re.I
)


def is_affirmative(answer: str) -> bool:
    """Whether a free-form answer means 'tick this box'."""
    a = (answer or "").strip()
    if NEGATIVE_RE.match(a):
        return False
    return bool(AFFIRMATIVE_RE.match(a))


def choose_option(answer: str, options: list[str]) -> Optional[str]:
    """Map a free-form answer onto one of the concrete options of a select/radio."""
    opts = [o for o in options if o and not is_placeholder_option(o)]
    if not opts or not answer:
        return None
    a = answer.strip().lower()

    for o in opts:  # exact match
        if o.strip().lower() == a:
            return o

    yes_no = None  # yes/no questions
    if re.match(r"^(yes|y|true|i am|i do|i have|i will)\b", a):
        yes_no = "yes"
    elif re.match(r"^(no|n|false|i am not|i do not|i don'?t|i have not|i haven'?t)\b", a):
        yes_no = "no"
    if yes_no:
        for o in opts:
            if o.strip().lower().startswith(yes_no):
                return o

    n = extract_number(answer)  # numeric answers against numeric / range options
    if n is not None and any(extract_number(o) is not None for o in opts):
        match = _numeric_option_match(n, opts)
        if match:
            return match

    contains = [o for o in opts if a in o.lower() or o.lower() in a]
    if contains:
        return min(contains, key=lambda o: abs(len(o) - len(a)))

    m = difflib.get_close_matches(answer, opts, n=1, cutoff=0.5)
    return m[0] if m else None


# --------------------------------------------------------------------------
# Human gate
# --------------------------------------------------------------------------


class HumanGate:
    """Blocks the pipeline until a human says "continue".

    The release can arrive from any of three places, whichever happens first:
      * Enter on the terminal (only when stdin is an interactive tty)
      * `release()`, which is what POST /admin/continue and the dashboard button call
      * a marker file appearing on disk, for supervised or containerised runs

    They race, so a run started from the web UI is releasable from the web UI *and*
    from the terminal it was launched in. Waiting on only one of them was a deadlock:
    a server blocked on `input()` never sees the button.
    """

    CONTINUE = "continue"
    SKIP = "skip"

    def __init__(self, mode: str = "terminal", marker_file: Optional[Path] = None):
        self.mode = mode
        self.marker_file = marker_file or Path("logs/CONTINUE")
        self.skip_file = self.marker_file.with_name("SKIP")
        self._event = asyncio.Event()
        self._outcome = self.CONTINUE
        self.paused = False
        self.reason = ""
        self.allow_skip = True
        self.paused_since: Optional[float] = None
        self.history: list[dict[str, Any]] = []

    # ---- waiting ------------------------------------------------------
    async def wait(self, reason: str, allow_skip: bool = True) -> str:
        """Block until a human continues or skips. Returns CONTINUE or SKIP.

        Skipping matters when a challenge cannot be solved: a Cloudflare interstitial
        that never clears would otherwise hold the whole run on one posting.
        """
        self.paused, self.reason, self.paused_since = True, reason, time.time()
        self.allow_skip = allow_skip
        self._outcome = self.CONTINUE
        self.history.append({"reason": reason, "at": time.time()})
        self._event.clear()
        self._print_banner(reason, allow_skip)
        log.warning("Paused for human: %s", reason)
        waiters = [asyncio.create_task(self._event.wait(), name="gate-release"),
                   asyncio.create_task(self._wait_marker(), name="gate-marker")]
        stdin_task = self._stdin_task()
        if stdin_task is not None:
            waiters.append(stdin_task)
        released_by = "unknown"
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            released_by = next(iter(done)).get_name()
        finally:
            # Cancelled here as well as on the normal path, because a caller may give up
            # on waiting: an unattended scan does. Leaving these behind would leak a file
            # poller and a stdin reader on every pause that nobody answers.
            for task in waiters:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*waiters, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            outcome = self._outcome if allow_skip else self.CONTINUE
            self.paused, self.reason, self.paused_since = False, "", None
            self._event.clear()
            self.marker_file.unlink(missing_ok=True)
            self.skip_file.unlink(missing_ok=True)
        log.info("Human %s the job (%s)", "skipped" if outcome == self.SKIP else "released",
                 released_by)
        print("Skipping this job.\n" if outcome == self.SKIP else "Continuing.\n", flush=True)
        return outcome

    def _print_banner(self, reason: str, allow_skip: bool = True) -> None:
        interactive = self._stdin_is_tty()
        how = []
        if interactive:
            how.append("press Enter here")
        how.append("click Continue in the dashboard")
        how.append(f"or create {self.marker_file}")
        skip_line = ""
        if allow_skip:
            skip_line = ("  If the challenge will not clear, skip this job instead: "
                         + ("type s then Enter, " if interactive else "")
                         + "click Skip, or create " + str(self.skip_file) + "\n")
        print(
            "\n" + "=" * 78 + "\n"
            "  HUMAN INPUT NEEDED\n"
            f"  {reason}\n"
            f"  -> Handle it in the browser window, then {', '.join(how)}.\n"
            + skip_line
            + "=" * 78 + "\a",
            flush=True,
        )

    @staticmethod
    def _stdin_is_tty() -> bool:
        try:
            return bool(sys.stdin) and sys.stdin.isatty()
        except (ValueError, AttributeError):
            return False

    def _stdin_task(self) -> Optional[asyncio.Task]:
        """A cancellable reader on stdin, or None when stdin is not interactive.

        `add_reader` is used rather than a thread running `input()`, because a thread
        blocked in `input()` cannot be cancelled when the release arrives elsewhere.
        """
        if not self._stdin_is_tty():
            return None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def on_readable() -> None:
            line = ""
            try:
                line = sys.stdin.readline()
            except Exception:
                pass
            if self.allow_skip and line.strip().lower() in ("s", "skip"):
                self._outcome = self.SKIP
            if not future.done():
                future.set_result(None)

        try:
            loop.add_reader(sys.stdin.fileno(), on_readable)
        except (NotImplementedError, OSError, ValueError) as exc:
            log.debug("stdin reader unavailable (%s); use the dashboard or the marker file", exc)
            return None

        async def waiter() -> None:
            try:
                await future
            finally:
                try:
                    loop.remove_reader(sys.stdin.fileno())
                except Exception:
                    pass

        return asyncio.create_task(waiter(), name="gate-stdin")

    async def _wait_marker(self) -> None:
        while True:
            if self.skip_file.exists():
                self.skip_file.unlink(missing_ok=True)
                self._outcome = self.SKIP
                return
            if self.marker_file.exists():
                self.marker_file.unlink(missing_ok=True)
                return
            await asyncio.sleep(1.5)

    # ---- releasing ----------------------------------------------------
    def release(self) -> None:
        """Continue with the current job. Safe whether or not anything is waiting."""
        self._outcome = self.CONTINUE
        self._event.set()

    def skip(self) -> None:
        """Abandon the current job and move to the next one."""
        self._outcome = self.SKIP
        self._event.set()

    def status(self) -> dict[str, Any]:
        return {"paused": self.paused, "reason": self.reason, "paused_since": self.paused_since,
                "allow_skip": self.allow_skip, "waited": len(self.history)}


# --------------------------------------------------------------------------
# Stealth browser
# --------------------------------------------------------------------------


class StealthBrowser:
    def __init__(self, settings: Settings, gate: HumanGate):
        self.s = settings
        self.gate = gate
        self._pw: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def __aenter__(self) -> "StealthBrowser":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def start(self) -> "StealthBrowser":
        self.s.ensure_dirs()
        self._pw = await async_playwright().start()
        kwargs: dict[str, Any] = dict(
            user_data_dir=str(self.s.user_data_dir),
            headless=self.s.headless,
            viewport={"width": self.s.viewport_width, "height": self.s.viewport_height},
            user_agent=self.s.user_agent,
            locale=self.s.locale,
            timezone_id=self.s.timezone_id,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-infobars",
            ],
            ignore_default_args=["--enable-automation"],
            accept_downloads=True,
        )
        if self.s.browser_channel:
            try:
                self.context = await self._pw.chromium.launch_persistent_context(
                    channel=self.s.browser_channel, **kwargs
                )
            except Exception as exc:  # channel not installed
                log.warning("Could not launch browser channel %r (%s); using bundled Chromium",
                            self.s.browser_channel, str(exc).splitlines()[0])
        if self.context is None:
            self.context = await self._pw.chromium.launch_persistent_context(**kwargs)
        await self.context.add_init_script(STEALTH_JS)
        self.context.set_default_timeout(self.s.action_timeout_ms)
        self.context.set_default_navigation_timeout(self.s.navigation_timeout_ms)
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        if self.s.start_url:
            # A bare about:blank window looks like the bot has hung; show something instead.
            try:
                await self.page.goto(self.s.start_url, wait_until="domcontentloaded", timeout=15_000)
            except Exception as exc:
                log.debug("start page %s did not load: %s", self.s.start_url, exc)
        log.info("Browser started (profile: %s)", self.s.user_data_dir)
        return self

    async def close(self) -> None:
        try:
            if self.context:
                await self.context.close()
        finally:
            if self._pw:
                await self._pw.stop()
            self.context = self.page = self._pw = None

    # ---- human-like behaviour ------------------------------------------
    async def sleep(self, mean: Optional[float] = None, std: Optional[float] = None, minimum: float = 0.25) -> None:
        mean = self.s.delay_mean_s if mean is None else mean
        std = self.s.delay_std_s if std is None else std
        await asyncio.sleep(max(minimum, random.gauss(mean, std)))

    async def human_scroll(self, page: Page, distance: Optional[int] = None, steps: Optional[int] = None) -> None:
        distance = distance or random.randint(400, 1400)
        steps = steps or random.randint(4, 9)
        per = distance / steps
        for _ in range(steps):
            await page.mouse.wheel(0, per * random.uniform(0.7, 1.3))
            await asyncio.sleep(random.uniform(0.05, 0.25))
        await self.sleep(0.5, 0.2)

    #: How far from the middle of the window an element may sit before it is worth
    #: scrolling again. Anything inside this band is already comfortably readable.
    SCROLL_DEAD_ZONE_PX = 200

    async def bring_into_view(self, locator: Locator) -> None:
        """Put an element in the middle of the window, once, and then leave it alone.

        `scroll_into_view_if_needed` moves by the smallest amount that makes an element
        visible, so an element sitting under a sticky header or footer is nudged again
        by every later call. With several calls per field, that reads as the page
        twitching up and down while you are trying to look at it.
        """
        try:
            await locator.evaluate(
                """(el, dead) => {
                    const r = el.getBoundingClientRect();
                    const middle = window.innerHeight / 2;
                    const centre = r.top + r.height / 2;
                    if (Math.abs(centre - middle) <= dead) return;   // close enough already
                    el.scrollIntoView({block: 'center', inline: 'nearest'});
                }""",
                self.SCROLL_DEAD_ZONE_PX,
            )
        except Exception:
            try:
                await locator.scroll_into_view_if_needed()
            except Exception:
                pass

    async def human_click(self, locator: Locator, timeout: Optional[int] = None) -> None:
        timeout = timeout or self.s.action_timeout_ms
        await self.bring_into_view(locator)
        await self.sleep(0.35, 0.12)
        try:
            await locator.hover(timeout=min(timeout, 3000))
        except Exception:
            pass
        await self.sleep(0.2, 0.08)
        await locator.click(timeout=timeout)
        await self.sleep()

    #: Above this length, typing character by character is pointless and slow: a
    #: 1,800-character cover letter at ~70ms a key takes over two minutes and hits the
    #: action timeout, which silently truncated letters to a couple of hundred characters.
    TYPE_CHAR_BY_CHAR_LIMIT = 220

    async def human_type(self, locator: Locator, text: str, clear: bool = True) -> None:
        """Enter text, keystroke by keystroke for ordinary fields and at once for long prose."""
        await self.bring_into_view(locator)
        await locator.click()
        if clear:
            await locator.fill("")
        if len(text) > self.TYPE_CHAR_BY_CHAR_LIMIT:
            await locator.fill(text)
            # A React textarea needs an input event to register a programmatic fill.
            try:
                await locator.dispatch_event("input")
            except Exception:
                pass
            await self.sleep(0.8, 0.25)
            return
        await locator.press_sequentially(text, delay=random.uniform(35, 110))
        await self.sleep(0.4, 0.15)

    # ---- navigation + guards -------------------------------------------
    async def goto(self, page: Page, url: str) -> str:
        """Navigate, then stop only for something that genuinely blocks the page."""
        await page.goto(url, wait_until="domcontentloaded")
        await self.sleep(1.6, 0.5)
        return await self.guard(page, blocking_only=True)

    async def detect_captcha(self, page: Page) -> Optional[str]:
        try:
            for sel in CAPTCHA_SELECTORS:
                loc = page.locator(sel)
                n = await loc.count()
                for i in range(min(n, 5)):
                    el = loc.nth(i)
                    if not await el.is_visible():
                        continue
                    box = await el.bounding_box()
                    if box and box["width"] >= 60 and box["height"] >= 40:
                        return f"CAPTCHA/challenge element detected ({sel})"
            url = page.url.lower()
            if "captcha" in url or "/challenge" in url or "checkpoint/challenge" in url:
                return f"Challenge URL: {page.url}"
            title = await page.title()
            if CAPTCHA_TEXT_RE.search(title or ""):
                return f"Challenge page title: {title!r}"
            text = await page.evaluate("() => document.body ? document.body.innerText.slice(0, 4000) : ''")
            m = CAPTCHA_TEXT_RE.search(text or "")
            if m:
                return f"Challenge text on page: {m.group(0)!r}"
        except Exception as exc:  # page navigating, closed, etc.
            log.debug("captcha detection error: %s", exc)
        return None

    async def detect_login_wall(self, page: Page) -> Optional[str]:
        try:
            url = page.url.lower()
            if any(h in url for h in LOGIN_URL_HINTS):
                return f"Login required: {page.url}"
            loc = page.locator('form#login, input#username[name="session_key"], input[name="session_password"]')
            if await loc.count() and await loc.first.is_visible():
                return "Login form visible"
        except Exception:
            pass
        return None

    async def has_fillable_form(self, page: Page) -> bool:
        """Whether the page is showing something worth filling in right now."""
        try:
            return bool(await page.evaluate(FILLABLE_FORM_JS))
        except Exception:
            return False

    async def guard(self, page: Page, max_rounds: int = 6, blocking_only: bool = False) -> str:
        """Pause for a human when something is genuinely in the way.

        `blocking_only` is used while navigating. An application form very often carries
        its own inline reCAPTCHA or Turnstile widget, and stopping for it on arrival
        means the form is never filled: the person sees a CAPTCHA and a set of empty
        boxes. So during navigation only a real wall counts, meaning a login page or a
        challenge on a page with no form on it. The full check runs before submitting,
        which is the only moment the CAPTCHA actually has to be solved.

        Returns HumanGate.CONTINUE or HumanGate.SKIP.
        """
        for _ in range(max_rounds):
            wall = await self.detect_login_wall(page)
            reason = wall
            if not reason:
                captcha = await self.detect_captcha(page)
                if captcha and blocking_only and await self.has_fillable_form(page):
                    log.info("%s, but a form is present: filling it first", captcha)
                    return HumanGate.CONTINUE
                reason = captcha
            if not reason:
                return HumanGate.CONTINUE
            outcome = await self.gate.wait(reason)
            if outcome == HumanGate.SKIP:
                return HumanGate.SKIP
            await asyncio.sleep(1.5)
        log.warning("Guard gave up after %d rounds, continuing anyway", max_rounds)
        return HumanGate.CONTINUE

    async def screenshot(self, page: Page, name: str) -> str:
        self.s.log_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name)[:80]
        path = self.s.log_dir / f"{int(time.time())}_{safe}.png"
        try:
            await page.screenshot(path=str(path), full_page=False)
            return str(path)
        except Exception as exc:
            log.debug("screenshot failed: %s", exc)
            return ""


# --------------------------------------------------------------------------
# Form model + discovery
# --------------------------------------------------------------------------


@dataclass
class FormField:
    kind: str  # text | textarea | number | email | tel | url | select | combobox | radio | checkbox | file
    label: str
    idx: str  # value of data-xapply-idx (empty for radio groups)
    options: list[dict[str, str]] = field(default_factory=list)  # {label, value, idx, dom_id}
    required: bool = False
    current_value: str = ""
    name: str = ""
    dom_id: str = ""
    has_error: bool = False
    error_text: str = ""
    hidden_select: bool = False
    #: Where this field sits on the page. Fields used to be filled in discovery order,
    #: which put every radio group last, so the page jumped back to the top halfway
    #: through and then worked down again. Filling in page order walks down once.
    order: int = 0
    #: False for a dropdown you can only click, such as a button that opens a list.
    #: Typing into one throws, which is why they were skipped rather than filled.
    typeable: bool = True

    @property
    def option_labels(self) -> list[str]:
        return [o["label"] for o in self.options]


@dataclass
class ResolvedAnswer:
    value: Optional[str]
    source: str
    confidence: float = 1.0
    needs_human: bool = False
    reasoning: str = ""


@dataclass
class StepResult:
    filled: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    fields_seen: int = 0


DISCOVER_JS = r"""
(root) => {
  const clean = s => (s || '').replace(/\s+/g, ' ').replace(/\s*\*\s*$/, '').replace(/\(?required\)?/ig, '').trim();
  const visible = el => {
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 || r.height > 0;
  };
  const textOfIds = ids => clean((ids || '').split(/\s+/).map(id => { const n = document.getElementById(id); return n ? n.innerText : ''; }).join(' '));
  const walkUp = (el, selector, maxDepth) => {
    let p = el.parentElement, depth = 0;
    while (p && depth < maxDepth) {
      const cands = p.querySelectorAll(selector);
      for (const c of cands) { if (!c.contains(el) && clean(c.innerText)) return clean(c.innerText); }
      p = p.parentElement; depth++;
    }
    return '';
  };
  const labelOf = el => {
    if (el.labels && el.labels.length) { const t = clean([...el.labels].map(l => l.innerText).join(' ')); if (t) return t; }
    const al = el.getAttribute('aria-label'); if (al && clean(al)) return clean(al);
    const lb = el.getAttribute('aria-labelledby'); if (lb) { const t = textOfIds(lb); if (t) return t; }
    const fs = el.closest('fieldset'); if (fs) { const lg = fs.querySelector('legend'); if (lg && clean(lg.innerText)) return clean(lg.innerText); }
    const ph = el.getAttribute('placeholder'); if (ph && clean(ph)) return clean(ph);
    const up = walkUp(el, 'label, legend, [class*="label" i]:not(input):not(select):not(textarea), [class*="question" i]:not(input):not(select):not(textarea)', 4);
    if (up) return up;
    return clean(el.getAttribute('name') || el.id || '');
  };
  const groupLabelOf = el => {
    const fs = el.closest('fieldset');
    if (fs) { const lg = fs.querySelector('legend'); if (lg && clean(lg.innerText)) return clean(lg.innerText); }
    const grp = el.closest('[role="radiogroup"], [role="group"]');
    if (grp) { const t = grp.getAttribute('aria-label') || textOfIds(grp.getAttribute('aria-labelledby')); if (clean(t)) return clean(t); }
    const up = walkUp(el, 'legend, [class*="label" i]:not(label):not(input), [class*="question" i]:not(label):not(input), h3, h4', 5);
    return up || clean(el.getAttribute('name') || '');
  };
  const errorOf = el => {
    if (el.getAttribute('aria-invalid') === 'true') {
      const d = el.getAttribute('aria-describedby'); const t = d ? textOfIds(d) : '';
      return t || 'invalid';
    }
    let p = el.parentElement, depth = 0;
    while (p && depth < 3) {
      const e = p.querySelector('[class*="error" i]:not(input):not(select):not(textarea), [role="alert"]');
      if (e && visible(e) && clean(e.innerText)) return clean(e.innerText);
      p = p.parentElement; depth++;
    }
    return '';
  };
  let idx = 0; let order = 0; const out = []; const radioGroups = new Map();
  const placeholderish = t => /^(select|choose|please|pick|-+|\s*)$/i.test((t || '').split(' ')[0]) && (t || '').length < 30;
  root.querySelectorAll('input, textarea, select').forEach(el => {
    const myOrder = order++;
    let type = (el.getAttribute('type') || '').toLowerCase();
    if (el.tagName === 'SELECT') type = 'select';
    else if (el.tagName === 'TEXTAREA') type = 'textarea';
    else if (!type) type = 'text';
    if (['hidden', 'submit', 'button', 'reset', 'image'].includes(type)) return;
    let hiddenSelect = false;
    if (type === 'select' && !visible(el)) {
      const p = el.parentElement;
      const widget = p && p.querySelector('.select2-container, [class*="select__control" i], [role="combobox"]');
      if (!widget || !visible(widget)) return;
      hiddenSelect = true;
    } else if (type !== 'file' && !visible(el)) return;
    if (el.disabled || el.readOnly) return;
    const id = 'x' + (idx++);
    el.setAttribute('data-xapply-idx', id);
    const required = !!(el.required || el.getAttribute('aria-required') === 'true');
    if (type === 'radio') {
      const key = el.name ? 'name:' + el.name : 'group:' + groupLabelOf(el);
      if (!radioGroups.has(key)) {
        radioGroups.set(key, { kind: 'radio', label: groupLabelOf(el), idx: '', name: el.name || '', dom_id: '',
                               required: required, options: [], current_value: '', has_error: false, error_text: '',
                               hidden_select: false, order: myOrder, typeable: false });
      }
      const g = radioGroups.get(key);
      const optLabel = clean((el.labels && el.labels[0] && el.labels[0].innerText) || el.getAttribute('aria-label') || el.value);
      g.options.push({ label: optLabel, value: el.value, idx: id, dom_id: el.id || '' });
      if (el.checked) g.current_value = optLabel;
      g.required = g.required || required;
      const err = errorOf(el); if (err) { g.has_error = true; g.error_text = err; }
      return;
    }
    const d = { kind: type, label: labelOf(el), idx: id, name: el.name || '', dom_id: el.id || '',
                required: required, options: [], current_value: '', has_error: false, error_text: '',
                hidden_select: hiddenSelect, order: myOrder, typeable: true };
    if (type === 'select') {
      d.options = [...el.options].map(o => ({ label: clean(o.text), value: o.value, idx: '', dom_id: '' }));
      const cur = el.selectedIndex >= 0 ? clean(el.options[el.selectedIndex].text) : '';
      d.current_value = placeholderish(cur) ? '' : cur;
      d.typeable = false;
    } else if (type === 'checkbox') {
      d.current_value = el.checked ? 'checked' : '';
    } else if (type === 'file') {
      d.current_value = el.files && el.files.length ? el.files[0].name : '';
    } else {
      d.current_value = el.value || '';
      if (el.getAttribute('role') === 'combobox' || el.getAttribute('aria-autocomplete') || el.getAttribute('aria-haspopup') === 'listbox') d.kind = 'combobox';
      if (type === 'number' || /numeric|number/i.test(el.id + ' ' + el.className)) d.kind = 'number';
    }
    const err = errorOf(el); if (err) { d.has_error = true; d.error_text = err; }
    out.push(d);
  });

  // Dropdowns built out of a button or a div, with no native control behind them.
  // Nothing above finds these, because they are not an input, a textarea or a select,
  // so on forms that use them the bot filled everything else and left them empty.
  root.querySelectorAll('[role="combobox"], [aria-haspopup="listbox"], [aria-haspopup="menu"]').forEach(el => {
    const tag = el.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;  // already captured
    if (el.querySelector('input:not([type=hidden]), textarea, select')) return;  // a wrapper, not the control
    if (el.hasAttribute('data-xapply-idx')) return;
    if (!visible(el)) return;
    if (el.getAttribute('aria-disabled') === 'true' || el.hasAttribute('disabled')) return;
    const id = 'x' + (idx++);
    el.setAttribute('data-xapply-idx', id);
    const shown = clean(el.innerText);
    const active = el.getAttribute('aria-activedescendant');
    const activeText = active ? clean((document.getElementById(active) || {}).innerText || '') : '';
    const d = { kind: 'combobox', label: labelOf(el), idx: id, name: el.getAttribute('name') || '',
                dom_id: el.id || '', required: el.getAttribute('aria-required') === 'true',
                options: [], current_value: '', has_error: false, error_text: '',
                hidden_select: false, order: order++, typeable: false };
    const cur = activeText || shown;
    d.current_value = placeholderish(cur) ? '' : cur;
    const err = errorOf(el); if (err) { d.has_error = true; d.error_text = err; }
    out.push(d);
  });

  return out.concat([...radioGroups.values()]);
}
"""


class FormFiller:
    """ATS-agnostic form filling. Works on any scope (page body, modal, form)."""

    def __init__(self, browser: StealthBrowser, resolver: "AnswerResolver", settings: Settings):
        self.b = browser
        self.resolver = resolver
        self.s = settings

    async def discover(self, scope: Locator) -> list[FormField]:
        raw = await scope.evaluate(DISCOVER_JS)
        # Sorted by position, not by discovery order. Radio groups are collected as they
        # are met but emitted last, so filling in discovery order sent the page back to
        # the top partway through and then down again.
        fields = sorted((FormField(**d) for d in raw), key=lambda f: f.order)
        log.debug("Discovered %d fields: %s", len(fields), [(f.kind, f.label[:40]) for f in fields])
        return fields

    def _loc(self, scope: Locator, idx: str) -> Locator:
        return scope.locator(f'[data-xapply-idx="{idx}"]')

    async def fill_step(
        self,
        scope: Locator,
        ctx: "ResolveContext",
        only_errors: bool = False,
        force_ai: bool = False,
    ) -> StepResult:
        """Fill every field in `scope`. Anything unanswerable is reported, never guessed."""
        result = StepResult()
        fields = await self.discover(scope)
        result.fields_seen = len(fields)
        for f in fields:
            if f.kind == "file":
                continue  # resume upload is handled by the applier
            if only_errors and not f.has_error:
                continue
            try:
                if f.kind == "checkbox":
                    entry = await self._handle_checkbox(scope, f, ctx)
                    if entry:
                        result.filled.append(entry)
                    elif f.required and not f.current_value:
                        result.unresolved.append(f.label)
                    continue
                if f.current_value and not f.has_error and not only_errors:
                    log.debug("Keeping prefilled %r = %r", f.label, f.current_value[:40])
                    continue
                if f.kind == "combobox" and not f.options:
                    f.options = await self._combobox_options(scope, f)
                answer = await self.resolver.resolve(f, ctx, force_ai=force_ai or f.has_error)
                if answer.value is None or answer.needs_human or answer.value == "":
                    if f.required:
                        result.unresolved.append(f.label)
                    log.info("Leaving %r blank (%s)", f.label, answer.reasoning or answer.source)
                    continue
                applied = await self._apply(scope, f, answer.value)
                entry = {
                    "label": f.label, "kind": f.kind, "value": applied or answer.value,
                    "source": answer.source, "confidence": round(answer.confidence, 2),
                    "ok": applied is not None,
                }
                result.filled.append(entry)
                if applied is None and f.required:
                    result.unresolved.append(f.label)
            except PlaywrightTimeout as exc:
                log.warning("Timeout filling %r: %s", f.label, str(exc).splitlines()[0])
                if f.required:
                    result.unresolved.append(f.label)
            except Exception as exc:
                log.warning("Error filling %r: %s", f.label, exc)
                if f.required:
                    result.unresolved.append(f.label)
        return result

    # ---- per-kind actions ----------------------------------------------
    async def _apply(self, scope: Locator, f: FormField, value: str) -> Optional[str]:
        """Enter `value` into the field. Returns the value actually applied or None."""
        # A form stores what is typed, so the same ATS-safe flattening applies here.
        value = ats_text(value)
        if f.kind in ("text", "textarea", "email", "tel", "url", "search", "password"):
            await self.b.human_type(self._loc(scope, f.idx), value)
            return value
        if f.kind == "number":
            n = extract_number(value)
            text = format_number(n) if n is not None else value
            await self.b.human_type(self._loc(scope, f.idx), text)
            return text
        if f.kind == "select":
            chosen = choose_option(value, f.option_labels)
            if not chosen:
                log.info("No select option matches %r for %r (options: %s)", value, f.label, f.option_labels[:8])
                return None
            opt = next(o for o in f.options if o["label"] == chosen)
            loc = self._loc(scope, f.idx)
            if f.hidden_select:
                # select2 / react-select wrapper: set the native select and notify the widget
                await loc.select_option(value=opt["value"], force=True)
                await loc.dispatch_event("change")
            else:
                await self.b.bring_into_view(loc)
                await loc.select_option(value=opt["value"])
            await self.b.sleep(0.5, 0.15)
            return chosen
        if f.kind == "radio":
            chosen = choose_option(value, f.option_labels)
            if not chosen:
                log.info("No radio option matches %r for %r (options: %s)", value, f.label, f.option_labels)
                return None
            opt = next(o for o in f.options if o["label"] == chosen)
            label_loc = scope.locator(f'label[for="{opt["dom_id"]}"]') if opt.get("dom_id") else None
            if label_loc is not None and await label_loc.count() and await label_loc.first.is_visible():
                await self.b.human_click(label_loc.first)
            else:
                await self._loc(scope, opt["idx"]).check(force=True)
                await self.b.sleep(0.4, 0.1)
            return chosen
        if f.kind == "combobox":
            return await self._fill_combobox(scope, f, value)
        log.debug("Unhandled field kind %r for %r", f.kind, f.label)
        return None

    async def _handle_checkbox(self, scope: Locator, f: FormField, ctx: "ResolveContext") -> Optional[dict[str, Any]]:
        """Tick a checkbox only when there is a truthful reason to.

        A checkbox being `required` is NOT such a reason: application forms mark whole
        groups required (spoken languages, for instance), and blanket-ticking them would
        claim skills the candidate does not have. Anything that is not a consent box is
        resolved like any other field, and left unticked when the answer is not clearly
        affirmative.
        """
        loc = self._loc(scope, f.idx)
        checked = bool(f.current_value)
        want: Optional[bool] = None
        source = "rule"
        if FOLLOW_RE.search(f.label) and not AGREE_RE.search(f.label):
            want = self.s.follow_companies
        elif AGREE_RE.search(f.label):
            want = True
        else:
            answer = await self.resolver.resolve(f, ctx)
            if answer.value and not answer.needs_human:
                want = is_affirmative(answer.value)
                source = answer.source
        if want is None:
            return None
        if want != checked:
            label_loc = scope.locator(f'label[for="{f.dom_id}"]') if f.dom_id else None
            if label_loc is not None and await label_loc.count() and await label_loc.first.is_visible():
                await self.b.human_click(label_loc.first)
            else:
                await loc.set_checked(want, force=True)
                await self.b.sleep(0.3, 0.1)
        return {"label": f.label, "kind": "checkbox", "value": "checked" if want else "unchecked",
                "source": source, "confidence": 1.0, "ok": True}

    #: Where an opened dropdown puts its choices, whatever it is built from.
    OPTION_SELECTOR = ('[role="option"], [role="listbox"] li, [role="menuitem"], '
                       '[role="menuitemradio"], [class*="option" i][id*="option" i], '
                       '.basic-typeahead__selectable')

    async def _open_menu(self, scope: Locator, f: FormField) -> Locator:
        """Click a dropdown open and return a locator for whatever it revealed."""
        loc = self._loc(scope, f.idx)
        await self.b.bring_into_view(loc)
        await loc.click()
        await asyncio.sleep(0.6)
        return scope.page.locator(self.OPTION_SELECTOR)

    async def _visible_option_texts(self, options: Locator, cap: int = 200) -> list[str]:
        out: list[str] = []
        for i in range(min(await options.count(), cap)):
            o = options.nth(i)
            try:
                if not await o.is_visible():
                    continue
                text = (await o.inner_text()).strip()
            except Exception:
                continue
            if text:
                out.append(re.sub(r"\s+", " ", text))
        return out

    async def _pick_from_menu(self, options: Locator, target: str) -> Optional[str]:
        """Click the choice closest to `target`, or None when nothing is close enough."""
        best: Optional[Locator] = None
        best_score = 0.0
        best_text = ""
        for i in range(min(await options.count(), 200)):
            o = options.nth(i)
            try:
                if not await o.is_visible():
                    continue
                text = (await o.inner_text()).strip()
            except Exception:
                continue
            if not text:
                continue
            if text.lower() == target.lower():
                best, best_score, best_text = o, 1.0, text
                break
            score = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
            if score > best_score:
                best, best_score, best_text = o, score, text
        if best is None or best_score < 0.4:
            return None
        await best.click()
        await self.b.sleep(0.4, 0.1)
        return best_text

    async def _combobox_options(self, scope: Locator, f: FormField) -> list[dict[str, str]]:
        """Open a custom (React) select, read its options, close it again."""
        try:
            opts = await self._open_menu(scope, f)
            page = scope.page
            labels = await self._visible_option_texts(opts)
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.2)
            return [{"label": l, "value": l, "idx": "", "dom_id": ""} for l in labels]
        except Exception as exc:
            log.debug("combobox option discovery failed for %r: %s", f.label, exc)
            return []

    async def _fill_combobox(self, scope: Locator, f: FormField, value: str) -> Optional[str]:
        page = scope.page
        loc = self._loc(scope, f.idx)
        target = choose_option(value, f.option_labels) if f.options else value
        if f.options and not target:
            # Typeahead style (e.g. location): type the raw value and pick the first suggestion.
            target = value

        if not f.typeable:
            # A dropdown made of a button or a div. There is nothing to type into, so it
            # is opened and the closest choice is clicked. Typing into one of these threw,
            # which is why these fields used to be left empty.
            try:
                options = await self._open_menu(scope, f)
                chosen = await self._pick_from_menu(options, target)
                if chosen:
                    return chosen
                log.info("No option in %r resembles %r (saw: %s)", f.label, target,
                         (await self._visible_option_texts(options, 8)))
                await page.keyboard.press("Escape")
            except Exception as exc:
                log.info("Could not work the dropdown %r: %s", f.label, exc)
                try:
                    await page.keyboard.press("Escape")
                except Exception:
                    pass
            return None

        await self.b.bring_into_view(loc)
        await loc.click()
        await loc.fill("")
        await loc.press_sequentially(target[:60], delay=random.uniform(40, 100))
        await asyncio.sleep(0.9)
        try:
            chosen = await self._pick_from_menu(page.locator(self.OPTION_SELECTOR), target)
            if chosen:
                return chosen
        except PlaywrightTimeout:
            pass
        # keyboard fallback
        await page.keyboard.press("ArrowDown")
        await page.keyboard.press("Enter")
        await self.b.sleep(0.4, 0.1)
        return target


# --------------------------------------------------------------------------
# Answer resolution
# --------------------------------------------------------------------------


@dataclass
class ResolveContext:
    profile: dict[str, Any]
    job: JobPosting
    analysis: Optional[JobAnalysis]


def _first_name(p: dict) -> str:
    return p.get("first_name") or (p.get("name", "").split(" ")[0] if p.get("name") else "")


def _last_name(p: dict) -> str:
    if p.get("last_name"):
        return p["last_name"]
    parts = p.get("name", "").split(" ")
    return " ".join(parts[1:]) if len(parts) > 1 else ""


def _current(p: dict, key: str) -> str:
    exp = p.get("experience") or []
    return exp[0].get(key, "") if exp else ""


def _sd(p: dict, key: str) -> str:
    v = (p.get("screening_defaults") or {}).get(key)
    return "" if v is None else str(v)


#: Contact fields carry short labels. Anything longer is a question, not a field.
MAX_CONTACT_LABEL = 64

PROFILE_RULES: list[tuple[re.Pattern, Any]] = [
    (re.compile(r"\bfirst\s*name|given name|\bforename", re.I), _first_name),
    (re.compile(r"\blast\s*name|surname|family name", re.I), _last_name),
    (re.compile(r"\bfull\s*name|^name$|^your name|legal name|preferred name", re.I), lambda p: p.get("name", "")),
    (re.compile(r"e-?mail", re.I), lambda p: p.get("email", "")),
    (re.compile(r"country code|phone country|dial code", re.I),
     lambda p: f"{p.get('phone_country', '')} ({p.get('phone_country_code', '')})".strip() if p.get("phone_country") else p.get("phone_country_code", "")),
    (re.compile(r"\b(mobile|phone|telephone|cell)\b", re.I), lambda p: p.get("phone", "")),
    (re.compile(r"linkedin", re.I), lambda p: p.get("linkedin", "")),
    (re.compile(r"github", re.I), lambda p: p.get("github", "")),
    (re.compile(r"portfolio|personal (web)?site|website|\burl\b|other website|blog", re.I),
     lambda p: p.get("website") or p.get("github", "")),
    (re.compile(r"\bcountry\b|nationality|country of residence", re.I),
     lambda p: p.get("country") or p.get("location", "")),
    (re.compile(r"\bcity\b|\blocation\b|where are you (located|based)|\baddress\b|reside", re.I),
     lambda p: p.get("city") or p.get("location", "")),
    (re.compile(r"(current|most recent|present) (company|employer|organi[sz]ation)|^company$|^employer$|\borg\b", re.I),
     lambda p: _current(p, "company")),
    (re.compile(r"(current|most recent|present) (title|role|position)|job title|^title$", re.I),
     lambda p: _current(p, "title")),
    (re.compile(r"headline", re.I), lambda p: p.get("headline", "")),
    (re.compile(r"pronouns", re.I), lambda p: _sd(p, "pronouns")),
]

SCREENING_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsponsor(ship)?\b", re.I), "requires_sponsorship"),
    (re.compile(r"authori[sz]ed|legally (able|permitted|allowed)|eligib|right to work|"
                r"work permit|work in the|work authorization", re.I), "work_authorization"),
    (re.compile(r"notice period|notice\b", re.I), "notice_period"),
    (re.compile(r"how soon|when (can|could|are you able to) (you )?start|start date|available to start|availability|earliest", re.I), "start_date"),
    (re.compile(r"salary|compensation|pay (expectation|range|rate)|expected (pay|rate)|\brate\b|\bctc\b", re.I), "expected_salary"),
    (re.compile(r"relocat", re.I), "willing_to_relocate"),
    (re.compile(r"remote|on-?site|hybrid|in[- ]office|commut|work from", re.I), "remote_preference"),
    (re.compile(r"travel", re.I), "willing_to_travel"),
    (re.compile(r"clearance", re.I), "security_clearance"),
    (re.compile(r"\bgender\b|\bsex\b", re.I), "gender"),
    (re.compile(r"race|ethnic", re.I), "race"),
    (re.compile(r"veteran|military", re.I), "veteran"),
    (re.compile(r"disabilit", re.I), "disability"),
    (re.compile(r"\b18\b|eighteen|age of majority|legal age", re.I), "over_18"),
    (re.compile(r"hear about|how did you (find|learn|hear)|referr(al|ed)|source", re.I), "hear_about_us"),
    (re.compile(r"english|language proficiency|fluen", re.I), "english_proficiency"),
    (re.compile(r"previously (worked|employed)|worked (here|for us|at .* before)|former employee", re.I), "previously_employed_here"),
]

YEARS_RE = re.compile(r"(how many )?years?\b.*?(experience|work(ing)?|using|with|in)\b|experience.*years?", re.I)
TECH_AFTER_RE = re.compile(r"(?:with|in|using|of)\s+([A-Za-z0-9+#./ \-]+?)(?:\?|$|\(|,|\.)", re.I)
UNKNOWN_VALUES = {"", "unknown", "n/a", "na", "none", "null"}


#: Words that carry no meaning when comparing a question to a form label.
QUESTION_STOPWORDS = {
    "do", "does", "did", "you", "your", "yours", "have", "has", "the", "a", "an", "of",
    "with", "are", "is", "was", "be", "been", "to", "in", "on", "for", "and", "or", "any",
    "will", "would", "can", "could", "please", "select", "choose", "what", "which", "how",
    "many", "much", "us", "this", "that", "at", "by", "from", "if", "now", "future",
}


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _content_words(s: str) -> list[str]:
    """The words worth comparing in a question or a field label."""
    return [w for w in _norm_text(s).split() if len(w) > 1 and w not in QUESTION_STOPWORDS]


class AnswerResolver:
    def __init__(self, ai: Optional[AIAgent], settings: Settings):
        self.ai = ai
        self.s = settings
        self._cache: dict[str, ResolvedAnswer] = {}

    async def resolve(self, f: FormField, ctx: ResolveContext, force_ai: bool = False) -> ResolvedAnswer:
        label = f.label or f.name or ""
        cache_key = f"{ctx.job.job_id}|{f.kind}|{_norm_text(label)}|{'|'.join(f.option_labels)}"
        if not force_ai and cache_key in self._cache:
            return self._cache[cache_key]

        answer: Optional[ResolvedAnswer] = None
        if not force_ai:
            # Screening questions are checked first because they are phrased as sentences
            # and would otherwise be captured by a loose contact-field rule. "Will you
            # require sponsorship in this location?" must not be answered with a city.
            answer = self._from_screening_defaults(f, ctx.profile) \
                or self._from_years(f, ctx.profile) \
                or self._from_profile(f, ctx.profile) \
                or self._from_predicted(f, ctx.analysis)
            if answer and f.options:
                if choose_option(answer.value or "", f.option_labels) is None:
                    log.debug("Rule answer %r does not fit options for %r, asking AI", answer.value, label)
                    answer = None
        if answer is None:
            answer = await self._from_ai(f, ctx)
        self._cache[cache_key] = answer
        return answer

    # ---- strategies ----------------------------------------------------
    def _from_profile(self, f: FormField, profile: dict) -> Optional[ResolvedAnswer]:
        """Contact-style fields, matched only against short, field-like labels.

        A real contact field is labelled "Email" or "Location (City)". A long sentence
        is a screening question that merely happens to contain one of these words, and
        answering it from the profile produces nonsense.
        """
        if f.kind in ("radio",):
            return None
        label = (f.label or "").strip()
        if len(label) > MAX_CONTACT_LABEL or label.rstrip().endswith("?"):
            return None
        for pattern, getter in PROFILE_RULES:
            if pattern.search(label):
                value = str(getter(profile) or "").strip()
                if value:
                    return ResolvedAnswer(value, "profile", 0.98)
                return None
        return None

    def _from_years(self, f: FormField, profile: dict) -> Optional[ResolvedAnswer]:
        if not YEARS_RE.search(f.label):
            return None
        years: dict[str, Any] = profile.get("years_of_experience") or {}
        if not years:
            return None
        label_n = _norm_text(f.label)
        m = TECH_AFTER_RE.search(f.label)
        candidates = [m.group(1)] if m else []
        candidates.append(f.label)
        best: Optional[tuple[int, str, Any]] = None
        for key, val in years.items():
            key_n = _norm_text(key)
            for cand in candidates:
                cand_n = _norm_text(cand)
                if key_n and (key_n in cand_n or cand_n == key_n):
                    score = len(key_n)
                    if best is None or score > best[0]:
                        best = (score, key, val)
        if best is None:
            if re.search(r"(total|overall|professional|software|engineering|work) experience", label_n):
                total = max((v for v in years.values() if isinstance(v, (int, float))), default=None)
                if total is not None:
                    return ResolvedAnswer(format_number(float(total)), "profile.years", 0.8)
            return None
        return ResolvedAnswer(format_number(float(best[2])), "profile.years", 0.95)

    def _from_predicted(self, f: FormField, analysis: Optional[JobAnalysis]) -> Optional[ResolvedAnswer]:
        """Match a form label against the questions predicted for this job.

        Comparison is stem-aware, because the model often answers in the vocabulary of
        the profile rather than of the form: a predicted "Work authorization" has to
        reach a field labelled "Are you legally authorized to work in the US?".
        """
        if analysis is None or not analysis.answers:
            return None
        label_tokens = _content_words(f.label)
        if not label_tokens:
            return None
        best_score, best_answer = 0.0, None
        for qa in analysis.answers:
            if qa.answer.strip().lower() in UNKNOWN_VALUES:
                continue
            q_tokens = _content_words(qa.question)
            if not q_tokens:
                continue
            overlap = sum(1 for q in q_tokens
                          if any(word_similarity(q, l) >= 0.85 for l in label_tokens))
            # Scored against the shorter side, so a terse "Work authorization" can still
            # answer a long sentence that contains the same idea.
            coverage = overlap / min(len(q_tokens), len(label_tokens))
            ratio = difflib.SequenceMatcher(None, _norm_text(f.label), _norm_text(qa.question)).ratio()
            score = max(coverage, ratio)
            if score > best_score:
                best_score, best_answer = score, qa.answer
        if best_answer and best_score >= 0.6:
            return ResolvedAnswer(best_answer.strip(), "ai_predicted", round(0.6 + 0.4 * best_score, 2))
        return None

    def _from_screening_defaults(self, f: FormField, profile: dict) -> Optional[ResolvedAnswer]:
        defaults = profile.get("screening_defaults") or {}
        for pattern, key in SCREENING_RULES:
            if pattern.search(f.label):
                value = defaults.get(key)
                if value is None or str(value).strip() == "":
                    return None
                value = str(value).strip()
                if key == "work_authorization" and not f.options and defaults.get("work_authorization_details"):
                    value = str(defaults["work_authorization_details"])
                return ResolvedAnswer(value, f"profile.screening_defaults.{key}", 0.9)
        return None

    #: A written answer shorter than this is a shrug, not a draft.
    MIN_DRAFT_CHARS = 40

    def _draft_worth_keeping(self, f: FormField, value: str) -> bool:
        """Whether a low-confidence written answer should still be entered.

        For a fact, low confidence means leave it blank: a wrong salary or visa status
        is worse than an empty box. For an essay question it means the opposite. The
        model rating its own prose at 0.5 is not evidence the prose is wrong, and an
        empty box helps nobody, so the draft goes in and you edit it.

        Not in auto mode, though. The argument above rests entirely on you reading it
        before it is sent, and in auto mode nobody does.
        """
        if self.s.fill_mode == "auto":
            return False
        return (f.kind == "textarea" and not f.options
                and len(value.strip()) >= self.MIN_DRAFT_CHARS)

    async def _from_ai(self, f: FormField, ctx: ResolveContext) -> ResolvedAnswer:
        if self.ai is None:
            return ResolvedAnswer(None, "none", 0.0, needs_human=True, reasoning="no AI agent configured")
        try:
            fa = await self.ai.answer_form_question(
                ctx.profile, ctx.job, ctx.analysis, f.label, f.kind,
                options=f.option_labels or None, current_value=f.current_value, error=f.error_text,
            )
        except Exception as exc:
            log.warning("AI field answer failed for %r: %s", f.label, exc)
            return ResolvedAnswer(None, "ai_live", 0.0, needs_human=True, reasoning=str(exc))
        value = fa.answer.strip()
        if value.lower() in UNKNOWN_VALUES and not (f.kind in ("textarea",) and value == ""):
            value = ""
        needs_human = fa.needs_human
        if fa.confidence < self.s.ai_min_confidence and not self._draft_worth_keeping(f, value):
            needs_human = True
        if f.options and value and choose_option(value, f.option_labels) is None:
            needs_human = True
        return ResolvedAnswer(value or None, "ai_live", fa.confidence, needs_human=needs_human, reasoning=fa.reasoning)
