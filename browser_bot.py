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
from models import JobPosting

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Stealth + CAPTCHA heuristics
# --------------------------------------------------------------------------

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

PLACEHOLDER_OPTION_RE = re.compile(r"^(select|choose|please (select|choose)|-+|\s*)", re.I)
AGREE_RE = re.compile(r"(agree|consent|acknowledge|certify|confirm|accept|i have read|authorize)", re.I)
FOLLOW_RE = re.compile(r"follow", re.I)
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


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


def choose_option(answer: str, options: list[str]) -> Optional[str]:
    """Map a free-form answer onto one of the concrete options of a select/radio."""
    opts = [o for o in options if o and not PLACEHOLDER_OPTION_RE.fullmatch(o.strip())]
    opts = [o for o in opts if not PLACEHOLDER_OPTION_RE.match(o.strip()) or len(o.strip()) > 20]
    if not opts or not answer:
        return None
    a = answer.strip().lower()
    for o in opts:
        if o.strip().lower() == a:
            return o
    yes_no = None
    if re.match(r"^(yes|y|true|i am|i do)\b", a):
        yes_no = "yes"
    elif re.match(r"^(no|n|false|i am not|i do not|i don't)\b", a):
        yes_no = "no"
    if yes_no:
        for o in opts:
            if o.strip().lower().startswith(yes_no):
                return o
    n = extract_number(answer)
    if n is not None:
        numbered = [(o, extract_number(o)) for o in opts]
        numbered = [(o, v) for o, v in numbered if v is not None]
        if numbered and not any(re.search(r"[a-z]", o.lower().replace("years", "").replace("year", "")) for o, _ in numbered):
            return min(numbered, key=lambda t: abs(t[1] - n))[0]
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

    mode="terminal": waits for Enter on stdin (falls back to a marker file when
                     stdin is not interactive).
    mode="api":      waits until `release()` is called (POST /admin/continue).
    """

    def __init__(self, mode: str = "terminal", marker_file: Optional[Path] = None):
        self.mode = mode
        self.marker_file = marker_file
        self._event = asyncio.Event()
        self.paused = False
        self.reason = ""
        self.paused_since: Optional[float] = None
        self.history: list[dict[str, Any]] = []

    async def wait(self, reason: str) -> None:
        self.paused, self.reason, self.paused_since = True, reason, time.time()
        self.history.append({"reason": reason, "at": time.time()})
        banner = (
            "\n" + "=" * 78 + "\n"
            "  HUMAN INPUT NEEDED\n"
            f"  {reason}\n"
            + ("  -> POST /admin/continue (or press Enter here) to resume.\n" if self.mode == "api"
               else "  -> Solve it in the browser window, then press Enter here to resume.\n")
            + "=" * 78 + "\a"
        )
        print(banner, flush=True)
        log.warning("Paused for human: %s", reason)
        try:
            if self.mode == "api":
                self._event.clear()
                await self._event.wait()
            else:
                await self._wait_terminal()
        finally:
            self.paused, self.reason, self.paused_since = False, "", None
            self._event.clear()
        log.info("Human released the gate")

    async def _wait_terminal(self) -> None:
        loop = asyncio.get_running_loop()
        prompt = "Press Enter after solving CAPTCHA / verifying form to continue... "
        try:
            await loop.run_in_executor(None, input, prompt)
        except (EOFError, OSError):
            marker = self.marker_file or Path("logs/CONTINUE")
            print(f"stdin is not interactive. Create the file {marker} (or POST /admin/continue) to resume.",
                  flush=True)
            while True:
                if marker.exists():
                    marker.unlink(missing_ok=True)
                    return
                if self._event.is_set():
                    return
                await asyncio.sleep(2)

    def release(self) -> None:
        self._event.set()

    def status(self) -> dict[str, Any]:
        return {"paused": self.paused, "reason": self.reason, "paused_since": self.paused_since}


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

    async def human_click(self, locator: Locator, timeout: Optional[int] = None) -> None:
        timeout = timeout or self.s.action_timeout_ms
        await locator.scroll_into_view_if_needed(timeout=timeout)
        await self.sleep(0.35, 0.12)
        try:
            await locator.hover(timeout=min(timeout, 3000))
        except Exception:
            pass
        await self.sleep(0.2, 0.08)
        await locator.click(timeout=timeout)
        await self.sleep()

    async def human_type(self, locator: Locator, text: str, clear: bool = True) -> None:
        await locator.scroll_into_view_if_needed()
        await locator.click()
        if clear:
            await locator.fill("")
        await locator.press_sequentially(text, delay=random.uniform(35, 110))
        await self.sleep(0.4, 0.15)

    # ---- navigation + guards -------------------------------------------
    async def goto(self, page: Page, url: str) -> None:
        await page.goto(url, wait_until="domcontentloaded")
        await self.sleep(1.6, 0.5)
        await self.guard(page)

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

    async def guard(self, page: Page, max_rounds: int = 6) -> None:
        """Pause for a human while a CAPTCHA / login wall is on screen."""
        for _ in range(max_rounds):
            reason = await self.detect_captcha(page) or await self.detect_login_wall(page)
            if not reason:
                return
            await self.gate.wait(reason)
            await asyncio.sleep(1.5)
        log.warning("Guard gave up after %d rounds, continuing anyway", max_rounds)

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
  let idx = 0; const out = []; const radioGroups = new Map();
  root.querySelectorAll('input, textarea, select').forEach(el => {
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
                               hidden_select: false });
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
                hidden_select: hiddenSelect };
    if (type === 'select') {
      d.options = [...el.options].map(o => ({ label: clean(o.text), value: o.value, idx: '', dom_id: '' }));
      const cur = el.selectedIndex >= 0 ? clean(el.options[el.selectedIndex].text) : '';
      d.current_value = /^(select|choose|please|-+|\s*)$/i.test(cur.split(' ')[0]) && cur.length < 30 ? '' : cur;
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
        fields = [FormField(**d) for d in raw]
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
                await loc.scroll_into_view_if_needed()
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
        loc = self._loc(scope, f.idx)
        checked = bool(f.current_value)
        want: Optional[bool] = None
        source = "rule"
        if FOLLOW_RE.search(f.label) and not AGREE_RE.search(f.label):
            want = self.s.follow_companies
        elif AGREE_RE.search(f.label):
            want = True
        elif f.required:
            want = True
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

    async def _combobox_options(self, scope: Locator, f: FormField) -> list[dict[str, str]]:
        """Open a custom (React) select, read its options, close it again."""
        loc = self._loc(scope, f.idx)
        try:
            await loc.scroll_into_view_if_needed()
            await loc.click()
            await asyncio.sleep(0.6)
            page = scope.page
            opts = page.locator('[role="option"], [role="listbox"] li, [class*="option" i][id*="option" i]')
            labels: list[str] = []
            n = min(await opts.count(), 200)
            for i in range(n):
                o = opts.nth(i)
                if await o.is_visible():
                    t = (await o.inner_text()).strip()
                    if t:
                        labels.append(re.sub(r"\s+", " ", t))
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
        await loc.scroll_into_view_if_needed()
        await loc.click()
        await loc.fill("")
        await loc.press_sequentially(target[:60], delay=random.uniform(40, 100))
        await asyncio.sleep(0.9)
        option = page.locator('[role="option"], [role="listbox"] li, .basic-typeahead__selectable')
        try:
            n = min(await option.count(), 50)
            best: Optional[Locator] = None
            best_score = 0.0
            for i in range(n):
                o = option.nth(i)
                if not await o.is_visible():
                    continue
                t = (await o.inner_text()).strip()
                score = difflib.SequenceMatcher(None, t.lower(), target.lower()).ratio()
                if t.lower() == target.lower():
                    best, best_score = o, 1.0
                    break
                if score > best_score:
                    best, best_score = o, score
            if best is not None and best_score >= 0.4:
                chosen_text = (await best.inner_text()).strip()
                await best.click()
                await self.b.sleep(0.4, 0.1)
                return chosen_text
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
    (re.compile(r"\bcity\b|location|where are you (located|based)|address|reside", re.I),
     lambda p: p.get("city") or p.get("location", "")),
    (re.compile(r"(current|most recent|present) (company|employer|organi[sz]ation)|^company$|^employer$|\borg\b", re.I),
     lambda p: _current(p, "company")),
    (re.compile(r"(current|most recent|present) (title|role|position)|job title|^title$", re.I),
     lambda p: _current(p, "title")),
    (re.compile(r"headline", re.I), lambda p: p.get("headline", "")),
    (re.compile(r"pronouns", re.I), lambda p: _sd(p, "pronouns")),
]

SCREENING_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sponsor", re.I), "requires_sponsorship"),
    (re.compile(r"authori[sz]ed|legally (able|permitted|allowed)|eligib|right to work|work permit|work in the", re.I), "work_authorization"),
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


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


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
            answer = self._from_profile(f, ctx.profile) or self._from_years(f, ctx.profile) \
                or self._from_predicted(f, ctx.analysis) or self._from_screening_defaults(f, ctx.profile)
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
        if f.kind in ("radio",):
            return None
        for pattern, getter in PROFILE_RULES:
            if pattern.search(f.label):
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
        if analysis is None or not analysis.answers:
            return None
        label_n = _norm_text(f.label)
        label_tokens = set(label_n.split())
        best_score, best_answer = 0.0, None
        for qa in analysis.answers:
            if qa.answer.strip().lower() in UNKNOWN_VALUES:
                continue
            q_n = _norm_text(qa.question)
            ratio = difflib.SequenceMatcher(None, label_n, q_n).ratio()
            q_tokens = set(q_n.split()) - {"do", "you", "have", "the", "a", "an", "of", "with", "your", "are", "is", "to", "in"}
            l_tokens = label_tokens - {"do", "you", "have", "the", "a", "an", "of", "with", "your", "are", "is", "to", "in"}
            jaccard = len(q_tokens & l_tokens) / len(q_tokens | l_tokens) if (q_tokens | l_tokens) else 0
            score = max(ratio, jaccard)
            if score > best_score:
                best_score, best_answer = score, qa.answer
        if best_answer and best_score >= 0.55:
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
        needs_human = fa.needs_human or fa.confidence < self.s.ai_min_confidence
        if f.options and value and choose_option(value, f.option_labels) is None:
            needs_human = True
        return ResolvedAnswer(value or None, "ai_live", fa.confidence, needs_human=needs_human, reasoning=fa.reasoning)
