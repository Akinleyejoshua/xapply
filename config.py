"""Central configuration.

Values are resolved in three layers, each overriding the one before:

  1. the defaults written below
  2. environment variables and `.env` (case-insensitive, so `AUTO_SUBMIT=true`
     sets `Settings.auto_submit`)
  3. `settings.local.json`, the choices you made in the web UI

Layer 3 is what makes the dashboard remember a model or a source list across a
restart. It only ever holds the keys in `PERSISTED_KEYS`, never a secret, and
deleting the file reverts everything to your `.env`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any, ClassVar, Iterable, Literal

from pydantic import Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

#: Settings the web UI may change and have remembered. Deliberately excludes every
#: path, credential and network binding: those stay under your control in `.env`.
PERSISTED_KEYS = (
    "llm_provider",
    "gemini_model",
    "nvidia_model",
    "opencode_model",
    "llm_extra_headers",
    "sources",
    "search_queries",
    "search_location",
    "remote_only",
    "seniority_levels",
    "countries",
    "title_match_threshold",
    "match_threshold",
    "fill_mode",
    "fill_all_at_once",
    "verify_after_fill",
    "email_apply",
    "hide_browser",
    "challenge_action",
    "headless",
    "max_applications_per_run",
    "max_jobs_per_company",
    "follow_companies",
    "resume_max_pages",
    "resume_may_drop_experience",
)


def _split_csv(value: Any) -> Any:
    """Allow `SEARCH_QUERIES=a,b,c` in .env instead of JSON lists."""
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_assignment=True,   # a bad value from the API is rejected, not stored
    )

    # ---- AI ----
    llm_provider: str = "gemini"  # gemini | nvidia | opencode

    # Google Gemini (https://aistudio.google.com/apikey)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # NVIDIA NIM (free key at https://build.nvidia.com, OpenAI-compatible)
    nvidia_api_key: str = ""
    #: NVIDIA's catalogue lists many models it has not deployed. This one answers today.
    #: The dashboard is where you change it, and it verifies the choice for you.
    nvidia_model: str = "openai/gpt-oss-20b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # OpenCode Zen (https://opencode.ai): one key, many vendors' models
    opencode_api_key: str = ""
    opencode_model: str = "claude-haiku-4-5"
    opencode_base_url: str = "https://opencode.ai/zen/v1"

    #: Extra HTTP headers sent with every LLM request, as JSON in .env or set in the UI.
    #: Some gateways want a User-Agent, a Referer or an app title to route a request.
    llm_extra_headers: Annotated[dict[str, str], NoDecode] = {}

    llm_timeout_s: float = 180.0
    llm_max_output_tokens: int = 8192
    match_threshold: int = Field(65, ge=0, le=100)
    ai_max_retries: int = 4
    ai_min_confidence: float = 0.55  # below this a live AI form answer is escalated to a human

    #: Which setting holds the model for each provider.
    MODEL_SETTING: ClassVar[dict[str, str]] = {
        "gemini": "gemini_model", "nvidia": "nvidia_model", "opencode": "opencode_model",
    }

    @property
    def active_model(self) -> str:
        return getattr(self, self.MODEL_SETTING.get(self.llm_provider.lower(), "gemini_model"))

    # ---- Execution mode ----
    #: documents = attach the resume and, when asked for, a cover letter, and leave every
    #:             other field to you
    #: assisted  = fill everything, then stop so you check it and press Submit
    #: auto      = fill everything and press Submit
    fill_mode: Literal["documents", "assisted", "auto"] = "assisted"
    auto_submit: bool = False  # kept in step with fill_mode; auto mode implies it
    #: Read the whole form, work out every answer at once, then fill it in one pass.
    #: Off by default: answers are worked out one at a time and typed at human speed,
    #: which is slower but looks like a person filling a form. On, a form with a dozen
    #: AI-answered questions takes about as long as its slowest single answer.
    fill_all_at_once: bool = False
    #: How many answers to work out at the same time in that mode. Higher is faster
    #: until the model provider starts refusing concurrent requests.
    fill_concurrency: int = Field(4, ge=1, le=16)
    #: After filling, read every field back and repair any that did not take. A click
    #: outside a box mid-typing, or a form that rejects a programmatic value, leaves a
    #: field half-filled and the form looks finished until somebody reads it.
    verify_after_fill: bool = True
    headless: bool = False  # keep False so a human can take over on CAPTCHAs
    #: Run with no window until a page actually needs you. Opening one means starting
    #: the browser again, which the profile survives, so you keep whatever you were
    #: signed into.
    hide_browser: bool = False
    #: What to do when a page puts a CAPTCHA or a login wall in the way.
    #:   wait  stop and let you deal with it, which needs a window on screen
    #:   show  open a window and start the posting again in it
    #:   skip  give up on this posting and move to the next one
    challenge_action: Literal["wait", "show", "skip"] = "wait"
    human_gate_mode: Literal["terminal", "api"] = "terminal"

    # ---- Applying by email ----
    #: Some roles never reach an applicant tracking system: the posting says "send your
    #: CV to careers@example.com" and that is the whole process. Off by default, because
    #: sending email is the one thing here that cannot be undone.
    email_apply: bool = False
    #: Send without stopping for you to read the draft. A separate decision from turning
    #: the feature on, and deliberately harder to reach.
    email_auto_send: bool = False
    #: Credentials. These stay in .env and are never written to settings.local.json.
    #: For Gmail this is an app password, not your account password.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    smtp_timeout_s: float = 30.0
    email_from: str = ""            # defaults to smtp_user
    email_reply_to: str = ""        # defaults to the profile's email

    # ---- Job search ----
    # linkedin | urls | greenhouse | lever | ashby | remoteok | himalayas | google
    sources: Annotated[list[str], NoDecode] = ["greenhouse", "ashby", "lever"]
    search_queries: Annotated[list[str], NoDecode] = ["Python Developer"]
    search_location: str = "Remote"
    posted_within_hours: int = 72  # 0 = any time
    max_jobs_per_query: int = 15
    max_applications_per_run: int = 10
    url_list_file: Path = BASE_DIR / "jobs.txt"
    follow_companies: bool = False
    remote_only: bool = False          # keep only postings that are genuinely remote (not hybrid)
    #: intern | junior | mid | senior | lead. Empty means every level.
    seniority_levels: Annotated[list[str], NoDecode] = []
    #: Country names from `countries.COUNTRIES`. Empty means anywhere.
    countries: Annotated[list[str], NoDecode] = []
    #: 0-1. How closely a title must resemble a search term to be worth scoring.
    #: Lower casts a wider net; the LLM still rejects poor fits afterwards.
    title_match_threshold: float = Field(0.45, ge=0.0, le=1.0)

    # ---- Discovery (public board APIs + aggregators) ----
    company_file: Path = BASE_DIR / "companies.json"
    max_jobs_per_company: int = 10     # cap per company board / aggregator feed
    discovery_timeout_s: float = 25.0
    #: A total deadline for one board request. `discovery_timeout_s` is httpx's
    #: per-operation timeout, which never fires on a slow but steady download: one
    #: Lever aggregator board returns 41 MB and takes over two minutes, during which
    #: the scan looks frozen. This bounds the whole request, not each read.
    board_fetch_timeout_s: float = 45.0
    discovery_delay_s: float = 0.35    # pause between API calls, to stay polite
    aggregator_page_size: int = 100
    max_browser_resolutions: int = 8   # aggregator links resolved through the browser per run
    follow_external_apply: bool = True  # LinkedIn "Apply" -> Greenhouse/Lever/Ashby hand-off

    # ---- Browser ----
    user_data_dir: Path = BASE_DIR / ".browser_profile"
    browser_channel: str = "chrome"  # installed Google Chrome; "" = bundled Chromium
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )
    viewport_width: int = 1440
    viewport_height: int = 900
    locale: str = "en-US"
    timezone_id: str = "America/Los_Angeles"
    delay_mean_s: float = 1.4
    delay_std_s: float = 0.5
    start_url: str = "https://www.google.com/robots.txt"  # avoids a blank window on launch
    action_timeout_ms: int = 12_000
    navigation_timeout_ms: int = 45_000
    max_form_steps: int = 15

    # ---- Storage ----
    db_path: Path = BASE_DIR / "applications.db"
    output_dir: Path = BASE_DIR / "output_resumes"
    #: How many pages a generated resume may run to. One page forces heavy trimming and
    #: can cost you whole roles; two is normal for anyone past a few years' experience.
    resume_max_pages: int = Field(2, ge=1, le=3)
    #: Work history is the substance of a CV, so it is only ever dropped as a last resort,
    #: and only when this allows it. Projects and bullet counts are trimmed first.
    resume_may_drop_experience: bool = False
    profile_path: Path = BASE_DIR / "profile.json"
    template_dir: Path = BASE_DIR / "templates"
    log_dir: Path = BASE_DIR / "logs"
    audit_dir: Path = BASE_DIR / "logs" / "applications"

    # ---- Persistence of UI choices ----
    overrides_path: Path = BASE_DIR / "settings.local.json"

    # ---- Admin API ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    admin_token: str = "change-me"

    @field_validator("sources", "search_queries", "seniority_levels", "countries", mode="before")
    @classmethod
    def _csv(cls, value: Any) -> Any:
        return _split_csv(value)

    _mode_ready: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _sync_mode(self) -> "Settings":
        """`fill_mode` is the single source of truth; `auto_submit` mirrors it.

        `AUTO_SUBMIT=true` predates the three-way mode and still appears in `.env` files,
        so on the first pass it is read as a request for auto mode. After that the mode
        decides, and nothing can leave the two disagreeing.
        """
        if not self._mode_ready:
            if self.auto_submit and self.fill_mode == "assisted":
                object.__setattr__(self, "fill_mode", "auto")
            self._mode_ready = True
        object.__setattr__(self, "auto_submit", self.fill_mode == "auto")
        return self

    @property
    def fills_every_field(self) -> bool:
        """False in documents mode, where only the attachments are the agent's job."""
        return self.fill_mode != "documents"

    @field_validator("llm_extra_headers", mode="before")
    @classmethod
    def _headers(cls, value: Any) -> Any:
        """Accept a JSON object from .env, or a dict from the API."""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM_EXTRA_HEADERS must be a JSON object: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("LLM_EXTRA_HEADERS must be a JSON object")
            return {str(k): str(v) for k, v in parsed.items()}
        return value

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.log_dir, self.audit_dir, self.user_data_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- persisted UI choices -----------------------------------------
    def load_overrides(self) -> dict[str, Any]:
        """Apply `settings.local.json` on top of the environment. Returns what was applied."""
        path = Path(self.overrides_path)
        if not path.exists():
            return {}
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Ignoring unreadable %s: %s", path, exc)
            return {}
        applied: dict[str, Any] = {}
        for key, value in (stored or {}).items():
            if key not in PERSISTED_KEYS:
                continue                      # never let the file widen its own scope
            try:
                setattr(self, key, value)
            except Exception as exc:          # a stale key from an older version
                log.warning("Ignoring saved setting %s=%r: %s", key, value, exc)
                continue
            applied[key] = getattr(self, key)
        if applied:
            log.info("Loaded %d saved setting(s) from %s", len(applied), path.name)
        return applied

    def save_overrides(self, keys: Iterable[str]) -> dict[str, Any]:
        """Merge `keys` into `settings.local.json` so they survive a restart."""
        path = Path(self.overrides_path)
        stored: dict[str, Any] = {}
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8")) or {}
            except (json.JSONDecodeError, OSError):
                stored = {}
        for key in keys:
            if key not in PERSISTED_KEYS:
                continue
            value = getattr(self, key)
            stored[key] = str(value) if isinstance(value, Path) else value
        path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return stored

    def clear_overrides(self) -> None:
        """Forget every saved UI choice and fall back to `.env` on the next start."""
        Path(self.overrides_path).unlink(missing_ok=True)

    def saved_overrides(self) -> dict[str, Any]:
        path = Path(self.overrides_path)
        if not path.exists():
            return {}
        try:
            return {k: v for k, v in (json.loads(path.read_text(encoding="utf-8")) or {}).items()
                    if k in PERSISTED_KEYS}
        except (json.JSONDecodeError, OSError):
            return {}


settings = Settings()
settings.load_overrides()
