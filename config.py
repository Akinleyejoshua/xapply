"""Central configuration.

Every value can be overridden through environment variables or a `.env` file
(see `.env.example`). Names are case-insensitive, so `AUTO_SUBMIT=true` sets
`Settings.auto_submit`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


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
    )

    # ---- AI ----
    llm_provider: str = "gemini"  # gemini | nvidia

    # Google Gemini (https://aistudio.google.com/apikey)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # NVIDIA NIM (free key at https://build.nvidia.com, OpenAI-compatible)
    nvidia_api_key: str = ""
    nvidia_model: str = "nvidia/nemotron-3-super-120b-a12b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    llm_timeout_s: float = 180.0
    llm_max_output_tokens: int = 8192
    match_threshold: int = Field(65, ge=0, le=100)
    ai_max_retries: int = 4
    ai_min_confidence: float = 0.55  # below this a live AI form answer is escalated to a human

    @property
    def active_model(self) -> str:
        return self.nvidia_model if self.llm_provider.lower() == "nvidia" else self.gemini_model

    # ---- Execution mode ----
    auto_submit: bool = False  # AUTO_SUBMIT=true -> bot clicks Submit itself
    headless: bool = False  # keep False so a human can take over on CAPTCHAs
    human_gate_mode: Literal["terminal", "api"] = "terminal"

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

    # ---- Discovery (public board APIs + aggregators) ----
    company_file: Path = BASE_DIR / "companies.json"
    max_jobs_per_company: int = 10     # cap per company board / aggregator feed
    discovery_timeout_s: float = 25.0
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
    profile_path: Path = BASE_DIR / "profile.json"
    template_dir: Path = BASE_DIR / "templates"
    log_dir: Path = BASE_DIR / "logs"
    audit_dir: Path = BASE_DIR / "logs" / "applications"

    # ---- Admin API ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    admin_token: str = "change-me"

    @field_validator("sources", "search_queries", "seniority_levels", mode="before")
    @classmethod
    def _csv(cls, value: Any) -> Any:
        return _split_csv(value)

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.log_dir, self.audit_dir, self.user_data_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
