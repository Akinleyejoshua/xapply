"""Settings chosen in the web UI must survive a tab switch, a reload and a restart."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PERSISTED_KEYS, Settings  # noqa: E402
from database import Database  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "t.db", output_dir=tmp_path, log_dir=tmp_path,
                    audit_dir=tmp_path, user_data_dir=tmp_path, template_dir=ROOT / "templates",
                    overrides_path=tmp_path / "settings.local.json",
                    profile_path=ROOT / "profile.example.json", company_file=ROOT / "companies.json")


def _client(settings: Settings):
    from fastapi.testclient import TestClient

    from api import create_app

    db = Database(settings.db_path)
    db.init()
    return TestClient(create_app(settings, db))


# ---- the settings file ----------------------------------------------------


def test_save_and_reload_in_a_new_process(settings: Settings) -> None:
    settings.llm_provider = "nvidia"
    settings.nvidia_model = "openai/gpt-oss-20b"
    settings.seniority_levels = ["mid", "senior"]
    settings.save_overrides(["llm_provider", "nvidia_model", "seniority_levels"])

    fresh = Settings(overrides_path=settings.overrides_path)     # simulates a restart
    assert fresh.llm_provider != "nvidia" or fresh.nvidia_model != "openai/gpt-oss-20b"
    applied = fresh.load_overrides()
    assert applied["llm_provider"] == "nvidia"
    assert fresh.active_model == "openai/gpt-oss-20b"
    assert fresh.seniority_levels == ["mid", "senior"]


def test_overrides_file_cannot_widen_its_own_scope(settings: Settings) -> None:
    """A hand-edited file must not be able to set paths, tokens or anything else."""
    settings.overrides_path.write_text(json.dumps({
        "match_threshold": 42,
        "admin_token": "stolen",                 # not persistable
        "db_path": "/tmp/somewhere-else.db",     # not persistable
        "gemini_api_key": "leaked",              # not persistable
    }), encoding="utf-8")
    applied = settings.load_overrides()
    assert applied == {"match_threshold": 42}
    assert settings.admin_token != "stolen"
    assert settings.gemini_api_key != "leaked"
    assert "somewhere-else" not in str(settings.db_path)


def test_bad_values_are_skipped_not_fatal(settings: Settings) -> None:
    settings.overrides_path.write_text(json.dumps({
        "match_threshold": 9000,          # out of range
        "llm_provider": "nvidia",         # fine
        "retired_setting": True,          # from an older version
    }), encoding="utf-8")
    applied = settings.load_overrides()
    assert applied == {"llm_provider": "nvidia"}
    assert settings.match_threshold <= 100


def test_unreadable_file_is_ignored(settings: Settings) -> None:
    settings.overrides_path.write_text("{ this is not json", encoding="utf-8")
    assert settings.load_overrides() == {}


def test_clear_overrides(settings: Settings) -> None:
    settings.match_threshold = 90
    settings.save_overrides(["match_threshold"])
    assert settings.overrides_path.exists()
    settings.clear_overrides()
    assert not settings.overrides_path.exists()
    assert settings.saved_overrides() == {}


def test_validate_assignment_rejects_bad_values(settings: Settings) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        settings.match_threshold = 900


# ---- through the API ------------------------------------------------------


def test_patch_config_persists_to_disk(settings: Settings) -> None:
    client = _client(settings)
    assert client.get("/api/config").json()["saved"] == []

    body = client.patch("/api/config", json={
        "llm_provider": "nvidia", "nvidia_model": "openai/gpt-oss-20b",
        "sources": ["greenhouse", "lever"], "seniority_levels": ["senior"],
        "remote_only": True, "match_threshold": 75,
    }).json()
    assert body["active_model"] == "openai/gpt-oss-20b"
    assert set(body["saved"]) == {"llm_provider", "nvidia_model", "sources",
                                  "seniority_levels", "remote_only", "match_threshold"}

    on_disk = json.loads(settings.overrides_path.read_text())
    assert on_disk["nvidia_model"] == "openai/gpt-oss-20b"
    assert on_disk["sources"] == ["greenhouse", "lever"]

    restarted = Settings(overrides_path=settings.overrides_path)
    restarted.load_overrides()
    assert restarted.active_model == "openai/gpt-oss-20b"
    assert restarted.remote_only is True


def test_patch_config_rejects_bad_values_without_saving(settings: Settings) -> None:
    client = _client(settings)
    assert client.patch("/api/config", json={"match_threshold": 900}).status_code == 422
    assert client.patch("/api/config", json={"llm_provider": "hal9000"}).status_code == 422
    assert client.patch("/api/config", json={"seniority_levels": ["wizard"]}).status_code == 422
    assert not settings.overrides_path.exists()


def test_reset_config_restores_env_defaults(settings: Settings) -> None:
    client = _client(settings)
    before = client.get("/api/config").json()
    client.patch("/api/config", json={"match_threshold": 99, "sources": ["ashby"],
                                      "seniority_levels": ["lead"]})
    assert client.get("/api/config").json()["match_threshold"] == 99

    after = client.post("/api/config/reset").json()
    assert after["saved"] == []
    assert after["match_threshold"] == before["match_threshold"]
    assert after["sources"] == before["sources"]
    assert not settings.overrides_path.exists()


def test_run_status_reports_the_saved_model(settings: Settings) -> None:
    """The header badge polls /api/run, so it has to agree with /api/config."""
    client = _client(settings)
    client.patch("/api/config", json={"llm_provider": "nvidia", "nvidia_model": "nv/demo"})
    run = client.get("/api/run").json()
    cfg = client.get("/api/config").json()
    assert run["provider"] == cfg["llm_provider"] == "nvidia"
    assert run["model"] == cfg["active_model"] == "nv/demo"


def test_persisted_keys_hold_no_secrets_or_paths() -> None:
    forbidden = ("key", "token", "password", "secret", "path", "dir", "host", "port", "url")
    leaky = [k for k in PERSISTED_KEYS if any(word in k for word in forbidden)]
    assert leaky == [], f"these would be written to settings.local.json: {leaky}"


def test_dashboard_renders_from_one_store() -> None:
    """Regression: each page used to render its own copy of the settings."""
    html = (ROOT / "static" / "index.html").read_text()
    assert "function renderConfig()" in html
    assert "async function saveConfig(" in html
    assert "/api/config/reset" in html
    # show() must re-render from the store on every tab switch
    show = html[html.index("function show(name)"):]
    assert "renderConfig();" in show[:600]
    # the old per-page renderers must be gone
    for stale in ("function renderSources(", "function renderLevels(", "async function loadSettings("):
        assert stale not in html, stale
