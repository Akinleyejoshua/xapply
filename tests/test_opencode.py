"""OpenCode Zen: a third provider, and the generic OpenAI-compatible client behind it."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from database import Database  # noqa: E402
from llm import (  # noqa: E402
    PROVIDERS,
    LLMError,
    NvidiaProvider,
    OpenAICompatibleProvider,
    OpencodeProvider,
    build_provider,
    llm_headers,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "t.db", output_dir=tmp_path, log_dir=tmp_path,
                    audit_dir=tmp_path, user_data_dir=tmp_path,
                    overrides_path=tmp_path / "settings.local.json",
                    profile_path=ROOT / "profile.example.json", company_file=ROOT / "companies.json")


# ---- registration ----------------------------------------------------------


def test_opencode_is_a_provider() -> None:
    assert "opencode" in PROVIDERS
    assert issubclass(OpencodeProvider, OpenAICompatibleProvider)
    assert issubclass(NvidiaProvider, OpenAICompatibleProvider)


def test_build_provider_needs_the_key(settings) -> None:
    settings.llm_provider = "opencode"
    settings.opencode_api_key = ""
    with pytest.raises(LLMError, match="OPENCODE_API_KEY"):
        build_provider(settings)


def test_active_model_follows_the_provider(settings) -> None:
    settings.gemini_model, settings.nvidia_model, settings.opencode_model = "g", "n/x", "claude-haiku-4-5"
    for provider, expected in (("gemini", "g"), ("nvidia", "n/x"), ("opencode", "claude-haiku-4-5")):
        settings.llm_provider = provider
        assert settings.active_model == expected


def test_opencode_model_is_persisted() -> None:
    from config import PERSISTED_KEYS

    assert "opencode_model" in PERSISTED_KEYS
    assert "llm_extra_headers" in PERSISTED_KEYS
    # the key itself must never be written to the overrides file
    assert "opencode_api_key" not in PERSISTED_KEYS


# ---- custom headers --------------------------------------------------------


def test_extra_headers_are_merged(settings) -> None:
    settings.llm_extra_headers = {"User-Agent": "Mozilla/5.0", "X-Title": "XApply"}
    headers = llm_headers(settings, "sk-abc")
    assert headers["Authorization"] == "Bearer sk-abc"
    assert headers["User-Agent"] == "Mozilla/5.0"
    assert headers["X-Title"] == "XApply"


def test_extra_headers_parse_from_env_json(monkeypatch) -> None:
    monkeypatch.setenv("LLM_EXTRA_HEADERS", '{"User-Agent": "Mozilla/5.0", "X-App": "1"}')
    assert Settings(_env_file=None).llm_extra_headers == {"User-Agent": "Mozilla/5.0", "X-App": "1"}


def test_bad_header_json_is_rejected(monkeypatch) -> None:
    from pydantic import ValidationError

    monkeypatch.setenv("LLM_EXTRA_HEADERS", "not json at all")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_empty_headers_are_fine(monkeypatch) -> None:
    monkeypatch.setenv("LLM_EXTRA_HEADERS", "")
    assert Settings(_env_file=None).llm_extra_headers == {}


# ---- the errors OpenCode actually returns ----------------------------------


@pytest.mark.parametrize("body,expected", [
    ('{"error":{"type":"FreeTierError","message":"OpenCode\'s free tier can only be used from '
     'within OpenCode"}}', "only works inside OpenCode"),
    ('{"error":{"type":"CreditsError","message":"No payment method."}}', "no payment method"),
    ('{"error":{"message":"something else"}}', "OpenCode rejected the request"),
])
def test_opencode_explains_its_own_errors(settings, body, expected) -> None:
    settings.opencode_api_key = "sk-test"
    provider = OpencodeProvider(settings)
    assert expected.lower() in provider.explain_auth_error(403, body).lower()


def test_opencode_distinguishes_a_model_problem_from_a_request_problem() -> None:
    assert OpencodeProvider.is_model_problem("Model is unavailable.") is True
    assert OpencodeProvider.is_model_problem("model not found") is True
    assert OpencodeProvider.is_model_problem("invalid temperature") is False
    # NVIDIA answers 404 only about models, so it treats any of these as a model problem
    assert NvidiaProvider.is_model_problem("anything") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("body,hint", [
    ('{"error":{"type":"FreeTierError","message":"free tier can only be used from within"}}',
     "only works inside"),
    ('{"error":{"type":"CreditsError","message":"No payment method."}}', "no payment method"),
    ('{"error":{"message":"Upstream request failed: Model is unavailable."}}', "unavailable"),
])
async def test_check_model_explains_opencode_refusals(monkeypatch, settings, body, hint) -> None:
    from llm import check_model

    settings.opencode_api_key = "sk-test"
    real = httpx.AsyncClient

    def fake(*a, **kw):
        transport = httpx.MockTransport(lambda r: httpx.Response(403, text=body))
        return real(transport=transport, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", fake)
    out = await check_model(settings, "opencode", "mimo-v2.5-free")
    assert out["ok"] is False
    assert hint.lower() in out["detail"].lower()


# ---- the API and the dashboard ---------------------------------------------


def test_switching_provider_is_never_blocked(monkeypatch, settings) -> None:
    """You have to reach a provider before you can choose one of its models."""
    from fastapi.testclient import TestClient

    from api import create_app

    async def fake_check(s, provider, model):
        return {"ok": False, "status": 401, "detail": "No payment method."}

    monkeypatch.setattr("llm.check_model", fake_check)
    settings.opencode_api_key = "sk-test"
    db = Database(settings.db_path)
    db.init()
    client = TestClient(create_app(settings, db))

    r = client.patch("/api/config", json={"llm_provider": "opencode"})
    assert r.status_code == 200
    assert r.json()["llm_provider"] == "opencode"
    assert r.json()["model_check"]["ok"] is False        # reported, not enforced

    # but picking a model that cannot answer is still refused
    bad = client.patch("/api/config", json={"opencode_model": "mimo-v2.5-free"})
    assert bad.status_code == 422


def test_config_reports_all_three_providers(settings) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    db = Database(settings.db_path)
    db.init()
    body = TestClient(create_app(settings, db)).get("/api/config").json()
    assert body["known_providers"] == ["gemini", "nvidia", "opencode"]
    for key in ("opencode_model", "has_opencode_key", "llm_extra_headers"):
        assert key in body


def test_dashboard_offers_opencode_and_headers() -> None:
    html = (ROOT / "static" / "index.html").read_text()
    assert html.count('<option value="opencode">') == 2      # dashboard and settings
    assert 'id="cfgHeaders"' in html and 'id="btnSaveHeaders"' in html
    assert "MODEL_KEY" in html and "opencode_model" in html


def test_opencode_key_is_not_in_a_committed_file() -> None:
    for name in (".env.example", "README.md"):
        text = (ROOT / name).read_text()
        assert "sk-UCZK" not in text, f"{name} contains a real OpenCode key"
