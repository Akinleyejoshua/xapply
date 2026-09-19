"""Choosing an LLM: which models are usable, and how the picker exposes them."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from database import Database  # noqa: E402
from llm import NON_CHAT_MARKERS, check_model, is_chat_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "t.db", output_dir=tmp_path, log_dir=tmp_path,
                    audit_dir=tmp_path, user_data_dir=tmp_path,
                    overrides_path=tmp_path / "settings.local.json",
                    profile_path=ROOT / "profile.example.json", company_file=ROOT / "companies.json")


# ---- which models can actually hold a conversation -------------------------


@pytest.mark.parametrize("model_id,usable", [
    ("nvidia/nemotron-3-super-120b-a12b", True),
    ("openai/gpt-oss-20b", True),
    ("z-ai/glm-5.3", True),
    ("mistralai/mistral-large-2-instruct", True),
    # NVIDIA lists these next to the chat models, and they cannot answer a prompt
    ("nvidia/nv-embedqa-mistral-7b-v2", False),
    ("nvidia/embed-qa-4", False),
    ("nvidia/llama-3.1-nemoguard-8b-content-safety", False),
    ("nvidia/nemotron-4-340b-reward", False),
    ("nvidia/nemotron-parse-2.0", False),
    ("nvidia/riva-translate-4b-instruct", False),
    ("nvidia/ai-synthetic-video-detector", False),
    ("nvidia/nvclip", False),
])
def test_is_chat_model(model_id, usable) -> None:
    assert is_chat_model(model_id) is usable


def test_non_chat_markers_are_lowercase() -> None:
    assert all(m == m.lower() for m in NON_CHAT_MARKERS)


# ---- verifying a model without guessing ------------------------------------


def _stub(status: int, body: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body or {"error": "nope"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,ok,hint", [
    (200, True, "Answered a test prompt"),
    (404, False, "No such model"),
    (410, False, "retired"),
    (401, False, "rejected"),
    (429, False, "Rate limited"),
    (503, False, "overloaded"),
])
async def test_check_model_explains_each_outcome(monkeypatch, settings, status, ok, hint) -> None:
    settings.nvidia_api_key = "nvapi-test"
    real = httpx.AsyncClient

    def fake(*a, **kw):
        return real(transport=_stub(status, {"choices": []} if status == 200 else None), **kw)

    monkeypatch.setattr(httpx, "AsyncClient", fake)
    out = await check_model(settings, "nvidia", "some/model")
    assert out["ok"] is ok
    assert hint.lower() in out["detail"].lower()


@pytest.mark.asyncio
async def test_check_model_needs_a_key(settings) -> None:
    settings.nvidia_api_key = ""
    out = await check_model(settings, "nvidia", "x/y")
    assert out["ok"] is False and "NVIDIA_API_KEY" in out["detail"]

    settings.gemini_api_key = ""
    out = await check_model(settings, "gemini", "x")
    assert out["ok"] is False and "GEMINI_API_KEY" in out["detail"]


def test_api_model_test_endpoint(monkeypatch, settings) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    async def fake_check(s, provider, model):
        return {"ok": model == "good/model", "model": model, "detail": "stubbed"}

    monkeypatch.setattr("llm.check_model", fake_check)
    db = Database(settings.db_path)
    db.init()
    client = TestClient(create_app(settings, db))
    assert client.post("/api/models/test", json={"model": "good/model"}).json()["ok"] is True
    assert client.post("/api/models/test", json={"model": "bad/model"}).json()["ok"] is False
    assert client.post("/api/models/test", json={}).status_code == 422


def test_models_endpoint_splits_chat_from_the_rest(monkeypatch, settings) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    payload = {"data": [{"id": "nvidia/nemotron-3-super-120b-a12b"},
                        {"id": "openai/gpt-oss-20b"},
                        {"id": "nvidia/embed-qa-4"},
                        {"id": "nvidia/nemotron-4-340b-reward"}]}
    real = httpx.AsyncClient

    def fake(*a, **kw):
        return real(transport=_stub(200, payload), **kw)

    monkeypatch.setattr(httpx, "AsyncClient", fake)
    db = Database(settings.db_path)
    db.init()
    body = TestClient(create_app(settings, db)).get("/api/models?provider=nvidia").json()
    assert len(body["models"]) == 4
    assert body["chat_models"] == ["nvidia/nemotron-3-super-120b-a12b", "openai/gpt-oss-20b"]
    assert "not listed" in body["note"]


# ---- the picker ------------------------------------------------------------


def test_dashboard_has_a_searchable_picker() -> None:
    html = (ROOT / "static" / "index.html").read_text()
    for marker in ("class ModelPicker", 'id="runModelList"', 'id="cfgModelList"',
                   'id="btnTestModel"', "MODEL_PICKERS", "combo-list"):
        assert marker in html, marker
    # it must be a text input, so an unlisted id can still be typed
    assert '<input id="cfgModel"' in html and "<select id=\"cfgModel\"" not in html
    assert '<input id="runModel"' in html and "<select id=\"runModel\"" not in html


def test_picker_opens_unfiltered() -> None:
    """Clicking the field must show every model, not filter by the one already chosen."""
    html = (ROOT / "static" / "index.html").read_text()
    picker = html[html.index("class ModelPicker"):]
    picker = picker[:picker.index("const MODEL_PICKERS")]
    assert "this.typed = false" in picker
    assert "this.typed ? this.input.value" in picker
