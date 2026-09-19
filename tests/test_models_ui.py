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


# ---- the picker must never change the model by itself ----------------------
#
# Regression: making the dropdown searchable introduced a blur handler that saved
# whatever text was in the box. Opening the list and clicking away selected
# "01-ai/yi-large", the first entry alphabetically, which NVIDIA lists but has never
# deployed. Every posting in the next run then failed with a 404.


def _picker_source() -> str:
    html = (ROOT / "static" / "index.html").read_text()
    start = html.index("class ModelPicker")
    return html[start:html.index("const MODEL_PICKERS")]


def test_picker_only_saves_a_deliberate_choice() -> None:
    src = _picker_source()
    assert "this.dirty = false" in src, "the picker must track whether a choice was made"
    commit = src[src.index("commit(){"):]
    commit = commit[:commit.index("\n  onKey")]
    assert "if(!this.dirty" in commit, "blurring without choosing must not save"
    assert "!this.looksLikeModelId(v)" in commit, "a half-typed filter must not be saved"
    assert "setValue(current, true)" in commit, "the current model must be put back"


def test_picker_accepts_an_unlisted_model_id() -> None:
    """Typing a full id the provider does not list must still work."""
    src = _picker_source()
    check = src[src.index("looksLikeModelId(v){"):]
    check = check[:check.index("\n  commit()")]
    assert "data.all.includes(v)" in check       # anything in the list
    assert "/" in check                           # or anything shaped like vendor/model


def test_changing_provider_does_not_save_a_model() -> None:
    html = (ROOT / "static" / "index.html").read_text()
    for handler in ("$('runProvider').onchange", "$('cfgProvider').onchange"):
        block = html[html.index(handler):]
        block = block[:block.index("};") + 2]
        assert "llm_provider: p" in block
        assert "nvidia_model" not in block and "gemini_model" not in block, \
            f"{handler} must change the provider only"


def test_picker_labels_verified_and_undeployed_models() -> None:
    src = _picker_source()
    assert "data.verified" in src and "data.broken" in src
    assert "works" in src and "not deployed" in src


# ---- a model that cannot be used stops the run -----------------------------


def test_model_unavailable_is_its_own_error() -> None:
    from llm import LLMError, ModelUnavailable

    assert issubclass(ModelUnavailable, LLMError)


def test_pipeline_checks_the_model_before_any_posting() -> None:
    src = (ROOT / "pipeline.py").read_text()
    run = src[src.index("async def run(self"):]
    run = run[:run.index("async def analyze_only")]
    assert "await self.preflight()" in run
    # and the check has to come before the browser and the sources
    assert run.index("await self.preflight()") < run.index("StealthBrowser")
    # a mid-run failure must stop rather than record one failure per posting
    assert "except ModelUnavailable:" in run
    assert "raise" in run[run.index("except ModelUnavailable:"):]


def test_a_404_raises_model_unavailable_not_a_generic_error() -> None:
    src = (ROOT / "llm.py").read_text()
    assert "raise ModelUnavailable(" in src
    block = src[src.index("if r.status_code in (404, 410)"):]
    block = block[:block.index("if r.status_code in RETRYABLE_STATUS")]
    assert "has not deployed" in block and "has retired" in block


def test_api_reports_a_model_blocker_at_run_level() -> None:
    src = (ROOT / "api.py").read_text()
    assert "app.state.blocker" in src
    assert '"/admin/clear-blocker"' in src
    html = (ROOT / "static" / "index.html").read_text()
    assert "st.blocker" in html and "fixModel()" in html


def test_verify_endpoint_exists() -> None:
    src = (ROOT / "api.py").read_text()
    assert '"/api/models/verify"' in src
    html = (ROOT / "static" / "index.html").read_text()
    assert 'id="btnVerifyModels"' in html


@pytest.mark.asyncio
async def test_verify_models_records_what_it_learned(monkeypatch, settings) -> None:
    from llm import verified_table, verify_models

    async def fake_check(s, provider, model):
        return {"ok": model.endswith("-good"), "status": 200 if model.endswith("-good") else 404}

    monkeypatch.setattr("llm.check_model", fake_check)
    out = await verify_models(settings, "nvidia", ["a/x-good", "b/y-bad"])
    assert out == {"a/x-good": True, "b/y-bad": False}
    assert verified_table("nvidia")["a/x-good"] is True
    assert verified_table("nvidia")["b/y-bad"] is False


@pytest.mark.asyncio
async def test_a_busy_model_still_counts_as_usable(monkeypatch, settings) -> None:
    """503 means deployed but overloaded, which is a wait rather than a wrong choice."""
    from llm import verify_models

    async def fake_check(s, provider, model):
        return {"ok": False, "status": 503, "detail": "overloaded"}

    monkeypatch.setattr("llm.check_model", fake_check)
    assert (await verify_models(settings, "nvidia", ["busy/model"]))["busy/model"] is True


#: What a real credential looks like, as opposed to a comment describing one.
SECRET_SHAPES = (
    r"nvapi-[A-Za-z0-9_\-]{20,}",      # NVIDIA
    r"AIza[A-Za-z0-9_\-]{20,}",        # Google
    r"sk-[A-Za-z0-9_\-]{20,}",         # OpenAI style
)


@pytest.mark.parametrize("path", [".env.example", "README.md", "companies.json", "profile.example.json"])
def test_committed_files_hold_no_secrets(path) -> None:
    """These files are meant to be committed, so a real key must never reach them."""
    import re

    text = (ROOT / path).read_text()
    for shape in SECRET_SHAPES:
        found = re.search(shape, text)
        assert not found, f"{path} contains something shaped like a real key: {found.group(0)[:12]}..."
    for line in text.splitlines():
        m = re.match(r"^\s*(GEMINI_API_KEY|NVIDIA_API_KEY|ADMIN_TOKEN)\s*=\s*(\S+)", line)
        if m:
            assert m.group(2) in ("", "change-me"), f"a real value is in {path}: {m.group(1)}"
