"""Pluggable LLM backends behind one structured-output interface.

Every caller asks for the same thing: "fill in this Pydantic model". The provider
is responsible for getting strict JSON back from whichever API it wraps.

  gemini  Google GenAI SDK, native `response_schema` structured outputs.
  nvidia  NVIDIA NIM (build.nvidia.com), OpenAI-compatible and free to start.
          NIM models vary in what they support, so the provider walks a ladder:
          json_schema -> nvext.guided_json -> json_object + schema in the prompt,
          remembering which rung worked so later calls go straight there.

Add a provider by implementing `LLMProvider.generate` and registering it in
`build_provider`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any, Optional, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a valid structured response."""


class ModelUnavailable(LLMError):
    """The configured model cannot be used at all, so the whole run should stop.

    Distinct from a transient failure: retrying other postings with the same model
    would fail identically, and would fill the database with useless failures.
    """


def extract_json(text: str) -> str:
    """Pull a JSON object out of a model response that may be fenced or chatty."""
    if not text:
        raise LLMError("empty response from model")
    fenced = FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    if start == -1:
        raise LLMError(f"no JSON object in response: {text[:200]!r}")
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise LLMError(f"unbalanced JSON in response: {text[:200]!r}")


def schema_of(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema with $defs inlined, which strict decoders generally require."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1], {})
                merged = {**inline(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(schema)


class LLMProvider:
    name = "base"

    def __init__(self, settings: Settings):
        self.s = settings
        self.max_retries = max(1, settings.ai_max_retries)

    async def generate(self, schema: type[T], system: str, prompt: str,
                       temperature: float = 0.2) -> T:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None

    def _validate(self, schema: type[T], raw_text: str) -> T:
        return schema.model_validate_json(extract_json(raw_text))


# --------------------------------------------------------------------------
# Google Gemini
# --------------------------------------------------------------------------


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from google import genai  # imported lazily so the nvidia path needs no google deps

        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=nvidia.")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    async def generate(self, schema: type[T], system: str, prompt: str, temperature: float = 0.2) -> T:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        delay = 2.0
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                parsed = response.parsed
                if isinstance(parsed, schema):
                    return parsed
                return self._validate(schema, response.text or "")
            except errors.APIError as exc:
                last = exc
                hard = _gemini_hard_failure(exc)
                if hard:
                    raise LLMError(hard) from exc
                if exc.code in RETRYABLE_STATUS and attempt < self.max_retries:
                    wait = delay + random.uniform(0, 1)
                    log.warning("Gemini %s (attempt %d/%d), retrying in %.1fs",
                                exc.code, attempt, self.max_retries, wait)
                    await asyncio.sleep(wait)
                    delay *= 2
                    continue
                raise LLMError(f"Gemini request failed: {exc}") from exc
            except (ValidationError, LLMError, ValueError) as exc:
                last = exc
                if attempt < self.max_retries:
                    log.warning("Gemini returned invalid JSON (attempt %d/%d): %s",
                                attempt, self.max_retries, exc)
                    await asyncio.sleep(1.0)
                    continue
                raise LLMError(f"Gemini returned invalid JSON: {exc}") from exc
        raise LLMError(f"Gemini failed after {self.max_retries} attempts: {last}")


def _gemini_hard_failure(exc: Any) -> Optional[str]:
    """Permanent problems worth failing fast on, with the fix spelled out."""
    code = getattr(exc, "code", None)
    blob = json.dumps(getattr(exc, "details", None) or {}) + str(exc)
    if code == 429 and ('"quota_limit_value": "0"' in blob or "'quota_limit_value': '0'" in blob):
        return (
            "Gemini rejected the request with quota limit 0: this API key has no Generative "
            "Language API quota (not temporary rate limiting).\n"
            "  1. Check the key at https://aistudio.google.com/apikey\n"
            "  2. Enable the API for its project: "
            "https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com\n"
            "  3. Or switch to the free NVIDIA backend: set LLM_PROVIDER=nvidia and NVIDIA_API_KEY in .env\n"
            f"  Raw error: {str(exc)[:200]}"
        )
    if code in (400, 401, 403) and "API_KEY" in str(exc).upper():
        return (
            "Gemini rejected the API key. Check GEMINI_API_KEY in .env "
            f"(https://aistudio.google.com/apikey). Raw error: {str(exc)[:200]}"
        )
    return None


# --------------------------------------------------------------------------
# NVIDIA NIM (build.nvidia.com) - OpenAI compatible
# --------------------------------------------------------------------------


class NvidiaProvider(LLMProvider):
    """OpenAI-compatible chat completions against https://integrate.api.nvidia.com/v1.

    Get a free key at https://build.nvidia.com (it starts with `nvapi-`).
    """

    name = "nvidia"
    MODES = ("json_schema", "guided_json", "json_object")

    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.nvidia_api_key:
            raise LLMError(
                "NVIDIA_API_KEY is not set. Create a free key at https://build.nvidia.com "
                "(it looks like nvapi-...) and put it in .env."
            )
        self.model = settings.nvidia_model
        self.base_url = settings.nvidia_base_url.rstrip("/")
        self._mode: Optional[str] = None  # remembered once a rung of the ladder works
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=20.0),
            headers={
                "Authorization": f"Bearer {settings.nvidia_api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, mode: str, schema: type[BaseModel], system: str, prompt: str,
                 temperature: float) -> dict[str, Any]:
        js = schema_of(schema)
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": self.s.llm_max_output_tokens,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True, "schema": js},
            }
        elif mode == "guided_json":
            body["nvext"] = {"guided_json": js}
        else:  # json_object: the schema has to travel in the prompt
            body["response_format"] = {"type": "json_object"}
            body["messages"][1]["content"] = (
                f"{prompt}\n\nRespond with ONLY a JSON object matching this JSON Schema. "
                f"No markdown, no commentary.\n\nSCHEMA:\n{json.dumps(js)}"
            )
        return body

    async def generate(self, schema: type[T], system: str, prompt: str, temperature: float = 0.2) -> T:
        modes = [self._mode] if self._mode else list(self.MODES)
        last: Exception | None = None
        for mode in modes:
            delay = 2.0
            for attempt in range(1, self.max_retries + 1):
                try:
                    r = await self._client.post("/chat/completions",
                                                json=self._payload(mode, schema, system, prompt, temperature))
                except httpx.HTTPError as exc:
                    last = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(delay + random.uniform(0, 1))
                        delay *= 2
                        continue
                    break
                if r.status_code in (401, 403):
                    raise LLMError(
                        "NVIDIA rejected the API key. Check NVIDIA_API_KEY in .env "
                        f"(free key at https://build.nvidia.com). Raw error: {r.text[:200]}"
                    )
                if r.status_code in (404, 410):
                    note_model_result("nvidia", self.model, False)
                    verb = "has retired" if r.status_code == 410 else "has not deployed"
                    raise ModelUnavailable(
                        f"NVIDIA {verb} {self.model!r}, so no application can be scored.\n"
                        f"  Its catalogue lists many models it does not actually serve.\n"
                        f"  Open Settings, press 'Check which models work', and pick one that "
                        f"passes, or run: python main.py models --verify"
                    )
                if r.status_code == 400:
                    # Usually this rung of the structured-output ladder is unsupported.
                    log.info("NVIDIA rejected %s mode for %s: %s", mode, self.model, r.text[:160])
                    last = LLMError(r.text[:300])
                    break
                if r.status_code in RETRYABLE_STATUS:
                    last = LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                    if attempt < self.max_retries:
                        wait = delay + random.uniform(0, 1)
                        log.warning("NVIDIA %s (attempt %d/%d), retrying in %.1fs",
                                    r.status_code, attempt, self.max_retries, wait)
                        await asyncio.sleep(wait)
                        delay *= 2
                        continue
                    break
                if r.status_code != 200:
                    last = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                    break
                try:
                    content = r.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, json.JSONDecodeError) as exc:
                    last = LLMError(f"unexpected response shape: {r.text[:200]}")
                    break
                if isinstance(content, list):  # some NIM models return content parts
                    content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
                try:
                    parsed = self._validate(schema, content)
                except (ValidationError, LLMError) as exc:
                    last = exc
                    if attempt < self.max_retries:
                        log.warning("NVIDIA returned invalid JSON in %s mode (attempt %d/%d): %s",
                                    mode, attempt, self.max_retries, str(exc)[:160])
                        await asyncio.sleep(1.0)
                        continue
                    break
                if self._mode != mode:
                    log.info("NVIDIA structured-output mode for %s: %s", self.model, mode)
                    self._mode = mode
                note_model_result("nvidia", self.model, True)
                return parsed
        raise LLMError(
            f"NVIDIA model {self.model!r} could not produce valid structured output "
            f"(tried {', '.join(m for m in modes if m)}). Last error: {str(last)[:300]}"
        )


# --------------------------------------------------------------------------


#: Model ids that cannot hold a chat conversation, so they can never produce the
#: structured JSON this app needs. NVIDIA lists them alongside the chat models.
NON_CHAT_MARKERS = (
    "embed", "embedqa", "rerank", "nvclip", "-parse", "parse-", "nemotron-parse",
    "guard", "safety", "reward", "content-safety", "topic-control",
    "translate", "detector", "calibration", "ocr",
)


def is_chat_model(model_id: str) -> bool:
    """Whether a model can answer a chat completion, and so be used here."""
    lower = (model_id or "").lower()
    return not any(marker in lower for marker in NON_CHAT_MARKERS)


async def check_model(settings: Settings, provider: str, model: str) -> dict[str, Any]:
    """Ask the provider to answer one tiny prompt, so a model choice can be verified.

    A listed model can still be retired or overloaded, and an unlisted one can still
    work, so the only reliable answer is to try it.
    """
    provider = (provider or settings.llm_provider).lower()
    if provider == "nvidia":
        if not settings.nvidia_api_key:
            return {"ok": False, "detail": "NVIDIA_API_KEY is not set in .env"}
        url = settings.nvidia_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {settings.nvidia_api_key}",
                   "Content-Type": "application/json"}
        body = {"model": model, "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 8, "temperature": 0}
    else:
        if not settings.gemini_api_key:
            return {"ok": False, "detail": "GEMINI_API_KEY is not set in .env"}
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
               f"?key={settings.gemini_api_key}")
        headers = {"Content-Type": "application/json"}
        body = {"contents": [{"parts": [{"text": "Reply with OK."}]}],
                "generationConfig": {"maxOutputTokens": 8}}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=body)
    except httpx.HTTPError as exc:
        return {"ok": False, "model": model, "detail": f"Could not reach the provider: {exc}"}
    if r.status_code == 200:
        return {"ok": True, "model": model, "detail": "Answered a test prompt"}
    detail = r.text[:300]
    hint = {
        404: "No such model on this endpoint.",
        410: "This model has been retired by the provider.",
        401: "The API key was rejected.",
        403: "The API key is not allowed to use this model.",
        429: "Rate limited or out of quota.",
        503: "The model exists but is overloaded right now. Try again shortly.",
    }.get(r.status_code, "")
    return {"ok": False, "model": model, "status": r.status_code,
            "detail": f"{hint} {detail}".strip()}


#: Models known to answer, refreshed by `verify_models`. NVIDIA's catalogue lists many
#: models it has not deployed, so "listed" and "usable" are different things.
_VERIFIED: dict[str, dict[str, bool]] = {}


async def verify_models(settings: Settings, provider: str, models: list[str],
                        concurrency: int = 6) -> dict[str, bool]:
    """Send a tiny prompt to each model and remember which ones answered."""
    provider = (provider or settings.llm_provider).lower()
    sem = asyncio.Semaphore(concurrency)

    async def one(model: str) -> tuple[str, bool]:
        async with sem:
            result = await check_model(settings, provider, model)
            # A 503 means it exists but is busy, which still counts as usable.
            return model, bool(result.get("ok") or result.get("status") == 503)

    pairs = await asyncio.gather(*(one(m) for m in models))
    table = _VERIFIED.setdefault(provider, {})
    table.update(dict(pairs))
    working = sum(1 for _, ok in pairs if ok)
    log.info("Verified %d/%d %s models", working, len(pairs), provider)
    return dict(pairs)


def verified_table(provider: str) -> dict[str, bool]:
    return dict(_VERIFIED.get((provider or "").lower(), {}))


def note_model_result(provider: str, model: str, ok: bool) -> None:
    """Record what a real call told us, so the picker learns without extra probing."""
    _VERIFIED.setdefault((provider or "").lower(), {})[model] = ok


PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "nvidia": NvidiaProvider,
}


def build_provider(settings: Settings) -> LLMProvider:
    key = (settings.llm_provider or "gemini").strip().lower()
    if key not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER {key!r}. Known providers: {', '.join(sorted(PROVIDERS))}")
    provider = PROVIDERS[key](settings)
    log.info("LLM provider: %s (%s)", provider.name,
             settings.nvidia_model if key == "nvidia" else settings.gemini_model)
    return provider
