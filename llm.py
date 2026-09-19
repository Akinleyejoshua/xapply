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
import difflib
import json
import logging
import random
import re
import ssl
from typing import Any, Optional, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
#: Connection faults, as opposed to anything the provider actually said. An SSL record
#: error is not an httpx exception, so it escaped the handler and ended the run.
TRANSPORT_FAULTS = (httpx.HTTPError, ssl.SSLError, ConnectionError, OSError)
#: How many times a model check re-dials before giving up on the network.
CHECK_ATTEMPTS = 3
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a valid structured response."""


class ModelUnavailable(LLMError):
    """The configured model cannot be used at all, so the whole run should stop.

    Distinct from a transient failure: retrying other postings with the same model
    would fail identically, and would fill the database with useless failures.
    """


#: How many opening braces are worth trying as the start of the real object.
MAX_JSON_STARTS = 4
#: How many times to step back to the previous field when closing a cut-off object.
MAX_REPAIR_STEPS = 8


def _balanced_object(text: str, start: int) -> Optional[str]:
    """The complete JSON object beginning at `start`, or None if it never closes."""
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
    return None


def _unfinished(fragment: str) -> tuple[int, bool]:
    """How many braces a fragment leaves open, and whether it stops inside a string."""
    depth, in_string, escape = 0, False, False
    for ch in fragment:
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
    return depth, in_string


def repair_truncated_json(text: str, start: int) -> Optional[str]:
    """Close an object the model stopped writing, when nothing was left half-said.

    Models sometimes stop mid-response. If the break falls between fields, the values
    already written are whole and the object is worth keeping: throwing it away costs
    a retry, and four failed retries leave the question unanswered altogether.

    If the break falls inside a string the answer itself is half a sentence, so it is
    refused. Closing the quote would put an unfinished sentence into an application,
    which is worse than asking again.
    """
    fragment = text[start:].rstrip()
    for _ in range(MAX_REPAIR_STEPS):
        candidate = fragment.rstrip().rstrip(",").rstrip()
        depth, in_string = _unfinished(candidate)
        if in_string or depth <= 0:
            return None
        closed = candidate + "}" * depth
        try:
            parsed = json.loads(closed)
        except json.JSONDecodeError:
            parsed = None
        # An empty object is not a repair, it is the wreckage of one. Stepping back far
        # enough always reaches "{}", which would parse and answer nothing.
        if isinstance(parsed, dict) and parsed:
            return closed
        # Step back over the field that was cut off and try again without it.
        cut = max(candidate.rfind(","), candidate.rfind("{"))
        if cut <= 0:
            return None
        fragment = candidate[:cut]
    return None


def extract_json(text: str) -> str:
    """Pull a JSON object out of a model response that may be fenced, chatty or cut short."""
    if not text:
        raise LLMError("empty response from model")
    fenced = FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    starts = [i for i, ch in enumerate(text) if ch == "{"][:MAX_JSON_STARTS]
    if not starts:
        raise LLMError(f"no JSON object in response: {text[:200]!r}")

    # A stray brace before the real object is common, so every opening brace is tried
    # as a starting point. Each one is fully exhausted before moving on, so that an
    # outer object that merely needs closing wins over an inner one that happens to
    # be complete.
    for start in starts:
        found = _balanced_object(text, start)
        if found is not None:
            return found
        repaired = repair_truncated_json(text, start)
        if repaired is not None:
            log.debug("closed a truncated JSON response from the model")
            return repaired
    raise LLMError(f"unbalanced JSON in response: {text[:200]!r}")


def llm_headers(settings: Settings, api_key: str) -> dict[str, str]:
    """Standard headers plus whatever the user configured, which wins on a clash."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    headers.update(settings.llm_extra_headers or {})
    return headers


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


class OpenAICompatibleProvider(LLMProvider):
    """Any service that speaks the OpenAI chat-completions API.

    Providers differ in how they enforce a JSON shape, so `MODES` is walked in order
    until one works and the winner is remembered for the rest of the session.
    """

    name = "openai-compatible"
    MODES = ("json_schema", "json_object")
    key_setting = ""
    model_setting = ""
    base_url_setting = ""
    key_help = ""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        api_key = getattr(settings, self.key_setting, "")
        if not api_key:
            raise LLMError(f"{self.key_setting.upper()} is not set. {self.key_help}")
        self.model = getattr(settings, self.model_setting)
        self.base_url = getattr(settings, self.base_url_setting).rstrip("/")
        self._mode: Optional[str] = None  # remembered once a rung of the ladder works
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=20.0),
            headers=llm_headers(settings, api_key),
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
                    raise LLMError(self.explain_auth_error(r.status_code, r.text))
                if r.status_code in (400, 404, 410) and self.is_model_problem(r.text):
                    note_model_result(self.name, self.model, False)
                    raise ModelUnavailable(self.explain_model_error(r.status_code, r.text))
                if r.status_code == 400:
                    # Usually this rung of the structured-output ladder is unsupported.
                    log.info("%s rejected %s mode for %s: %s", self.name, mode, self.model, r.text[:160])
                    last = LLMError(r.text[:300])
                    break
                if r.status_code in RETRYABLE_STATUS:
                    last = LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                    if attempt < self.max_retries:
                        wait = delay + random.uniform(0, 1)
                        log.warning("%s %s (attempt %d/%d), retrying in %.1fs",
                                    self.name, r.status_code, attempt, self.max_retries, wait)
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
                note_model_result(self.name, self.model, True)
                return parsed
        raise LLMError(
            f"{self.name} model {self.model!r} could not produce valid structured output "
            f"(tried {', '.join(m for m in modes if m)}). Last error: {str(last)[:300]}"
        )

    # ---- wording each provider can sharpen -----------------------------
    @staticmethod
    def is_model_problem(body: str) -> bool:
        """Whether an error is about the model rather than the request."""
        return True

    def explain_auth_error(self, status: int, body: str) -> str:
        return (f"{self.name} rejected the request ({status}). Check "
                f"{self.key_setting.upper()} in .env. Raw error: {body[:220]}")

    def explain_model_error(self, status: int, body: str) -> str:
        verb = "has retired" if status == 410 else "does not serve"
        return (f"{self.name} {verb} {self.model!r}, so nothing can be scored.\n"
                f"  Open Settings, press 'Check which models work', and pick one that passes,\n"
                f"  or run: python main.py models --verify\n  Raw error: {body[:200]}")


class NvidiaProvider(OpenAICompatibleProvider):
    """NVIDIA NIM. Free key at https://build.nvidia.com, starting with `nvapi-`."""

    name = "nvidia"
    MODES = ("json_schema", "guided_json", "json_object")
    key_setting = "nvidia_api_key"
    model_setting = "nvidia_model"
    base_url_setting = "nvidia_base_url"
    key_help = "Create a free key at https://build.nvidia.com and put it in .env."

    def _payload(self, mode, schema, system, prompt, temperature):
        body = super()._payload(mode, schema, system, prompt, temperature)
        if mode == "guided_json":
            body.pop("response_format", None)
            body["nvext"] = {"guided_json": schema_of(schema)}
        return body

    def explain_auth_error(self, status: int, body: str) -> str:
        return ("NVIDIA rejected the API key. Check NVIDIA_API_KEY in .env "
                f"(free key at https://build.nvidia.com). Raw error: {body[:200]}")

    def explain_model_error(self, status: int, body: str) -> str:
        verb = "has retired" if status == 410 else "has not deployed"
        return (f"NVIDIA {verb} {self.model!r}, so no application can be scored.\n"
                f"  Its catalogue lists many models it does not actually serve.\n"
                f"  Open Settings, press 'Check which models work', and pick one that passes,\n"
                f"  or run: python main.py models --verify")


class OpencodeProvider(OpenAICompatibleProvider):
    """OpenCode Zen, a gateway to Claude, GPT, Gemini, DeepSeek, Qwen and others.

    Two things to know about it. Models whose id ends in `-free` are reserved for
    OpenCode's own client and answer 403 to anything else, and the rest need a payment
    method on the workspace. Both are reported plainly rather than retried.
    """

    name = "opencode"
    MODES = ("json_schema", "json_object")
    key_setting = "opencode_api_key"
    model_setting = "opencode_model"
    base_url_setting = "opencode_base_url"
    key_help = "Create one at https://opencode.ai and put it in .env as OPENCODE_API_KEY."

    @staticmethod
    def is_model_problem(body: str) -> bool:
        return "unavailable" in body.lower() or "not found" in body.lower()

    def explain_auth_error(self, status: int, body: str) -> str:
        low = body.lower()
        if "freetiererror" in low or "free tier can only be used" in low:
            return (f"{self.model!r} is on OpenCode's free tier, which only works inside "
                    f"OpenCode's own client and refuses other applications.\n"
                    f"  Pick a model whose id does not end in -free.")
        if "creditserror" in low or "no payment method" in low:
            return (f"OpenCode has no payment method on this workspace, so {self.model!r} "
                    f"cannot be used.\n  Add one at https://opencode.ai, or switch provider "
                    f"to NVIDIA, which is free.")
        return (f"OpenCode rejected the request ({status}). Check OPENCODE_API_KEY in .env. "
                f"Raw error: {body[:200]}")

    def explain_model_error(self, status: int, body: str) -> str:
        return (f"OpenCode cannot serve {self.model!r} right now.\n"
                f"  Raw error: {body[:200]}")


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
    if provider in ("nvidia", "opencode"):
        key_attr = f"{provider}_api_key"
        api_key = getattr(settings, key_attr, "")
        if not api_key:
            return {"ok": False, "detail": f"{key_attr.upper()} is not set in .env"}
        base = getattr(settings, f"{provider}_base_url").rstrip("/")
        url = base + "/chat/completions"
        headers = llm_headers(settings, api_key)
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
    # A connection can break mid-handshake or mid-stream. Those faults are not a verdict
    # on the model, and one of them used to end the whole run with a stack trace, so each
    # attempt gets a fresh connection.
    last: Optional[Exception] = None
    for attempt in range(1, CHECK_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=body)
            break
        except TRANSPORT_FAULTS as exc:
            last = exc
            log.warning("Model check attempt %d/%d could not reach %s: %s",
                        attempt, CHECK_ATTEMPTS, provider, exc)
            if attempt < CHECK_ATTEMPTS:
                await asyncio.sleep(1.5 * attempt)
    else:
        return {"ok": False, "model": model, "transient": True,
                "detail": f"Could not reach {provider} after {CHECK_ATTEMPTS} attempts: {last}. "
                          f"This looks like the network rather than the model."}
    if r.status_code == 200:
        return {"ok": True, "model": model, "detail": "Answered a test prompt"}
    detail = r.text[:300]
    low = detail.lower()
    if "modelerror" in low or "is not supported" in low:
        hint = "This endpoint does not serve that model."
    elif "free tier can only be used" in low or "freetiererror" in low:
        hint = ("This model is on the provider's free tier, which only works inside their own "
                "client and refuses other applications. No header changes that.")
    elif "no payment method" in low or "creditserror" in low:
        hint = "The account has no payment method, so this model cannot be billed."
    elif "model is unavailable" in low:
        hint = "The provider reports this model as unavailable right now."
    else:
        hint = {
            404: "No such model on this endpoint.",
            410: "This model has been retired by the provider.",
            401: "The API key was rejected.",
            403: "The API key is not allowed to use this model.",
            429: "Rate limited or out of quota.",
            503: "The model exists but is overloaded right now. Try again shortly.",
        }.get(r.status_code, "")
    out = {"ok": False, "model": model, "status": r.status_code,
           "detail": f"{hint} {detail}".strip()}
    if r.status_code == 404 or "not supported" in low or "no such model" in hint.lower():
        near = await nearest_models(settings, provider, model)
        if near:
            out["suggestions"] = near
            out["detail"] += " Closest ids it does serve: " + ", ".join(near) + "."
        else:
            total = len(await list_model_ids(settings, provider))
            out["detail"] += (f" Nothing similar among the {total} models {provider} serves, so "
                              f"this one is not reachable through its API.")
    return out


async def list_model_ids(settings: Settings, provider: str) -> list[str]:
    """The model ids a provider's own endpoint reports."""
    provider = (provider or "").lower()
    try:
        if provider == "nvidia":
            url = settings.nvidia_base_url.rstrip("/") + "/models"
            headers = {"Accept": "application/json"}
        elif provider == "opencode":
            url = settings.opencode_base_url.rstrip("/") + "/models"
            headers = llm_headers(settings, settings.opencode_api_key)
        else:
            url = "https://generativelanguage.googleapis.com/v1beta/models"
            headers = {}
        async with httpx.AsyncClient(timeout=25) as client:
            params = {"key": settings.gemini_api_key} if provider == "gemini" else None
            r = await client.get(url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        log.debug("could not list %s models: %s", provider, exc)
        return []
    if provider == "gemini":
        return [m["name"].split("/")[-1] for m in data.get("models", [])]
    return [m["id"] for m in data.get("data", [])]


async def nearest_models(settings: Settings, provider: str, wanted: str, limit: int = 3) -> list[str]:
    """Ids that look like the one asked for.

    A provider's website can name a model differently from its API, so a 404 is often a
    spelling difference rather than a missing model. Showing the closest real ids turns
    "I can see it on their site" into an answerable question.
    """
    ids = await list_model_ids(settings, provider)
    if not ids:
        return []
    target = re.sub(r"[^a-z0-9]", "", (wanted or "").lower())
    scored = []
    for candidate in ids:
        flat = re.sub(r"[^a-z0-9]", "", candidate.lower())
        ratio = difflib.SequenceMatcher(None, target, flat).ratio()
        tail = wanted.split("/")[-1].lower()
        if tail and tail in candidate.lower():
            ratio = max(ratio, 0.9)
        scored.append((ratio, candidate))
    scored.sort(reverse=True)
    return [c for score, c in scored[:limit] if score >= 0.45]


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
    "opencode": OpencodeProvider,
}


def build_provider(settings: Settings) -> LLMProvider:
    key = (settings.llm_provider or "gemini").strip().lower()
    if key not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER {key!r}. Known providers: {', '.join(sorted(PROVIDERS))}")
    provider = PROVIDERS[key](settings)
    log.info("LLM provider: %s (%s)", provider.name, settings.active_model)
    return provider
