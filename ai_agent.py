"""Gemini-powered analysis via the Google GenAI SDK.

Two responsibilities:
  1. `analyze_job`            -> JobAnalysis  (match score, tailored resume content,
                                                predicted screening answers)
  2. `answer_form_question`   -> FieldAnswer   (live answer for a form field the
                                                predictions did not cover)

Both use Pydantic `response_schema` so Gemini returns strict JSON (no markdown
fences, no broken syntax) that is validated before anything else touches it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Literal, Optional, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

from config import Settings
from config import settings as default_settings
from models import JobPosting

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# --------------------------------------------------------------------------
# Structured output schemas
# NOTE: Gemini's schema subset does not support free-form dict/additionalProperties,
# so the "answers" map is modelled as a list of question/answer pairs and exposed
# as a dict through `JobAnalysis.answers_map`.
# --------------------------------------------------------------------------


class ScreeningAnswer(BaseModel):
    question: str = Field(
        description="Screening question, phrased the way an application form would ask it."
    )
    answer: str = Field(
        description="Short literal answer. Numbers as digits (e.g. '6'), yes/no as 'Yes'/'No'. "
        "Use 'UNKNOWN' if the master profile does not contain the information."
    )


class TailoredBulletGroup(BaseModel):
    kind: Literal["experience", "project"] = Field(
        description="'experience' for a job in the profile, 'project' for a project in the profile."
    )
    name: str = Field(
        description="Company name (experience) or project name (project), copied VERBATIM from the master profile."
    )
    title: str = Field(
        default="", description="Job title exactly as written in the master profile (empty for projects)."
    )
    bullets: list[str] = Field(
        description="3-5 rewritten bullets for this entry. Each must describe real work already in the "
        "profile; only wording, ordering and emphasis may change. One line each, start with a verb, "
        "keep the original metrics."
    )


class JobAnalysis(BaseModel):
    job_title: str = Field(description="Job title as stated in the job description.")
    company_name: str = Field(description="Hiring company as stated in the job description, or 'Unknown'.")
    match_score: int = Field(
        ge=0, le=100,
        description="0-100 fit between the candidate's REAL profile and this job. "
        "85+: strong fit, 65-84: good fit, 40-64: partial fit, <40: poor fit. Be calibrated, not generous.",
    )
    match_rationale: str = Field(description="2-3 sentences: strongest overlaps and the biggest genuine gaps.")
    missing_requirements: list[str] = Field(
        default_factory=list,
        description="Hard requirements from the JD that the candidate does NOT meet (verbatim phrases).",
    )
    tailored_summary: str = Field(
        description="Exactly two sentences targeted at this JD. No first-person pronouns."
    )
    highlighted_skills: list[str] = Field(
        description="8-14 skills that appear in the master profile, ordered by relevance to the JD. "
        "Never include a skill that is absent from the profile."
    )
    tailored_bullets: list[TailoredBulletGroup] = Field(
        description="One group per relevant profile experience/project, most relevant first."
    )
    answers: list[ScreeningAnswer] = Field(
        description="Predicted answers to screening questions likely for this role: years of experience with "
        "each key technology in the JD, work authorization, visa sponsorship, notice period / start date, "
        "salary expectation, remote/on-site/relocation, and anything the JD explicitly asks."
    )

    @property
    def answers_map(self) -> dict[str, str]:
        return {a.question: a.answer for a in self.answers}


class FieldAnswer(BaseModel):
    answer: str = Field(
        description="Value to enter. For choice fields it MUST be one of the provided options, verbatim. "
        "Numbers as digits. Empty string if the field should be left blank."
    )
    confidence: float = Field(
        ge=0, le=1, description="0-1 confidence that this is truthful and what the candidate would answer."
    )
    needs_human: bool = Field(
        description="True when the master profile lacks the information needed to answer truthfully."
    )
    reasoning: str = Field(description="One sentence.")


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert technical recruiter and resume writer acting on behalf of ONE candidate.
You receive the candidate's MASTER PROFILE (the single source of truth) and a JOB DESCRIPTION.

STRICT GUARDRAILS - violating any of these makes the output unusable:
- Do not invent companies, degrees, certifications, dates, titles, or technical proficiencies.
  Only emphasize or reframe genuine experiences found in the master profile.
- Never claim more years of experience with a technology than the profile supports.
- Never move a skill from "familiar" to "expert" or add tools the candidate has not used.
- If the job requires something the candidate genuinely lacks, say so in match_rationale and
  missing_requirements and lower match_score accordingly. Do not paper over gaps.
- Rewritten bullets must stay truthful to the original bullet: mirror the job description's
  vocabulary only where it genuinely describes the same work. Keep all quantified results.
- Screening answers must agree with the profile's `screening_defaults` and `years_of_experience`.
  When the profile lacks the information, answer exactly "UNKNOWN" instead of guessing.
- Output plain text values: no markdown, no bullet symbols inside strings.
"""

ANALYSIS_PROMPT = """MASTER PROFILE (JSON, source of truth):
{profile}

JOB POSTING
Title: {title}
Company: {company}
Location: {location}
URL: {url}

JOB DESCRIPTION:
\"\"\"
{description}
\"\"\"

TASK
1. Score how well the candidate's real profile matches this job (match_score, match_rationale, missing_requirements).
2. Write a two-sentence tailored_summary for this posting.
3. Pick highlighted_skills strictly from the profile, most relevant first.
4. For each relevant experience/project in the profile, rewrite its bullets to mirror this JD (tailored_bullets).
   Use the exact company/project name and title from the profile so the bullets can be mapped back.
5. Predict screening answers (answers) using screening_defaults and years_of_experience from the profile.
"""

FIELD_PROMPT = """You are filling ONE field of an online job application on behalf of the candidate.

MASTER PROFILE (JSON, source of truth):
{profile}

JOB: {title} at {company}
Previously predicted screening answers (JSON): {predicted}

FIELD
Label: {label}
Type: {field_type}
Options (choose exactly one, verbatim, if non-empty): {options}
Current value: {current_value}
Validation error shown by the form (if any): {error}

Answer truthfully from the profile. If the profile does not contain the information, set needs_human=true
and keep confidence low. For yes/no questions answer with the option that matches the profile.
For "years of experience" questions use `years_of_experience` from the profile (digits only).
For free-text motivation/cover-letter fields write 2-4 concise, factual sentences grounded in the profile.
"""

RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _hard_quota_failure(exc: errors.APIError) -> Optional[str]:
    """Return an actionable message when a 429 is a permanent quota problem, not rate limiting.

    A free-tier key that has never been enabled for the Generative Language API comes back as
    429 RESOURCE_EXHAUSTED with quota_limit_value "0". Retrying that forever is pointless.
    """
    if exc.code != 429:
        return None
    blob = json.dumps(getattr(exc, "details", None) or {}) + str(exc)
    if '"quota_limit_value": "0"' not in blob and "'quota_limit_value': '0'" not in blob:
        return None
    return (
        "Gemini rejected the request with quota limit 0, which means this API key has no "
        "Generative Language API quota (not temporary rate limiting).\n"
        "  1. Open https://aistudio.google.com/apikey and confirm the key is active.\n"
        "  2. Make sure the key's Google Cloud project has the Generative Language API enabled:\n"
        "     https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com\n"
        "  3. If you are on a paid project, check quotas for your region.\n"
        f"  Raw error: {str(exc)[:300]}"
    )



class AIAgent:
    """Thin async wrapper around google-genai with structured outputs and retries."""

    def __init__(self, settings: Settings = default_settings):
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env or export it in the environment."
            )
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.max_retries = max(1, settings.ai_max_retries)

    # ---- core call ----------------------------------------------------
    async def _generate(self, schema: type[T], prompt: str, temperature: float = 0.2) -> T:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            # we never pass tools; disabling AFC silences the SDK's recommendation warning
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        delay = 2.0
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                parsed = response.parsed
                if isinstance(parsed, schema):
                    return parsed
                # Belt and braces: validate the raw JSON text ourselves.
                return schema.model_validate_json(response.text or "")
            except errors.APIError as exc:
                last_error = exc
                hard = _hard_quota_failure(exc)
                if hard:
                    raise RuntimeError(hard) from exc
                if exc.code in (400, 401, 403) and "API_KEY" in str(exc).upper():
                    raise RuntimeError(
                        "Gemini rejected the API key. Check GEMINI_API_KEY in .env "
                        f"(https://aistudio.google.com/apikey). Raw error: {str(exc)[:200]}"
                    ) from exc
                if exc.code in RETRYABLE_CODES and attempt < self.max_retries:
                    wait = delay + random.uniform(0, 1)
                    log.warning("Gemini API error %s (attempt %d/%d), retrying in %.1fs",
                                exc.code, attempt, self.max_retries, wait)
                    await asyncio.sleep(wait)
                    delay *= 2
                    continue
                raise
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    log.warning("Gemini returned invalid structured output (attempt %d/%d): %s",
                                attempt, self.max_retries, exc)
                    await asyncio.sleep(1.0)
                    continue
                raise
        raise RuntimeError(f"Gemini call failed after {self.max_retries} attempts: {last_error}")

    # ---- public API ---------------------------------------------------
    async def analyze_job(self, profile: dict, job: JobPosting) -> JobAnalysis:
        prompt = ANALYSIS_PROMPT.format(
            profile=json.dumps(profile, ensure_ascii=False, indent=1),
            title=job.title or "(unknown)",
            company=job.company or "(unknown)",
            location=job.location or "(unknown)",
            url=job.url,
            description=(job.description or "")[:14_000],
        )
        analysis = await self._generate(JobAnalysis, prompt, temperature=0.2)
        log.info("Gemini analysis: %s @ %s -> score %d", analysis.job_title,
                 analysis.company_name, analysis.match_score)
        return analysis

    async def answer_form_question(
        self,
        profile: dict,
        job: JobPosting,
        analysis: Optional[JobAnalysis],
        label: str,
        field_type: str,
        options: Optional[list[str]] = None,
        current_value: str = "",
        error: str = "",
    ) -> FieldAnswer:
        prompt = FIELD_PROMPT.format(
            profile=json.dumps(profile, ensure_ascii=False),
            title=job.title or (analysis.job_title if analysis else ""),
            company=job.company or (analysis.company_name if analysis else ""),
            predicted=json.dumps(analysis.answers_map, ensure_ascii=False) if analysis else "{}",
            label=label,
            field_type=field_type,
            options=json.dumps(options or [], ensure_ascii=False),
            current_value=current_value or "(empty)",
            error=error or "(none)",
        )
        answer = await self._generate(FieldAnswer, prompt, temperature=0.1)
        log.info("Gemini field answer for %r -> %r (conf %.2f, human=%s)",
                 label, answer.answer, answer.confidence, answer.needs_human)
        return answer
