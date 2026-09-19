"""Job analysis and form answering, on whichever LLM backend is configured.

Two responsibilities:
  1. `analyze_job`            -> JobAnalysis  (match score, tailored resume content,
                                                predicted screening answers)
  2. `answer_form_question`   -> FieldAnswer   (live answer for a form field the
                                                predictions did not cover)

Both go through `llm.build_provider`, which returns strict JSON validated against
the Pydantic schema before anything else touches it. Switch backends with
`LLM_PROVIDER=gemini|nvidia`; the prompts and schemas are identical either way.
"""
from __future__ import annotations

import json
import logging
from typing import Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from config import Settings
from config import settings as default_settings
from llm import LLMError, LLMProvider, build_provider
from models import JobPosting

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def settings_model(settings: Settings) -> str:
    """Name of the model actually in use, for logs and the admin API."""
    return settings.active_model

# --------------------------------------------------------------------------
# Structured output schemas
# NOTE: Gemini's schema subset does not support free-form dict/additionalProperties,
# so the "answers" map is modelled as a list of question/answer pairs and exposed
# as a dict through `JobAnalysis.answers_map`.
# --------------------------------------------------------------------------


class ScreeningAnswer(BaseModel):
    question: str = Field(
        description="The question as an application form would print it, in full and in plain "
        "English, ending in a question mark. For example 'How many years of experience do you have "
        "with Python?' or 'Will you now or in the future require visa sponsorship?'. "
        "Never return a settings key such as 'requires_sponsorship' or 'notice_period'."
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
        description="3-5 rewritten bullets for this entry, ordered most relevant to this job first. "
        "Each must describe real work already in the profile; only wording, ordering and emphasis "
        "may change. Lead each bullet with the part of the work this job cares about, and use the "
        "job description's own vocabulary wherever it genuinely names the same thing. Keep every "
        "number and metric from the original. One line each, starting with a past-tense verb. "
        "Do NOT return a bullet that differs from the original only in punctuation or hyphenation: "
        "either re-emphasise it for this job or leave the original wording alone. "
        "Use plain ASCII punctuation: ordinary hyphens and straight quotes."
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
        description="8-14 predicted screening questions for THIS role, each written as a form would "
        "ask it, with the candidate's answer. Cover: years of experience with each key technology "
        "named in the job description, work authorization, visa sponsorship, notice period and "
        "start date, salary expectation, remote or on-site and relocation, and anything the job "
        "description explicitly asks about. Prefer job-specific questions over generic ones."
    )

    @property
    def answers_map(self) -> dict[str, str]:
        return {a.question: a.answer for a in self.answers}


class CoverLetter(BaseModel):
    """A cover letter written for one specific posting."""

    greeting: str = Field(
        description="Salutation line, e.g. 'Dear Hiring Team,'. Use the team or company name only "
        "if the job description names one. Never invent a person's name."
    )
    opening: str = Field(
        description="2-3 sentences naming the role and company and saying, concretely, why this "
        "candidate fits. No filler such as 'I am writing to apply'."
    )
    body: list[str] = Field(
        description="2-3 paragraphs, each 2-4 sentences. Each paragraph takes one requirement from "
        "the job description and answers it with a specific thing the candidate actually did, "
        "drawn from the master profile, with its real numbers. No claim may go beyond the profile."
    )
    closing: str = Field(
        description="1-2 sentences: what the candidate would bring, and a plain willingness to talk. "
        "No begging, no exclamation marks."
    )
    signature: str = Field(
        description="Sign-off and the candidate's name only, e.g. 'Sincerely,\nJane Doe'. "
        "Never append an address block, email, phone number or links: the letter is filed "
        "alongside the resume, which already carries them."
    )

    def as_text(self) -> str:
        """The letter as plain text, for a textarea on an application form."""
        parts = [self.greeting, "", self.opening, ""]
        for paragraph in self.body:
            parts += [paragraph, ""]
        parts += [self.closing, "", self.signature]
        return "\n".join(parts).strip()

    @property
    def paragraphs(self) -> list[str]:
        return [self.opening, *self.body, self.closing]


class FieldAnswer(BaseModel):
    answer: str = Field(
        description="Value to enter. For choice fields it MUST be one of the provided options, verbatim. "
        "Numbers as digits. Empty string if the field should be left blank."
    )
    confidence: float = Field(
        ge=0, le=1, description="0-1 confidence that this is truthful and what the candidate would answer."
    )
    needs_human: bool = Field(
        description="True when the profile lacks the information needed to answer truthfully. "
        "An open question asking the candidate to write about their own experience is not such "
        "a case: compose the answer from the profile instead of setting this."
    )
    #: Informational only, so a model that omits it is not worth a whole retry. The
    #: decisions above are not defaulted: a missing `needs_human` is a real omission,
    #: and guessing it either escalates everything or silently answers for the candidate.
    reasoning: str = Field(default="", description="One sentence.")


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
- Output plain text with ASCII punctuation only: ordinary hyphens, straight quotes, no em dashes
  or non-breaking hyphens. No markdown, no bullet symbols inside strings.
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
4. For each relevant experience/project in the profile, rewrite its bullets for this JD
   (tailored_bullets). Use the exact company/project name and title from the profile so the
   bullets can be mapped back. Reorder so the work this employer cares about comes first, and
   reword so their vocabulary appears wherever it truthfully describes the same work. A bullet
   that comes back with only its punctuation changed is a wasted bullet.
5. Predict screening answers (answers), each phrased as a real form would ask it, using
   screening_defaults and years_of_experience from the profile.

WRITING RULES
- Plain ASCII punctuation only: ordinary hyphens (-), straight quotes (' and "), no em dashes,
  no non-breaking hyphens, no ellipsis characters. Applicant tracking systems mis-parse them.
- No markdown, no bullet symbols inside strings.
"""

COVER_LETTER_PROMPT = """Write a cover letter for this candidate and this specific job.

MASTER PROFILE (JSON, source of truth):
{profile}

JOB
Title: {title}
Company: {company}
Location: {location}

JOB DESCRIPTION:
\"\"\"
{description}
\"\"\"

WHAT THE ANALYSIS ALREADY FOUND
Match score: {score}
Strengths and gaps: {rationale}
Requirements the candidate does not meet: {gaps}

RULES
- Every claim must be traceable to the master profile. No invented employers, tools, degrees,
  numbers or achievements. If the profile does not support it, leave it out.
- Do not claim, or apologise for, anything in the list of unmet requirements. Simply write about
  what the candidate has actually done.
- Be specific. Name the real project or employer and the real result. A sentence that could
  appear in anyone's letter is a wasted sentence.
- No flattery about the company, no "passionate about", no "I believe I would be a great fit".
- Plain ASCII punctuation only: ordinary hyphens and straight quotes.
- Around 250-320 words in total.
- End with the sign-off and the name. Do not add a contact block, address, email, phone
  number or links; the resume carries those already.
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

There are two kinds of field, and they are answered differently.

1. A FACT about the candidate: name, email, location, years of experience, visa status,
   salary expectation, notice period, a yes/no screening question, or any choice field.
   Answer it from the profile alone. If the profile does not contain it, set
   needs_human=true and keep confidence low. Never guess a fact.
   For "years of experience" questions use `years_of_experience` (digits only).
   For yes/no questions answer with the option that matches the profile.

2. An OPEN QUESTION asking the candidate to write something: "tell us about a time...",
   "describe a project...", "what metric did you define", "why this role", "what interests
   you", or any motivation or cover-letter box. These are the candidate's own words, so
   write them. The profile will never contain the finished sentence, and that is not a
   reason to refuse. Build the answer like this:

   - Start from the job title above. Pick the entries in `experience` and `projects`
     closest to it and use them together: the role gives the setting, the project gives
     the detail. A project that belongs to one of those roles is the strongest material
     available, so prefer it.
   - Elaborate. The profile stores work as short summaries, so expand them into full
     sentences that sound like the candidate speaking, not like a list read out loud.
   - You may state the ordinary, standard way that work is done where it plainly follows
     from what the entry already says. Someone who built a reporting dashboard chose what
     went on it and agreed that with whoever asked for it. That is method, and it is safe
     to say.
   - You may NOT invent outcomes. Percentages, revenue, user counts, hours saved, headcount
     and rankings are claims, not method. Where the profile gives no figure, say what
     changed in words and leave the number out. An answer with no number is fine; an answer
     with a number the candidate cannot defend in an interview is not.
   - Keep the candidate's real scale. Do not turn a personal project into company work, a
     contributor into a lead, or a small internal tool into a platform.
   - Answer every part of the question, in the order asked, in the first person, 3 to 6
     sentences. No headings, no bullets.
   - Set needs_human=true only when the profile holds nothing relevant at all.

Answer truthfully. Inventing an employer, a metric or a result is worse than leaving the
field for the candidate to fill in.
"""

class AIAgent:
    """Prompting, schemas and guardrails. Transport lives in `llm.py`."""

    def __init__(self, settings: Settings = default_settings, provider: Optional[LLMProvider] = None):
        self.settings = settings
        self.provider = provider or build_provider(settings)

    @property
    def model(self) -> str:
        return settings_model(self.settings)

    async def aclose(self) -> None:
        await self.provider.aclose()

    async def _generate(self, schema: type[T], prompt: str, temperature: float = 0.2) -> T:
        return await self.provider.generate(schema, SYSTEM_PROMPT, prompt, temperature=temperature)

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

    async def write_cover_letter(self, profile: dict, job: JobPosting,
                                 analysis: Optional[JobAnalysis] = None) -> CoverLetter:
        """Write a cover letter for one posting, grounded in the profile."""
        prompt = COVER_LETTER_PROMPT.format(
            profile=json.dumps(profile, ensure_ascii=False, indent=1),
            title=job.title or (analysis.job_title if analysis else ""),
            company=job.company or (analysis.company_name if analysis else ""),
            location=job.location or "(unspecified)",
            description=(job.description or "")[:12_000],
            score=analysis.match_score if analysis else "(not scored)",
            rationale=analysis.match_rationale if analysis else "(not scored)",
            gaps=", ".join(analysis.missing_requirements) if analysis else "(not scored)",
        )
        letter = await self._generate(CoverLetter, prompt, temperature=0.35)
        log.info("Cover letter written for %s @ %s (%d paragraphs)",
                 job.title, job.company, len(letter.paragraphs))
        return letter

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
