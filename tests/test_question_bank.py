"""Recognising which of the usual application questions is being asked.

A form asks the same handful of things in a thousand wordings. Knowing which one is in
front of you is what lets the answer be built from the right part of the profile: a
question about remote working is answered from which roles were remote, and one about
open source from your public repositories. Guessing wrong is worse than not guessing,
so a weak match is no match.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from question_bank import BANK_PATH, Archetype, QuestionBank, normalise  # noqa: E402

REAL_QUESTIONS = {
    "remote_async_working":
        "Tell us about your experience working in an async and/or remote environment. "
        "What practices or approaches have worked well for you? What challenges have you faced?",
    "open_source":
        "Have you made any open source contributions in the past that you'd like to share with us?",
    "why_this_company": "Why do you want to work at Acme?",
    "behavioural_story": "Tell us about a time you disagreed with a colleague.",
    "impact_or_metric":
        "Tell us about a metric you defined that helped your business make better decisions.",
    "how_did_you_hear": "How did you hear about this role?",
    "anything_else": "Is there anything else you would like us to know?",
}


@pytest.fixture(scope="module")
def bank() -> QuestionBank:
    return QuestionBank.load()


@pytest.mark.parametrize("expected,question", list(REAL_QUESTIONS.items()))
def test_real_questions_are_recognised(bank: QuestionBank, expected: str, question: str) -> None:
    found = bank.match(question)
    assert found is not None and found.id == expected, bank.explain(question)


@pytest.mark.parametrize("label", [
    "First name", "Email", "Phone", "LinkedIn Profile", "Upload your resume",
    "Are you legally authorized to work in the United States?",
    "What is your expected salary?", "Preferred pronouns",
])
def test_ordinary_fields_are_left_alone(bank: QuestionBank, label: str) -> None:
    """These are facts, answered from the profile or the screening defaults. Steering
    them towards an essay archetype would turn a one-word answer into a paragraph."""
    assert bank.match(label) is None, bank.explain(label)


def test_a_longer_phrase_wins_over_a_stray_word() -> None:
    """"remote" alone appears in half of all postings, so it must not outrank a phrase."""
    bank = QuestionBank([
        Archetype(id="vague", phrases=["remote"]),
        Archetype(id="specific", phrases=["remote environment"]),
    ])
    assert bank.match("Describe working in a remote environment").id == "specific"


def test_guidance_names_the_profile_fields_that_answer_it(bank: QuestionBank) -> None:
    text = bank.guidance_for(REAL_QUESTIONS["open_source"])
    assert "github" in text.lower() and "projects" in text.lower()
    assert "never invent" in text.lower(), "the honesty rule has to reach the model"


def test_an_unrecognised_question_gets_no_guidance(bank: QuestionBank) -> None:
    assert bank.guidance_for("What is your favourite colour?") == ""


def test_wording_differences_do_not_matter(bank: QuestionBank) -> None:
    for phrasing in ("Why do you want to work here?",
                     "Why do you want to work with us?",
                     "WHY DO YOU WANT TO WORK AT OUR COMPANY?"):
        assert bank.match(phrasing).id == "why_this_company", phrasing


def test_normalise_strips_what_should_not_affect_a_match() -> None:
    assert normalise("  Why do you WANT to work here??  ") == "why do you want to work here"


# ---- the file itself -------------------------------------------------------

def test_the_bank_holds_no_written_answers() -> None:
    """The whole point: answers are composed from your own experience every time, so
    nothing in this file can claim something you did not do."""
    raw = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    for entry in raw["archetypes"]:
        assert set(entry) <= {"id", "asks", "phrases", "evidence", "guidance", "style"}, entry["id"]
        assert entry.get("style", "prose") in ("prose", "short"), entry["id"]
        assert "answer" not in entry, "an archetype must not carry a ready-made answer"


def test_every_archetype_is_usable() -> None:
    bank = QuestionBank.load()
    assert len(bank.archetypes) >= 10
    seen = set()
    for a in bank.archetypes:
        assert a.id and a.id not in seen, f"duplicate archetype {a.id}"
        seen.add(a.id)
        assert a.asks and a.guidance, f"{a.id} says nothing useful"
        assert a.phrases, f"{a.id} can never match anything"


def test_a_missing_or_broken_bank_is_survivable(tmp_path: Path) -> None:
    """Open questions still get answered, just without the extra steer."""
    assert QuestionBank.load(tmp_path / "gone.json").archetypes == []
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert QuestionBank.load(broken).guidance_for("anything") == ""


def test_the_agent_puts_the_guidance_in_the_prompt() -> None:
    from ai_agent import FIELD_PROMPT

    assert "{guidance}" in FIELD_PROMPT
    assert "WHAT THIS QUESTION IS ASKING" in FIELD_PROMPT


# ---- catching an answer that invented its details --------------------------

PROFILE = {
    "name": "Jane Doe",
    "experience": [{"company": "BLNR (Open Source)", "title": "Backend Developer",
                    "location": "Remote",
                    "bullets": ["Led backend architecture on Node.js, Express and MongoDB.",
                                "Reviewed community contributions."]}],
    "projects": [{"name": "DataBI", "tech": ["Next.js", "MongoDB"]}],
}


def _job():
    from models import JobPosting

    return JobPosting(job_id="t", url="https://example.com", title="Full Stack Developer",
                      company="Acme", description="We run Kubernetes and Postgres.")


def test_a_tool_the_profile_never_mentions_is_caught() -> None:
    """Seen live: the model claimed Slack, Confluence and Notion, none of which the
    candidate has ever named."""
    from ai_agent import unsupported_names

    said = ("I rely on written updates in Slack and shared documentation in "
            "Confluence or Notion.")

    assert unsupported_names(said, PROFILE, _job()) == ["confluence", "notion", "slack"]


def test_what_the_profile_does_name_is_left_alone() -> None:
    from ai_agent import unsupported_names

    said = "At BLNR I led backend architecture on Node.js, Express and MongoDB."

    assert unsupported_names(said, PROFILE, _job()) == []


def test_generic_wording_passes() -> None:
    """The honest way to say it: describe the practice, not a product."""
    from ai_agent import unsupported_names

    said = "I rely on written updates, pull-request descriptions and shared documentation."

    assert unsupported_names(said, PROFILE, _job()) == []


def test_a_tool_the_posting_itself_names_is_allowed() -> None:
    """A question can ask about the team's own stack, so the posting is fair game."""
    from ai_agent import unsupported_names

    said = "I have deployed services with Kubernetes and stored data in Postgres."

    assert unsupported_names(said, PROFILE, _job()) == []


def test_one_name_is_not_mistaken_for_another() -> None:
    from ai_agent import unsupported_names

    assert "java" not in unsupported_names("I write JavaScript daily.", PROFILE, _job())


def test_an_empty_answer_is_not_an_invention() -> None:
    from ai_agent import unsupported_names

    assert unsupported_names("", PROFILE, _job()) == []


def test_the_story_guidance_forbids_inventing_an_incident() -> None:
    """The live failure was an invented payment module and token format."""
    bank = QuestionBank.load()
    text = bank.guidance_for("Tell us about a time a project did not go to plan.")
    assert "invented" in text.lower() or "do not invent" in text.lower()


# ---- a lookup rule must never answer a written question --------------------

REMOTE_ESSAY = ("Tell us about your experience working in an async and/or remote "
                "environment. What practices or approaches have worked well for you? "
                "What challenges have you faced?")


def test_an_essay_about_remote_work_is_not_a_remote_preference(bank: QuestionBank) -> None:
    """The bug: the rule for remote-or-onsite working fires on any label containing
    "remote", and answered this question with the one word "Remote"."""
    assert bank.wants_prose(REMOTE_ESSAY) is True


def test_a_yes_or_no_about_remote_work_still_is_one(bank: QuestionBank) -> None:
    """Matching the same archetype must not turn a short question into an essay."""
    assert bank.wants_prose("Are you open to remote work?") is False
    assert bank.wants_prose("Preferred work arrangement") is False


@pytest.mark.parametrize("label", [
    "What is your notice period?", "What is your expected salary?",
    "Will you now or in the future require sponsorship?", "Email", "LinkedIn Profile",
    "How did you hear about this role?",
])
def test_ordinary_screening_questions_stay_lookups(bank: QuestionBank, label: str) -> None:
    assert bank.wants_prose(label) is False, bank.explain(label)


def test_a_wording_the_bank_has_never_seen_is_judged_by_shape(bank: QuestionBank) -> None:
    """Nothing in the bank matches this, and it still plainly wants writing."""
    unknown = ("Describe how you would explain a difficult technical trade-off to "
               "somebody who does not write software.")
    assert bank.match(unknown) is None
    assert bank.wants_prose(unknown) is True


@pytest.mark.asyncio
async def test_the_model_answers_the_essay_and_the_rules_answer_the_facts() -> None:
    """End to end through the resolver, which is where the wrong answer came from."""
    from ai_agent import FieldAnswer
    from browser_bot import AnswerResolver, FormField, ResolveContext
    from config import Settings
    from models import JobPosting

    profile = {"email": "you@example.com",
               "screening_defaults": {"remote_preference": "Remote", "notice_period": "2 weeks"}}

    class SpyAI:
        def __init__(self) -> None:
            self.asked: list[str] = []

        async def answer_form_question(self, profile, job, analysis, label, kind, **kw):
            self.asked.append(label)
            return FieldAnswer(answer="I have worked remotely for three years, relying on "
                                      "written updates and documented handovers.",
                               confidence=0.9, needs_human=False, reasoning="")

    ai = SpyAI()
    resolver = AnswerResolver(ai, Settings(_env_file=None))
    ctx = ResolveContext(profile, JobPosting.from_url("https://jobs.lever.co/a/b"), None)

    essay = await resolver.resolve(FormField(kind="textarea", label=REMOTE_ESSAY, idx="x0"), ctx)
    short = await resolver.resolve(
        FormField(kind="text", label="Are you open to remote work?", idx="x1"), ctx)

    assert REMOTE_ESSAY in ai.asked, "the essay never reached the model"
    assert essay.value.startswith("I have worked remotely")
    assert short.value == "Remote" and "Are you open to remote work?" not in ai.asked


@pytest.mark.asyncio
async def test_a_multi_line_box_is_always_treated_as_writing() -> None:
    """Whatever it is labelled. A textarea is not where a stored one-liner belongs."""
    from browser_bot import AnswerResolver, FormField
    from config import Settings

    resolver = AnswerResolver(None, Settings(_env_file=None))

    assert resolver.wants_writing(FormField(kind="textarea", label="Notes", idx="x0")) is True
    assert resolver.wants_writing(FormField(kind="text", label="Notes", idx="x1")) is False


@pytest.mark.asyncio
async def test_a_dropdown_is_never_treated_as_writing() -> None:
    from browser_bot import AnswerResolver, FormField
    from config import Settings

    resolver = AnswerResolver(None, Settings(_env_file=None))
    field = FormField(kind="textarea", label=REMOTE_ESSAY, idx="x0",
                      options=[{"label": "Yes", "value": "y", "idx": "", "dom_id": ""}])

    assert resolver.wants_writing(field) is False


def test_the_screening_rules_refuse_an_essay_on_their_own() -> None:
    """Belt and braces: these rules match on a single word, and a stored value dropped
    into an essay box is the most visible way the bot embarrasses you."""
    from browser_bot import AnswerResolver, FormField
    from config import Settings

    resolver = AnswerResolver(None, Settings(_env_file=None))
    profile = {"screening_defaults": {"remote_preference": "Remote"}}

    assert resolver._from_screening_defaults(
        FormField(kind="text", label=REMOTE_ESSAY, idx="x0"), profile) is None
    assert resolver._from_screening_defaults(
        FormField(kind="text", label="Are you open to remote work?", idx="x1"),
        profile).value == "Remote"
