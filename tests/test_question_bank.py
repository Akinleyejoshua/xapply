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
        assert set(entry) <= {"id", "asks", "phrases", "evidence", "guidance"}, entry["id"]
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
