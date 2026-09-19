"""Recognising the open questions application forms ask.

An application form asks the same handful of things in a thousand different wordings.
Knowing which of them is being asked is what lets an answer be built from the right
part of your profile: a question about remote working is answered from which roles
were remote, and one about open source from your public repositories.

The bank holds no written answers. An archetype says what the question is after, which
part of the profile answers it, and how to build the answer. The words are composed
from your own experience every time, so nothing here can claim something you did not
do. `question_bank.json` is yours to edit.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

BANK_PATH = Path(__file__).resolve().parent / "question_bank.json"

#: A phrase has to be worth more than a stray word, or "time" in "part time" would
#: drag in every behavioural archetype.
PHRASE_WEIGHT = 1.0
#: Below this a match is a coincidence, and guessing wrong is worse than not guessing.
MIN_SCORE = 1.0

_NORMALISE = re.compile(r"[^a-z0-9 ]+")


def normalise(text: str) -> str:
    """Lowercased, punctuation removed, single spaces, so wordings compare fairly."""
    return re.sub(r"\s+", " ", _NORMALISE.sub(" ", (text or "").lower())).strip()


#: A question that wants writing rather than a fact. The distinction decides whether a
#: lookup rule may answer at all: "Tell us about working in a remote environment" was
#: being answered with the profile's remote preference, the single word "Remote",
#: because the rule for remote-or-onsite saw the word and never let the model near it.
PROSE = "prose"
SHORT = "short"

#: Questions the bank does not know, recognised by shape instead. Long, and asking to
#: be told something.
ESSAY_SHAPE = re.compile(
    r"\btell (us|me)\b|\bdescribe\b|\bexplain\b|\bwalk (us|me) through\b|"
    r"\bin your own words\b|\bwhat (practices|approaches|challenges|steps)\b|"
    r"\bshare (an|a|your)\b|\bgive (an|us an) example\b|\belaborate\b|"
    r"\bwhy do you\b|\bwhat (interests|excites|motivates) you\b", re.I)
#: Under this length a question is a field label, however it is phrased.
ESSAY_MIN_CHARS = 60


def looks_like_an_essay(label: str) -> bool:
    """Whether a question wants writing, judged from its shape alone."""
    text = (label or "").strip()
    return len(text) >= ESSAY_MIN_CHARS and bool(ESSAY_SHAPE.search(text))


@dataclass
class Archetype:
    id: str
    asks: str = ""
    phrases: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    guidance: str = ""
    style: str = PROSE

    def score(self, label: str) -> float:
        """How strongly this archetype claims `label`, already normalised."""
        total = 0.0
        for phrase in self.phrases:
            needle = normalise(phrase)
            if needle and needle in label:
                # A longer phrase is a stronger claim: "remote environment" is a better
                # signal than "remote", which appears in every remote-first posting.
                total += PHRASE_WEIGHT + len(needle) / 100.0
        return total

    def brief(self) -> str:
        """The part that goes to the model, as plain instructions."""
        lines = [f"This question is asking: {self.asks}", f"How to answer it: {self.guidance}"]
        if self.evidence:
            lines.append("Answer it from these parts of the profile: " + ", ".join(self.evidence))
        return "\n".join(lines)


class QuestionBank:
    def __init__(self, archetypes: list[Archetype]):
        self.archetypes = archetypes

    @classmethod
    def load(cls, path: Path | str = BANK_PATH) -> "QuestionBank":
        path = Path(path)
        if not path.exists():
            log.warning("No question bank at %s; open questions still get answered, "
                        "just without the extra steer", path)
            return cls([])
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Ignoring unreadable %s: %s", path.name, exc)
            return cls([])
        out: list[Archetype] = []
        for entry in raw.get("archetypes", []):
            try:
                out.append(Archetype(
                    id=str(entry["id"]), asks=entry.get("asks", ""),
                    phrases=list(entry.get("phrases") or []),
                    evidence=list(entry.get("evidence") or []),
                    guidance=entry.get("guidance", ""),
                    style=entry.get("style", PROSE)))
            except (KeyError, TypeError) as exc:
                log.warning("Skipping a malformed question archetype: %s", exc)
        return cls(out)

    def match(self, label: str) -> Optional[Archetype]:
        """The archetype that best fits this question, or None when none clearly does."""
        if not label or not self.archetypes:
            return None
        normalised = normalise(label)
        best, best_score = None, 0.0
        for archetype in self.archetypes:
            score = archetype.score(normalised)
            if score > best_score:
                best, best_score = archetype, score
        return best if best_score >= MIN_SCORE else None

    def guidance_for(self, label: str) -> str:
        """What to tell the model about this question, or an empty string."""
        found = self.match(label)
        return found.brief() if found else ""

    def wants_prose(self, label: str) -> bool:
        """Whether this question has to be written rather than looked up.

        A lookup rule answers by spotting a word: the rule for remote-or-onsite working
        fires on any label containing "remote". That is right for "Are you open to remote
        work?" and badly wrong for "Tell us about your experience working in a remote
        environment", which it answered with the single word "Remote". So a question that
        wants writing is never offered to the rules at all.
        """
        found = self.match(label)
        if found is not None and found.style == SHORT:
            return False
        if looks_like_an_essay(label):
            return True
        # Matching a prose archetype is not enough on its own. "Are you open to remote
        # work?" matches the remote archetype and is still a yes or no, so the question
        # has to be long enough to be asking for writing.
        return found is not None and len((label or "").strip()) >= ESSAY_MIN_CHARS

    def explain(self, label: str) -> dict[str, Any]:
        """Every archetype's score, for working out a surprising match."""
        normalised = normalise(label)
        scored = sorted(((a.id, round(a.score(normalised), 3)) for a in self.archetypes),
                        key=lambda kv: -kv[1])
        found = self.match(label)
        return {"label": label, "matched": found.id if found else None,
                "wants_prose": self.wants_prose(label), "scores": scored[:5]}
