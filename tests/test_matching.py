"""Title relevance: does a posting resemble what the user searched for?

Regression suite for the bug where a search for "Data Analytics" returned nothing.
Of 93 real intern postings on the configured boards, the old word-count rule matched
0, including obvious hits like "Analytics Engineer Intern" and "Business Analyst Intern".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matching import (  # noqa: E402
    best_relevance,
    concept_of,
    explain,
    relevance,
    stem_equal,
    tokenize,
    weight_of,
    word_similarity,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_QUERY = ["Data Analytics", "Data Analysis"]


# ---- the reported bug -----------------------------------------------------


@pytest.mark.parametrize("title", [
    "Analytics Engineer Intern",
    "Data Engineer Intern",
    "Data Science Intern",
    "People Analytics Intern",
    "Business Analyst Intern (Summer 2027)",
    "Brokerage Risk Analyst Intern (Summer 2027)",
    "PeopleX Insights & Analytics Intern (Summer 2027)",
    "Data Science Intern (Winter 2027)",
    "Investment Analyst Intern (Summer 2027)",
])
def test_intern_data_roles_are_found(title) -> None:
    """These are real postings the old matcher dropped outright."""
    assert best_relevance(title, DATA_QUERY) >= 0.45, title


@pytest.mark.parametrize("title", [
    "Software Engineer, Intern",
    "Accounting Intern",
    "University Recruiter",
    "Crypto Inventory Operations Intern",
    "Product Management Intern (Summer 2027)",
    "Business Controller Intern",
])
def test_unrelated_interns_are_still_rejected(title) -> None:
    assert best_relevance(title, DATA_QUERY) < 0.45, title


def test_exact_titles_score_near_one() -> None:
    assert best_relevance("Data Analyst", DATA_QUERY) > 0.9
    assert best_relevance("Senior Data Analyst", DATA_QUERY) > 0.9
    assert best_relevance("Financial Data Analyst", DATA_QUERY) > 0.9


# ---- generic words must not drag in unrelated roles ------------------------


def test_engineer_alone_is_not_enough() -> None:
    """The counterweight to being generous: 'engineer' barely narrows a search."""
    assert relevance("Backend Engineer", "Backend Engineer") == 1.0
    for noise in ("Sales Engineer", "Solutions Engineer", "Frontend Engineer",
                  "Engineering Manager", "Software Engineer, Intern"):
        assert relevance(noise, "Backend Engineer") < 0.45, noise


def test_weights_favour_specific_words() -> None:
    assert weight_of("engineer") < weight_of("backend")
    assert weight_of("manager") < weight_of("kubernetes")
    assert weight_of("intern") < weight_of("python")


# ---- concepts and inflections ---------------------------------------------


@pytest.mark.parametrize("word,concept", [
    ("analyst", "analytics"), ("analytics", "analytics"), ("analysis", "analytics"),
    ("insights", "analytics"), ("bi", "analytics"),
    ("engineer", "software"), ("developer", "software"), ("swe", "software"),
    ("ml", "ml"), ("ai", "ml"), ("nlp", "ml"),
    ("kubernetes", "devops"), ("sre", "devops"),
    ("intern", "internship"), ("apprentice", "internship"),
])
def test_concept_of(word, concept) -> None:
    assert concept_of(word) == concept


def test_unknown_words_have_no_concept() -> None:
    assert concept_of("zzzqqq") is None
    assert concept_of("brokerage") is None
    # but an inflection of a known word still resolves: "peoplex" is a people-team posting
    assert concept_of("peoplex") == "people"


@pytest.mark.parametrize("a,b", [
    ("analytics", "analyst"), ("engineer", "engineering"), ("develop", "developer"),
    ("science", "scientist"), ("design", "designer"),
])
def test_stem_equal(a, b) -> None:
    assert stem_equal(a, b) and stem_equal(b, a)


@pytest.mark.parametrize("a,b", [("support", "supply"), ("backend", "frontend"), ("cloud", "clinical")])
def test_stem_keeps_different_words_apart(a, b) -> None:
    assert not stem_equal(a, b)


def test_word_similarity_is_ordered() -> None:
    assert word_similarity("data", "data") == 1.0
    assert 0.9 < word_similarity("engineer", "engineering") < 1.0
    assert 0.8 < word_similarity("analyst", "insights") < 0.9     # same concept
    assert word_similarity("data", "sales") == 0.0


def test_multiword_concepts_collapse() -> None:
    """'machine learning' is one idea, not two, so it is not double counted."""
    assert relevance("ML Engineer", "Machine Learning Engineer") > 0.85
    assert relevance("AI Engineer", "Machine Learning Engineer") > 0.85
    assert "machinelearning" in tokenize("Machine Learning Engineer")
    assert "fullstack" in tokenize("Full Stack Developer")


def test_full_stack_variants() -> None:
    for title in ("Full Stack Engineer", "Fullstack Developer", "Full-Stack Engineer"):
        assert relevance(title, "Full Stack Developer") > 0.85, title


# ---- edges ----------------------------------------------------------------


def test_empty_query_matches_everything() -> None:
    assert relevance("Anything", "") == 1.0
    assert best_relevance("Anything", []) == 1.0


def test_empty_title_matches_nothing() -> None:
    assert relevance("", "Data Analytics") == 0.0


def test_score_is_bounded() -> None:
    for title in ("Data Data Data Analytics Analytics", "Data Analyst", "x"):
        assert 0.0 <= relevance(title, "Data Analytics") <= 1.0


def test_explain_breaks_the_score_down() -> None:
    out = explain("Analytics Engineer Intern", "Data Analytics")
    assert out["score"] == relevance("Analytics Engineer Intern", "Data Analytics")
    words = {w["query_word"]: w for w in out["words"]}
    assert words["analytics"]["matched"] == "analytics"
    assert words["data"]["similarity"] == 0.0        # nothing in the title answers "data"


# ---- threshold plumbing ----------------------------------------------------


def test_title_matches_respects_the_threshold() -> None:
    from discovery import title_matches

    title = "Analytics Engineer Intern"
    assert title_matches(title, DATA_QUERY, 0.3) is True
    assert title_matches(title, DATA_QUERY, 0.45) is True
    assert title_matches(title, DATA_QUERY, 0.8) is False
    assert title_matches(title, [], 0.9) is True      # no search terms means no filter


def test_threshold_setting_is_persisted(tmp_path: Path) -> None:
    from config import PERSISTED_KEYS, Settings

    assert "title_match_threshold" in PERSISTED_KEYS
    s = Settings(overrides_path=tmp_path / "s.json", db_path=tmp_path / "t.db")
    s.title_match_threshold = 0.3
    s.save_overrides(["title_match_threshold"])
    fresh = Settings(overrides_path=s.overrides_path, db_path=tmp_path / "t.db")
    fresh.load_overrides()
    assert fresh.title_match_threshold == 0.3


def test_dashboard_exposes_sensitivity_and_score() -> None:
    html = (ROOT / "static" / "index.html").read_text()
    for marker in ('id="scanThreshold"', 'id="thrLabel"', "title_match_threshold", "j.relevance"):
        assert marker in html, marker
