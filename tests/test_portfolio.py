"""Keeping the profile in step with the portfolio site, without losing your wording.

The portfolio already holds the projects and roles you keep up to date, so they are
read from it rather than retyped. The risk is the obvious one: an import that overwrites
the bullets you tuned by hand, or that adds a second copy of a project because the site
gives it a longer title than the profile does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio import (Changes, as_project, as_role, import_portfolio,  # noqa: E402
                       merge_entries, same_entry, sentences, short_name)

SITE_PROJECT = {
    "title": "xMachine: Browser-Based Deep Learning & Inference",
    "description": "A browser-based deep learning platform. It trains models client side.",
    "technologies": ["TensorFlow.js", "Next.js"],
    "liveUrl": "https://xmachine.example", "githubUrl": "https://github.com/jo/xmachine",
    "isVisible": True,
}
SITE_ROLE = {
    "role": "Backend Developer", "company": "BLNR (Open Source)",
    "description": ["Led backend architecture.", "Reviewed community contributions."],
    "startDate": "2024-01-01T00:00:00.000Z", "endDate": "2025-12-31T00:00:00.000Z",
    "isCurrent": False,
}


# ---- reading what the site gives -------------------------------------------

def test_a_site_project_becomes_a_profile_project() -> None:
    project = as_project(SITE_PROJECT)

    assert project["name"] == "xMachine: Browser-Based Deep Learning & Inference"
    assert project["tech"] == ["TensorFlow.js", "Next.js"]
    assert project["github"].endswith("xmachine")
    assert project["bullets"][0].startswith("A browser-based deep learning platform")


def test_a_hidden_project_is_not_imported() -> None:
    assert as_project({**SITE_PROJECT, "isVisible": False}) is None
    assert as_project({"title": "   "}) is None


def test_a_site_role_becomes_a_profile_role() -> None:
    role = as_role(SITE_ROLE)

    assert role["company"] == "BLNR (Open Source)" and role["title"] == "Backend Developer"
    assert role["start"] == "2024" and role["end"] == "2025"
    assert role["bullets"] == ["Led backend architecture.", "Reviewed community contributions."]


def test_a_current_role_says_present() -> None:
    assert as_role({**SITE_ROLE, "isCurrent": True})["end"] == "Present"


def test_descriptions_are_accepted_however_the_site_stores_them() -> None:
    """Projects give one block of prose; roles give a list of lines."""
    assert sentences("One thing. Then another.") == ["One thing.", "Then another."]
    assert sentences(["One thing", "Another"]) == ["One thing.", "Another."]
    assert sentences("") == [] and sentences(None) == []


def test_a_paragraph_is_not_a_bullet() -> None:
    assert sentences("word " * 80) == []


# ---- matching the same thing written at different lengths ------------------

@pytest.mark.parametrize("profile_name,site_name", [
    ("xMachine", "xMachine: Browser-Based Deep Learning & Inference"),
    ("DataBI", "DataBI: AI-Powered BI & Financial Automation"),
    ("Cloud Gallery", "Cloud Gallery: Next-Gen Cloud Storage Platform"),
    ("Blogrr - Social Media App", "Blogrr"),
])
def test_the_same_project_is_recognised(profile_name: str, site_name: str) -> None:
    """Otherwise the import adds a second copy of everything you already had."""
    assert same_entry(profile_name, site_name) is True


@pytest.mark.parametrize("one,other", [
    ("xRec", "xSearchPro - AI-Powered Enterprise Search"),
    ("Noketa", "NotePad: Dual-Engine"),
    ("AI Tool", "AI Platform"),
])
def test_different_projects_stay_different(one: str, other: str) -> None:
    assert same_entry(one, other) is False


def test_a_name_is_what_comes_before_the_colon() -> None:
    assert short_name("xMachine: Browser-Based Deep Learning") == short_name("xMachine")


# ---- merging without losing your work --------------------------------------

def test_your_own_wording_is_never_overwritten() -> None:
    """The whole risk of an import. Bullets you tuned for a kind of role must survive."""
    mine = [{"name": "DataBI", "bullets": ["My careful wording."], "tech": []}]
    theirs = [{"name": "DataBI: AI-Powered BI", "bullets": ["The site's blurb."],
               "tech": ["Next.js"], "url": "https://databi.example"}]

    merged, added, updated = merge_entries(mine, theirs, "name")

    assert added == [] and updated == ["DataBI: AI-Powered BI"]
    assert merged[0]["bullets"] == ["My careful wording."], "your bullets stay"
    assert merged[0]["tech"] == ["Next.js"], "what was empty is filled in"
    assert merged[0]["url"] == "https://databi.example"


def test_something_new_is_added() -> None:
    merged, added, _ = merge_entries([], [{"name": "Noketa", "bullets": ["New."]}], "name")

    assert added == ["Noketa"] and len(merged) == 1


def test_importing_twice_changes_nothing_the_second_time() -> None:
    theirs = [{"name": "Noketa", "bullets": ["New."], "tech": ["Next.js"]}]

    once, _, _ = merge_entries([], theirs, "name")
    twice, added, updated = merge_entries(once, theirs, "name")

    assert twice == once and added == [] and updated == []


# ---- the whole thing -------------------------------------------------------

class FakeSite:
    """The portfolio, answering the endpoints it has and 404ing the ones it has not."""

    def __init__(self, data: dict) -> None:
        self.data = data
        self.asked: list[str] = []


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "name": "Joshua",
        "projects": [{"name": "DataBI", "bullets": ["My wording."], "tech": [], "url": ""}],
        "experience": [{"company": "BLNR (Open Source)", "title": "Backend Developer",
                        "bullets": ["My wording."], "start": "2024", "end": "2025"}],
        "skills": {"backend": ["Node.js"]},
        "screening_defaults": {"notice_period": "2 weeks"},
    }, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch) -> FakeSite:
    fake = FakeSite({
        "projects": [SITE_PROJECT],
        "product-projects": [{**SITE_PROJECT, "title": "DataBI: AI-Powered BI",
                              "technologies": ["Next.js"], "githubUrl": ""}],
        "experience": [SITE_ROLE],
        "skills": [{"name": "Python", "category": "backend", "isVisible": True},
                   {"name": "Node.js", "category": "backend"},
                   {"name": "Hidden", "category": "backend", "isVisible": False}],
    })

    async def fetch(site_url: str, name: str):
        fake.asked.append(name)
        return fake.data.get(name)

    import portfolio
    monkeypatch.setattr(portfolio, "fetch", fetch)
    return fake


@pytest.mark.asyncio
async def test_an_import_adds_what_is_new_and_keeps_what_you_wrote(profile: Path, site) -> None:
    changes = await import_portfolio(profile)

    after = json.loads(profile.read_text(encoding="utf-8"))
    names = [p["name"] for p in after["projects"]]

    assert "xMachine: Browser-Based Deep Learning & Inference" in names
    assert len([n for n in names if n.lower().startswith("databi")]) == 1, "no second copy"
    databi = [p for p in after["projects"] if p["name"].lower().startswith("databi")][0]
    assert databi["bullets"] == ["My wording."]
    assert changes.touched is True


@pytest.mark.asyncio
async def test_the_rest_of_your_profile_is_left_alone(profile: Path, site) -> None:
    before = json.loads(profile.read_text(encoding="utf-8"))

    await import_portfolio(profile)

    after = json.loads(profile.read_text(encoding="utf-8"))
    assert after["screening_defaults"] == before["screening_defaults"]
    assert after["name"] == before["name"]


@pytest.mark.asyncio
async def test_a_copy_is_kept_before_anything_is_written(profile: Path, site) -> None:
    changes = await import_portfolio(profile)

    assert changes.backup is not None and changes.backup.exists()
    assert "DataBI" in changes.backup.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_dry_run_writes_nothing(profile: Path, site) -> None:
    before = profile.read_bytes()

    changes = await import_portfolio(profile, dry_run=True)

    assert profile.read_bytes() == before
    assert changes.backup is None and changes.touched is True


@pytest.mark.asyncio
async def test_importing_twice_leaves_the_file_alone(profile: Path, site) -> None:
    await import_portfolio(profile)
    after_once = profile.read_bytes()

    changes = await import_portfolio(profile)

    assert profile.read_bytes() == after_once
    assert changes.touched is False


@pytest.mark.asyncio
async def test_hidden_skills_are_not_imported(profile: Path, site) -> None:
    await import_portfolio(profile)

    skills = json.loads(profile.read_text(encoding="utf-8"))["skills"]
    assert "Python" in skills["backend"]
    assert "Hidden" not in skills["backend"]
    assert skills["backend"].count("Node.js") == 1, "already there, not added twice"


@pytest.mark.asyncio
async def test_an_endpoint_that_is_not_there_is_not_a_failure(profile: Path,
                                                              monkeypatch) -> None:
    """The site 404s some of these, and that must not stop the rest importing."""
    async def fetch(site_url: str, name: str):
        return [SITE_PROJECT] if name == "projects" else None

    import portfolio
    monkeypatch.setattr(portfolio, "fetch", fetch)

    changes = await import_portfolio(profile)

    assert changes.added.get("project(s)")
    assert "experience" in changes.unreachable and "skills" in changes.unreachable


@pytest.mark.asyncio
async def test_both_project_endpoints_are_read(profile: Path, site) -> None:
    """The site lists projects under two names, and both hold real work."""
    await import_portfolio(profile)

    assert "projects" in site.asked and "product-projects" in site.asked
