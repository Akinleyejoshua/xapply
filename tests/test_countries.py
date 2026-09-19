"""Country filtering: postings name cities and regions far more often than countries."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from countries import ANYWHERE, COUNTRIES, country_matches, countries_for  # noqa: E402
from database import Database  # noqa: E402
from discovery import AshbyBoardSource, location_matches  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("location,wanted,expected", [
    # the country is named outright
    ("Remote, United States", ["United States"], True),
    ("Lagos, Nigeria", ["Nigeria"], True),
    ("Bangalore, India", ["India"], True),
    # only a city is named
    ("San Francisco", ["United States"], True),
    ("London", ["United Kingdom"], True),
    ("Lagos", ["Nigeria"], True),
    ("Bengaluru", ["India"], True),
    ("Sao Paulo", ["Brazil"], True),
    # regional shorthand covers its members
    ("Remote, EMEA", ["Germany"], True),
    ("Remote, EMEA", ["India"], False),
    ("Remote - APAC", ["Singapore"], True),
    ("Remote, LATAM", ["Brazil"], True),
    # several locations in one string
    ("Remote, Canada; Remote, United Kingdom", ["United Kingdom"], True),
    ("Remote, Canada; Remote, United Kingdom", ["Nigeria"], False),
    # wrong country
    ("Berlin", ["United States"], False),
    ("Tokyo, Japan", ["Nigeria"], False),
    # anywhere
    ("Anywhere", [ANYWHERE], True),
    ("San Francisco", [ANYWHERE], False),
    # no filter at all
    ("anything", [], True),
    ("anything", None, True),
])
def test_country_matches(location, wanted, expected) -> None:
    assert country_matches(location, wanted) is expected


def test_country_matching_is_word_bounded() -> None:
    """A substring must not count: 'Usually remote' is not the United States."""
    assert country_matches("Usually remote", ["United States"]) is False
    assert country_matches("Indiana Township", ["India"]) is False
    assert country_matches("Chinatown, San Francisco", ["China"]) is False


def test_countries_for_reports_everything_named() -> None:
    found = countries_for("Remote, Canada; Remote, United Kingdom; Remote, Israel")
    assert set(found) == {"Canada", "United Kingdom", "Israel"}
    assert countries_for("San Francisco, Seattle, NYC") == ["United States"]
    assert countries_for("") == []


def test_country_list_is_sane() -> None:
    assert COUNTRIES[0] == ANYWHERE                 # the catch-all sorts first
    assert COUNTRIES[1:] == sorted(COUNTRIES[1:])   # the rest are alphabetical
    assert len(set(COUNTRIES)) == len(COUNTRIES)    # no duplicates
    for name in ("United States", "United Kingdom", "Nigeria", "India", "Germany"):
        assert name in COUNTRIES


# ---- how it stacks with the other location rules -------------------------


@pytest.mark.parametrize("location,workplace,remote_only,countries,expected", [
    ("Remote, United Kingdom", None, True, ["United Kingdom"], True),
    ("London", "Hybrid", True, ["United Kingdom"], False),     # right country, not remote
    ("Remote, United States", None, True, ["United Kingdom"], False),  # remote, wrong country
    ("London", "Hybrid", False, ["United Kingdom"], True),     # not filtering on remote
    ("Bangalore, India", None, False, ["India"], True),
])
def test_location_matches_stacks_remote_and_country(location, workplace, remote_only,
                                                    countries, expected) -> None:
    assert location_matches(location, "Remote", remote_only, workplace, countries) is expected


def test_country_filter_overrides_the_location_text() -> None:
    """With countries chosen, the free-text location is ignored so the two cannot fight."""
    assert location_matches("Berlin", "Lisbon", False, None, ["Germany"]) is True
    assert location_matches("Berlin", "Berlin", False, None, ["Portugal"]) is False


@pytest.mark.asyncio
async def test_board_source_applies_the_country_filter(monkeypatch, tmp_path: Path) -> None:
    from tests.test_discovery_llm import patch_client

    payload = {"jobs": [
        {"id": "uk", "title": "Software Engineer", "isListed": True, "location": "London",
         "workplaceType": "Remote", "descriptionPlain": "a " * 200,
         "jobUrl": "https://jobs.ashbyhq.com/acme/uk"},
        {"id": "ng", "title": "Software Engineer", "isListed": True, "location": "Lagos, Nigeria",
         "workplaceType": "Remote", "descriptionPlain": "b " * 200,
         "jobUrl": "https://jobs.ashbyhq.com/acme/ng"},
        {"id": "us", "title": "Software Engineer", "isListed": True, "location": "San Francisco",
         "workplaceType": "Remote", "descriptionPlain": "c " * 200,
         "jobUrl": "https://jobs.ashbyhq.com/acme/us"},
    ]}
    base = dict(output_dir=tmp_path, log_dir=tmp_path, audit_dir=tmp_path, user_data_dir=tmp_path,
                overrides_path=tmp_path / "s.json", company_file=ROOT / "companies.json",
                search_queries=["Software Engineer"], max_jobs_per_company=50, discovery_delay_s=0)

    async def discover(**kw):
        s = Settings(db_path=tmp_path / f"{kw.get('countries', ['x'])[0]}.db", **base, **kw)
        d = Database(s.db_path)
        d.init()
        src = AshbyBoardSource(s, d, None, tokens=["acme"])
        patch_client(monkeypatch, src, {"posting-api/job-board/acme": payload})
        return {j.job_id[-2:] for j in await src.discover()}

    assert await discover(countries=["Nigeria"]) == {"ng"}
    assert await discover(countries=["United Kingdom"]) == {"uk"}
    assert await discover(countries=["Nigeria", "United Kingdom"]) == {"ng", "uk"}


def test_api_validates_and_persists_countries(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from api import create_app

    s = Settings(db_path=tmp_path / "t.db", output_dir=tmp_path, log_dir=tmp_path,
                 audit_dir=tmp_path, user_data_dir=tmp_path,
                 overrides_path=tmp_path / "settings.local.json",
                 profile_path=ROOT / "profile.example.json", company_file=ROOT / "companies.json")
    db = Database(s.db_path)
    db.init()
    client = TestClient(create_app(s, db))

    cfg = client.get("/api/config").json()
    assert cfg["countries"] == []
    assert "Nigeria" in cfg["known_countries"]

    body = client.patch("/api/config", json={"countries": ["Nigeria", "United Kingdom"]}).json()
    assert body["countries"] == ["Nigeria", "United Kingdom"]
    assert "countries" in body["saved"]

    bad = client.patch("/api/config", json={"countries": ["Atlantis"]})
    assert bad.status_code == 422 and "Atlantis" in bad.json()["detail"]
    assert client.get("/api/config").json()["countries"] == ["Nigeria", "United Kingdom"]

    restarted = Settings(overrides_path=s.overrides_path)
    restarted.load_overrides()
    assert restarted.countries == ["Nigeria", "United Kingdom"]


def test_cli_resolves_country_aliases() -> None:
    from main import resolve_countries

    assert resolve_countries("nigeria, uk, USA") == ["Nigeria", "United Kingdom", "United States"]
    assert resolve_countries("anywhere") == [ANYWHERE]
    assert resolve_countries("Atlantis") == []          # unknown names are dropped, with a message


def test_dashboard_has_the_country_picker() -> None:
    html = (ROOT / "static" / "index.html").read_text()
    for marker in ('id="countryPick"', 'id="countryChips"', 'id="cfgCountryChips"',
                   "function renderCountries()", "known_countries"):
        assert marker in html, marker
