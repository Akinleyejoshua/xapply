"""What makes two links the same job, and what makes them different ones.

A posting is reachable at several addresses: the board page, the apply page and the
embedded form all describe one job. The identifier has to come from the posting, never
from the page being looked at. Taking the last path segment broke that: an Ashby link
ending "/application" reduced to "ashby-application", so every Ashby posting shared one
identifier. The first was recorded, and the next one you picked was refused as already
applied, naming a job you had never seen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import STATUS_SUBMITTED, Database  # noqa: E402
from models import JobPosting, job_id_from_url  # noqa: E402

GOODY = "https://jobs.ashbyhq.com/goody/df70c3ea-1233-463c-8a7f-70e4ebe3342d"
REVENUECAT = "https://jobs.ashbyhq.com/revenuecat/aaaaaaaa-1111-2222-3333-444444444444"
GITLAB = "https://job-boards.greenhouse.io/gitlab/jobs/8773006002"
ASPEN = "https://job-boards.greenhouse.io/aspenviewtech/jobs/4389940009"


def test_two_different_ashby_postings_are_two_different_jobs() -> None:
    """The bug: both ended in "/application", so both were the same job."""
    assert job_id_from_url(GOODY + "/application") != job_id_from_url(REVENUECAT + "/application")


@pytest.mark.parametrize("suffix", ["", "/application", "/apply", "/"])
def test_one_posting_keeps_one_identity_across_its_addresses(suffix: str) -> None:
    assert job_id_from_url(GOODY + suffix) == job_id_from_url(GOODY)


def test_an_embedded_greenhouse_form_is_the_same_job_as_its_board_page() -> None:
    """The embed URL names the job in its query, and "job_app" in its path."""
    embed = "https://job-boards.greenhouse.io/embed/job_app?for=gitlab&token=8773006002"
    assert job_id_from_url(embed) == job_id_from_url(GITLAB)
    assert job_id_from_url(GITLAB + "?gh_jid=8773006002") == job_id_from_url(GITLAB)


def test_the_page_name_never_becomes_the_identity() -> None:
    for url in (GOODY + "/application", GITLAB + "/application",
                "https://jobs.lever.co/acme/bbbbbbbb-1111-2222-3333-555555555555/apply"):
        assert not job_id_from_url(url).endswith(("-application", "-apply", "-job_app"))


def test_different_greenhouse_postings_stay_different() -> None:
    assert job_id_from_url(GITLAB) != job_id_from_url(ASPEN)


def test_an_address_with_nothing_identifying_is_its_own_identity() -> None:
    a = job_id_from_url("https://careers.example.com/roles/apply")
    b = job_id_from_url("https://careers.example.com/other/apply")
    assert a != b and a.startswith("url-")


# ---- repairing what was already stored ------------------------------------

def test_rows_saved_under_the_page_name_are_re_keyed(tmp_path: Path) -> None:
    """Without this the bad row keeps refusing every other Ashby posting."""
    db = Database(tmp_path / "t.db")
    db.init()
    db.record(JobPosting(job_id="ashby-application", url=GOODY + "/application",
                         apply_url=GOODY + "/application", company="Goody",
                         title="Data Analyst", source="urls"), STATUS_SUBMITTED)

    db.init()                                    # as a restart would

    assert db.find_job("urls", "ashby-application") is None
    repaired = db.find_job("urls", job_id_from_url(GOODY))
    assert repaired is not None and repaired["company"] == "Goody"


def test_a_different_posting_is_no_longer_refused(tmp_path: Path) -> None:
    """The symptom this fixes, stated as the user met it."""
    db = Database(tmp_path / "t.db")
    db.init()
    db.record(JobPosting(job_id="ashby-application", url=GOODY + "/application",
                         company="Goody", title="Data Analyst", source="urls"),
              STATUS_SUBMITTED)
    db.init()

    assert not db.has_job("urls", job_id_from_url(REVENUECAT + "/application"))


def test_the_repair_leaves_correctly_keyed_rows_alone(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init()
    db.record(JobPosting(job_id=job_id_from_url(GITLAB), url=GITLAB, company="GitLab",
                         title="Engineer", source="urls"), STATUS_SUBMITTED)
    before = db.list()

    db.init()

    assert db.list() == before


def test_the_repair_is_safe_to_run_twice(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init()
    db.record(JobPosting(job_id="ashby-application", url=GOODY + "/application",
                         company="Goody", title="Data Analyst", source="urls"),
              STATUS_SUBMITTED)

    db.init()
    once = db.list()
    db.init()

    assert db.list() == once and len(once) == 1
