"""Applying by email, for postings that ask for one instead of offering a form.

Sending email is the one thing this tool does that cannot be undone. So it is off by
default, it never invents a recipient, and it stops for you to read the draft before
anything leaves. These tests hold those three things in place.
"""
from __future__ import annotations

import sys
from email import message_from_bytes, policy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from email_apply import (EmailApplier, NotConfigured, find_address, find_addresses,  # noqa: E402
                         plausible)
from models import JobPosting  # noqa: E402

POSTING = """
We are hiring a Data Analyst. Send your CV to Careers@Example-Corp.com.
Questions about our privacy policy go to privacy@example-corp.com, and please do not
reply to noreply@example-corp.com. Our recruiter is jane.doe@example-corp.com.
<a href="mailto:jobs@example-corp.com">Apply here</a>
"""

PROFILE = {"name": "Joshua Akinleye", "email": "joshua@example.com",
           "phone": "+2348131519518", "github": "https://github.com/jo",
           "linkedin": "https://linkedin.com/in/jo"}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, log_dir=tmp_path, output_dir=tmp_path,
                    audit_dir=tmp_path, db_path=tmp_path / "t.db",
                    user_data_dir=tmp_path, overrides_path=tmp_path / "s.json")


@pytest.fixture
def job() -> JobPosting:
    return JobPosting(job_id="j1", url="https://example-corp.com/jobs/1",
                      title="Data Analyst", company="Example Corp", description=POSTING)


# ---- finding who to write to ----------------------------------------------

def test_the_address_meant_for_applications_wins(job: JobPosting) -> None:
    """A posting carries several addresses and only one is being offered."""
    assert find_address(job) == "careers@example-corp.com"


def test_every_usable_address_is_ranked(job: JobPosting) -> None:
    assert find_addresses(job.description) == [
        "careers@example-corp.com", "jobs@example-corp.com", "jane.doe@example-corp.com"]


@pytest.mark.parametrize("address", [
    "noreply@example-corp.com", "no-reply@x.com", "privacy@x.com", "support@x.com",
    "legal@x.com", "someone@example.com", "not-an-address", "",
])
def test_addresses_that_are_never_an_application_are_refused(address: str) -> None:
    assert plausible(address) is False


def test_a_posting_with_no_address_yields_none() -> None:
    job = JobPosting(job_id="x", url="https://x.com", description="Apply on our website.")
    assert find_address(job) is None


# ---- composing -------------------------------------------------------------

def test_the_message_carries_the_role_and_the_documents(settings, job, tmp_path) -> None:
    resume = tmp_path / "Joshua_Resume.pdf"
    resume.write_bytes(b"%PDF-1.4 resume")
    letter = tmp_path / "Joshua_Cover.pdf"
    letter.write_bytes(b"%PDF-1.4 letter")
    settings.email_from = "joshua@example.com"

    draft = EmailApplier(settings).compose(
        job, "careers@example-corp.com", PROFILE, "I would like to apply.", [resume, letter])

    assert draft.to == "careers@example-corp.com"
    assert "Data Analyst" in draft.subject and "Example Corp" in draft.subject
    assert [p.name for p in draft.attachments] == ["Joshua_Resume.pdf", "Joshua_Cover.pdf"]
    assert "I would like to apply." in draft.body
    assert "joshua@example.com" in draft.body, "a recruiter needs a way to reply"


def test_an_attachment_that_is_not_there_is_left_out(settings, job, tmp_path) -> None:
    settings.email_from = "joshua@example.com"
    draft = EmailApplier(settings).compose(job, "careers@example-corp.com", PROFILE,
                                           "Hello.", [tmp_path / "gone.pdf"])
    assert draft.attachments == []


def test_the_message_is_ascii_clean(settings, job, tmp_path) -> None:
    """A typographic dash in a subject line is the sort of thing that trips a filter."""
    settings.email_from = "joshua@example.com"
    draft = EmailApplier(settings).compose(
        job, "careers@example-corp.com", PROFILE, "I built a data‑quality score — it helped.", [])
    assert "‑" not in draft.body and "—" not in draft.body


def test_the_built_message_is_a_real_email(settings, job, tmp_path) -> None:
    resume = tmp_path / "cv.pdf"
    resume.write_bytes(b"%PDF-1.4")
    settings.email_from = "joshua@example.com"

    draft = EmailApplier(settings).compose(job, "careers@example-corp.com", PROFILE,
                                           "Hello.", [resume])
    parsed = message_from_bytes(bytes(draft.as_message()), policy=policy.default)

    assert parsed["To"] == "careers@example-corp.com"
    assert parsed["From"] == "joshua@example.com"
    assert parsed["Reply-To"] == "joshua@example.com"
    names = [p.get_filename() for p in parsed.iter_attachments()]
    assert names == ["cv.pdf"]


# ---- refusing to send ------------------------------------------------------

def test_it_says_what_is_missing_before_anything_is_attempted(settings) -> None:
    assert EmailApplier(settings).missing_settings() == [
        "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "EMAIL_FROM"]


@pytest.mark.asyncio
async def test_sending_without_a_mail_server_is_refused(settings, job, tmp_path) -> None:
    draft = EmailApplier(settings).compose(job, "careers@example-corp.com", PROFILE, "Hi.", [])

    with pytest.raises(NotConfigured) as caught:
        await EmailApplier(settings).send(draft)

    assert "SMTP_HOST" in str(caught.value)


@pytest.mark.asyncio
async def test_a_message_with_no_recipient_is_refused(settings, job) -> None:
    settings.smtp_host, settings.smtp_user = "smtp.example.com", "me@example.com"
    settings.smtp_password, settings.email_from = "secret", "me@example.com"
    draft = EmailApplier(settings).compose(job, "", PROFILE, "Hi.", [])

    with pytest.raises(ValueError):
        await EmailApplier(settings).send(draft)


@pytest.mark.asyncio
async def test_what_is_sent_is_what_was_composed(settings, job, tmp_path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """The one test that touches sending. No socket is opened."""
    settings.smtp_host, settings.smtp_user = "smtp.example.com", "me@example.com"
    settings.smtp_password, settings.email_from = "secret", "me@example.com"
    sent = {}

    mailer = EmailApplier(settings)
    monkeypatch.setattr(mailer, "_send_blocking", lambda draft: sent.update(draft=draft))
    draft = mailer.compose(job, "careers@example-corp.com", PROFILE, "Hello.", [])

    await mailer.send(draft)

    assert sent["draft"].to == "careers@example-corp.com"


def test_the_draft_is_written_where_you_can_read_it(settings, job, tmp_path) -> None:
    settings.email_from = "joshua@example.com"
    draft = EmailApplier(settings).compose(job, "careers@example-corp.com", PROFILE, "Hi.", [])

    path = EmailApplier(settings).save_preview(draft, job)

    assert path.exists() and path.suffix == ".eml"
    assert b"careers@example-corp.com" in path.read_bytes()


# ---- the settings themselves -----------------------------------------------

def test_it_is_off_until_you_turn_it_on(settings) -> None:
    assert settings.email_apply is False
    assert settings.email_auto_send is False, "sending unread is a separate decision"


def test_credentials_are_never_written_to_the_ui_settings_file() -> None:
    from config import PERSISTED_KEYS

    for secret in ("smtp_password", "smtp_user", "smtp_host", "email_from",
                   "email_reply_to", "email_auto_send"):
        assert secret not in PERSISTED_KEYS, secret


# ---- finding the postings in the first place -------------------------------

def test_an_aggregator_listing_with_no_form_is_kept_when_email_is_on(settings,
                                                                     tmp_path: Path) -> None:
    """These listings used to be discarded as dead ends. Many are not: the posting says
    to email a CV, which is the whole hiring process for a lot of smaller employers."""
    from database import Database
    from discovery import RemoteOKSource

    db = Database(tmp_path / "t.db")
    db.init()
    settings.email_apply = True
    source = RemoteOKSource(settings, db, None)

    assert source.email_route("Send your CV to careers@example-corp.com", "") == \
        "careers@example-corp.com"


def test_those_listings_stay_discarded_when_email_is_off(settings, tmp_path: Path) -> None:
    from database import Database
    from discovery import RemoteOKSource

    db = Database(tmp_path / "t.db")
    db.init()
    settings.email_apply = False

    assert RemoteOKSource(settings, db, None).email_route(
        "Send your CV to careers@example-corp.com", "") is None


def test_a_listing_with_no_address_is_still_a_dead_end(settings, tmp_path: Path) -> None:
    from database import Database
    from discovery import HimalayasSource

    db = Database(tmp_path / "t.db")
    db.init()
    settings.email_apply = True

    assert HimalayasSource(settings, db, None).email_route(
        "Apply through our website.", "https://example.com/jobs/1") is None
