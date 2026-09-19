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


# ---- sending through a signed-in Gmail instead of a mail server ------------

def test_gmail_needs_no_credentials_at_all(settings) -> None:
    """The session is the credential, so there is no password anywhere in the project."""
    settings.email_transport = "gmail"

    assert EmailApplier(settings).missing_settings() == []


def test_smtp_still_says_what_it_needs(settings) -> None:
    settings.email_transport = "smtp"

    assert "SMTP_HOST" in EmailApplier(settings).missing_settings()


@pytest.mark.asyncio
async def test_gmail_without_a_browser_is_refused(settings, job) -> None:
    """It drives a real page, so there has to be one."""
    settings.email_transport = "gmail"
    draft = EmailApplier(settings).compose(job, "careers@example-corp.com", PROFILE, "Hi.", [])

    with pytest.raises(NotConfigured) as caught:
        await EmailApplier(settings, browser=None).send(draft)

    assert "browser" in str(caught.value).lower()


@pytest.mark.asyncio
async def test_a_lapsed_gmail_session_is_reported_not_guessed_at(settings) -> None:
    from email_apply import GmailTransport

    class SignedOutPage:
        url = "https://accounts.google.com/ServiceLogin?continue=mail"

    class Browser:
        page = SignedOutPage()

        async def goto(self, page, url):
            return url

    with pytest.raises(NotConfigured) as caught:
        await GmailTransport(settings, Browser()).open_mail()

    assert "make gmail-login" in str(caught.value)


@pytest.mark.asyncio
async def test_a_good_session_is_accepted(settings) -> None:
    from email_apply import GmailTransport

    class InboxPage:
        url = "https://mail.google.com/mail/u/0/#inbox"

    class Browser:
        page = InboxPage()

        async def goto(self, page, url):
            return url

    assert await GmailTransport(settings, Browser()).open_mail() is Browser.page


def test_the_smtp_message_still_says_gmail_is_an_option(settings, job) -> None:
    """Someone stuck on app passwords should be told there is another way."""
    settings.email_transport = "smtp"
    import asyncio

    draft = EmailApplier(settings).compose(job, "careers@example-corp.com", PROFILE, "Hi.", [])
    with pytest.raises(NotConfigured) as caught:
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            EmailApplier(settings).send(draft))

    assert "EMAIL_TRANSPORT=gmail" in str(caught.value)


# ---- the action flow: reaching all of this on purpose ---------------------

def _api(tmp_path: Path, **over):
    from api import create_app
    from database import Database

    settings = Settings(_env_file=None, db_path=tmp_path / "t.db",
                        overrides_path=tmp_path / "s.json", log_dir=tmp_path / "l",
                        audit_dir=tmp_path / "l" / "a", output_dir=tmp_path / "o",
                        user_data_dir=tmp_path / "p", company_file=tmp_path / "c.json",
                        template_dir=Path(__file__).resolve().parents[1] / "templates",
                        **over)
    db = Database(settings.db_path)
    db.init()
    from fastapi.testclient import TestClient

    return TestClient(create_app(settings, db)), settings


def test_there_is_a_way_to_apply_by_email_on_purpose(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Before this it could only be reached by chance, mid-scan, when a posting happened
    to have no form."""
    client, settings = _api(tmp_path, email_apply=True)
    asked = {}

    class StubPipeline:
        def __init__(self, *a, **k) -> None:
            self.gate = None

        async def email_one(self, url, to=None):
            asked.update(url=url, to=to)
            return {"status": "submitted", "to": to or "careers@example.com",
                    "job": {}, "resume_path": "cv.pdf"}

    import pipeline
    monkeypatch.setattr(pipeline, "Pipeline", StubPipeline)

    out = client.post("/api/email/apply",
                      json={"url": "example.com/jobs/1", "to": "careers@example.com"})

    assert out.status_code == 200 and out.json()["started"] is True
    assert out.json()["url"] == "https://example.com/jobs/1", "a bare host is made a URL"


def test_applying_by_email_is_refused_while_the_feature_is_off(tmp_path: Path) -> None:
    client, _ = _api(tmp_path, email_apply=False)

    out = client.post("/api/email/apply", json={"url": "https://example.com/j"})

    assert out.status_code == 400 and "Settings" in out.json()["detail"]


def test_a_link_is_required(tmp_path: Path) -> None:
    client, _ = _api(tmp_path, email_apply=True)

    assert client.post("/api/email/apply", json={"url": "   "}).status_code == 400


def test_the_email_account_can_be_checked_without_sending(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise the only way to learn the session has lapsed is a failed application."""
    client, _ = _api(tmp_path, email_apply=True, email_transport="gmail")

    class StubPipeline:
        def __init__(self, *a, **k) -> None:
            self.gate = None

        async def check_email_account(self):
            return {"transport": "gmail", "ready": False,
                    "detail": "Not signed in to Gmail in this browser profile."}

    import pipeline
    monkeypatch.setattr(pipeline, "Pipeline", StubPipeline)

    out = client.post("/api/email/check").json()

    assert out["ready"] is False and "Gmail" in out["detail"]


def test_an_smtp_setup_is_checked_without_opening_anything(settings) -> None:
    """There is no session to look at, so this is answered from the settings alone."""
    import asyncio

    from pipeline import Pipeline

    settings.email_transport = "smtp"
    p = Pipeline.__new__(Pipeline)
    p.s = settings

    out = asyncio.run(p.check_email_account())

    assert out["transport"] == "smtp" and out["ready"] is False
    assert "SMTP_HOST" in out["detail"]


# ---- which address a page is actually offering -----------------------------

def test_the_address_has_to_sit_next_to_the_wording_that_offers_it() -> None:
    """Otherwise any address on any page counts, including a blog author's."""
    from email_apply import application_address

    offered = "We are hiring. To apply, send your CV to careers@northwind.com."
    incidental = ("An article about hiring. " + "Filler sentence. " * 40 +
                  "Reach the author at hello@somebody.blog.")

    assert application_address(offered, "https://northwind.com/j") == "careers@northwind.com"
    assert application_address(incidental, "https://somebody.blog/p") is None


def test_a_job_board_is_never_the_employer() -> None:
    """The reported problem: scans returned board pages with nobody to write to."""
    from email_apply import application_address, is_job_board

    board = ("Data Analyst jobs. Interested in working at Indeed? Send your CV to "
             "careers@indeed.com.")

    assert application_address(board, "https://www.indeed.com/q-data-analyst") is None
    assert application_address("Apply: jobs@greenhouse.io", "https://a.blog/p") is None
    assert is_job_board("https://www.linkedin.com/jobs/1") is True
    assert is_job_board("https://northwind.com/jobs/1") is False


def test_the_address_named_for_applications_is_preferred() -> None:
    from email_apply import application_address

    page = "For press: press@nw.com. To apply, send your CV to careers@nw.com."

    assert application_address(page, "https://nw.com/j") == "careers@nw.com"


def test_an_address_that_is_never_an_application_is_still_refused() -> None:
    from email_apply import application_address

    assert application_address("To apply send your CV to noreply@nw.com",
                               "https://nw.com/j") is None


def test_a_page_offering_nothing_yields_nothing() -> None:
    from email_apply import application_address

    assert application_address("Apply through our portal.", "https://nw.com/j") is None
    assert application_address("", "https://nw.com/j") is None


def test_the_wording_may_come_after_the_address() -> None:
    from email_apply import application_address

    assert application_address("careers@nw.com is where applications go.",
                               "https://nw.com/j") == "careers@nw.com"


# ---- never say sent without looking ----------------------------------------

class GmailPage:
    """Enough of the Gmail compose window to act out each ending."""

    url = "https://mail.google.com/mail/u/0/#inbox"

    def __init__(self, ending: str) -> None:
        self.ending = ending
        self.sent = False
        self.keys: list[str] = []
        self.keyboard = self._Keyboard(self)

    class _Keyboard:
        def __init__(self, page: "GmailPage") -> None:
            self.page = page

        async def press(self, key: str) -> None:
            self.page.keys.append(key)

    class _Loc:
        def __init__(self, sel: str, page: "GmailPage") -> None:
            self.sel, self.page = sel, page

        @property
        def first(self):
            return self

        @property
        def last(self):
            return self

        async def wait_for(self, **_k):
            if not await self.count():
                raise RuntimeError("not on the page")

        async def count(self) -> int:
            page, sel = self.page, self.sel
            if "alertdialog" in sel or "Kj-JD" in sel or "one recipient" in sel:
                return 1 if (page.ending == "refused" and page.sent) else 0
            if "Message Body" in sel or 'role="textbox"' in sel:
                return 0 if (page.sent and page.ending == "sent") else 1
            if "Message sent" in sel or "Sending" in sel or "has-text" in sel:
                return 0
            return 1

        async def is_visible(self) -> bool:
            return bool(await self.count())

        async def inner_text(self) -> str:
            return ("Please specify at least one recipient."
                    if "alertdialog" in self.sel else "")

        async def click(self) -> None:
            if "Send" in self.sel:
                self.page.sent = True

        async def fill(self, _v: str) -> None:
            return None

        async def type(self, _v: str, delay: int = 0) -> None:
            return None

        async def set_input_files(self, _f) -> None:
            return None

    def locator(self, sel: str):
        return self._Loc(sel, self)


class GmailBrowser:
    def __init__(self, page: GmailPage) -> None:
        self.page = page

    async def goto(self, page, url):
        return url


async def _send(settings, ending: str):
    from email_apply import GmailTransport

    page = GmailPage(ending)
    GmailTransport.CONFIRM_TIMEOUT = 1.5
    draft = EmailApplier(settings).compose(
        JobPosting(job_id="j", url="https://x.com/j", title="Analyst", company="X"),
        "careers@example.com", PROFILE, "Hello.", [])
    await GmailTransport(settings, GmailBrowser(page)).send(draft)
    return page


@pytest.mark.asyncio
async def test_a_message_gmail_actually_sent_is_reported_sent(settings) -> None:
    page = await _send(settings, "sent")

    assert page.sent is True


@pytest.mark.asyncio
async def test_a_message_gmail_refused_is_not_reported_sent(settings) -> None:
    """The bug: Send was clicked and success logged without looking, so a message
    sitting in Drafts was recorded as an application you had made."""
    from email_apply import SendRefused

    with pytest.raises(SendRefused) as caught:
        await _send(settings, "refused")

    assert "at least one recipient" in str(caught.value)


@pytest.mark.asyncio
async def test_silence_from_gmail_is_not_taken_as_success(settings) -> None:
    from email_apply import SendRefused

    with pytest.raises(SendRefused) as caught:
        await _send(settings, "stuck")

    assert "Drafts" in str(caught.value)
    assert "Nothing has been recorded as sent" in str(caught.value)


@pytest.mark.asyncio
async def test_the_recipient_is_committed_before_sending(settings) -> None:
    """Gmail only accepts a recipient once it becomes a chip. Filling the box leaves
    the text uncommitted and Send then refuses the whole message."""
    page = await _send(settings, "sent")

    assert "Tab" in page.keys


@pytest.mark.asyncio
async def test_a_refusal_is_recorded_as_failed_not_submitted(settings, tmp_path: Path,
                                                             monkeypatch) -> None:
    """The whole point: the database must not say you applied when you did not."""
    from database import STATUS_FAILED, Database
    from email_apply import SendRefused
    from pipeline import Pipeline

    db = Database(tmp_path / "t.db")
    db.init()
    p = Pipeline.__new__(Pipeline)
    p.s, p.db, p.gate, p.stats = settings, db, None, {}
    p.profile = PROFILE
    p.resumes = None
    monkeypatch.setattr(p, "_bump", lambda status: None)
    monkeypatch.setattr(p, "write_audit",
                        lambda *a, **k: Path(tmp_path / "audit.json"))

    class Refusing:
        def __init__(self, *a, **k) -> None:
            pass

        transport = "gmail"

        def missing_settings(self):
            return []

        def compose(self, *a, **k):
            from email_apply import Draft

            return Draft(to="careers@example.com", subject="s", body="b", attachments=[])

        def save_preview(self, *a, **k):
            return tmp_path / "preview.eml"

        async def send(self, draft):
            raise SendRefused("Gmail did not confirm sending.")

    import email_apply
    monkeypatch.setattr(email_apply, "EmailApplier", Refusing)
    settings.email_auto_send = True

    async def cover():
        return tmp_path / "cover.pdf", "letter"

    cover.cache = {}
    from ai_agent import JobAnalysis

    analysis = JobAnalysis(job_title="Analyst", company_name="X", match_score=70,
                           match_rationale="ok", missing_requirements=[],
                           highlighted_skills=[], tailored_summary="s",
                           tailored_bullets=[], answers=[])
    resume = tmp_path / "cv.pdf"
    resume.write_bytes(b"%PDF")

    status = await p._apply_by_email(
        JobPosting(job_id="j", url="https://x.com/j", title="Analyst", company="X"),
        analysis, resume, "nothing trimmed", "careers@example.com", cover)

    assert status == STATUS_FAILED
    assert "did not confirm" in db.list()[0]["notes"]
