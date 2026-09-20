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
    """Preferences are saved. Credentials are not, and stay in .env."""
    from config import PERSISTED_KEYS

    for secret in ("smtp_password", "smtp_user", "smtp_host", "email_from",
                   "email_reply_to", "gmail_url"):
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


def test_a_job_boards_own_address_is_never_the_employers() -> None:
    """A board index page offers only its own address, so it yields nothing."""
    from email_apply import application_address

    board = ("Data Analyst jobs. Interested in working at Indeed? Send your CV to "
             "careers@indeed.com.")

    assert application_address(board, "https://www.indeed.com/q-data-analyst") is None
    assert application_address("Apply: jobs@greenhouse.io", "https://a.blog/p") is None


def test_a_recruiters_post_is_a_posting() -> None:
    """The reported failure: rejecting a page by its domain threw away LinkedIn and X
    posts, which is exactly where somebody writes "send your CV to"."""
    from email_apply import application_address

    post = "URGENT: hiring a Data Engineer. Send your CV to jane@ascendion.com"

    assert application_address(
        post, "https://www.linkedin.com/posts/y_urgent") == "jane@ascendion.com"
    assert application_address(
        post, "https://x.com/r/status/210117") == "jane@ascendion.com"


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
        self.visited: list[str] = []
        self.keyboard = self._Keyboard(self)

    #: What the Sent row will say, set by the test that needs it to differ.
    subject = "Application for Analyst at X - Joshua Akinleye"

    async def goto(self, url: str, **_k) -> None:
        """Gmail is asked what it actually sent, so the Sent view is navigated to."""
        self.visited.append(url)

    async def reload(self, **_k) -> None:
        """Hash navigation does not always re-run the search, so it is forced."""
        return None

    class _Keyboard:
        def __init__(self, page: "GmailPage") -> None:
            self.page = page

        async def press(self, key: str) -> None:
            self.page.keys.append(key)
            # Gmail's own send shortcut, which is how the transport sends by default.
            if key.endswith("+Enter"):
                self.page.sent = True

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
            if sel == "tr.zA":            # a row in the Sent list
                return 1 if (page.sent and page.ending == "sent") else 0
            return 1

        async def all_inner_texts(self) -> list:
            page = self.page
            if self.sel != "tr.zA" or not (page.sent and page.ending == "sent"):
                return []
            return [f"To: careers, {page.subject}, has attachment, 11:25 PM"]

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
    page.subject = draft.subject
    GmailTransport.SENT_TIMEOUT = 3.0
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


def test_the_compose_window_must_really_be_open_before_anything_is_claimed() -> None:
    """A compose that never opened looks exactly like one that closed after sending,
    which is how a message that was never written got reported as sent."""
    import inspect

    from email_apply import GmailTransport

    source = inspect.getsource(GmailTransport.send)
    assert "_present(page, self.BODY)" in source
    assert source.index("_present(page, self.BODY)") < source.index("_dispatch")


def test_gmail_is_asked_what_it_actually_sent() -> None:
    """Reading the compose window is not evidence. The Sent folder is."""
    import inspect

    from email_apply import GmailTransport

    assert "verify_in_sent" in inspect.getsource(GmailTransport._dispatch)
    assert "in%3Asent" in GmailTransport.SENT_SEARCH


@pytest.mark.asyncio
async def test_nothing_in_sent_means_it_was_not_sent(settings) -> None:
    from email_apply import GmailTransport, SendRefused

    page = GmailPage("sent")
    page.sent = True

    class Empty(GmailPage._Loc):
        async def count(self):
            return 0

        async def all_inner_texts(self):
            return []

    page.locator = lambda sel: Empty(sel, page)
    GmailTransport.SENT_TIMEOUT = 2.0
    draft = EmailApplier(settings).compose(
        JobPosting(job_id="j", url="https://x.com/j", title="Analyst", company="X"),
        "careers@example.com", PROFILE, "Hello.", [])

    with pytest.raises(SendRefused) as caught:
        await GmailTransport(settings, GmailBrowser(page)).verify_in_sent(page, draft)

    assert "nothing in Sent" in str(caught.value)


def test_whether_to_ask_before_sending_is_yours_and_is_remembered() -> None:
    from config import PERSISTED_KEYS

    assert "email_auto_send" in PERSISTED_KEYS


def test_asking_first_is_still_the_shipped_default() -> None:
    """Your own choice is saved separately. Out of the box it stops and shows you."""
    assert Settings(_env_file=None).email_auto_send is False


@pytest.mark.asyncio
async def test_a_different_message_in_sent_is_not_this_one(settings) -> None:
    """Counting rows was the weak check that misled twice. A Sent folder full of older
    applications has rows whatever happened just now."""
    from email_apply import GmailTransport, SendRefused

    page = GmailPage("sent")
    page.sent = True
    page.subject = "Application for Something Else Entirely at Another Company"
    GmailTransport.SENT_TIMEOUT = 2.0
    draft = EmailApplier(settings).compose(
        JobPosting(job_id="j", url="https://x.com/j", title="Analyst", company="X"),
        "careers@example.com", PROFILE, "Hello.", [])

    with pytest.raises(SendRefused):
        await GmailTransport(settings, GmailBrowser(page)).verify_in_sent(page, draft)


def test_wrapping_and_ellipses_do_not_break_the_match() -> None:
    from email_apply import _comparable

    subject = "Application for Researcher and Data Analyst Role at London Plus"
    row = ("To: lucy,  Application for Researcher\nand Data Analyst Role at London "
           "Plus… , has attachment")

    assert _comparable(subject)[:50] in _comparable(row)


@pytest.mark.asyncio
async def test_the_keyboard_shortcut_is_tried_when_the_button_does_not_send(settings) -> None:
    """Seen live: clicking Send closed the compose window and left the message in
    Drafts, which looks exactly like success from the window and is not."""
    from email_apply import GmailTransport, SendRefused

    tried: list[str] = []

    class Transport(GmailTransport):
        async def _click_send(self, page):
            tried.append("button")

        async def _press_send(self, page):
            tried.append("shortcut")

        async def confirm_sent(self, page, draft):
            return None

        async def verify_in_sent(self, page, draft):
            if tried == ["shortcut"]:
                raise SendRefused("nothing in Sent")   # the first way did not send it

        async def _reopen_draft(self, page):
            return None

    draft = EmailApplier(settings).compose(
        JobPosting(job_id="j", url="https://x.com/j", title="Analyst", company="X"),
        "careers@example.com", PROFILE, "Hello.", [])

    await Transport(settings, GmailBrowser(GmailPage("sent")))._dispatch(None, draft)

    assert tried == ["shortcut", "button"], "both ways must be tried"


@pytest.mark.asyncio
async def test_when_neither_way_works_the_draft_is_left_for_you(settings) -> None:
    from email_apply import GmailTransport, SendRefused

    class Transport(GmailTransport):
        async def _click_send(self, page):
            return None

        async def _press_send(self, page):
            return None

        async def confirm_sent(self, page, draft):
            return None

        async def verify_in_sent(self, page, draft):
            raise SendRefused("nothing in Sent")

        async def _reopen_draft(self, page):
            return None

    draft = EmailApplier(settings).compose(
        JobPosting(job_id="j", url="https://x.com/j", title="Analyst", company="X"),
        "careers@example.com", PROFILE, "Hello.", [])

    with pytest.raises(SendRefused) as caught:
        await Transport(settings, GmailBrowser(GmailPage("stuck")))._dispatch(None, draft)

    assert "in your Drafts" in str(caught.value)
    assert "Nothing was recorded as sent" in str(caught.value)


# ---- saying nothing rather than saying "Unknown" ---------------------------

def test_a_company_nobody_knows_is_left_out_of_the_subject(settings, tmp_path) -> None:
    """A model asked for a company it cannot know answers "Unknown", and repeating it
    puts "at Unknown" in front of a recruiter."""
    settings.email_from = "joshua@example.com"

    for company in ("Unknown", "", "N/A", "?"):
        job = JobPosting(job_id="j", url="https://x.com/j", title="Data Analyst",
                         company=company)
        subject = EmailApplier(settings).compose(job, "a@b.com", PROFILE, "Hi.", []).subject
        assert subject == "Application for Data Analyst - Joshua Akinleye", company


def test_a_company_that_is_known_is_named(settings) -> None:
    settings.email_from = "joshua@example.com"
    job = JobPosting(job_id="j", url="https://x.com/j", title="Data Analyst",
                     company="Northwind")

    subject = EmailApplier(settings).compose(job, "a@b.com", PROFILE, "Hi.", []).subject

    assert subject == "Application for Data Analyst at Northwind - Joshua Akinleye"


def test_a_whole_advert_is_not_used_as_a_subject(settings) -> None:
    """A post scraped from social media puts the entire advert in the page title."""
    settings.email_from = "joshua@example.com"
    job = JobPosting(job_id="j", url="https://x.com/j", company="",
                     title="Ganesh Reddy on X: 'https://t.co/mN93 is hiring Role: SDE 1 "
                           "Experience: 1-3 Years Full-Time Apply Here: https://t.co/Uw'")

    subject = EmailApplier(settings).compose(job, "a@b.com", PROFILE, "Hi.", []).subject

    assert len(subject) < 120 and subject.endswith("- Joshua Akinleye")


def test_the_role_the_model_found_beats_a_page_title() -> None:
    from models import best_title

    assert best_title("Yevgeniya Tsernoh's Post", "Data Analyst") == "Data Analyst"
    assert best_title("Ganesh Reddy on X", "Software Engineer") == "Software Engineer"
    assert best_title("Data Analyst | LinkedIn", "Data Analyst") == "Data Analyst"


def test_a_real_role_is_not_mistaken_for_a_page_title() -> None:
    """"Post" appears in real job titles, so the rule must not fire on those."""
    from models import best_title

    assert best_title("Data Analyst, Post Sales", "Analyst") == "Data Analyst, Post Sales"
    assert best_title("Post Production Coordinator", "X") == "Post Production Coordinator"
    assert best_title("Senior Data Analyst", "Analyst") == "Senior Data Analyst"


def test_a_file_is_not_named_after_something_nobody_knows(tmp_path) -> None:
    from ai_agent import JobAnalysis
    from resume_builder import ResumeBuilder

    settings = Settings(_env_file=None, output_dir=tmp_path)
    analysis = JobAnalysis(job_title="Data Analyst", company_name="Unknown", match_score=70,
                           match_rationale="x", missing_requirements=[],
                           highlighted_skills=[], tailored_summary="x",
                           tailored_bullets=[], answers=[])
    job = JobPosting(job_id="j", url="https://x.com/j", title="Data Analyst",
                     company="Unknown")

    assert ResumeBuilder(settings).output_path(job, analysis).name == "Data_Analyst.pdf"


def test_a_file_falls_back_to_something_rather_than_nothing(tmp_path) -> None:
    from resume_builder import ResumeBuilder

    settings = Settings(_env_file=None, output_dir=tmp_path)
    job = JobPosting(job_id="j", url="https://x.com/j", title="", company="")

    assert ResumeBuilder(settings).output_path(job, None).name == "application.pdf"


# ---- an address that ran into the words after it --------------------------

def test_an_address_glued_to_the_next_word_is_cut_back() -> None:
    """Seen live. Page text comes out of the browser as "jobs@care247.inincluding",
    and writing to that fails."""
    from email_apply import trim_tld

    assert trim_tld("jobs@care247.inincluding") == "jobs@care247.in"
    assert trim_tld("hr@example.comand") == "hr@example.com"
    assert trim_tld("hr@firm.consultingplease") == "hr@firm.consulting"


@pytest.mark.parametrize("address", [
    "careers@acme.com", "careers@acme.co.uk", "hr@company.ng", "a@b.technology",
    "x@y.solutions", "a@b.healthcare", "x@thing.museum",
])
def test_a_real_address_is_left_exactly_as_it_is(address: str) -> None:
    """Truncating a working address is worse than keeping an odd-looking one."""
    from email_apply import trim_tld

    assert trim_tld(address) == address


def test_the_trimming_reaches_the_addresses_a_scan_finds() -> None:
    from email_apply import application_address, find_addresses

    page = "To apply, send your CV to jobs@care247.inincluding your notice period."

    assert find_addresses(page) == ["jobs@care247.in"]
    assert application_address(page, "https://care247.in/jobs") == "jobs@care247.in"


def test_something_that_is_not_an_address_is_untouched() -> None:
    from email_apply import trim_tld

    assert trim_tld("not-an-address") == "not-an-address"
    assert trim_tld("") == ""
