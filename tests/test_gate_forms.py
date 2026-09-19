"""Human-in-the-loop gate, checkbox truthfulness, and the lazy browser."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from browser_bot import FormField, HumanGate, is_affirmative  # noqa: E402
from config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "t.db", output_dir=tmp_path, log_dir=tmp_path,
                    audit_dir=tmp_path, user_data_dir=tmp_path, template_dir=ROOT / "templates",
                    overrides_path=tmp_path / "settings.local.json")


# ---- human gate -----------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_released_by_api_call(tmp_path: Path) -> None:
    """The dashboard's Continue button must free a gate the server is waiting on."""
    gate = HumanGate("terminal", tmp_path / "CONTINUE")

    async def releaser() -> None:
        await asyncio.sleep(0.2)
        assert gate.paused is True
        assert "CAPTCHA" in gate.status()["reason"]
        gate.release()

    await asyncio.wait_for(asyncio.gather(gate.wait("CAPTCHA on review page"), releaser()), timeout=5)
    assert gate.paused is False
    assert gate.status()["waited"] == 1


@pytest.mark.asyncio
async def test_gate_released_by_marker_file(tmp_path: Path) -> None:
    marker = tmp_path / "CONTINUE"
    gate = HumanGate("api", marker)

    async def toucher() -> None:
        await asyncio.sleep(0.2)
        marker.write_text("go")

    await asyncio.wait_for(asyncio.gather(gate.wait("login wall"), toucher()), timeout=8)
    assert gate.paused is False
    assert not marker.exists()          # consumed, so it cannot release the next pause


@pytest.mark.asyncio
async def test_gate_can_pause_repeatedly(tmp_path: Path) -> None:
    """Regression: a released gate used to stay stuck and never pause again."""
    gate = HumanGate("terminal", tmp_path / "CONTINUE")
    for i in range(3):
        async def releaser() -> None:
            await asyncio.sleep(0.1)
            gate.release()

        await asyncio.wait_for(asyncio.gather(gate.wait(f"pause {i}"), releaser()), timeout=5)
        assert gate.paused is False
    assert len(gate.history) == 3


def test_gate_release_without_a_waiter_is_safe(tmp_path: Path) -> None:
    HumanGate("api", tmp_path / "CONTINUE").release()


# ---- checkbox truthfulness ------------------------------------------------


@pytest.mark.parametrize("answer,expected", [
    ("Yes", True), ("yes", True), ("true", True), ("I agree", True), ("Fluent", True),
    ("Native or bilingual", True), ("Authorized to work", True),
    ("No", False), ("false", False), ("I do not", False), ("Decline to self-identify", False),
    ("UNKNOWN", False), ("N/A", False), ("", False), ("Maybe someday", False),
])
def test_is_affirmative(answer, expected) -> None:
    assert is_affirmative(answer) is expected


@pytest.mark.asyncio
async def test_required_checkbox_is_not_blanket_ticked(settings: Settings) -> None:
    """A required checkbox group (spoken languages) must not all be ticked.

    Regression: a real Lever form marks every language checkbox required, and the old
    rule ticked anything required, claiming languages the candidate does not speak.
    """
    from browser_bot import AnswerResolver, ResolveContext
    from models import JobPosting

    profile = {"name": "Jane Doe", "email": "j@x.com", "experience": [],
               "screening_defaults": {"english_proficiency": "Fluent", "agree_to_terms": "Yes"}}
    resolver = AnswerResolver(None, settings)      # no AI, so only profile rules apply
    ctx = ResolveContext(profile, JobPosting.from_url("https://jobs.lever.co/a/b"), None)

    def box(label: str) -> FormField:
        return FormField(kind="checkbox", label=label, idx="x0", required=True)

    # Nothing in the profile says the candidate speaks these, so none may be ticked.
    for label in ("Khmer (KHM)", "Hmong (HMN)", "American Sign Language (ASL)", "Mandarin (MAN)"):
        answer = await resolver.resolve(box(label), ctx)
        ticked = bool(answer.value) and not answer.needs_human and is_affirmative(answer.value)
        assert ticked is False, f"{label} would have been ticked"

    # A language the profile does vouch for is still affirmative, and so are consent boxes.
    english = await resolver.resolve(box("English (ENG)"), ctx)
    assert english.value == "Fluent" and is_affirmative(english.value) is True
    from browser_bot import AGREE_RE
    assert AGREE_RE.search("I agree to the terms") is not None


# ---- lazy browser ---------------------------------------------------------


def test_lazy_browser_reports_not_started(settings: Settings) -> None:
    from job_search import BROWSER_FALLBACK_SOURCES, BROWSER_SOURCES, LazyBrowser

    lazy = LazyBrowser(settings, HumanGate("api", settings.log_dir / "CONTINUE"))
    assert lazy.started is False
    with pytest.raises(RuntimeError, match="has not been started"):
        _ = lazy.page
    with pytest.raises(AttributeError, match="not been started"):
        _ = lazy.human_click
    assert "greenhouse" not in BROWSER_SOURCES | BROWSER_FALLBACK_SOURCES
    assert {"linkedin", "google", "urls"} == BROWSER_SOURCES


@pytest.mark.asyncio
async def test_lazy_browser_gives_no_page_to_api_sources(settings: Settings) -> None:
    """A Greenhouse/Lever/Ashby scan must never open a browser window."""
    from job_search import LazyBrowser

    lazy = LazyBrowser(settings, HumanGate("api", settings.log_dir / "CONTINUE"))
    for name in ("greenhouse", "lever", "ashby"):
        assert await lazy.page_for(name) is None
    assert lazy.started is False
    await lazy.close()


# ---- skipping a job you cannot get past ------------------------------------


@pytest.mark.asyncio
async def test_gate_can_skip_instead_of_continue(tmp_path: Path) -> None:
    """A Cloudflare challenge that never clears must not hold the whole run."""
    gate = HumanGate("api", tmp_path / "CONTINUE")

    async def decide() -> None:
        await asyncio.sleep(0.15)
        gate.skip()

    outcome, _ = await asyncio.wait_for(
        asyncio.gather(gate.wait("Challenge will not clear"), decide()), timeout=5)
    assert outcome == HumanGate.SKIP


@pytest.mark.asyncio
async def test_gate_continue_is_the_default_outcome(tmp_path: Path) -> None:
    gate = HumanGate("api", tmp_path / "CONTINUE")

    async def decide() -> None:
        await asyncio.sleep(0.15)
        gate.release()

    outcome, _ = await asyncio.wait_for(
        asyncio.gather(gate.wait("CAPTCHA"), decide()), timeout=5)
    assert outcome == HumanGate.CONTINUE


@pytest.mark.asyncio
async def test_skip_marker_file(tmp_path: Path) -> None:
    gate = HumanGate("api", tmp_path / "CONTINUE")

    async def touch() -> None:
        await asyncio.sleep(0.2)
        gate.skip_file.write_text("x")

    outcome, _ = await asyncio.wait_for(
        asyncio.gather(gate.wait("challenge"), touch()), timeout=8)
    assert outcome == HumanGate.SKIP
    assert not gate.skip_file.exists()          # consumed, so the next pause is unaffected


@pytest.mark.asyncio
async def test_skip_can_be_disallowed(tmp_path: Path) -> None:
    """Some pauses are not skippable, for instance a login the run depends on."""
    gate = HumanGate("api", tmp_path / "CONTINUE")

    async def decide() -> None:
        await asyncio.sleep(0.15)
        gate.skip()

    outcome, _ = await asyncio.wait_for(
        asyncio.gather(gate.wait("Log in first", allow_skip=False), decide()), timeout=5)
    assert outcome == HumanGate.CONTINUE


def test_gate_status_reports_skippability(tmp_path: Path) -> None:
    gate = HumanGate("api", tmp_path / "CONTINUE")
    assert "allow_skip" in gate.status()


def test_api_exposes_skip() -> None:
    from pathlib import Path as _P

    api_src = (_P(__file__).resolve().parents[1] / "api.py").read_text()
    assert '"/admin/skip"' in api_src
    html = (_P(__file__).resolve().parents[1] / "static" / "index.html").read_text()
    assert "skipJob()" in html and "/admin/skip" in html


# ---- fill the form before worrying about the CAPTCHA -----------------------


def test_guard_signature_allows_blocking_only() -> None:
    """During navigation only a real wall should stop the agent."""
    import inspect

    from browser_bot import StealthBrowser

    params = inspect.signature(StealthBrowser.guard).parameters
    assert "blocking_only" in params
    assert params["blocking_only"].default is False


def test_goto_uses_the_lenient_guard() -> None:
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[1] / "browser_bot.py").read_text()
    goto = src[src.index("async def goto(self"):]
    goto = goto[:goto.index("async def detect_captcha")]
    assert "blocking_only=True" in goto, "navigation must not stop for an inline form CAPTCHA"


def test_appliers_run_the_full_guard_before_submitting() -> None:
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[1] / "appliers.py").read_text()
    # the submit paths check the gate result so a skip is honoured
    assert src.count("HumanGate.SKIP") >= 3
    assert "await self.b.guard(page) == HumanGate.SKIP" in src
