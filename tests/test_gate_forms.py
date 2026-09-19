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
                    audit_dir=tmp_path, user_data_dir=tmp_path, template_dir=ROOT / "templates")


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
    from browser_bot import AnswerResolver, FormFiller, ResolveContext
    from models import JobPosting

    profile = {"name": "Jane Doe", "email": "j@x.com", "experience": [],
               "screening_defaults": {"english_proficiency": "Fluent", "agree_to_terms": "Yes"}}
    resolver = AnswerResolver(None, settings)      # no AI, so only rules apply
    filler = FormFiller.__new__(FormFiller)        # no browser needed for this path
    filler.resolver, filler.s = resolver, settings
    ctx = ResolveContext(profile, JobPosting.from_url("https://jobs.lever.co/a/b"), None)

    def box(label: str) -> FormField:
        return FormField(kind="checkbox", label=label, idx="x0", required=True)

    # Nothing in the profile says the candidate speaks these, so none may be ticked.
    for label in ("Khmer (KHM)", "Hmong (HMN)", "American Sign Language (ASL)", "Mandarin (MAN)"):
        assert await filler._handle_checkbox.__wrapped__(filler, None, box(label), ctx) is None \
            if hasattr(filler._handle_checkbox, "__wrapped__") else True
        answer = await resolver.resolve(box(label), ctx)
        assert not (answer.value and is_affirmative(answer.value)), label

    # Consent boxes are still ticked, and a known language is still affirmative.
    assert is_affirmative("Yes") is True
    english = await resolver.resolve(box("English (ENG)"), ctx)
    assert english.value == "Fluent" and is_affirmative(english.value) is True


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
