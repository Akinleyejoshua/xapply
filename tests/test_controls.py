"""Every kind of control a form is built from, not just the native ones.

Discovery used to look only at input, textarea and select. Everything else was
invisible to it: a checkbox the design hides behind a styled span, a div with
role="checkbox", a switch, a group of divs acting as radios, a rich text box. Those
questions were left blank without comment, which is the worst failure this tool has,
because the form looks filled until somebody reads it.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from browser_bot import FormFiller, HumanGate, ResolvedAnswer, StealthBrowser  # noqa: E402
from config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FORM = """<!doctype html><html><head><meta charset="utf-8"><title>t</title><style>
.f{margin:40px 0}
.menu{display:none;position:absolute;background:#fff;border:1px solid #ccc;z-index:9}
.menu.open{display:block}.menu li{padding:6px;list-style:none}
.pretty input{position:absolute;opacity:0;width:0;height:0}
.pretty .box{display:inline-block;width:16px;height:16px;border:2px solid #555}
.editor{border:1px solid #ccc;min-height:60px;padding:8px}
.sw{width:42px;height:22px;display:inline-block;background:#bbb}
</style></head><body>
<div class="f"><label for="fn">First name</label><input id="fn" name="first_name"></div>
<div class="f pretty"><label><input type="checkbox" id="terms" name="terms">
  <span class="box"></span> I agree to the privacy policy</label></div>
<div class="f"><span id="l2">Subscribe to the careers newsletter</span>
  <div id="news" role="checkbox" aria-checked="false" aria-labelledby="l2" tabindex="0"
       style="width:16px;height:16px;border:2px solid #555;display:inline-block"></div></div>
<div class="f"><span id="l3">Open to relocation</span>
  <div id="reloc" class="sw" role="switch" aria-checked="false" aria-labelledby="l3" tabindex="0"></div></div>
<div class="f"><span id="l4">Do you require visa sponsorship?</span>
  <div role="radiogroup" aria-labelledby="l4" id="spon">
    <div role="radio" aria-checked="false" tabindex="0" data-v="Yes">Yes</div>
    <div role="radio" aria-checked="false" tabindex="0" data-v="No">No</div></div></div>
<div class="f"><label for="loc">Location</label>
  <input id="loc" role="combobox" aria-autocomplete="list" autocomplete="off">
  <ul class="menu" role="listbox" id="locmenu"></ul></div>
<div class="f"><span id="l6">Why do you want to join us?</span>
  <div id="why" class="editor" contenteditable="true" role="textbox" aria-labelledby="l6"></div></div>
<div class="f"><label for="ln">Last name</label><input id="ln" name="last_name"></div>
<script>
function toggle(id){var e=document.getElementById(id);
e.addEventListener('click',function(){e.setAttribute('aria-checked',
e.getAttribute('aria-checked')==='true'?'false':'true');});}
toggle('news');toggle('reloc');
document.querySelectorAll('#spon [role=radio]').forEach(function(r){
r.addEventListener('click',function(){
document.querySelectorAll('#spon [role=radio]').forEach(x=>x.setAttribute('aria-checked','false'));
r.setAttribute('aria-checked','true');});});
var CITIES=['Lagos, Nigeria','Lagos, Portugal','London, United Kingdom'];
var loc=document.getElementById('loc'),lm=document.getElementById('locmenu');
loc.addEventListener('input',function(){
var q=loc.value.toLowerCase();
var hits=q?CITIES.filter(c=>c.toLowerCase().indexOf(q)===0):[];
lm.innerHTML=hits.map(c=>'<li role="option">'+c+'</li>').join('');
lm.className=hits.length?'menu open':'menu';
lm.querySelectorAll('li').forEach(function(li){
li.addEventListener('click',function(){loc.value=li.textContent;lm.className='menu';});});});
</script></body></html>"""

ANSWERS = {
    "First name": "Joshua",
    "Last name": "Akinleye",
    "I agree to the privacy policy": "Yes",
    "Subscribe to the careers newsletter": "Yes",
    "Open to relocation": "Yes",
    "Do you require visa sponsorship?": "Yes",
    "Location": "Lagos, Nigeria (Remote, Worldwide)",
    "Why do you want to join us?": "Because the work is close to what I already build.",
}


class Stub:
    async def resolve(self, f, ctx, force_ai: bool = False):
        return ResolvedAnswer(ANSWERS.get(f.label), "stub", 1.0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=tmp_path / "t.db", output_dir=tmp_path,
                    log_dir=tmp_path, audit_dir=tmp_path,
                    user_data_dir=Path(tempfile.mkdtemp()),
                    template_dir=ROOT / "templates", headless=True,
                    overrides_path=tmp_path / "settings.local.json")


@pytest.fixture
async def form(settings: Settings, tmp_path: Path):
    page_file = tmp_path / "wild.html"
    page_file.write_text(FORM, encoding="utf-8")
    async with StealthBrowser(settings, HumanGate(mode="api")) as browser:
        page = browser.page or await browser.context.new_page()
        await page.goto(page_file.as_uri())
        yield page, FormFiller(browser, Stub(), settings)


@pytest.mark.asyncio
async def test_every_control_on_the_form_is_found(form) -> None:
    page, filler = form
    labels = [f.label for f in await filler.discover(page.locator("body"))]

    assert labels == [
        "First name",
        "I agree to the privacy policy",
        "Subscribe to the careers newsletter",
        "Open to relocation",
        "Do you require visa sponsorship?",
        "Location",
        "Why do you want to join us?",
        "Last name",
    ]


@pytest.mark.asyncio
async def test_each_control_is_read_as_the_right_kind(form) -> None:
    page, filler = form
    kinds = {f.label: (f.kind, f.native) for f in await filler.discover(page.locator("body"))}

    assert kinds["I agree to the privacy policy"] == ("checkbox", True), "a real input, only hidden"
    assert kinds["Subscribe to the careers newsletter"] == ("checkbox", False)
    assert kinds["Open to relocation"] == ("checkbox", False), "a switch is a checkbox to us"
    assert kinds["Do you require visa sponsorship?"] == ("radio", False)
    assert kinds["Why do you want to join us?"] == ("textarea", False)


@pytest.mark.asyncio
async def test_a_hidden_input_is_still_the_thing_that_gets_set(form) -> None:
    """The design hides the input and styles a span. Clicking the span sets the input."""
    page, filler = form
    await filler.fill_step(page.locator("body"), ctx=None)

    assert await page.is_checked("#terms") is True


@pytest.mark.asyncio
async def test_controls_that_are_not_inputs_at_all_get_set(form) -> None:
    page, filler = form
    await filler.fill_step(page.locator("body"), ctx=None)

    assert await page.get_attribute("#news", "aria-checked") == "true"
    assert await page.get_attribute("#reloc", "aria-checked") == "true"
    assert await page.get_attribute("#spon [data-v='Yes']", "aria-checked") == "true"
    assert await page.get_attribute("#spon [data-v='No']", "aria-checked") == "false"


@pytest.mark.asyncio
async def test_a_rich_text_box_receives_the_answer(form) -> None:
    """fill() sets a value, and a contenteditable has no value to set."""
    page, filler = form
    await filler.fill_step(page.locator("body"), ctx=None)

    assert "already build" in await page.inner_text("#why")


@pytest.mark.asyncio
async def test_a_location_box_picks_a_real_suggestion(form) -> None:
    """The profile writes a location as a person would. A suggestion box matches on a
    prefix and offers nothing for that, so the whole string sat in the box unmatched."""
    page, filler = form
    await filler.fill_step(page.locator("body"), ctx=None)

    assert await page.input_value("#loc") == "Lagos, Nigeria"


@pytest.mark.asyncio
async def test_nothing_is_reported_unresolved(form) -> None:
    page, filler = form
    result = await filler.fill_step(page.locator("body"), ctx=None)

    assert result.unresolved == []
    assert len(result.filled) == 8


def test_a_typeahead_query_is_narrowed_until_something_matches() -> None:
    assert FormFiller.typeahead_queries("Lagos, Nigeria (Remote, Worldwide)") == [
        "Lagos, Nigeria (Remote, Worldwide)", "Lagos, Nigeria", "Lagos"]
    assert FormFiller.typeahead_queries("Berlin - Germany") == ["Berlin - Germany", "Berlin"]
    assert FormFiller.typeahead_queries("London") == ["London"]
    assert FormFiller.typeahead_queries("") == []


# ---- reading the form back ------------------------------------------------

BROKEN = """<!doctype html><html><head><meta charset="utf-8"><title>t</title></head><body>
<label for="a">First name</label><input id="a">
<label for="b">Why do you want to join us?</label><textarea id="b"></textarea>
<label for="c">Phone</label><input id="c">
<script>
// Losing focus cuts the answer short, once, the way clicking outside a box while it is
// being typed into leaves the rest of the sentence unsent.
document.getElementById('b').addEventListener('blur', function(){
  if (!this.dataset.cut && this.value.length > 12) {
    this.dataset.cut = '1';
    this.value = this.value.slice(0, 12);
  }});
</script></body></html>"""

BROKEN_ANSWERS = {
    "First name": "Joshua",
    "Why do you want to join us?": "Because the work is close to what I already build every day.",
    "Phone": "+2348131519518",
}


class BrokenStub:
    async def resolve(self, f, ctx, force_ai: bool = False):
        return ResolvedAnswer(BROKEN_ANSWERS.get(f.label), "stub", 1.0)


@pytest.fixture
async def broken_form(settings: Settings, tmp_path: Path):
    page_file = tmp_path / "broken.html"
    page_file.write_text(BROKEN, encoding="utf-8")
    async with StealthBrowser(settings, HumanGate(mode="api")) as browser:
        page = browser.page or await browser.context.new_page()
        await page.goto(page_file.as_uri())
        yield page, FormFiller(browser, BrokenStub(), settings)


@pytest.mark.asyncio
async def test_a_field_that_lost_its_value_is_filled_again(broken_form, settings) -> None:
    """Clicking outside a box while it is being typed into stops it part way. The form
    then looks complete, and only whoever reads the application finds out it is not."""
    page, filler = broken_form
    settings.verify_after_fill = True

    result = await filler.fill_step(page.locator("body"), ctx=None)

    assert "Why do you want to join us?" in result.repaired
    assert await page.input_value("#b") == BROKEN_ANSWERS["Why do you want to join us?"]
    assert result.mismatched == []


@pytest.mark.asyncio
async def test_fields_that_were_fine_are_not_touched_again(broken_form, settings) -> None:
    page, filler = broken_form
    settings.verify_after_fill = True

    result = await filler.fill_step(page.locator("body"), ctx=None)

    assert "First name" not in result.repaired and "Phone" not in result.repaired


@pytest.mark.asyncio
async def test_the_check_can_be_turned_off(broken_form, settings) -> None:
    page, filler = broken_form
    settings.verify_after_fill = False

    result = await filler.fill_step(page.locator("body"), ctx=None)

    assert result.repaired == []
    assert await page.input_value("#b") == "Because the ", "left half-typed, as before"


def test_a_truncated_answer_is_not_mistaken_for_a_match() -> None:
    """A cut-off value is a substring of the full one, so containment alone says yes."""
    full = "Because the work is close to what I already build every day."

    assert FormFiller.holds(full, full[:30]) is False
    assert FormFiller.holds(full, full) is True
    assert FormFiller.holds("Yes", "Ye") is False


def test_harmless_reformatting_is_not_treated_as_a_failure() -> None:
    """Refilling a field the form merely tidied would fight it forever."""
    assert FormFiller.holds("2 weeks", "2 Weeks") is True
    assert FormFiller.holds("yes", "Yes") is True
    assert FormFiller.holds("5", "5 years") is True


def test_a_field_the_form_removed_counts_as_lost() -> None:
    assert FormFiller.holds("Joshua", None) is False


# ---- working with no window on screen --------------------------------------

def _browser(**over):
    from browser_bot import HumanGate, StealthBrowser
    from config import Settings

    return StealthBrowser(Settings(_env_file=None, **over), HumanGate(mode="api"))


def test_the_window_can_be_hidden_until_something_needs_you() -> None:
    assert _browser().hidden is False, "a window by default"
    assert _browser(hide_browser=True).hidden is True


def test_a_challenge_can_open_a_window_or_be_skipped() -> None:
    assert _browser(hide_browser=True, challenge_action="show").on_challenge() == "show"
    assert _browser(hide_browser=True, challenge_action="skip").on_challenge() == "skip"
    assert _browser(challenge_action="wait").on_challenge() == "wait"


def test_waiting_with_nothing_on_screen_becomes_a_skip() -> None:
    """Otherwise the run stops for a person who cannot see what it stopped for, which
    looks exactly like a hang."""
    assert _browser(hide_browser=True, challenge_action="wait").on_challenge() == "skip"


def test_fully_headless_never_opens_a_window() -> None:
    """`headless` means no window, whatever else is asked for."""
    hidden = _browser(headless=True, challenge_action="show")

    assert hidden.can_reveal is False
    assert hidden.on_challenge() == "skip"


@pytest.mark.asyncio
async def test_revealing_is_a_no_op_when_a_window_is_already_up() -> None:
    assert await _browser().reveal("test") is False


@pytest.mark.asyncio
async def test_a_hidden_browser_stays_hidden() -> None:
    """Hiding it means you do not want to see it. Nothing opens one behind your back."""
    hidden = _browser(hide_browser=True)

    await hidden.attention("Submit it yourself")

    assert hidden.hidden is True


@pytest.mark.asyncio
async def test_a_pause_with_no_window_still_waits_for_you() -> None:
    """The dashboard's Continue and Skip work whether or not a window exists, so a
    hidden browser is not a reason to give up on a posting. Only a CAPTCHA is, because
    that genuinely cannot be solved without one."""
    hidden = _browser(hide_browser=True)
    skipped = []
    hidden.gate.skip = lambda: skipped.append(True)

    await hidden.attention("Documents attached, submit it yourself")

    assert skipped == []


@pytest.mark.asyncio
async def test_a_pause_brings_the_window_forward() -> None:
    """The thing that needs doing should be on screen, not behind another window."""
    from browser_bot import HumanGate

    gate = HumanGate("api")
    asked: list[str] = []

    async def on_pause(reason: str) -> None:
        asked.append(reason)
        gate.release()

    gate.on_pause = on_pause
    await asyncio.wait_for(gate.wait("Submit it yourself"), timeout=5)

    assert asked == ["Submit it yourself"]


@pytest.mark.asyncio
async def test_a_pause_still_works_when_the_window_cannot_be_shown() -> None:
    """A failure to raise the window must not stop you being asked."""
    from browser_bot import HumanGate

    gate = HumanGate("api")

    async def broken(reason: str) -> None:
        raise RuntimeError("no display")

    gate.on_pause = broken

    async def release() -> None:
        await asyncio.sleep(0.2)
        gate.release()

    outcome, _ = await asyncio.wait_for(
        asyncio.gather(gate.wait("Submit it yourself"), release()), timeout=5)
    assert outcome == HumanGate.CONTINUE


