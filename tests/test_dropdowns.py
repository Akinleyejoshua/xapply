"""Dropdowns the bot could not see, and a page that would not sit still.

Two faults on real application forms. Dropdowns built from a button or a div were never
discovered at all, because discovery only looked at inputs, textareas and selects, so
those questions were silently left blank. And fields were filled in discovery order,
which emits every radio group last, so the page scrolled to the bottom and then jumped
back up partway through.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from browser_bot import FormFiller, HumanGate, StealthBrowser  # noqa: E402
from config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FORM = """<!doctype html><html><head><meta charset="utf-8"><title>t</title>
<style>.f{margin:50px 0}.menu{display:none;position:absolute;background:#fff;border:1px solid #ccc}
.menu.open{display:block}.menu li{padding:6px;list-style:none}</style></head><body>
<div class="f"><label for="fn">First name</label><input id="fn" name="first_name"></div>
<div class="f"><label for="nat">Work authorisation</label>
  <select id="nat" name="native"><option value="">Select...</option>
  <option value="y">Yes</option><option value="n">No</option></select></div>
<fieldset class="f"><legend>Do you require visa sponsorship?</legend>
  <label><input type="radio" name="spon" value="yes"> Yes</label>
  <label><input type="radio" name="spon" value="no"> No</label></fieldset>
<div class="f"><label for="em">Email</label><input id="em" name="email" type="email"></div>
<div class="f"><span id="lb">How did you hear about us?</span>
  <button type="button" id="btn" aria-haspopup="listbox" aria-labelledby="lb">Select...</button>
  <ul class="menu" role="listbox" id="m1"><li role="option">A friend</li>
  <li role="option">LinkedIn</li><li role="option">Job board</li></ul></div>
<div class="f"><span id="lc">Preferred work arrangement</span>
  <div id="dv" role="combobox" aria-haspopup="listbox" aria-labelledby="lc" tabindex="0">Choose one</div>
  <ul class="menu" role="listbox" id="m2"><li role="option">Fully remote</li>
  <li role="option">Hybrid</li><li role="option">On site</li></ul></div>
<div class="f"><label for="ln">Last name</label><input id="ln" name="last_name"></div>
<script>
function wire(t,m){const a=document.getElementById(t),b=document.getElementById(m);
a.addEventListener('click',()=>b.classList.toggle('open'));
b.querySelectorAll('li').forEach(li=>li.addEventListener('click',()=>{
a.textContent=li.textContent;b.classList.remove('open');}));}
wire('btn','m1');wire('dv','m2');</script></body></html>"""


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=tmp_path / "t.db", output_dir=tmp_path,
                    log_dir=tmp_path, audit_dir=tmp_path,
                    user_data_dir=Path(tempfile.mkdtemp()),
                    template_dir=ROOT / "templates", headless=True,
                    overrides_path=tmp_path / "settings.local.json")


@pytest.fixture
async def form(settings: Settings, tmp_path: Path):
    """The test form open in a real browser, with a filler pointed at it."""
    page_file = tmp_path / "form.html"
    page_file.write_text(FORM, encoding="utf-8")
    async with StealthBrowser(settings, HumanGate(mode="api")) as browser:
        page = browser.page or await browser.context.new_page()
        await page.goto(page_file.as_uri())
        yield browser, page, FormFiller(browser, None, settings)


@pytest.mark.asyncio
async def test_dropdowns_with_no_select_behind_them_are_found(form) -> None:
    """A button or a div that opens a list is a question, and it has to be answered."""
    _browser, _page, filler = form
    fields = await filler.discover(_page.locator("body"))
    labels = {f.label: f for f in fields}

    assert "How did you hear about us?" in labels, "the button dropdown was invisible to us"
    assert "Preferred work arrangement" in labels, "the div dropdown was invisible to us"
    assert labels["How did you hear about us?"].kind == "combobox"
    assert labels["Preferred work arrangement"].kind == "combobox"


@pytest.mark.asyncio
async def test_a_dropdown_that_cannot_be_typed_into_says_so(form) -> None:
    """Typing into one throws, which is why they were skipped rather than filled."""
    _browser, page, filler = form
    fields = {f.label: f for f in await filler.discover(page.locator("body"))}

    assert fields["How did you hear about us?"].typeable is False
    assert fields["Work authorisation"].typeable is False, "a native select is chosen, not typed"
    assert fields["First name"].typeable is True


@pytest.mark.asyncio
async def test_fields_come_back_in_the_order_they_appear_on_the_page(form) -> None:
    """Out of order, the page scrolls down, jumps back to the top and works down again."""
    _browser, page, filler = form
    fields = await filler.discover(page.locator("body"))

    assert [f.label for f in fields] == [
        "First name",
        "Work authorisation",
        "Do you require visa sponsorship?",
        "Email",
        "How did you hear about us?",
        "Preferred work arrangement",
        "Last name",
    ]


@pytest.mark.asyncio
async def test_a_click_only_dropdown_is_actually_filled(form) -> None:
    _browser, page, filler = form
    body = page.locator("body")
    fields = {f.label: f for f in await filler.discover(body)}

    chosen = await filler._fill_combobox(body, fields["How did you hear about us?"], "LinkedIn")

    assert chosen == "LinkedIn"
    assert (await page.inner_text("#btn")) == "LinkedIn"


@pytest.mark.asyncio
async def test_the_nearest_choice_is_taken_when_the_wording_differs(form) -> None:
    """The profile says "Remote"; the form offers "Fully remote"."""
    _browser, page, filler = form
    body = page.locator("body")
    fields = {f.label: f for f in await filler.discover(body)}

    chosen = await filler._fill_combobox(body, fields["Preferred work arrangement"], "Remote")

    assert chosen == "Fully remote"
    assert (await page.inner_text("#dv")) == "Fully remote"


@pytest.mark.asyncio
async def test_nothing_is_chosen_when_no_option_fits(form) -> None:
    """Picking the least-bad option would put a false answer on the form."""
    _browser, page, filler = form
    body = page.locator("body")
    fields = {f.label: f for f in await filler.discover(body)}

    chosen = await filler._fill_combobox(body, fields["How did you hear about us?"],
                                         "Carrier pigeon")

    assert chosen is None
    assert (await page.inner_text("#btn")) == "Select...", "the dropdown is left as it was"


@pytest.mark.asyncio
async def test_the_options_of_a_click_only_dropdown_can_be_read(form) -> None:
    """The AI needs the choices before it can answer, and reading them must not
    leave the menu hanging open over the rest of the form."""
    _browser, page, filler = form
    body = page.locator("body")
    fields = {f.label: f for f in await filler.discover(body)}

    options = await filler._combobox_options(body, fields["Preferred work arrangement"])

    assert [o["label"] for o in options] == ["Fully remote", "Hybrid", "On site"]


@pytest.mark.asyncio
async def test_scrolling_settles_instead_of_twitching(form) -> None:
    """Bringing the same field into view twice must not move the page the second time."""
    _browser, page, filler = form
    body = page.locator("body")
    field = {f.label: f for f in await filler.discover(body)}["Last name"]
    loc = filler._loc(body, field.idx)

    await _browser.bring_into_view(loc)
    settled = await page.evaluate("() => window.scrollY")
    await _browser.bring_into_view(loc)
    await _browser.bring_into_view(loc)

    assert await page.evaluate("() => window.scrollY") == settled
