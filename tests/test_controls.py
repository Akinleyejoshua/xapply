"""Every kind of control a form is built from, not just the native ones.

Discovery used to look only at input, textarea and select. Everything else was
invisible to it: a checkbox the design hides behind a styled span, a div with
role="checkbox", a switch, a group of divs acting as radios, a rich text box. Those
questions were left blank without comment, which is the worst failure this tool has,
because the form looks filled until somebody reads it.
"""
from __future__ import annotations

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
