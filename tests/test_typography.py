"""The project typeface, in the dashboard and in the generated resume."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from resume_builder import FONT_DIR, ResumeBuilder, skill_group_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_font_files_are_present() -> None:
    for name, _ in ResumeBuilder.FONT_WEIGHTS:
        path = FONT_DIR / name
        assert path.exists(), f"missing {path}"
        assert path.stat().st_size > 20_000, f"{name} looks truncated"
    assert (FONT_DIR / "BricolageGrotesque.woff2").exists()      # the dashboard's copy
    assert (FONT_DIR / "README.md").exists(), "the licence note must ship with the fonts"


def test_font_licence_is_recorded() -> None:
    text = (FONT_DIR / "README.md").read_text()
    assert "Open Font License" in text
    assert "Bricolage Grotesque" in text


def test_font_face_embeds_every_weight() -> None:
    css = ResumeBuilder.font_face()
    assert css.count("@font-face") == len(ResumeBuilder.FONT_WEIGHTS)
    assert "Bricolage Grotesque" in css
    assert "base64," in css, "the PDF must not depend on a network fetch"
    for _, weight in ResumeBuilder.FONT_WEIGHTS:
        assert f"font-weight:{weight}" in css


def test_pdf_uses_static_weights_not_the_variable_font() -> None:
    """Chromium renders a variable font into a PDF at its lightest instance."""
    names = [n for n, _ in ResumeBuilder.FONT_WEIGHTS]
    assert "BricolageGrotesque.ttf" not in names
    assert any("Regular" in n for n in names) and any("Bold" in n for n in names)


def test_template_declares_the_typeface() -> None:
    html = (ROOT / "templates" / "resume.html").read_text()
    assert "{{ font_face | safe }}" in html, "autoescaping would mangle the embedded CSS"
    body = html[html.index("body {"):html.index("h1 {")]
    assert '"Bricolage Grotesque"' in body
    assert "Helvetica" in body, "a fallback is needed if the font files go missing"


def test_dashboard_serves_the_font_itself() -> None:
    html = (ROOT / "static" / "index.html").read_text()
    assert "@font-face" in html
    assert "/assets/fonts/BricolageGrotesque.woff2" in html
    assert '"Bricolage Grotesque"' in html
    assert "fonts.googleapis.com" not in html, "the dashboard must not depend on a CDN"
    api = (ROOT / "api.py").read_text()
    assert '"/assets/fonts/{filename}"' in api


def test_font_route_refuses_a_path_outside_the_font_directory(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from api import create_app
    from database import Database

    settings = Settings(db_path=tmp_path / "t.db", output_dir=tmp_path, log_dir=tmp_path,
                        audit_dir=tmp_path, user_data_dir=tmp_path,
                        overrides_path=tmp_path / "s.json",
                        profile_path=ROOT / "profile.example.json",
                        company_file=ROOT / "companies.json")
    db = Database(settings.db_path)
    db.init()
    client = TestClient(create_app(settings, db))
    assert client.get("/assets/fonts/BricolageGrotesque.woff2").status_code == 200
    assert client.get("/assets/fonts/BricolageGrotesque.woff2").headers["content-type"] == "font/woff2"
    assert client.get("/assets/fonts/nope.woff2").status_code == 404
    assert client.get("/assets/fonts/..%2F..%2F.env").status_code == 404


# ---- skills headings -------------------------------------------------------


@pytest.mark.parametrize("key,label", [
    ("ai_ml", "AI/ML"),                 # was rendering as "Ai Ml"
    ("data_analytics", "Data & Analytics"),
    ("languages_tools", "Languages & Tools"),
    ("cloud_devops", "Cloud & DevOps"),
    ("nlp", "NLP"),
    ("web3", "Web3"),
    ("frontend", "Frontend"),
    ("database", "Databases"),
    ("my_custom_group", "My Custom Group"),
    ("", ""),
])
def test_skill_group_label(key, label) -> None:
    assert skill_group_label(key) == label
