"""Guard rails shared by the whole suite.

The tests exercise the real FastAPI app, and the app persists settings to disk. Without
this guard a fixture that forgets `overrides_path` writes into the developer's own
`settings.local.json`, silently changing the model or turning auto-submit on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REAL_OVERRIDES = ROOT / "settings.local.json"


@pytest.fixture(autouse=True)
def never_touch_the_real_settings_file():
    """Fail loudly if a test writes to the repository's own settings file."""
    before = REAL_OVERRIDES.read_bytes() if REAL_OVERRIDES.exists() else None
    yield
    after = REAL_OVERRIDES.read_bytes() if REAL_OVERRIDES.exists() else None
    if before != after:
        if before is None:
            REAL_OVERRIDES.unlink(missing_ok=True)
        else:
            REAL_OVERRIDES.write_bytes(before)
        pytest.fail(
            "settings.local.json changed during this test.\n"
            "If the test builds a Settings, give it "
            "`overrides_path=tmp_path / 'settings.local.json'`.\n"
            "If it does not, a dashboard left running on this machine is writing to the "
            "file while the suite runs; stop it, or run the suite with the server down."
        )
