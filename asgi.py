"""ASGI entry point, used by `make dev` for autoreload.

`make serve` builds the app in-process and hands the object to uvicorn, which is
simple but cannot reload: the reloader has to re-import the app itself after a file
changes, so it needs an import string rather than an object. This module is that
import string.

The settings are read at import time, exactly as `main.py serve` reads them, so a
reload also picks up a changed `.env` or `settings.local.json`.
"""
from __future__ import annotations

from api import create_app
from config import settings

app = create_app(settings)
