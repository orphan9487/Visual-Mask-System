"""Runtime feature flags that select which optional subsystems load.

One codebase ships as two profiles:

  * ``competition`` - LINE Messaging webhook + talking-face video + TTS.
    Relies on external services (LINE channel, talking-face sidecar, TTS).
  * ``thesis``      - self-hosted WebSocket chat only.  No external
    dependencies, so it runs on a machine with none of the LINE/talking-face
    credentials or the ``line-bot-sdk`` package installed.

Select a profile with ``VMS_PROFILE=competition|thesis`` (defaults to
``thesis`` - the safe, dependency-free variant).  Any individual flag can be
overridden regardless of profile with ``VMS_ENABLE_LINE`` /
``VMS_ENABLE_TALKING_FACE`` / ``VMS_ENABLE_TTS`` set to ``0`` or ``1``.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


PROFILE = os.getenv("VMS_PROFILE", "thesis").strip().lower()
_is_competition = PROFILE == "competition"


def _flag(name: str, default: bool) -> bool:
    """Read a 0/1 env override, falling back to the profile default."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip() == "1"


# LINE webhook frontend (competition only). When off, its module - and the
# ``line-bot-sdk`` import it carries - is never loaded.
LINE_ENABLED = _flag("VMS_ENABLE_LINE", _is_competition)

# Talking-face video sidecar. Only ever invoked from the LINE path today.
TALKING_FACE_ENABLED = _flag("VMS_ENABLE_TALKING_FACE", _is_competition)

# Text-to-speech for talking-face audio.
TTS_ENABLED = _flag("VMS_ENABLE_TTS", _is_competition)
