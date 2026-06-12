"""Persist sandbox state to disk across server restarts and code reloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_FILE = Path(__file__).resolve().parent / "sandbox_state.json"


def save_snapshot(snapshot: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(snapshot, indent=2))


def load_snapshot() -> dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != 1:
        return None
    return data


def clear_snapshot() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()
