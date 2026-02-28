# -*- coding: utf-8 -*-
"""
CoWriter Module — Shared utilities: history persistence and tool-call storage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

USER_DIR = Path(__file__).parent.parent.parent.parent / "data" / "user" / "co-writer"
HISTORY_FILE = USER_DIR / "history.json"
TOOL_CALLS_DIR = USER_DIR / "tool_calls"


def ensure_dirs() -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    TOOL_CALLS_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> list:
    ensure_dirs()
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history: list) -> None:
    ensure_dirs()
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_tool_call(call_id: str, tool_type: str, data: dict[str, Any]) -> str:
    """Save tool call result to disk; return file path string."""
    ensure_dirs()
    filepath = TOOL_CALLS_DIR / f"{call_id}_{tool_type}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(filepath)


__all__ = [
    "USER_DIR",
    "HISTORY_FILE",
    "TOOL_CALLS_DIR",
    "ensure_dirs",
    "load_history",
    "save_history",
    "save_tool_call",
]
