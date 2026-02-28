# -*- coding: utf-8 -*-
"""
Guide Module — LangGraph State Definition
==========================================

GuideState replaces the GuidedSession dataclass, making the session
fully serialisable for LangGraph MemorySaver checkpointer.

The guide workflow is stateful and multi-turn:
  - Each "turn" the user can: start / next / chat / fix_html
  - The graph uses a MemorySaver with thread_id=session_id so state
    persists across WebSocket reconnections.
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict


def _concat_lists(a: list, b: list) -> list:
    return a + b


class KnowledgePointDict(TypedDict):
    """Mirrors the knowledge point dict produced by LocateAgent."""
    title: str
    summary: str
    difficulty: str


class ChatMessageDict(TypedDict):
    """A single chat message."""
    role: str     # user | assistant
    content: str


class GuideState(TypedDict):
    """
    Full session state for the Guide LangGraph workflow.

    Maps 1-to-1 with GuidedSession fields so the existing GuideManager
    session JSON format is preserved.
    """

    # ── Session metadata ─────────────────────────────────────────────────────
    session_id: str
    notebook_id: str
    notebook_name: str
    language: str

    # ── Action requested by this turn ────────────────────────────────────────
    # One of: "create" | "start" | "next" | "chat" | "fix_html"
    action: str
    user_message: str      # question text for "chat" action
    notebook_records: list[dict[str, Any]]  # raw records for "create" action

    # ── Knowledge points (from LocateAgent) ──────────────────────────────────
    knowledge_points: list[KnowledgePointDict]
    current_index: int

    # ── HTML learning page (from InteractiveAgent) ────────────────────────────
    current_html: str

    # ── Chat history ─────────────────────────────────────────────────────────
    chat_history: Annotated[list[ChatMessageDict], _concat_lists]

    # ── Session lifecycle ────────────────────────────────────────────────────
    # "initialized" | "learning" | "completed"
    status: str

    # ── Summary (from SummaryAgent, on completion) ────────────────────────────
    summary: str

    # ── Streaming / progress ─────────────────────────────────────────────────
    streaming_events: Annotated[list[dict], _concat_lists]


__all__ = ["GuideState", "KnowledgePointDict", "ChatMessageDict"]
