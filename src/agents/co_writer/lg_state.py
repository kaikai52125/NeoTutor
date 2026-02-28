# -*- coding: utf-8 -*-
"""
CoWriter Module — LangGraph State Definition
=============================================

CoWriterState covers both the edit and narrate pipelines:
  edit:    retrieve_context_node → edit_node
  narrate: narrate_node (single node)
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict


def _concat_lists(a: list, b: list) -> list:
    return a + b


class CoWriterState(TypedDict):
    """State for the CoWriter LangGraph pipelines."""

    # ── Inputs (edit) ───────────────────────────────────────────────────────
    text: str
    instruction: str
    action: str          # rewrite | shorten | expand
    source: str          # none | rag | web
    kb_name: str
    language: str
    operation_id: str

    # ── Inputs (narrate) ────────────────────────────────────────────────────
    content: str
    style: str           # friendly | academic | concise
    voice: str
    skip_audio: bool

    # ── Intermediate ────────────────────────────────────────────────────────
    context: str         # retrieved RAG/web context for edit

    # ── Outputs (edit) ──────────────────────────────────────────────────────
    edited_text: str

    # ── Outputs (narrate) ───────────────────────────────────────────────────
    script: str
    key_points: list[str]
    has_audio: bool
    audio_url: str

    # ── Streaming ───────────────────────────────────────────────────────────
    streaming_events: Annotated[list[dict], _concat_lists]


__all__ = ["CoWriterState"]
