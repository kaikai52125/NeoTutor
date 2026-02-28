# -*- coding: utf-8 -*-
"""
Question Module — LangGraph State Definition
=============================================

QuestionState covers both batch question generation and single-question
generation with relevance analysis.
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict


def _concat_lists(a: list, b: list) -> list:
    return a + b


class QuestionState(TypedDict):
    """State for the Question LangGraph pipeline."""

    # ── Inputs ──────────────────────────────────────────────────────────────
    kb_name: str
    language: str
    num_questions: int
    requirement: dict[str, Any]   # difficulty, type, topic, etc.
    focus: str                    # optional focus hint

    # ── Phase 1: Retrieve ───────────────────────────────────────────────────
    knowledge_context: str        # merged retrieval result
    retrieval_queries: list[str]

    # ── Phase 2: Generate ───────────────────────────────────────────────────
    generated_questions: list[dict[str, Any]]

    # ── Phase 3: Relevance ──────────────────────────────────────────────────
    relevance_results: list[dict[str, Any]]

    # ── Streaming ───────────────────────────────────────────────────────────
    streaming_events: Annotated[list[dict], _concat_lists]


__all__ = ["QuestionState"]
