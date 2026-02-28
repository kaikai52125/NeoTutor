# -*- coding: utf-8 -*-
"""
IdeaGen Module — LangGraph State Definition
============================================

IdeaGenState covers the five-stage idea-generation pipeline:
  extract → loose_filter → explore → strict_filter → statement
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict


def _concat_lists(a: list, b: list) -> list:
    return a + b


class IdeaGenState(TypedDict):
    """State for the IdeaGen LangGraph pipeline."""

    # ── Inputs ──────────────────────────────────────────────────────────────
    notebook_records: list[dict[str, Any]]
    user_thoughts: str
    kb_name: str
    language: str
    run_id: str

    # ── Stage 1: Extract knowledge points ───────────────────────────────────
    knowledge_points: list[dict[str, Any]]    # [{knowledge_point, description}]

    # ── Stage 2: Loose filter ───────────────────────────────────────────────
    filtered_points: list[dict[str, Any]]

    # ── Stage 3: Explore ideas (per knowledge point) ────────────────────────
    explored_ideas: list[dict[str, Any]]      # [{knowledge_point, ideas: [str]}]

    # ── Stage 4: Strict filter ──────────────────────────────────────────────
    strict_filtered_ideas: list[dict[str, Any]]

    # ── Stage 5: Generate statements ────────────────────────────────────────
    idea_results: list[dict[str, Any]]        # Final [{id, knowledge_point, statement, ...}]

    # ── Streaming ───────────────────────────────────────────────────────────
    streaming_events: Annotated[list[dict], _concat_lists]


__all__ = ["IdeaGenState"]
