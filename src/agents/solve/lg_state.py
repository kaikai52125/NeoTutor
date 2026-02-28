# -*- coding: utf-8 -*-
"""
Solve Module — LangGraph State Definition
==========================================

SolveState replaces the three Memory classes:
  - InvestigateMemory  → knowledge_chain + analysis_* fields
  - SolveMemory        → solve_steps (SolveChainStep list)
  - CitationMemory     → citations

All sub-structures are plain dicts (JSON-serialisable) so LangGraph
checkpointers can persist them.
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict


def _concat_lists(a: list, b: list) -> list:
    return a + b


# ---------------------------------------------------------------------------
# Sub-structures (plain dicts — mirrors the original dataclasses)
# ---------------------------------------------------------------------------


class KnowledgeItemDict(TypedDict):
    """Mirrors InvestigateMemory.KnowledgeItem."""
    cite_id: str
    tool_type: str       # rag_hybrid / rag_naive / web_search / ...
    query: str
    raw_result: str
    summary: str


class ToolCallRecordDict(TypedDict):
    """Mirrors SolveMemory.ToolCallRecord."""
    call_id: str
    tool_type: str
    query: str
    cite_id: str          # may be empty string
    raw_answer: str       # may be empty string
    summary: str          # may be empty string
    status: str           # pending | running | success | failed


class SolveStepDict(TypedDict):
    """Mirrors SolveMemory.SolveChainStep."""
    step_id: str
    step_target: str       # goal for this step
    available_cite: list[str]
    tool_calls: list[ToolCallRecordDict]
    step_response: str     # may be empty string
    status: str            # undone | in_progress | waiting_response | done
    used_citations: list[str]


class CitationRecordDict(TypedDict):
    """Mirrors CitationMemory entry."""
    cite_id: str
    tool_type: str
    source: str
    query: str
    content: str


# ---------------------------------------------------------------------------
# SolveState
# ---------------------------------------------------------------------------


class SolveState(TypedDict):
    """
    Unified state for the Solve LangGraph dual-loop pipeline.

    Mirrors the full execution context originally spread across:
      MainSolver fields, InvestigateMemory, SolveMemory, CitationMemory.
    """

    # ── Inputs ──────────────────────────────────────────────────────────────
    question: str
    kb_name: str
    language: str
    output_dir: str

    # ── Analysis Loop (InvestigateMemory) ────────────────────────────────────
    knowledge_chain: list[KnowledgeItemDict]
    analysis_iteration: int
    max_analysis_iterations: int
    analysis_should_stop: bool
    new_knowledge_ids: list[str]        # cite_ids discovered this iteration

    # ── Solve Loop (SolveMemory) ─────────────────────────────────────────────
    solve_steps: list[SolveStepDict]
    current_step_index: int
    solve_iteration: int                # iterations within the current step
    max_solve_iterations: int
    finish_requested: bool              # SolveAgent signalled step done

    # ── Citations (CitationMemory) ────────────────────────────────────────────
    citations: list[CitationRecordDict]

    # ── Output ───────────────────────────────────────────────────────────────
    final_answer: str

    # ── Streaming / progress ─────────────────────────────────────────────────
    streaming_events: Annotated[list[dict], _concat_lists]


__all__ = [
    "KnowledgeItemDict",
    "ToolCallRecordDict",
    "SolveStepDict",
    "CitationRecordDict",
    "SolveState",
]
