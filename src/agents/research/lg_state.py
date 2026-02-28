# -*- coding: utf-8 -*-
"""
Research Module — LangGraph State Definition
=============================================

ResearchState is the single source of truth for the LangGraph-based
Research pipeline, replacing the combination of:
  - DynamicTopicQueue       → topic_blocks + custom reducer
  - Per-block ToolTrace list → TopicBlock.tool_traces
  - Ad-hoc progress dict    → streaming_events
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Sub-structures (plain dicts — JSON-serializable for checkpointer)
# ---------------------------------------------------------------------------


class ToolTraceDict(TypedDict):
    """Mirrors data_structures.ToolTrace but as a plain dict."""
    tool_id: str
    citation_id: str
    tool_type: str          # rag_naive / rag_hybrid / web_search / paper_search / ...
    query: str
    raw_answer: str
    summary: str


class TopicBlockDict(TypedDict):
    """Mirrors data_structures.TopicBlock but as a plain dict."""
    block_id: str
    sub_topic: str
    overview: str
    status: str             # pending | researching | completed | failed
    tool_traces: list[ToolTraceDict]
    iteration_count: int
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Reducer: merge topic blocks by block_id (replaces DynamicTopicQueue logic)
# ---------------------------------------------------------------------------


def _merge_topic_blocks(
    existing: list[TopicBlockDict],
    updates: list[TopicBlockDict],
) -> list[TopicBlockDict]:
    """
    Custom reducer for topic_blocks.

    Merges updates into the existing list by block_id:
      - New block_ids are appended.
      - Existing block_ids have their entry replaced.

    This replicates DynamicTopicQueue.add_block() / mark_completed() semantics
    while remaining JSON-serializable for LangGraph checkpointers.
    """
    existing_map: dict[str, TopicBlockDict] = {b["block_id"]: b for b in existing}
    for block in updates:
        existing_map[block["block_id"]] = block
    return list(existing_map.values())


def _concat_lists(a: list, b: list) -> list:
    """Simple list concatenation reducer for streaming_events."""
    return a + b


# ---------------------------------------------------------------------------
# ResearchState TypedDict
# ---------------------------------------------------------------------------


class ResearchState(TypedDict):
    """
    Full state for the Research LangGraph pipeline.

    Corresponds to the three phases of ResearchPipeline:
      Phase 1 — Planning:     optimized_topic, topic_blocks (initial set)
      Phase 2 — Researching:  topic_blocks (updated per block), citations
      Phase 3 — Reporting:    final_report, report_path
    """

    # ── Inputs ──────────────────────────────────────────────────────────────
    topic: str
    kb_name: str
    research_id: str         # unique task/run ID
    language: str

    # Plan-mode config (mirrors router plan_mode logic)
    initial_subtopics: int   # how many subtopics to request in decompose phase
    max_iterations: int      # max research iterations per block
    plan_mode: str           # quick / medium / deep / auto
    skip_rephrase: bool

    # Enabled tools list (subset of: rag, web, paper, query_item, code)
    enabled_tools: list[str]

    # ── Phase 1 outputs ──────────────────────────────────────────────────────
    optimized_topic: str

    # ── Phase 2 state ────────────────────────────────────────────────────────
    # Custom reducer: blocks are merged/updated by block_id
    topic_blocks: Annotated[list[TopicBlockDict], _merge_topic_blocks]

    # Flat list of all citations accumulated during research
    citations: Annotated[list[dict], _concat_lists]

    # ── Phase 3 output ───────────────────────────────────────────────────────
    final_report: str
    report_path: str

    # ── Streaming / progress ─────────────────────────────────────────────────
    streaming_events: Annotated[list[dict], _concat_lists]


__all__ = [
    "ToolTraceDict",
    "TopicBlockDict",
    "ResearchState",
]
