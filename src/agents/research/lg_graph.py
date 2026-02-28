# -*- coding: utf-8 -*-
"""
Research Module — LangGraph Graph Definition
=============================================

Builds the Research StateGraph which maps the original three-phase pipeline:

  Phase 1 — Planning:
    START → rephrase_node → decompose_node

  Phase 2 — Researching (Map-Reduce parallel via Send API):
    decompose_node → [Send per pending block] → research_block_node (×N, parallel)

  Phase 3 — Reporting:
    research_block_node (all) → report_node → END

The Send API (LangGraph 1.x) replaces the original asyncio.Semaphore-based
parallel loop in ResearchPipeline._phase2_researching_parallel().
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .lg_nodes import decompose_node, report_node, research_block_node, rephrase_node
from .lg_state import ResearchState, TopicBlockDict


# ---------------------------------------------------------------------------
# Condition / distributor functions
# ---------------------------------------------------------------------------


def distribute_blocks(state: ResearchState) -> list[Send]:
    """
    Map-Reduce distributor: spawn one research_block_node per pending block.

    Each Send injects `_current_block` into the node's state so the node
    knows which block to research without scanning the entire list.

    This replaces asyncio.gather + semaphore in the original pipeline.
    """
    pending: list[TopicBlockDict] = [
        b for b in state.get("topic_blocks", []) if b["status"] == "pending"
    ]
    if not pending:
        # No pending blocks → skip straight to report
        return [Send("report", state)]

    return [
        Send(
            "research_block",
            {**state, "_current_block": block},
        )
        for block in pending
    ]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_research_graph():
    """
    Build and compile the Research LangGraph.

    Returns:
        A compiled CompiledStateGraph (no checkpointer — research runs are
        stateless within a single pipeline execution).
    """
    builder = StateGraph(ResearchState)

    # Register nodes
    builder.add_node("rephrase", rephrase_node)
    builder.add_node("decompose", decompose_node)
    builder.add_node("research_block", research_block_node)
    builder.add_node("report", report_node)

    # Phase 1: planning edges
    builder.add_edge(START, "rephrase")
    builder.add_edge("rephrase", "decompose")

    # Phase 2: Map-Reduce — decompose fans out to N parallel research_block nodes
    builder.add_conditional_edges(
        "decompose",
        distribute_blocks,
        ["research_block", "report"],  # possible target nodes
    )

    # Phase 3: all research_block completions converge on report
    builder.add_edge("research_block", "report")
    builder.add_edge("report", END)

    # No checkpointer: research runs are one-shot (resumability is handled by
    # the original JSON save logic inside report_node / research_block_node).
    return builder.compile()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_research_graph: Any = None


def get_research_graph():
    """
    Return the singleton compiled Research graph, creating it on first call.

    Returns:
        Compiled CompiledStateGraph ready for astream_events() / ainvoke().
    """
    global _research_graph
    if _research_graph is None:
        _research_graph = build_research_graph()
    return _research_graph


__all__ = ["build_research_graph", "get_research_graph"]
