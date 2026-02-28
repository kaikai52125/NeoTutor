# -*- coding: utf-8 -*-
"""
Guide Module — LangGraph Graph Definition
=========================================

The Guide graph is multi-turn and stateful.  Each API call (start / next /
chat / fix_html) updates the persisted state via MemorySaver checkpointer
using thread_id=session_id, equivalent to the original GuidedSession JSON files.

Graph topology:

  START → route_action_node ─(conditional edge)─►  locate_node       → END
                                                ►  interactive_node   → END
                                                ►  advance_index_node → check_complete ─►  interactive_node → END
                                                                                        ►  summary_node     → END
                                                ►  chat_node          → END
                                                ►  fix_html_node      → END

The `action` field in GuideState drives the conditional routing.
"""

from __future__ import annotations

import logging
from typing import Any, Union

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .lg_nodes import (
    advance_index_node,
    chat_node,
    fix_html_node,
    interactive_node,
    locate_node,
    route_action_node,
    summary_node,
)
from .lg_state import GuideState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition / routing functions
# ---------------------------------------------------------------------------


def decide_action(state: GuideState) -> str:
    """
    Route to the appropriate work node based on state["action"].

    Called after route_action_node.
    """
    action = state.get("action", "")
    if action == "create":
        return "locate"
    if action == "start":
        return "interactive"
    if action == "next":
        return "advance_index"
    if action == "chat":
        return "chat"
    if action == "fix_html":
        return "fix_html"
    # Unknown action → go straight to END
    return END


def check_complete(state: GuideState) -> str:
    """
    After advancing the index, decide whether to show the next learning page
    or generate a summary (if all points are done).
    """
    idx = state.get("current_index", 0)
    total = len(state.get("knowledge_points", []))
    logger.debug("check_complete: current_index=%d, total=%d", idx, total)
    if idx >= total:
        return "summary"
    return "interactive"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_guide_graph():
    """
    Build and compile the Guide LangGraph with MemorySaver checkpointer.

    Returns:
        Compiled CompiledStateGraph with per-session memory.
    """
    builder = StateGraph(GuideState)

    # Register nodes
    builder.add_node("route_action", route_action_node)
    builder.add_node("locate", locate_node)
    builder.add_node("interactive", interactive_node)
    builder.add_node("advance_index", advance_index_node)
    builder.add_node("summary", summary_node)
    builder.add_node("chat", chat_node)
    builder.add_node("fix_html", fix_html_node)

    # Entry
    builder.add_edge(START, "route_action")

    # Conditional dispatch based on action
    builder.add_conditional_edges(
        "route_action",
        decide_action,
        {
            "locate":        "locate",
            "interactive":   "interactive",
            "advance_index": "advance_index",
            "chat":          "chat",
            "fix_html":      "fix_html",
            END:             END,
        },
    )

    # After advance_index: check if session is complete
    builder.add_conditional_edges(
        "advance_index",
        check_complete,
        {
            "interactive": "interactive",
            "summary":     "summary",
        },
    )

    # All terminal nodes → END
    for terminal in ("locate", "interactive", "summary", "chat", "fix_html"):
        builder.add_edge(terminal, END)

    # MemorySaver: each thread_id (= session_id) is an isolated conversation.
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_guide_graph: Any = None


def get_guide_graph():
    """Return the singleton compiled Guide graph."""
    global _guide_graph
    if _guide_graph is None:
        _guide_graph = build_guide_graph()
    return _guide_graph


__all__ = ["build_guide_graph", "get_guide_graph"]
