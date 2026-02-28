# -*- coding: utf-8 -*-
"""
CoWriter Module — LangGraph Graph Definitions
=============================================

Two separate graphs for the two CoWriter pipelines:

Edit graph:
  START → retrieve_context_node → edit_node → END

Narrate graph:
  START → narrate_node → END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .lg_nodes import edit_node, narrate_node, retrieve_context_node
from .lg_state import CoWriterState


def build_edit_graph():
    """
    Build and compile the CoWriter Edit LangGraph.

    Returns:
        Compiled CompiledStateGraph.
    """
    builder = StateGraph(CoWriterState)

    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("edit", edit_node)

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "edit")
    builder.add_edge("edit", END)

    return builder.compile()


def build_narrate_graph():
    """
    Build and compile the CoWriter Narrate LangGraph.

    Returns:
        Compiled CompiledStateGraph.
    """
    builder = StateGraph(CoWriterState)

    builder.add_node("narrate", narrate_node)

    builder.add_edge(START, "narrate")
    builder.add_edge("narrate", END)

    return builder.compile()


# ── Singletons ────────────────────────────────────────────────────────────────

_edit_graph: Any = None
_narrate_graph: Any = None


def get_edit_graph():
    """Return the singleton compiled Edit graph."""
    global _edit_graph
    if _edit_graph is None:
        _edit_graph = build_edit_graph()
    return _edit_graph


def get_narrate_graph():
    """Return the singleton compiled Narrate graph."""
    global _narrate_graph
    if _narrate_graph is None:
        _narrate_graph = build_narrate_graph()
    return _narrate_graph


__all__ = [
    "build_edit_graph",
    "build_narrate_graph",
    "get_edit_graph",
    "get_narrate_graph",
]
