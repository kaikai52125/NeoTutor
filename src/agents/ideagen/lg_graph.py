# -*- coding: utf-8 -*-
"""
IdeaGen Module — LangGraph Graph Definition
===========================================

Graph topology (linear pipeline):
  START → extract_node → loose_filter_node → explore_node
        → strict_filter_node → statement_node → END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .lg_nodes import (
    explore_node,
    extract_node,
    loose_filter_node,
    statement_node,
    strict_filter_node,
)
from .lg_state import IdeaGenState


def build_ideagen_graph():
    """
    Build and compile the IdeaGen LangGraph.

    Returns:
        Compiled CompiledStateGraph (no checkpointer — stateless per request).
    """
    builder = StateGraph(IdeaGenState)

    builder.add_node("extract", extract_node)
    builder.add_node("loose_filter", loose_filter_node)
    builder.add_node("explore", explore_node)
    builder.add_node("strict_filter", strict_filter_node)
    builder.add_node("statement", statement_node)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "loose_filter")
    builder.add_edge("loose_filter", "explore")
    builder.add_edge("explore", "strict_filter")
    builder.add_edge("strict_filter", "statement")
    builder.add_edge("statement", END)

    return builder.compile()


_ideagen_graph: Any = None


def get_ideagen_graph():
    """Return the singleton compiled IdeaGen graph."""
    global _ideagen_graph
    if _ideagen_graph is None:
        _ideagen_graph = build_ideagen_graph()
    return _ideagen_graph


__all__ = ["build_ideagen_graph", "get_ideagen_graph"]
