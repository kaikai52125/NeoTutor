# -*- coding: utf-8 -*-
"""
Question Module — LangGraph Graph Definition
=============================================

Graph topology:
  START → retrieve_node → generate_node → relevance_node → END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .lg_nodes import generate_node, relevance_node, retrieve_node
from .lg_state import QuestionState


def build_question_graph():
    """
    Build and compile the Question LangGraph.

    Returns:
        Compiled CompiledStateGraph (no checkpointer — stateless per request).
    """
    builder = StateGraph(QuestionState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("relevance", relevance_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "relevance")
    builder.add_edge("relevance", END)

    return builder.compile()


_question_graph: Any = None


def get_question_graph():
    """Return the singleton compiled Question graph."""
    global _question_graph
    if _question_graph is None:
        _question_graph = build_question_graph()
    return _question_graph


__all__ = ["build_question_graph", "get_question_graph"]
