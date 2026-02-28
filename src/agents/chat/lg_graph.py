# -*- coding: utf-8 -*-
"""
Chat Module — LangGraph Graph Definition
=========================================

Builds and compiles the Chat StateGraph:

  START → retrieve_context_node → chat_node → END

The MemorySaver checkpointer provides per-session message history using
thread_id=session_id, replacing the original SessionManager.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .lg_nodes import chat_node, retrieve_context_node
from .lg_state import ChatState


def build_chat_graph():
    """
    Build and compile the Chat LangGraph.

    Graph topology:
        START
          └─► retrieve_context_node   (optional RAG / web retrieval)
                └─► chat_node         (LLM invocation)
                      └─► END

    Returns:
        A compiled CompiledStateGraph with MemorySaver checkpointer.
    """
    builder = StateGraph(ChatState)

    # Register nodes
    builder.add_node("retrieve", retrieve_context_node)
    builder.add_node("chat", chat_node)

    # Wire edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "chat")
    builder.add_edge("chat", END)

    # MemorySaver: in-memory checkpointer; each thread_id is an isolated session.
    # For persistence across restarts, swap with:
    #   from langgraph.checkpoint.sqlite import SqliteSaver
    #   checkpointer = SqliteSaver.from_conn_string("data/user/chat/checkpoints.db")
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ── Singleton ─────────────────────────────────────────────────────────────────
# One compiled graph instance is reused across all requests (thread-safe).

_chat_graph = None


def get_chat_graph():
    """
    Return the singleton compiled Chat graph, creating it on first call.

    Returns:
        Compiled CompiledStateGraph ready for astream_events() / ainvoke().
    """
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = build_chat_graph()
    return _chat_graph


__all__ = ["build_chat_graph", "get_chat_graph"]
