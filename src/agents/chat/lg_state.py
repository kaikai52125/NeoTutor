# -*- coding: utf-8 -*-
"""
Chat Module — LangGraph State Definition
==========================================

Defines the TypedDict state used by the Chat LangGraph StateGraph.

Key design decisions:
  - `messages` uses the built-in `add_messages` reducer so that each node
    simply returns a list of new messages to append (no manual concatenation).
  - History truncation is handled automatically by LangGraph's checkpointer:
    the graph accumulates history across invocations within the same thread_id.
  - `streaming_events` is a custom reducer that concatenates lists, enabling
    nodes to emit progress events that are forwarded to the WebSocket.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _concat_lists(a: list, b: list) -> list:
    """Reducer that concatenates two lists (used for streaming_events)."""
    return a + b


class ChatState(TypedDict):
    """
    State shared across all nodes in the Chat LangGraph graph.

    Fields:
        messages:           Conversation messages (auto-appended via add_messages).
        kb_name:            Knowledge base name for RAG retrieval.
        enable_rag:         Whether to perform RAG retrieval before responding.
        enable_web_search:  Whether to perform web search before responding.
        language:           UI language code ('zh' | 'en').
        session_id:         Session identifier (used as checkpointer thread_id).
        rag_context:        Retrieved RAG/web context injected into the prompt.
        sources:            Citation metadata from RAG / web search.
        streaming_events:   Custom progress events emitted by nodes for WebSocket.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    kb_name: str
    enable_rag: bool
    enable_web_search: bool
    language: str
    session_id: str
    rag_context: str
    sources: dict
    streaming_events: Annotated[list[dict], _concat_lists]


__all__ = ["ChatState"]
