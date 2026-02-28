# -*- coding: utf-8 -*-
"""
LangGraph → FastAPI WebSocket Adapter
======================================

Bridges a compiled LangGraph graph's `astream_events(version="v2")` event stream
to a FastAPI WebSocket connection, mapping LangGraph events to the frontend's
existing message protocol.

Frontend message types (preserved for backwards compatibility):
  {"type": "session",  "session_id": str}            — sent before streaming starts
  {"type": "status",   "stage": str, "message": str} — node lifecycle events
  {"type": "stream",   "content": str}               — LLM token chunks
  {"type": "progress", "stage": str, ...}            — custom progress events
  {"type": "result",   ...}                          — final answer (sent by caller)
  {"type": "error",    "message": str}               — error notification

Usage:
    final_state = await stream_graph_to_websocket(
        graph=compiled_graph,
        initial_state={"question": ..., ...},
        websocket=websocket,
        config={"configurable": {"thread_id": session_id}},
    )
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)

# Maps LangGraph node names → human-readable status strings sent to the frontend.
# Extend this dict as new nodes are added.
_NODE_STATUS_MAP: dict[str, str] = {
    # Solve module — Analysis Loop
    "investigate": "analyzing",
    "exec_analysis_tools": "executing_analysis_tools",
    "note": "taking_notes",
    # Solve module — Solve Loop
    "plan": "planning",
    "exec_step_tools": "executing_tools",
    "solve_step": "solving",
    "response": "generating_response",
    "advance_step": "advancing",
    "finalize": "finalizing",
    # Research module
    "rephrase": "rephrasing_topic",
    "decompose": "decomposing_topic",
    "research_block": "researching",
    "report": "generating_report",
    # Chat module
    "retrieve": "retrieving_context",
    "chat": "responding",
    # Guide module
    "locate": "locating_knowledge_point",
    "interactive": "interactive_learning",
    "summary": "summarizing",
    # Question module
    "retrieve_context": "retrieving_context",
    "generate": "generating_questions",
    "analyze": "analyzing_questions",
    # IdeaGen module
    "organize_material": "organizing_material",
    "generate_ideas": "generating_ideas",
    # CoWriter module
    "edit": "editing",
    "narrate": "narrating",
}


async def stream_graph_to_websocket(
    graph: Any,  # CompiledStateGraph — avoid import cycle at module level
    initial_state: dict[str, Any],
    websocket: WebSocket,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Stream a LangGraph graph execution to a FastAPI WebSocket.

    Handles the following LangGraph event types (version="v2"):
      - on_chat_model_stream → forwards each token chunk as {"type":"stream"}
      - on_chain_start       → sends status update when a named node starts
      - on_chain_end         → extracts and forwards custom streaming_events from
                               node output
      - on_tool_start        → sends progress event with tool name and query

    After streaming completes, the full accumulated state is retrieved via
    graph.aget_state() so that checkpointer-managed history (messages, etc.)
    is returned correctly — individual node outputs are partial updates only.

    Args:
        graph:         Compiled LangGraph graph (from builder.compile()).
        initial_state: Initial state dict to pass to the graph.
        websocket:     Active FastAPI WebSocket connection.
        config:        LangGraph run config (e.g. {"configurable": {"thread_id": ...}}).

    Returns:
        The full accumulated state dict from the checkpointer after graph completes.
    """
    _config = config or {}
    # Fallback state accumulated from all on_chain_end outputs for graphs without a checkpointer.
    # We merge each node's partial output so the final dict is the full accumulated state.
    _accumulated_state: dict[str, Any] = {}
    _last_output: dict[str, Any] = {}  # kept for backwards compat reference

    async def safe_send(data: dict[str, Any]) -> bool:
        """Send JSON to WebSocket, silently ignore connection errors."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
                return True
        except Exception as exc:
            logger.debug("WebSocket send failed (connection may be closed): %s", exc)
        return False

    try:
        async for event in graph.astream_events(
            initial_state,
            config=_config,
            version="v2",
        ):
            etype: str = event.get("event", "")
            ename: str = event.get("name", "")
            edata: dict[str, Any] = event.get("data", {})

            # ── LLM token chunks ────────────────────────────────────────────
            if etype == "on_chat_model_stream":
                chunk = edata.get("chunk")
                if chunk is not None:
                    content = getattr(chunk, "content", None)
                    if content:
                        await safe_send({"type": "stream", "content": content})

            # ── Node started ─────────────────────────────────────────────────
            elif etype == "on_chain_start":
                status_msg = _NODE_STATUS_MAP.get(ename)
                if status_msg:
                    await safe_send({
                        "type": "status",
                        "stage": ename,
                        "message": status_msg,
                    })

            # ── Node finished ────────────────────────────────────────────────
            elif etype == "on_chain_end":
                output = edata.get("output")
                if isinstance(output, dict):
                    # Forward any custom progress events emitted by the node
                    for progress_event in output.get("streaming_events", []):
                        if isinstance(progress_event, dict):
                            await safe_send(progress_event)
                    # Merge node's partial output into accumulated state so that
                    # graphs without a checkpointer still return the full state.
                    _accumulated_state.update(output)
                    _last_output = output

            # ── Tool invocation started ──────────────────────────────────────
            elif etype == "on_tool_start":
                tool_input = edata.get("input", {})
                query = ""
                if isinstance(tool_input, dict):
                    query = str(tool_input.get("query", ""))
                await safe_send({
                    "type": "progress",
                    "stage": "tool_call",
                    "tool": ename,
                    "query": query,
                })

    except Exception as exc:
        logger.exception("Error during LangGraph streaming: %s", exc)
        await safe_send({"type": "error", "message": str(exc)})
        raise

    # Retrieve the full accumulated state from the checkpointer.
    # Individual node on_chain_end outputs are partial updates; only aget_state()
    # returns the complete merged state including all messages accumulated via reducers.
    # Graphs without a checkpointer (question, research, etc.) fall back to the
    # accumulated state merged from all node on_chain_end outputs.
    try:
        snapshot = await graph.aget_state(_config)
        return dict(snapshot.values) if snapshot and hasattr(snapshot, "values") else _accumulated_state
    except Exception as exc:
        logger.warning("aget_state failed, falling back to last streamed output: %s", exc)
        logger.debug(
            "accumulated_state keys=%s, relevance_results=%d, generated_questions=%d",
            list(_accumulated_state.keys()),
            len(_accumulated_state.get("relevance_results", [])),
            len(_accumulated_state.get("generated_questions", [])),
        )
        return _accumulated_state


__all__ = ["stream_graph_to_websocket"]
