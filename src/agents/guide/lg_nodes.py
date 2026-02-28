# -*- coding: utf-8 -*-
"""
Guide Module — LangGraph Node Implementations
=============================================

Nodes:
  locate_node       — LocateAgent: extract knowledge points from notebook records
  interactive_node  — InteractiveAgent: generate interactive HTML for current point
  chat_node         — ChatAgent: answer a question about the current knowledge point
  summary_node      — SummaryAgent: produce a learning summary report
  fix_html_node     — InteractiveAgent with regeneration mode (same as interactive_node)
  route_action_node — Lightweight router: decide which node to call next based on action

Only route_action_node is wired to conditional edges; the others are leaf nodes.
"""

from __future__ import annotations

import logging
from typing import Any

from .lg_state import GuideState

logger = logging.getLogger(__name__)


def _get_llm_cfg():
    from src.services.llm.config import get_llm_config
    return get_llm_config()


# ---------------------------------------------------------------------------
# Node: locate  (create session → extract knowledge points)
# ---------------------------------------------------------------------------


async def locate_node(state: GuideState) -> dict[str, Any]:
    """
    Extract structured knowledge points from notebook records using LocateAgent.

    Returns:
        {"knowledge_points": list, "status": "initialized", "streaming_events": list}
    """
    records: list = state.get("notebook_records", []) or []
    language: str = state.get("language", "en")
    notebook_id: str = state.get("notebook_id", "") or "cross_notebook"
    notebook_name: str = state.get("notebook_name", "") or "Notebook"

    knowledge_points: list = []
    try:
        from src.agents.guide.agents.locate_agent import LocateAgent
        cfg = _get_llm_cfg()
        agent = LocateAgent(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            api_version=getattr(cfg, "api_version", None),
            language=language,
            binding=getattr(cfg, "binding", "openai"),
        )
        result = await agent.process(
            notebook_id=notebook_id,
            notebook_name=notebook_name,
            records=records,
        )
        knowledge_points = result.get("knowledge_points", []) if isinstance(result, dict) else []
        logger.debug("locate_node: %d knowledge points extracted", len(knowledge_points))
    except Exception as exc:
        logger.warning("locate_node failed: %s", exc)

    return {
        "knowledge_points": knowledge_points,
        "current_index": 0,
        "status": "initialized",
        "streaming_events": [
            {"type": "status", "stage": "locate",
             "message": "knowledge_points_extracted",
             "knowledge_points": knowledge_points}
        ],
    }


# ---------------------------------------------------------------------------
# Node: interactive  (start / next → generate HTML for current point)
# ---------------------------------------------------------------------------


async def interactive_node(state: GuideState) -> dict[str, Any]:
    """
    Generate an interactive HTML learning page for the current knowledge point.

    Returns:
        {"current_html": str, "status": "learning", "streaming_events": list}
    """
    points: list = state.get("knowledge_points", [])
    idx: int = state.get("current_index", 0)
    language: str = state.get("language", "en")

    current_point = points[idx] if points and idx < len(points) else {}
    html = ""
    try:
        from src.agents.guide.agents.interactive_agent import InteractiveAgent
        cfg = _get_llm_cfg()
        agent = InteractiveAgent(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            api_version=getattr(cfg, "api_version", None),
            language=language,
            binding=getattr(cfg, "binding", "openai"),
        )
        result = await agent.process(current_point)
        # process() returns dict {"success": bool, "html": str, ...}
        if isinstance(result, dict):
            html = result.get("html", "")
        else:
            html = str(result)
        logger.debug("interactive_node: HTML generated (%d chars)", len(html))
    except Exception as exc:
        logger.warning("interactive_node failed: %s", exc)
        html = f"<div><h2>{current_point.get('knowledge_title', 'Learning')}</h2><p>{current_point.get('knowledge_summary', '')}</p></div>"

    return {
        "current_html": html,
        "status": "learning",
        "streaming_events": [
            {"type": "html", "stage": "interactive",
             "html": html,
             "knowledge_point": current_point,
             "index": idx,
             "total": len(points)}
        ],
    }


# ---------------------------------------------------------------------------
# Node: chat  (answer user question about current knowledge point)
# ---------------------------------------------------------------------------


async def chat_node(state: GuideState) -> dict[str, Any]:
    """
    Answer the user's question using ChatAgent.

    Returns:
        {"chat_history": [new messages], "streaming_events": list}
    """
    points: list = state.get("knowledge_points", [])
    idx: int = state.get("current_index", 0)
    history: list = state.get("chat_history", [])
    user_message: str = state.get("user_message", "") or ""
    language: str = state.get("language", "en")

    current_point = points[idx] if points and idx < len(points) else {}
    answer = ""
    try:
        from src.agents.guide.agents.chat_agent import ChatAgent
        cfg = _get_llm_cfg()
        agent = ChatAgent(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            api_version=getattr(cfg, "api_version", None),
            language=language,
            binding=getattr(cfg, "binding", "openai"),
        )
        result = await agent.process(
            knowledge=current_point,
            chat_history=history[-10:],  # last 10 messages
            user_question=user_message,
        )
        if isinstance(result, dict):
            answer = result.get("answer", "")
        else:
            answer = str(result)
        logger.debug("chat_node: answer=%d chars", len(answer))
    except Exception as exc:
        logger.warning("chat_node failed: %s", exc)
        answer = f"[Error generating answer: {exc}]"

    new_messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]
    return {
        "chat_history": new_messages,   # _concat_lists reducer appends these
        "streaming_events": [
            {"type": "chat_response", "stage": "chat",
             "answer": answer, "question": user_message}
        ],
    }


# ---------------------------------------------------------------------------
# Node: summary  (all points done → generate learning report)
# ---------------------------------------------------------------------------


async def summary_node(state: GuideState) -> dict[str, Any]:
    """
    Generate a personalised learning summary report using SummaryAgent.

    Returns:
        {"summary": str, "status": "completed", "streaming_events": list}
    """
    points: list = state.get("knowledge_points", [])
    history: list = state.get("chat_history", [])
    notebook_name: str = state.get("notebook_name", "") or ""
    language: str = state.get("language", "en")

    summary = ""
    try:
        from src.agents.guide.agents.summary_agent import SummaryAgent
        cfg = _get_llm_cfg()
        agent = SummaryAgent(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            api_version=getattr(cfg, "api_version", None),
            language=language,
            binding=getattr(cfg, "binding", "openai"),
        )
        result = await agent.process(
            notebook_name=notebook_name,
            knowledge_points=points,
            chat_history=history,
        )
        if isinstance(result, dict):
            summary = result.get("summary", "")
        else:
            summary = str(result)
        logger.debug("summary_node: summary=%d chars", len(summary))
    except Exception as exc:
        logger.warning("summary_node failed: %s", exc)
        summary = f"Learning session completed. Covered {len(points)} knowledge points."

    return {
        "summary": summary,
        "status": "completed",
        "streaming_events": [
            {"type": "summary", "stage": "summary",
             "summary": summary, "knowledge_points": points}
        ],
    }


# ---------------------------------------------------------------------------
# Node: fix_html  (user reported HTML bug → regenerate)
# ---------------------------------------------------------------------------


async def fix_html_node(state: GuideState) -> dict[str, Any]:
    """
    Regenerate the current HTML page (user reported a bug).

    Returns:
        {"current_html": str, "streaming_events": list}
    """
    # Delegate to interactive_node (same logic, fresh HTML generation)
    return await interactive_node(state)


# ---------------------------------------------------------------------------
# Node: route_action  (decide which work node to invoke)
# ---------------------------------------------------------------------------


async def route_action_node(state: GuideState) -> dict[str, Any]:
    """
    No-op pass-through node.  The actual routing is done by the conditional
    edge function `decide_action()` in lg_graph.py which reads state["action"].

    This node exists so the graph has a named entry point after START.
    """
    return {}


# ---------------------------------------------------------------------------
# Node: advance_index  (called on "next" action to increment current_index)
# ---------------------------------------------------------------------------


async def advance_index_node(state: GuideState) -> dict[str, Any]:
    """
    Increment the current_index counter before generating the next HTML page.

    Returns:
        {"current_index": int, "streaming_events": list}
    """
    idx = state.get("current_index", 0) + 1
    total = len(state.get("knowledge_points", []))
    logger.debug("advance_index_node: %d → %d (total=%d)", idx - 1, idx, total)
    return {
        "current_index": idx,
        "streaming_events": [
            {"type": "status", "stage": "advance",
             "message": "moving_to_next_point",
             "index": idx, "total": total}
        ],
    }


__all__ = [
    "locate_node",
    "interactive_node",
    "chat_node",
    "summary_node",
    "fix_html_node",
    "route_action_node",
    "advance_index_node",
]
