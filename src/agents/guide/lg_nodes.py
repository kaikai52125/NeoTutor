# -*- coding: utf-8 -*-
"""
Guide Module — LangGraph Node Implementations
=============================================

All LLM calls use LangChain's BaseChatModel via get_chat_model_from_env().
No BaseAgent / custom agent wrapper is used.

Nodes (in graph order):
  route_action_node  — No-op entry point for conditional routing
  locate_node        — Extract knowledge points from notebook records (LLM → JSON)
  interactive_node   — Generate interactive HTML for current knowledge point
  chat_node          — Answer user question about current knowledge point
  summary_node       — Generate learning summary report
  fix_html_node      — Regenerate HTML (delegates to interactive_node)
  advance_index_node — Increment current_index counter
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .lg_state import GuideState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_prompts(agent_name: str, language: str) -> dict[str, Any]:
    from src.services.prompt import get_prompt_manager
    try:
        return get_prompt_manager().load_prompts(
            module_name="guide",
            agent_name=agent_name,
            language=language,
        ) or {}
    except Exception as exc:
        logger.warning("Failed to load prompts %s/%s: %s", agent_name, language, exc)
        return {}


def _p(prompts: dict, key: str, default: str = "") -> str:
    return (prompts.get(key) or default).strip()


# ---------------------------------------------------------------------------
# Node: route_action  (no-op entry point for conditional routing)
# ---------------------------------------------------------------------------


async def route_action_node(state: GuideState) -> dict[str, Any]:
    """
    No-op pass-through.  The actual routing is done by the conditional
    edge function `decide_action()` in lg_graph.py which reads state["action"].
    """
    return {}


# ---------------------------------------------------------------------------
# Node: locate  (create session → extract knowledge points)
# ---------------------------------------------------------------------------


async def locate_node(state: GuideState) -> dict[str, Any]:
    """
    Extract structured knowledge points from notebook records using LangChain LLM.

    Returns:
        {"knowledge_points": list, "current_index": 0, "status": "initialized", "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    records: list = state.get("notebook_records", []) or []
    language: str = state.get("language", "en")
    notebook_id: str = state.get("notebook_id", "") or "cross_notebook"
    notebook_name: str = state.get("notebook_name", "") or "Notebook"

    knowledge_points: list = []

    try:
        prompts = _load_prompts("locate_agent", language)
        llm = get_chat_model_from_env()

        # Format records as readable text
        # NotebookRecord uses user_query / output; legacy records may use question/answer
        records_lines = []
        for i, rec in enumerate(records, 1):
            q = rec.get("user_query", rec.get("question", rec.get("query", rec.get("user_message", ""))))
            a = rec.get("output", rec.get("answer", rec.get("response", rec.get("assistant_message", ""))))
            title = rec.get("title", "")
            line = f"[{i}]{f' {title}' if title else ''}\n    Q: {q}\n    A: {a}"
            records_lines.append(line)
        records_content = "\n\n".join(records_lines) or "No records available."

        system_text = _p(
            prompts, "system",
            "You are an experienced Learning Planner. Analyze notebook records and extract knowledge points."
        )
        user_tpl = _p(
            prompts, "user_template",
            "Notebook ID: {notebook_id}\nNotebook Name: {notebook_name}\nRecords: {record_count}\n\n"
            "{records_content}\n\nExtract 3-5 knowledge points as a JSON array with fields: "
            "knowledge_title, knowledge_summary, user_difficulty."
        )
        user_text = user_tpl.format(
            notebook_id=notebook_id,
            notebook_name=notebook_name,
            record_count=len(records),
            records_content=records_content,
        )

        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw = response.content if hasattr(response, "content") else str(response)

        # Extract JSON array from response
        raw_clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", raw)
        # Try markdown code block first
        block_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw_clean)
        json_str = block_match.group(1).strip() if block_match else raw_clean.strip()
        # Try to find JSON array
        arr_match = re.search(r"\[[\s\S]*\]", json_str)
        if arr_match:
            json_str = arr_match.group(0)
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            knowledge_points = [
                kp for kp in parsed
                if isinstance(kp, dict) and kp.get("knowledge_title")
            ]
        logger.debug("locate_node: %d knowledge points extracted", len(knowledge_points))

    except Exception as exc:
        logger.warning("locate_node failed: %s", exc)

    return {
        "knowledge_points": knowledge_points,
        "current_index": 0,
        "status": "initialized",
        "streaming_events": [
            {
                "type": "status",
                "stage": "locate",
                "message": "knowledge_points_extracted",
                "knowledge_points": knowledge_points,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Node: interactive  (start / next → generate HTML for current point)
# ---------------------------------------------------------------------------

_FALLBACK_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{title}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#f8fafc;margin:0;padding:1.5rem;box-sizing:border-box;}}
.card{{background:#fff;border-radius:.75rem;padding:1.5rem;
  box-shadow:0 1px 3px rgba(0,0,0,.1);max-width:100%;}}
h2{{color:#1e40af;margin-top:0;}}p{{color:#374151;line-height:1.6;}}
.difficulty{{background:#eff6ff;border-left:4px solid #3b82f6;
  padding:.75rem 1rem;border-radius:0 .5rem .5rem 0;margin-top:1rem;font-size:.9rem;}}
</style></head>
<body>
<div class="card">
  <h2>{title}</h2>
  <p>{summary}</p>
  <div class="difficulty"><strong>Note:</strong> {difficulty}</div>
</div>
</body></html>"""


async def interactive_node(state: GuideState) -> dict[str, Any]:
    """
    Generate an interactive HTML learning page for the current knowledge point.

    Returns:
        {"current_html": str, "status": "learning", "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    points: list = state.get("knowledge_points", [])
    idx: int = state.get("current_index", 0)
    language: str = state.get("language", "en")
    user_message: str = state.get("user_message", "") or ""  # bug description when fix_html

    current_point: dict = points[idx] if points and idx < len(points) else {}
    html = ""

    try:
        prompts = _load_prompts("interactive_agent", language)
        llm = get_chat_model_from_env()

        system_text = _p(
            prompts, "system",
            "You are an Interactive Learning Designer. Generate complete, self-contained HTML pages for knowledge points."
        )
        user_tpl = _p(
            prompts, "user_template",
            "Title: {knowledge_title}\nContent: {knowledge_summary}\nDifficulties: {user_difficulty}\n\n"
            "Generate a complete interactive HTML page. Output only HTML, no markdown markers."
        )
        user_text = user_tpl.format(
            knowledge_title=current_point.get("knowledge_title", ""),
            knowledge_summary=current_point.get("knowledge_summary", ""),
            user_difficulty=current_point.get("user_difficulty", ""),
        )
        if user_message:
            user_text = f"[Bug reported: {user_message}]\n\n{user_text}"

        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw = response.content if hasattr(response, "content") else str(response)

        # Extract HTML from markdown code block if present
        block_match = re.search(r"```(?:html)?\s*\n?([\s\S]*?)```", raw, re.DOTALL)
        html = block_match.group(1).strip() if block_match else raw.strip()

        # Basic validation — must contain <html or <!DOCTYPE
        if not re.search(r"<!DOCTYPE|<html", html, re.IGNORECASE):
            raise ValueError("LLM response does not look like valid HTML")

        logger.debug("interactive_node: HTML generated (%d chars)", len(html))

    except Exception as exc:
        logger.warning("interactive_node failed, using fallback: %s", exc)
        html = _FALLBACK_HTML_TEMPLATE.format(
            title=current_point.get("knowledge_title", "Learning"),
            summary=current_point.get("knowledge_summary", ""),
            difficulty=current_point.get("user_difficulty", ""),
        )

    return {
        "current_html": html,
        "status": "learning",
        "streaming_events": [
            {
                "type": "html",
                "stage": "interactive",
                "html": html,
                "knowledge_point": current_point,
                "index": idx,
                "total": len(points),
            }
        ],
    }


# ---------------------------------------------------------------------------
# Node: chat  (answer user question about current knowledge point)
# ---------------------------------------------------------------------------


async def chat_node(state: GuideState) -> dict[str, Any]:
    """
    Answer the user's question about the current knowledge point.

    Returns:
        {"chat_history": [new messages], "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    points: list = state.get("knowledge_points", [])
    idx: int = state.get("current_index", 0)
    history: list = state.get("chat_history", [])
    user_message: str = state.get("user_message", "") or ""
    language: str = state.get("language", "en")

    current_point: dict = points[idx] if points and idx < len(points) else {}
    answer = ""

    try:
        if not user_message.strip():
            raise ValueError("Empty user message")

        prompts = _load_prompts("chat_agent", language)
        llm = get_chat_model_from_env()

        # Format last 10 messages of history
        recent = history[-10:]
        history_text = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in recent
        ) or "No previous conversation."

        system_text = _p(
            prompts, "system",
            "You are an Intelligent Learning Assistant. Answer questions about the current knowledge point."
        )
        user_tpl = _p(
            prompts, "user_template",
            "Knowledge Point: {knowledge_title}\nContent: {knowledge_summary}\n"
            "Difficulties: {user_difficulty}\n\nHistory:\n{chat_history}\n\n"
            "Question: {user_question}\n\nPlease answer clearly and helpfully."
        )
        user_text = user_tpl.format(
            knowledge_title=current_point.get("knowledge_title", ""),
            knowledge_summary=current_point.get("knowledge_summary", ""),
            user_difficulty=current_point.get("user_difficulty", ""),
            chat_history=history_text,
            user_question=user_message,
        )

        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        answer = response.content if hasattr(response, "content") else str(response)
        logger.debug("chat_node: answer=%d chars", len(answer))

    except Exception as exc:
        logger.warning("chat_node failed: %s", exc)
        answer = f"[Error generating answer: {exc}]"

    new_messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]
    return {
        "chat_history": new_messages,  # _concat_lists reducer appends these
        "streaming_events": [
            {
                "type": "chat_response",
                "stage": "chat",
                "answer": answer,
                "question": user_message,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Node: summary  (all points done → generate learning report)
# ---------------------------------------------------------------------------


async def summary_node(state: GuideState) -> dict[str, Any]:
    """
    Generate a personalised learning summary report.

    Returns:
        {"summary": str, "status": "completed", "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    points: list = state.get("knowledge_points", [])
    history: list = state.get("chat_history", [])
    notebook_name: str = state.get("notebook_name", "") or ""
    language: str = state.get("language", "en")

    summary = ""

    try:
        prompts = _load_prompts("summary_agent", language)
        llm = get_chat_model_from_env()

        # Format all knowledge points
        kp_lines = []
        for i, kp in enumerate(points, 1):
            kp_lines.append(
                f"{i}. {kp.get('knowledge_title', '')}\n"
                f"   Summary: {kp.get('knowledge_summary', '')}\n"
                f"   Difficulties: {kp.get('user_difficulty', '')}"
            )
        all_knowledge_points = "\n\n".join(kp_lines) or "No knowledge points."

        # Format full chat history
        history_lines = []
        for m in history:
            role = "User" if m.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {m.get('content', '')}")
        full_chat_history = "\n".join(history_lines) or "No conversation history."

        system_text = _p(
            prompts, "system",
            "You are a Learning Summary Expert. Generate a comprehensive learning summary report."
        )
        user_tpl = _p(
            prompts, "user_template",
            "Notebook: {notebook_name}\nKnowledge Points: {total_points}\n\n"
            "{all_knowledge_points}\n\nConversation:\n{full_chat_history}\n\n"
            "Generate a detailed learning summary in Markdown format."
        )
        user_text = user_tpl.format(
            notebook_name=notebook_name,
            total_points=len(points),
            all_knowledge_points=all_knowledge_points,
            full_chat_history=full_chat_history,
        )

        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw = response.content if hasattr(response, "content") else str(response)

        # Strip markdown code block wrapper if present
        block_match = re.search(r"```(?:markdown)?\s*\n?([\s\S]*?)```", raw, re.DOTALL)
        summary = block_match.group(1).strip() if block_match else raw.strip()
        logger.debug("summary_node: summary=%d chars", len(summary))

    except Exception as exc:
        logger.warning("summary_node failed, using fallback: %s", exc)
        summary = (
            f"# Learning Summary\n\n"
            f"You completed {len(points)} knowledge point(s) in **{notebook_name}**.\n\n"
            + "\n".join(f"- {kp.get('knowledge_title', '')}" for kp in points)
        )

    return {
        "summary": summary,
        "status": "completed",
        "streaming_events": [
            {
                "type": "summary",
                "stage": "summary",
                "summary": summary,
                "knowledge_points": points,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Node: fix_html  (user reported HTML bug → regenerate)
# ---------------------------------------------------------------------------


async def fix_html_node(state: GuideState) -> dict[str, Any]:
    """
    Regenerate the current HTML page (user reported a bug).
    Delegates to interactive_node — bug description is in state["user_message"].
    """
    return await interactive_node(state)


# ---------------------------------------------------------------------------
# Node: advance_index  (increment current_index on "next" action)
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
            {
                "type": "status",
                "stage": "advance",
                "message": "moving_to_next_point",
                "index": idx,
                "total": total,
            }
        ],
    }


__all__ = [
    "route_action_node",
    "locate_node",
    "interactive_node",
    "chat_node",
    "summary_node",
    "fix_html_node",
    "advance_index_node",
]
