# -*- coding: utf-8 -*-
"""
CoWriter Module — LangGraph Node Implementations
=================================================

Edit pipeline nodes:
  retrieve_context_node  — fetch RAG/web context (if source != "none")
  edit_node              — EditAgent.process()

Narrate pipeline node:
  narrate_node           — NarratorAgent.narrate()
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from .lg_state import CoWriterState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edit pipeline
# ---------------------------------------------------------------------------


async def retrieve_context_node(state: CoWriterState) -> dict[str, Any]:
    """
    Optionally retrieve RAG or web context before editing.

    Returns:
        {"context": str, "streaming_events": list}
    """
    source: str = state.get("source", "none") or "none"
    kb_name: str = state.get("kb_name", "") or ""
    text: str = state.get("text", "") or ""
    instruction: str = state.get("instruction", "") or ""

    context = ""
    if source == "rag" and kb_name:
        try:
            from src.tools.rag_tool import rag_search
            result = await rag_search(
                query=f"{instruction}\n\n{text[:200]}",
                kb_name=kb_name,
                mode="hybrid",
            )
            context = result.get("answer", "") or result.get("content", "")
            logger.debug("retrieve_context_node (rag): %d chars", len(context))
        except Exception as exc:
            logger.warning("retrieve_context_node RAG failed: %s", exc)

    elif source == "web":
        try:
            from src.tools.web_search import web_search
            results = await web_search(query=f"{instruction} {text[:100]}")
            if isinstance(results, list):
                context = "\n\n".join(
                    r.get("content", r.get("snippet", "")) for r in results[:3]
                )
            elif isinstance(results, str):
                context = results
            logger.debug("retrieve_context_node (web): %d chars", len(context))
        except Exception as exc:
            logger.warning("retrieve_context_node web failed: %s", exc)

    return {
        "context": context,
        "streaming_events": [
            {"type": "status", "stage": "retrieve_context",
             "message": f"retrieving_{source}_context"}
        ] if source != "none" else [],
    }


async def edit_node(state: CoWriterState) -> dict[str, Any]:
    """
    Edit text using EditAgent.

    Returns:
        {"edited_text": str, "operation_id": str, "streaming_events": list}
    """
    text: str = state.get("text", "") or ""
    instruction: str = state.get("instruction", "") or ""
    action: str = state.get("action", "rewrite") or "rewrite"
    source: str = state.get("source", "none") or "none"
    kb_name: str = state.get("kb_name", "") or ""
    context: str = state.get("context", "") or ""
    language: str = state.get("language", "en")
    operation_id: str = state.get("operation_id", "") or str(uuid.uuid4())

    edited_text = text
    try:
        from src.agents.co_writer.edit_agent import EditAgent

        agent = EditAgent(language=language)
        result = await agent.process(
            text=text,
            instruction=instruction,
            action=action,
            source=source if source != "none" else None,
            kb_name=kb_name or None,
        )
        edited_text = result.get("edited_text", text)
        operation_id = result.get("operation_id", operation_id)
        logger.debug("edit_node: %d → %d chars", len(text), len(edited_text))
    except Exception as exc:
        logger.warning("edit_node failed: %s", exc)

    return {
        "edited_text": edited_text,
        "operation_id": operation_id,
        "streaming_events": [
            {"type": "result", "stage": "edit",
             "edited_text": edited_text, "operation_id": operation_id}
        ],
    }


# ---------------------------------------------------------------------------
# Narrate pipeline
# ---------------------------------------------------------------------------


async def narrate_node(state: CoWriterState) -> dict[str, Any]:
    """
    Generate narration script (and optionally TTS audio) using NarratorAgent.

    Returns:
        {"script": str, "key_points": list, "has_audio": bool, "audio_url": str,
         "streaming_events": list}
    """
    content: str = state.get("content", "") or ""
    style: str = state.get("style", "friendly") or "friendly"
    voice: str = state.get("voice", "") or ""
    skip_audio: bool = state.get("skip_audio", True)
    language: str = state.get("language", "en")

    script = ""
    key_points: list[str] = []
    has_audio = False
    audio_url = ""
    try:
        from src.agents.co_writer.narrator_agent import NarratorAgent

        agent = NarratorAgent(language=language)
        result = await agent.narrate(
            content=content,
            style=style,
            voice=voice or None,
            skip_audio=skip_audio,
        )
        script = result.get("script", "")
        key_points = result.get("key_points", [])
        has_audio = result.get("has_audio", False)
        audio_url = result.get("audio_url", "")
        logger.debug("narrate_node: script=%d chars, has_audio=%s", len(script), has_audio)
    except Exception as exc:
        logger.warning("narrate_node failed: %s", exc)

    return {
        "script": script,
        "key_points": key_points,
        "has_audio": has_audio,
        "audio_url": audio_url,
        "streaming_events": [
            {"type": "result", "stage": "narrate",
             "script": script, "key_points": key_points,
             "has_audio": has_audio, "audio_url": audio_url}
        ],
    }


__all__ = [
    "retrieve_context_node",
    "edit_node",
    "narrate_node",
]
