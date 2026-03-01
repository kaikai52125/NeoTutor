# -*- coding: utf-8 -*-
"""
Chat Module — LangGraph Node Implementations
=============================================

Each function is a LangGraph node: it receives the current ChatState and returns
a dict of state updates.

Nodes:
  retrieve_context_node  — Optional RAG / web-search retrieval
  chat_node              — Main conversation node using LangChain ChatModel
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from .lg_state import ChatState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node: retrieve_context
# ---------------------------------------------------------------------------


async def retrieve_context_node(state: ChatState) -> dict[str, Any]:
    """
    Optionally retrieve context from the knowledge base and/or the web.

    Corresponds to the ChatAgent.retrieve_context() method in the original
    implementation. The results are stored in state["rag_context"] and
    state["sources"] so that chat_node can incorporate them into the prompt.

    Args:
        state: Current ChatState.

    Returns:
        Partial state update with "rag_context", "sources", and
        optionally "streaming_events" (progress notifications).
    """
    enable_rag: bool = state.get("enable_rag", False)  # type: ignore[assignment]
    enable_web: bool = state.get("enable_web_search", False)  # type: ignore[assignment]
    kb_name: str = state.get("kb_name", "") or ""
    sources: dict = {"rag": [], "web": []}
    context_parts: list[str] = []
    events: list[dict] = []

    # Last human message text (used as search query)
    last_human_text = ""
    for msg in reversed(state.get("messages", [])):  # type: ignore[arg-type]
        if hasattr(msg, "content") and getattr(msg, "type", "") == "human":
            last_human_text = str(msg.content)
            break
    if not last_human_text:
        for msg in reversed(state.get("messages", [])):  # type: ignore[arg-type]
            if hasattr(msg, "content"):
                last_human_text = str(msg.content)
                break

    # RAG retrieval
    if enable_rag and kb_name:
        events.append({"type": "status", "stage": "rag",
                        "message": f"Searching knowledge base: {kb_name}..."})
        try:
            from src.tools.rag_tool import rag_search  # noqa: PLC0415

            rag_result = await rag_search(query=last_human_text, kb_name=kb_name, mode="hybrid")
            rag_answer: str = rag_result.get("answer", "") or ""
            if rag_answer:
                context_parts.append(f"[Knowledge Base: {kb_name}]\n{rag_answer}")
                sources["rag"].append({
                    "kb_name": kb_name,
                    "content": rag_answer[:500] + ("..." if len(rag_answer) > 500 else ""),
                })
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)

    # Web search
    if enable_web:
        events.append({"type": "status", "stage": "web", "message": "Searching the web..."})
        try:
            import asyncio  # noqa: PLC0415
            from src.tools.web_search import web_search  # noqa: PLC0415

            web_result = await asyncio.to_thread(
                web_search, query=last_human_text, verbose=False
            )
            web_answer: str = web_result.get("answer", "") or ""
            web_citations: list = web_result.get("citations", [])
            if web_answer:
                context_parts.append(f"[Web Search Results]\n{web_answer}")
                sources["web"] = web_citations[:5]
            elif web_citations:
                # jina returns citations/search_results but no LLM answer;
                # build a brief context from the top snippets
                snippets = [
                    f"- {c.get('title', '')}: {c.get('snippet', '') or c.get('content', '')[:200]}"
                    for c in web_citations[:3]
                    if c.get("title") or c.get("snippet") or c.get("content")
                ]
                if snippets:
                    context_parts.append("[Web Search Results]\n" + "\n".join(snippets))
                sources["web"] = web_citations[:5]
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
        else:
            # Extract image URLs from search results
            images: list[str] = []
            # Tavily: to_dict() promotes metadata keys to top level, so images is at root
            for img in web_result.get("images", []):
                url_str = img if isinstance(img, str) else img.get("url", "")
                if url_str and url_str.startswith("http"):
                    images.append(url_str)
            # Jina: each citation may have attributes.images (dict alt->url)
            for c in web_citations:
                for img_url in (c.get("attributes") or {}).get("images", {}).values():
                    if img_url and img_url.startswith("http"):
                        images.append(img_url)
            if images:
                sources["images"] = images[:6]  # 最多展示 6 张
                # 把图片列表追加到上下文，让 LLM 在回答中直接用 Markdown 语法嵌入
                img_lines = "\n".join(
                    f"图片{i+1} URL: {url}" for i, url in enumerate(images[:6])
                )
                context_parts.append(
                    "[Web Search Images]\n"
                    "以下是搜索到的相关图片URL，请在回答的合适位置直接用 Markdown 语法嵌入，"
                    "格式必须是 ![简短描述](完整URL)，不要写图1、图2等文字占位符：\n"
                    + img_lines
                )

    return {
        "rag_context": "\n\n".join(context_parts),
        "sources": sources,
        "streaming_events": events,
    }


# ---------------------------------------------------------------------------
# Node: chat
# ---------------------------------------------------------------------------


async def chat_node(state: ChatState) -> dict[str, Any]:
    """
    Main conversation node.

    Builds the message list (system prompt + optional context + history + latest
    human message) and calls the LangChain ChatModel via ainvoke().  The
    AIMessage returned is appended to state["messages"] by the add_messages reducer.

    Args:
        state: Current ChatState.

    Returns:
        Partial state update containing the new AIMessage in "messages".
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env  # noqa: PLC0415
    from src.services.settings.interface_settings import get_ui_language  # noqa: PLC0415

    language = state.get("language") or get_ui_language(default="en")

    # ── System prompt ────────────────────────────────────────────────────────
    # Try to load the prompt from the ChatAgent's YAML; fall back to a default.
    system_text = _get_system_prompt(language)

    # ── Build message list ───────────────────────────────────────────────────
    messages_for_llm: list = [SystemMessage(content=system_text)]

    rag_context: str = state.get("rag_context", "") or ""
    if rag_context:
        # 如果上下文中包含图片，附加使用说明
        has_images = "[Web Search Images]" in rag_context
        img_instruction = (
            "\n\n【强制要求】上下文中已提供图片URL，你必须在回答的对应位置直接嵌入图片，"
            "格式为 ![简短描述](完整URL)。"
            "严禁写图1、图2等文字占位符，必须用真实URL。"
            if has_images else ""
        )
        messages_for_llm.append(
            SystemMessage(content=f"Reference context:\n{rag_context}{img_instruction}")
        )

    # Append conversation history (already accumulated by add_messages)
    messages_for_llm.extend(state.get("messages", []))  # type: ignore[arg-type]

    # ── Call LLM ─────────────────────────────────────────────────────────────
    llm = get_chat_model_from_env()
    response: AIMessage = await llm.ainvoke(messages_for_llm)  # type: ignore[assignment]

    logger.debug("Chat response: %d chars", len(str(response.content)))

    return {
        "messages": [response],  # add_messages reducer will append this
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_system_prompt(language: str) -> str:
    """
    Load the system prompt for the chat agent from the YAML prompt file.
    Falls back to a sensible default if loading fails.
    """
    default = (
        "You are a helpful AI learning assistant. "
        "Answer questions clearly and concisely."
    )
    try:
        from src.services.prompt import get_prompt_manager  # noqa: PLC0415

        prompts = get_prompt_manager().load_prompts(
            module_name="chat",
            agent_name="chat_agent",
            language=language,
        )
        return prompts.get("system", default) or default
    except Exception:
        return default


__all__ = ["retrieve_context_node", "chat_node"]
