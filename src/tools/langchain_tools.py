# -*- coding: utf-8 -*-
"""
LangChain Tool Wrappers
========================

Wraps existing NeoTutor tool functions as LangChain 1.x @tool objects so they
can be bound to ChatModels via `llm.bind_tools(...)` and used inside LangGraph nodes.

IMPORTANT: The original tool implementations in src/tools/ are NOT modified.
This module is purely an adapter layer.

Tool groups:
  SOLVE_TOOLS    — rag_search, web_search, code_execution, query_item
  RESEARCH_TOOLS — rag_search, web_search, paper_search, code_execution, query_item
  CHAT_TOOLS     — rag_search, web_search
"""

from __future__ import annotations

import json
from typing import Sequence

from langchain_core.tools import BaseTool, tool


# ---------------------------------------------------------------------------
# Individual tool definitions
# ---------------------------------------------------------------------------


@tool
async def rag_search_tool(
    query: str,
    kb_name: str = "ai_textbook",
    mode: str = "hybrid",
) -> str:
    """
    Search the knowledge base for information relevant to the query.

    Use this tool when the answer is likely contained in the local knowledge base.

    Args:
        query:   Natural-language search query.
        kb_name: Knowledge base name (default: ai_textbook).
        mode:    Retrieval mode — "hybrid" (vector + keyword) or "naive" (vector only).

    Returns:
        Retrieved knowledge as plain text.
    """
    from src.tools.rag_tool import rag_search  # noqa: PLC0415

    result = await rag_search(query=query, kb_name=kb_name, mode=mode)
    return result.get("answer", "") or result.get("content", "") or ""


@tool
async def web_search_tool(query: str) -> str:
    """
    Search the internet for up-to-date information.

    Use this tool when the answer is not in the knowledge base or requires
    real-time data (e.g. recent events, live statistics).

    Args:
        query: Search query string.

    Returns:
        Web search results as plain text.
    """
    from src.tools.web_search import web_search  # noqa: PLC0415

    result = await web_search(query=query, verbose=False)
    return result.get("answer", "") or result.get("content", "") or ""


@tool
async def paper_search_tool(query: str, max_results: int = 3) -> str:
    """
    Search academic databases for relevant research papers.

    Use this tool when looking for citations, literature reviews, or
    specific academic findings.

    Args:
        query:       Keywords or title fragment to search for.
        max_results: Maximum number of papers to return (default: 3).

    Returns:
        JSON string with a "papers" list, each entry containing title, authors,
        abstract, and URL.
    """
    from src.tools.paper_search_tool import PaperSearchTool  # noqa: PLC0415

    papers = await PaperSearchTool().search_papers(query=query, max_results=max_results)
    return json.dumps({"papers": papers}, ensure_ascii=False)


@tool
async def code_execution_tool(code: str, language: str = "python") -> str:
    """
    Execute code and return the output.

    Use this tool to verify mathematical computations, run simulations, or
    generate numerical results that require precise calculation.

    Args:
        code:     Source code to execute.
        language: Programming language (default: "python").

    Returns:
        JSON string with "stdout", "stderr", "return_code", and "error" fields.
    """
    from src.tools.code_executor import run_code  # noqa: PLC0415

    result = await run_code(language=language, code=code)
    return json.dumps(result, ensure_ascii=False)


@tool
async def query_item_tool(identifier: str, kb_name: str = "ai_textbook") -> str:
    """
    Look up a numbered item in the knowledge base by its identifier.

    Use this tool to retrieve specific theorems, equations, definitions, or
    figures referenced by their number (e.g. "Theorem 3.1", "Equation (2.5)").

    Args:
        identifier: Item identifier string (e.g. "3.1", "Figure 5", "Theorem 2").
        kb_name:    Knowledge base name (default: ai_textbook).

    Returns:
        JSON string with the matched item's content and metadata.
    """
    from src.tools.query_item_tool import query_numbered_item  # noqa: PLC0415

    result = await query_numbered_item(identifier=identifier, kb_name=kb_name)
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool groups (used by LangGraph nodes to bind the appropriate subset)
# ---------------------------------------------------------------------------

SOLVE_TOOLS: Sequence[BaseTool] = [
    rag_search_tool,
    web_search_tool,
    code_execution_tool,
    query_item_tool,
]

RESEARCH_TOOLS: Sequence[BaseTool] = [
    rag_search_tool,
    web_search_tool,
    paper_search_tool,
    code_execution_tool,
    query_item_tool,
]

CHAT_TOOLS: Sequence[BaseTool] = [
    rag_search_tool,
    web_search_tool,
]

ALL_TOOLS: Sequence[BaseTool] = [
    rag_search_tool,
    web_search_tool,
    paper_search_tool,
    code_execution_tool,
    query_item_tool,
]

__all__ = [
    "rag_search_tool",
    "web_search_tool",
    "paper_search_tool",
    "code_execution_tool",
    "query_item_tool",
    "SOLVE_TOOLS",
    "RESEARCH_TOOLS",
    "CHAT_TOOLS",
    "ALL_TOOLS",
]
