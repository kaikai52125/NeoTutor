# -*- coding: utf-8 -*-
"""
Question Module — LangGraph Node Implementations
=================================================

Nodes (in pipeline order):
  retrieve_node    — Use RetrieveAgent to fetch KB context
  generate_node    — Use GenerateAgent to produce questions
  relevance_node   — Use RelevanceAnalyzer to classify questions
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .lg_state import QuestionState

logger = logging.getLogger(__name__)


async def retrieve_node(state: QuestionState) -> dict[str, Any]:
    """
    Retrieve knowledge context for question generation using RetrieveAgent.

    Returns:
        {"knowledge_context": str, "retrieval_queries": list, "streaming_events": list}
    """
    kb_name: str = state.get("kb_name", "") or ""
    requirement: dict = state.get("requirement", {}) or {}
    language: str = state.get("language", "en")

    knowledge_context = ""
    queries: list[str] = []
    try:
        from src.agents.question.agents import RetrieveAgent
        from src.services.llm.config import get_llm_config

        cfg = get_llm_config()
        # kb_name and language are constructor parameters, not process() parameters
        agent = RetrieveAgent(
            kb_name=kb_name,
            language=language,
            config={},
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            api_version=getattr(cfg, "api_version", None),
        )
        result = await agent.process(
            requirement=requirement,
        )
        knowledge_context = result.get("summary", "") or ""
        queries = result.get("queries", [])
        logger.debug("retrieve_node: %d queries, context=%d chars", len(queries), len(knowledge_context))
    except Exception as exc:
        logger.warning("retrieve_node failed: %s", exc)

    return {
        "knowledge_context": knowledge_context,
        "retrieval_queries": queries,
        "streaming_events": [
            {"type": "status", "stage": "retrieve", "message": "retrieving_context"}
        ],
    }


async def generate_node(state: QuestionState) -> dict[str, Any]:
    """
    Generate questions using GenerateAgent.

    GenerateAgent.process() generates one question at a time.
    We call it num_questions times in parallel.

    Returns:
        {"generated_questions": list, "streaming_events": list}
    """
    knowledge_context: str = state.get("knowledge_context", "") or ""
    requirement: dict = state.get("requirement", {}) or {}
    focus: str = state.get("focus", "") or ""
    num_questions: int = state.get("num_questions", 5)
    language: str = state.get("language", "en")

    questions: list[dict] = []
    try:
        from src.agents.question.agents import GenerateAgent
        from src.services.llm.config import get_llm_config

        cfg = get_llm_config()
        # language is a constructor parameter, not a process() parameter
        agent = GenerateAgent(
            language=language,
            config={},
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            api_version=getattr(cfg, "api_version", None),
        )

        # GenerateAgent generates one question per call — run num_questions times in parallel
        focus_arg = {"focus": focus} if focus else None
        tasks = [
            agent.process(
                requirement=requirement,
                knowledge_context=knowledge_context,
                focus=focus_arg,
            )
            for _ in range(num_questions)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                logger.warning("generate_node: one question failed: %s", res)
                continue
            if res.get("success") and res.get("question"):
                questions.append(res["question"])

        logger.debug("generate_node: %d questions generated", len(questions))
    except Exception as exc:
        logger.warning("generate_node failed: %s", exc)

    return {
        "generated_questions": questions,
        "streaming_events": [
            {"type": "status", "stage": "generate", "message": "generating_questions",
             "count": len(questions)}
        ],
    }


async def relevance_node(state: QuestionState) -> dict[str, Any]:
    """
    Classify question relevance using RelevanceAnalyzer.

    Returns:
        {"relevance_results": list, "streaming_events": list}
    """
    questions: list[dict] = state.get("generated_questions", [])
    knowledge_context: str = state.get("knowledge_context", "") or ""
    language: str = state.get("language", "en")

    relevance_results: list[dict] = []
    try:
        from src.agents.question.agents import RelevanceAnalyzer
        from src.services.llm.config import get_llm_config

        cfg = get_llm_config()
        # language is a constructor parameter, not a process() parameter
        agent = RelevanceAnalyzer(
            language=language,
            config={},
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            api_version=getattr(cfg, "api_version", None),
        )
        for q in questions:
            result = await agent.process(
                question=q,
                knowledge_context=knowledge_context,
            )
            relevance_results.append({**q, **result})
        logger.debug("relevance_node: %d questions classified", len(relevance_results))
    except Exception as exc:
        logger.warning("relevance_node failed, using raw questions: %s", exc)
        relevance_results = questions

    return {
        "relevance_results": relevance_results,
        "streaming_events": [
            {"type": "status", "stage": "relevance", "message": "analysing_relevance",
             "count": len(relevance_results)}
        ],
    }


__all__ = ["retrieve_node", "generate_node", "relevance_node"]
