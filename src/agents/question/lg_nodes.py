# -*- coding: utf-8 -*-
"""
Question Module — LangGraph Node Implementations
=================================================

All LLM calls use LangChain's BaseChatModel via get_chat_model_from_env().
No BaseAgent / custom agent wrapper is used.

Nodes (in pipeline order):
  retrieve_node    — Generate RAG queries and retrieve KB context
  generate_node    — Generate questions from context
  relevance_node   — Classify question-KB relevance
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .lg_state import QuestionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_prompts(agent_name: str, language: str) -> dict[str, Any]:
    from src.services.prompt import get_prompt_manager
    try:
        return get_prompt_manager().load_prompts(
            module_name="question",
            agent_name=agent_name,
            language=language,
        ) or {}
    except Exception as exc:
        logger.warning("Failed to load prompts %s/%s: %s", agent_name, language, exc)
        return {}


def _p(prompts: dict, key: str, default: str = "") -> str:
    return (prompts.get(key) or default).strip()


def _parse_json_robust(text: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response, handling markdown code blocks."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    content = match.group(1).strip() if match else text.strip()
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    obj_match = re.search(r"\{[\s\S]*\}", content)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    fixed = re.sub(r",\s*([}\]])", r"\1", content)
    fixed = re.sub(r'"""([\s\S]*?)"""', lambda m: json.dumps(m.group(1)), fixed)
    return json.loads(fixed)


# ---------------------------------------------------------------------------
# retrieve_node
# ---------------------------------------------------------------------------


async def retrieve_node(state: QuestionState) -> dict[str, Any]:
    """
    Generate RAG queries using LLM, then retrieve KB context in parallel.

    Returns:
        {"knowledge_context": str, "retrieval_queries": list, "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env
    from src.tools.rag_tool import rag_search

    kb_name: str = state.get("kb_name", "") or ""
    requirement: dict = state.get("requirement", {}) or {}
    language: str = state.get("language", "en")
    num_queries = 3

    knowledge_context = ""
    queries: list[str] = []

    try:
        prompts = _load_prompts("retrieve_agent", language)
        llm = get_chat_model_from_env()

        requirement_text = json.dumps(requirement, ensure_ascii=False, indent=2)

        system_text = _p(prompts, "system", "You are a knowledge base retrieval assistant.")
        user_tpl = _p(
            prompts, "generate_queries",
            'Extract {num_queries} knowledge point names for retrieval from:\n{requirement_text}\n\nReturn JSON: {{"queries": ["point1", ...]}}',
        )
        user_text = user_tpl.format(requirement_text=requirement_text, num_queries=num_queries)

        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw = response.content if hasattr(response, "content") else str(response)

        try:
            data = _parse_json_robust(raw)
            queries_raw = data.get("queries", [])
            if isinstance(queries_raw, dict):
                queries_raw = list(queries_raw.values())
            elif not isinstance(queries_raw, list):
                queries_raw = [str(queries_raw)]
            queries = [q.strip() for q in queries_raw if q and str(q).strip()][:num_queries]
        except Exception as exc:
            logger.warning("retrieve_node: failed to parse query JSON: %s", exc)

        if not queries:
            queries = [requirement_text[:100]]

        logger.debug("retrieve_node: %d queries generated", len(queries))

        async def _search(q: str) -> dict[str, Any]:
            try:
                result = await rag_search(query=q, kb_name=kb_name, mode="naive", only_need_context=True)
                return {"query": q, "answer": result.get("answer", "")}
            except Exception as exc:
                logger.warning("RAG search failed for '%s': %s", q, exc)
                return {"query": q, "answer": ""}

        retrievals = await asyncio.gather(*[_search(q) for q in queries])
        retrievals = [r for r in retrievals if r.get("answer")]

        if retrievals:
            lines = []
            for item in retrievals:
                lines.append(f"=== Query: {item['query']} ===")
                answer = item["answer"]
                if len(answer) > 2000:
                    answer = answer[:2000] + "...[truncated]"
                lines.append(answer)
                lines.append("")
            knowledge_context = "\n".join(lines)
        else:
            knowledge_context = "No retrieval context available."

        logger.debug("retrieve_node: context=%d chars", len(knowledge_context))

    except Exception as exc:
        logger.warning("retrieve_node failed: %s", exc)

    return {
        "knowledge_context": knowledge_context,
        "retrieval_queries": queries,
        "streaming_events": [
            {"type": "status", "stage": "retrieve", "message": "retrieving_context"}
        ],
    }


# ---------------------------------------------------------------------------
# generate_node
# ---------------------------------------------------------------------------


def _parse_question(response: str) -> dict[str, Any]:
    """Parse LLM response into a validated question dict."""
    if not response or not response.strip():
        raise ValueError("LLM returned empty response")

    question = _parse_json_robust(response)

    if "question" not in question:
        raise ValueError("Question response missing 'question' field")
    if "question_type" not in question:
        question["question_type"] = "written"

    if question.get("question_type") == "choice":
        options = question.get("options")
        if not options:
            question["options"] = {
                "A": "Option A (placeholder)", "B": "Option B (placeholder)",
                "C": "Option C (placeholder)", "D": "Option D (placeholder)",
            }
        elif isinstance(options, list):
            question["options"] = {chr(65 + i): str(opt) for i, opt in enumerate(options[:4])}
        elif not isinstance(options, dict):
            question["options"] = {"A": str(options)}

    return question


async def generate_node(state: QuestionState) -> dict[str, Any]:
    """
    Generate questions using LangChain LLM. Runs num_questions times in parallel.

    Returns:
        {"generated_questions": list, "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    knowledge_context: str = state.get("knowledge_context", "") or ""
    requirement: dict = state.get("requirement", {}) or {}
    focus: str = state.get("focus", "") or ""
    num_questions: int = state.get("num_questions", 5)
    language: str = state.get("language", "en")

    questions: list[dict] = []

    try:
        prompts = _load_prompts("generate_agent", language)
        llm = get_chat_model_from_env()

        requirements_str = json.dumps(requirement, ensure_ascii=False, indent=2)
        focus_str = (
            f"Focus: {focus}\nType: {requirement.get('question_type', 'written')}" if focus
            else f"Type: {requirement.get('question_type', 'written')}"
        )
        knowledge_snippet = knowledge_context[:4000] if len(knowledge_context) > 4000 else knowledge_context

        system_text = _p(prompts, "system", "You are a professional Question Generation Agent.")
        user_tpl = _p(
            prompts, "generate",
            "Generate a question based on:\nRequirements: {requirements}\nFocus: {focus}\nKnowledge: {knowledge}\n\nReturn JSON with question_type, question, correct_answer, explanation.",
        )
        user_text = user_tpl.format(
            requirements=requirements_str,
            focus=focus_str,
            knowledge=knowledge_snippet,
        )

        async def _generate_one() -> dict[str, Any] | None:
            try:
                resp = await llm.ainvoke([
                    SystemMessage(content=system_text),
                    HumanMessage(content=user_text),
                ])
                raw = resp.content if hasattr(resp, "content") else str(resp)
                q = _parse_question(raw)
                q["knowledge_point"] = requirement.get("knowledge_point", "")
                return q
            except Exception as exc:
                logger.warning("generate_node: one question failed: %s", exc)
                return None

        results = await asyncio.gather(*[_generate_one() for _ in range(num_questions)])
        questions = [q for q in results if q is not None]
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


# ---------------------------------------------------------------------------
# relevance_node
# ---------------------------------------------------------------------------


async def relevance_node(state: QuestionState) -> dict[str, Any]:
    """
    Classify question-KB relevance using LangChain LLM.

    Returns:
        {"relevance_results": list, "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    questions: list[dict] = state.get("generated_questions", [])
    knowledge_context: str = state.get("knowledge_context", "") or ""
    language: str = state.get("language", "en")

    relevance_results: list[dict] = []

    try:
        prompts = _load_prompts("relevance_analyzer", language)
        llm = get_chat_model_from_env()

        knowledge_snippet = (
            knowledge_context[:4000] + "...[truncated]"
            if len(knowledge_context) > 4000
            else knowledge_context
        )

        system_text = _p(prompts, "system", "You are an educational content analyst.")
        user_tpl = _p(
            prompts, "analyze_relevance",
            'Analyze relevance:\nQuestion:\n{question}\n\nKnowledge:\n{knowledge}\n\nReturn JSON: {{"relevance":"high"/"partial","kb_coverage":"...","extension_points":"..."}}',
        )

        async def _analyze(q: dict) -> dict[str, Any]:
            try:
                question_str = json.dumps(q, ensure_ascii=False, indent=2)
                resp = await llm.ainvoke([
                    SystemMessage(content=system_text),
                    HumanMessage(content=user_tpl.format(
                        question=question_str, knowledge=knowledge_snippet
                    )),
                ])
                raw = resp.content if hasattr(resp, "content") else str(resp)
                result = _parse_json_robust(raw)
                relevance = result.get("relevance", "partial")
                if relevance not in ("high", "partial"):
                    relevance = "partial"
                return {
                    **q,
                    "relevance": relevance,
                    "kb_coverage": result.get("kb_coverage", ""),
                    "extension_points": result.get("extension_points", "") if relevance == "partial" else "",
                }
            except Exception as exc:
                logger.warning("relevance_node: analysis failed: %s", exc)
                return {**q, "relevance": "partial", "kb_coverage": "", "extension_points": str(exc)}

        relevance_results = list(await asyncio.gather(*[_analyze(q) for q in questions]))
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
