# -*- coding: utf-8 -*-
"""
IdeaGen Module — LangGraph Node Implementations
================================================

All LLM calls use LangChain's BaseChatModel via get_chat_model_from_env().
No BaseAgent / custom agent wrapper is used.

Nodes (in pipeline order):
  extract_node        — Extract knowledge points from notebook records
  loose_filter_node   — Remove obviously unsuitable knowledge points
  explore_node        — Generate research ideas per knowledge point
  strict_filter_node  — Keep only high-quality idea candidates
  statement_node      — Generate final markdown research statements
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .lg_state import IdeaGenState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_prompts(agent_name: str, language: str) -> dict[str, Any]:
    from src.services.prompt import get_prompt_manager
    try:
        return get_prompt_manager().load_prompts(
            module_name="ideagen",
            agent_name=agent_name,
            language=language,
        ) or {}
    except Exception as exc:
        logger.warning("Failed to load prompts %s/%s: %s", agent_name, language, exc)
        return {}


def _p(prompts: dict, key: str, default: str = "") -> str:
    return (prompts.get(key) or default).strip()


def _parse_json(raw: str) -> Any:
    """Parse JSON from LLM response, handling markdown code blocks and control chars."""
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", raw)
    block = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", clean)
    text = block.group(1).strip() if block else clean.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Node: extract  (Stage 1 — knowledge point extraction)
# ---------------------------------------------------------------------------


async def extract_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Extract structured knowledge points from notebook records.

    Returns:
        {"knowledge_points": list[{knowledge_point, description}], "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    records: list[dict] = state.get("notebook_records", []) or []
    user_thoughts: str = state.get("user_thoughts", "") or ""
    language: str = state.get("language", "en")

    points: list[dict] = []

    # Text-only mode: create virtual knowledge point from user_thoughts
    if not records and user_thoughts.strip():
        points = [{"knowledge_point": "User Research Topic", "description": user_thoughts.strip()}]
        logger.debug("extract_node: text-only mode, created virtual knowledge point")
        return {
            "knowledge_points": points,
            "streaming_events": [
                {"type": "status", "stage": "extracting",
                 "message": "extracting_knowledge_points", "count": len(points)}
            ],
        }

    try:
        prompts = _load_prompts("material_organizer", language)
        llm = get_chat_model_from_env()

        # Format records as materials text (matches material_organizer.yaml template vars)
        materials_text = ""
        for i, rec in enumerate(records, 1):
            materials_text += f"\n\n=== Record {i} ===\n"
            materials_text += f"Type: {rec.get('type', '')}\n"
            materials_text += f"Title: {rec.get('title', '')}\n"
            materials_text += f"User Query: {rec.get('user_query', rec.get('question', rec.get('query', '')))}\n"
            materials_text += f"System Response: {rec.get('output', rec.get('answer', rec.get('response', '')))}\n"

        user_thoughts_text = ""
        if user_thoughts.strip():
            user_thoughts_text = f"\n\nUser Additional Thoughts:\n{user_thoughts}"

        system_text = _p(
            prompts, "system",
            "You are a knowledge organization expert. Extract independent, researchable knowledge points "
            "from the notebook records. Output JSON: {\"knowledge_points\": [{\"knowledge_point\": str, \"description\": str}]}"
        )
        user_tpl = _p(
            prompts, "user_template",
            "Please analyze the following notebook records and extract relatively independent knowledge points:\n\n"
            "{materials_text}{user_thoughts_text}\n\nPlease output the extracted knowledge points in JSON format."
        )
        user_text = user_tpl.format(
            materials_text=materials_text,
            user_thoughts_text=user_thoughts_text,
        )

        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw = response.content if hasattr(response, "content") else str(response)

        result = _parse_json(raw)
        raw_points = result.get("knowledge_points", []) if isinstance(result, dict) else []

        for pt in raw_points:
            if not isinstance(pt, dict):
                continue
            kp = str(pt.get("knowledge_point", "")).strip()
            desc = str(pt.get("description", "")).strip()
            if kp and desc and len(desc) >= 10:
                points.append({"knowledge_point": kp, "description": desc})

        logger.debug("extract_node: %d knowledge points extracted", len(points))

    except Exception as exc:
        logger.warning("extract_node primary failed: %s", exc)

    # Fallback: try again with simpler prompt if nothing extracted
    if not points and records:
        try:
            prompts = _load_prompts("material_organizer", language)
            llm = get_chat_model_from_env()

            materials_text = ""
            for i, rec in enumerate(records, 1):
                materials_text += (
                    f"\nRecord {i}: {rec.get('title', '')} - "
                    f"{str(rec.get('user_query', rec.get('question', '')))[:100]}"
                )

            user_thoughts_str = f"User thoughts: {user_thoughts}" if user_thoughts else ""
            system_text = _p(
                prompts, "fallback_system",
                "You are a knowledge extraction expert. Extract at least 1 knowledge point. "
                "Output JSON: {\"knowledge_points\": [{\"knowledge_point\": str, \"description\": str}]}"
            )
            user_tpl = _p(
                prompts, "fallback_user_template",
                "Please extract knowledge points from the following records:\n\n"
                "{materials_text}\n\n{user_thoughts}\n\nOutput JSON, must extract at least 1 knowledge point."
            )
            user_text = user_tpl.format(
                materials_text=materials_text,
                user_thoughts=user_thoughts_str,
            )

            response = await llm.ainvoke([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            result = _parse_json(raw)
            raw_points = result.get("knowledge_points", []) if isinstance(result, dict) else []

            for pt in raw_points:
                if not isinstance(pt, dict):
                    continue
                kp = str(pt.get("knowledge_point", "")).strip()
                desc = str(pt.get("description", "")).strip()
                if kp and desc:
                    points.append({"knowledge_point": kp, "description": desc})

            logger.debug("extract_node fallback: %d points", len(points))
        except Exception as exc2:
            logger.warning("extract_node fallback also failed: %s", exc2)

    # Last-resort synthetic point
    if not points and records:
        points = [{
            "knowledge_point": "Comprehensive Knowledge Point",
            "description": (
                f"Comprehensive knowledge content based on {len(records)} records, "
                "containing multiple research directions and concepts."
            ),
        }]

    return {
        "knowledge_points": points,
        "streaming_events": [
            {"type": "status", "stage": "extracting",
             "message": "extracting_knowledge_points", "count": len(points)}
        ],
    }


# ---------------------------------------------------------------------------
# Node: loose_filter  (Stage 2 — remove obviously unsuitable points)
# ---------------------------------------------------------------------------


async def loose_filter_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Loose filter: remove obviously unsuitable knowledge points.

    Returns:
        {"filtered_points": list, "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    points: list[dict] = state.get("knowledge_points", []) or []
    language: str = state.get("language", "en")

    filtered: list[dict] = points  # default: pass all through

    try:
        prompts = _load_prompts("idea_generation", language)
        llm = get_chat_model_from_env()

        points_text = ""
        for i, pt in enumerate(points, 1):
            points_text += f"\n{i}. {pt['knowledge_point']}\n   Description: {pt['description']}\n"

        system_text = _p(
            prompts, "loose_filter_system",
            "You are a research screening expert. Filter out clearly unsuitable knowledge points using loose criteria. "
            "Output JSON: {\"filtered_points\": [{\"knowledge_point\": str, \"description\": str}]}"
        )
        user_tpl = _p(
            prompts, "loose_filter_user_template",
            "Please filter the following knowledge points using loose criteria, only retaining those suitable for research:\n\n"
            "{points_text}\n\nPlease output the filtered knowledge points in JSON format."
        )
        user_text = user_tpl.format(points_text=points_text)

        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw = response.content if hasattr(response, "content") else str(response)

        result = _parse_json(raw)
        if isinstance(result, dict):
            candidates = result.get("filtered_points", [])
            if candidates:
                filtered = [
                    pt for pt in candidates
                    if isinstance(pt, dict) and pt.get("knowledge_point")
                ]
            # If LLM filtered everything out, fall back to original
            if not filtered:
                logger.warning("loose_filter_node: all filtered out, keeping originals")
                filtered = points

        logger.debug("loose_filter_node: %d → %d points", len(points), len(filtered))

    except Exception as exc:
        logger.warning("loose_filter_node failed, using all points: %s", exc)

    return {
        "filtered_points": filtered,
        "streaming_events": [
            {"type": "status", "stage": "filtering",
             "message": "loose_filtering", "count": len(filtered)}
        ],
    }


# ---------------------------------------------------------------------------
# Node: explore  (Stage 3 — generate research ideas per knowledge point)
# ---------------------------------------------------------------------------


async def explore_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Explore research ideas for each filtered knowledge point.

    Returns:
        {"explored_ideas": list[{knowledge_point, description, ideas}], "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    filtered: list[dict] = state.get("filtered_points", []) or []
    language: str = state.get("language", "en")

    explored: list[dict] = []

    prompts = _load_prompts("idea_generation", language)
    system_text = _p(
        prompts, "explore_ideas_system",
        "You are a research idea generation expert. Generate at least 5 feasible research ideas based on the given "
        "knowledge point. Output JSON: {\"research_ideas\": [str, ...]}"
    )
    user_tpl = _p(
        prompts, "explore_ideas_user_template",
        "Based on the following knowledge point, generate at least 5 feasible research ideas:\n\n"
        "Knowledge Point: {knowledge_point}\nDescription: {description}\n\nPlease output research ideas in JSON format."
    )

    for kp in filtered:
        ideas: list[str] = []
        try:
            llm = get_chat_model_from_env()
            user_text = user_tpl.format(
                knowledge_point=kp.get("knowledge_point", ""),
                description=kp.get("description", ""),
            )
            response = await llm.ainvoke([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            result = _parse_json(raw)
            if isinstance(result, dict):
                ideas = [str(x) for x in result.get("research_ideas", []) if x]
            ideas = ideas[:10]  # cap at 10
            logger.debug("explore_node: %d ideas for '%s'", len(ideas), kp.get("knowledge_point", ""))
        except Exception as exc:
            logger.warning("explore_node failed for '%s': %s", kp.get("knowledge_point", ""), exc)

        explored.append({
            "knowledge_point": kp.get("knowledge_point", ""),
            "description": kp.get("description", ""),
            "ideas": ideas,
        })

    return {
        "explored_ideas": explored,
        "streaming_events": [
            {"type": "status", "stage": "exploring",
             "message": "exploring_ideas", "count": len(explored)}
        ],
    }


# ---------------------------------------------------------------------------
# Node: strict_filter  (Stage 4 — keep only high-quality idea candidates)
# ---------------------------------------------------------------------------


async def strict_filter_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Strict filter: keep only high-quality idea candidates.

    Returns:
        {"strict_filtered_ideas": list, "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    explored: list[dict] = state.get("explored_ideas", []) or []
    language: str = state.get("language", "en")

    strict: list[dict] = []

    prompts = _load_prompts("idea_generation", language)
    system_text = _p(
        prompts, "strict_filter_system",
        "You are a strict research review expert. Evaluate research ideas with strict criteria. "
        "Must eliminate at least 2 and retain at least 1. "
        "Output JSON: {\"kept_ideas\": [str], \"rejected_ideas\": [str], \"reasons\": {}}"
    )
    user_tpl = _p(
        prompts, "strict_filter_user_template",
        "Based on the following knowledge point and research ideas, evaluate using strict criteria:\n\n"
        "Knowledge Point: {knowledge_point}\nDescription: {description}\n\nResearch Ideas:\n{ideas_text}\n\n"
        "Output JSON. Remember: must eliminate at least 2, retain at least 1."
    )

    for idea_group in explored:
        ideas: list[str] = idea_group.get("ideas", [])
        kept: list[str] = ideas  # default: keep all

        if len(ideas) > 1:
            try:
                llm = get_chat_model_from_env()
                ideas_text = "".join(f"{i}. {idea}\n" for i, idea in enumerate(ideas, 1))
                user_text = user_tpl.format(
                    knowledge_point=idea_group.get("knowledge_point", ""),
                    description=idea_group.get("description", ""),
                    ideas_text=ideas_text,
                )
                response = await llm.ainvoke([
                    SystemMessage(content=system_text),
                    HumanMessage(content=user_text),
                ])
                raw = response.content if hasattr(response, "content") else str(response)
                result = _parse_json(raw)
                if isinstance(result, dict):
                    kept_raw = result.get("kept_ideas", [])
                    rejected_raw = result.get("rejected_ideas", [])
                    kept = [str(x) for x in kept_raw if x]
                    rejected = [str(x) for x in rejected_raw if x]

                    # Enforce: at least 1 kept
                    if not kept:
                        kept = [ideas[0]]
                        rejected = ideas[1:]
                    # Enforce: at least 2 rejected (if original had ≥ 3)
                    elif len(rejected) < 2 and len(ideas) >= 3 and len(kept) > 1:
                        rejected.extend(kept[1:])
                        kept = [kept[0]]

                logger.debug(
                    "strict_filter_node: %d → %d ideas for '%s'",
                    len(ideas), len(kept), idea_group.get("knowledge_point", "")
                )
            except Exception as exc:
                logger.warning(
                    "strict_filter_node failed for '%s': %s",
                    idea_group.get("knowledge_point", ""), exc
                )
                kept = ideas[:1] if ideas else []

        strict.append({
            "knowledge_point": idea_group.get("knowledge_point", ""),
            "description": idea_group.get("description", ""),
            "ideas": kept,
        })

    return {
        "strict_filtered_ideas": strict,
        "streaming_events": [
            {"type": "status", "stage": "strict_filtering",
             "message": "strict_filtering", "count": len(strict)}
        ],
    }


# ---------------------------------------------------------------------------
# Node: statement  (Stage 5 — generate final research statements)
# ---------------------------------------------------------------------------


async def statement_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Generate final markdown research statements for each idea group.

    Returns:
        {"idea_results": list, "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    strict: list[dict] = state.get("strict_filtered_ideas", []) or []
    language: str = state.get("language", "en")
    run_id: str = state.get("run_id", "") or str(uuid.uuid4())

    results: list[dict] = []

    prompts = _load_prompts("idea_generation", language)
    system_text = _p(
        prompts, "generate_statement_system",
        "You are a research statement generation expert. Generate a high-quality markdown-formatted statement "
        "for the given knowledge point and research ideas."
    )
    user_tpl = _p(
        prompts, "generate_statement_user_template",
        "Please generate a markdown-formatted statement for the following knowledge point and research ideas:\n\n"
        "Knowledge Point: {knowledge_point}\nDescription: {description}\n\nRetained Research Ideas:\n{ideas_text}\n\n"
        "Please generate a markdown-formatted statement."
    )

    for idx, idea_group in enumerate(strict):
        ideas: list[str] = idea_group.get("ideas", [])
        statement = ""
        try:
            llm = get_chat_model_from_env()
            ideas_text = "".join(f"{i}. {idea}\n" for i, idea in enumerate(ideas, 1))
            user_text = user_tpl.format(
                knowledge_point=idea_group.get("knowledge_point", ""),
                description=idea_group.get("description", ""),
                ideas_text=ideas_text,
            )
            response = await llm.ainvoke([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            # Strip markdown code block wrapper if present
            block = re.search(r"```(?:markdown)?\s*\n?([\s\S]*?)```", raw, re.DOTALL)
            statement = block.group(1).strip() if block else raw.strip()
            logger.debug("statement_node: '%s' (%d chars)",
                         idea_group.get("knowledge_point", ""), len(statement))
        except Exception as exc:
            logger.warning("statement_node failed for '%s': %s",
                           idea_group.get("knowledge_point", ""), exc)
            statement = (
                f"## {idea_group.get('knowledge_point', 'Research Topic')}\n\n" +
                "\n".join(f"- {idea}" for idea in ideas)
            )

        results.append({
            "id": f"{run_id}_{idx}",
            "knowledge_point": idea_group.get("knowledge_point", ""),
            "description": idea_group.get("description", ""),
            "research_ideas": ideas,
            "statement": statement,
            "expanded": False,
        })

    return {
        "idea_results": results,
        "streaming_events": [
            {"type": "status", "stage": "statement",
             "message": "generating_statements", "count": len(results)}
        ],
    }


__all__ = [
    "extract_node",
    "loose_filter_node",
    "explore_node",
    "strict_filter_node",
    "statement_node",
]
