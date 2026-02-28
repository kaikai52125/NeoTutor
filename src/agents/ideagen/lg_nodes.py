# -*- coding: utf-8 -*-
"""
IdeaGen Module — LangGraph Node Implementations
================================================

Nodes (in pipeline order):
  extract_node        — MaterialOrganizerAgent: extract knowledge points
  loose_filter_node   — IdeaGenerationWorkflow.loose_filter()
  explore_node        — IdeaGenerationWorkflow.explore_ideas() per point
  strict_filter_node  — IdeaGenerationWorkflow.strict_filter()
  statement_node      — IdeaGenerationWorkflow.generate_statement() per point
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from .lg_state import IdeaGenState

logger = logging.getLogger(__name__)


def _get_workflow(cfg, language: str = "en"):
    from src.agents.ideagen.idea_generation_workflow import IdeaGenerationWorkflow
    return IdeaGenerationWorkflow(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        api_version=getattr(cfg, "api_version", None),
        language=language,
    )


async def extract_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Stage 1 — Extract knowledge points from notebook records.

    Returns:
        {"knowledge_points": list, "streaming_events": list}
    """
    records: list[dict] = state.get("notebook_records", []) or []
    user_thoughts: str = state.get("user_thoughts", "") or ""
    language: str = state.get("language", "en")

    points: list[dict] = []
    try:
        from src.agents.ideagen.material_organizer_agent import MaterialOrganizerAgent
        from src.services.llm.config import get_llm_config

        cfg = get_llm_config()
        agent = MaterialOrganizerAgent(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            api_version=getattr(cfg, "api_version", None),
            language=language,
        )
        result = await agent.process(
            records=records,
            user_thoughts=user_thoughts,
        )
        points = result if isinstance(result, list) else result.get("knowledge_points", [])
        logger.debug("extract_node: %d knowledge points extracted", len(points))
    except Exception as exc:
        logger.warning("extract_node failed: %s", exc)

    return {
        "knowledge_points": points,
        "streaming_events": [
            {"type": "status", "stage": "extracting",
             "message": "extracting_knowledge_points", "count": len(points)}
        ],
    }


async def loose_filter_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Stage 2 — Loose filter: remove obviously unsuitable knowledge points.

    Returns:
        {"filtered_points": list, "streaming_events": list}
    """
    points: list[dict] = state.get("knowledge_points", []) or []
    language: str = state.get("language", "en")

    filtered: list[dict] = points  # default: pass all through
    try:
        from src.services.llm.config import get_llm_config
        cfg = get_llm_config()
        workflow = _get_workflow(cfg, language)
        filtered = await workflow.loose_filter(knowledge_points=points)
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


async def explore_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Stage 3 — Explore research ideas for each filtered knowledge point.

    Returns:
        {"explored_ideas": list[{knowledge_point, ideas}], "streaming_events": list}
    """
    filtered: list[dict] = state.get("filtered_points", []) or []
    language: str = state.get("language", "en")

    explored: list[dict] = []
    try:
        from src.services.llm.config import get_llm_config
        cfg = get_llm_config()
        workflow = _get_workflow(cfg, language)

        for kp in filtered:
            ideas = await workflow.explore_ideas(knowledge_point=kp)
            explored.append({
                "knowledge_point": kp.get("knowledge_point", ""),
                "description": kp.get("description", ""),
                "ideas": ideas if isinstance(ideas, list) else [],
            })
        logger.debug("explore_node: %d points explored", len(explored))
    except Exception as exc:
        logger.warning("explore_node failed: %s", exc)
        explored = [
            {"knowledge_point": kp.get("knowledge_point", ""), "description": kp.get("description", ""), "ideas": []}
            for kp in filtered
        ]

    return {
        "explored_ideas": explored,
        "streaming_events": [
            {"type": "status", "stage": "exploring",
             "message": "exploring_ideas", "count": len(explored)}
        ],
    }


async def strict_filter_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Stage 4 — Strict filter: keep only high-quality idea candidates.

    Returns:
        {"strict_filtered_ideas": list, "streaming_events": list}
    """
    explored: list[dict] = state.get("explored_ideas", []) or []
    language: str = state.get("language", "en")

    strict: list[dict] = explored
    try:
        from src.services.llm.config import get_llm_config
        cfg = get_llm_config()
        workflow = _get_workflow(cfg, language)

        strict = []
        for idea_group in explored:
            kp_dict = {
                "knowledge_point": idea_group.get("knowledge_point", ""),
                "description": idea_group.get("description", ""),
            }
            ideas = idea_group.get("ideas", [])
            filtered_ideas = await workflow.strict_filter(
                knowledge_point=kp_dict,
                research_ideas=ideas,
            )
            strict.append({
                "knowledge_point": idea_group.get("knowledge_point", ""),
                "description": idea_group.get("description", ""),
                "ideas": filtered_ideas,
            })
        logger.debug("strict_filter_node: %d groups processed", len(strict))
    except Exception as exc:
        logger.warning("strict_filter_node failed, using explored ideas: %s", exc)
        strict = explored

    return {
        "strict_filtered_ideas": strict,
        "streaming_events": [
            {"type": "status", "stage": "strict_filtering",
             "message": "strict_filtering", "count": len(strict)}
        ],
    }


async def statement_node(state: IdeaGenState) -> dict[str, Any]:
    """
    Stage 5 — Generate final research statements for each idea.

    Returns:
        {"idea_results": list, "streaming_events": list}
    """
    strict: list[dict] = state.get("strict_filtered_ideas", []) or []
    language: str = state.get("language", "en")
    run_id: str = state.get("run_id", "") or str(uuid.uuid4())

    results: list[dict] = []
    try:
        from src.services.llm.config import get_llm_config
        cfg = get_llm_config()
        workflow = _get_workflow(cfg, language)

        for idx, idea_group in enumerate(strict):
            kp_dict = {
                "knowledge_point": idea_group.get("knowledge_point", ""),
                "description": idea_group.get("description", ""),
            }
            ideas = idea_group.get("ideas", [])
            statement = await workflow.generate_statement(
                knowledge_point=kp_dict,
                research_ideas=ideas,
            )
            results.append({
                "id": f"{run_id}_{idx}",
                "knowledge_point": idea_group.get("knowledge_point", ""),
                "description": idea_group.get("description", ""),
                "research_ideas": ideas,
                "statement": statement if isinstance(statement, str) else str(statement),
                "expanded": False,
            })
        logger.debug("statement_node: %d statements generated", len(results))
    except Exception as exc:
        logger.warning("statement_node failed: %s", exc)

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
