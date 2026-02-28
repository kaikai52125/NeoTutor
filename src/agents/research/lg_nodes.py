# -*- coding: utf-8 -*-
"""
Research Module — LangGraph Node Implementations
==================================================

Each function is a LangGraph node: receives ResearchState, returns a dict of
state updates.

Nodes (in pipeline order):
  rephrase_node         — Phase 1: optimise topic via RephraseAgent
  decompose_node        — Phase 1: break topic into TopicBlocks via DecomposeAgent
  research_block_node   — Phase 2: research a single TopicBlock (runs in parallel
                          via LangGraph Send API, one invocation per pending block)
  report_node           — Phase 3: generate final report via ReportingAgent

Pipeline sharing:
  A single ResearchPipeline instance is created once per research_id and cached
  in _PIPELINE_REGISTRY so all nodes within the same graph run share it.
  This avoids re-initialising six agents per node and lets research_block_node
  share the queue and citation_manager with report_node.
  The registry entry is removed after report_node completes.

Streaming events emitted by each node are typed to match the frontend
useResearchReducer event types exactly:
  planning_started, rephrase_completed, decompose_completed, planning_completed,
  researching_started, block_started, block_completed, block_failed,
  reporting_started, reporting_completed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from .lg_state import ResearchState, TopicBlockDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline registry — keyed by research_id, shared across nodes in one run
# ---------------------------------------------------------------------------

_PIPELINE_REGISTRY: dict[str, Any] = {}
_REGISTRY_LOCK = asyncio.Lock()


async def _get_or_create_pipeline(state: ResearchState):
    """
    Return the ResearchPipeline for this research_id, creating it on first call.
    All subsequent calls with the same research_id reuse the same instance.
    """
    from pathlib import Path
    from src.agents.research.research_pipeline import ResearchPipeline
    from src.services.config import load_config_with_main
    from src.services.llm import get_llm_config

    research_id: str = state.get("research_id") or str(uuid.uuid4())

    async with _REGISTRY_LOCK:
        if research_id in _PIPELINE_REGISTRY:
            return _PIPELINE_REGISTRY[research_id]

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        config = load_config_with_main("research_config.yaml", project_root)

        language = state.get("language", "en")
        kb_name = state.get("kb_name", "") or ""
        enabled_tools: list[str] = state.get("enabled_tools", ["RAG"]) or ["RAG"]
        # Normalise to upper-case for comparison
        tools_upper = {t.upper() for t in enabled_tools}

        config.setdefault("system", {})["language"] = language
        if kb_name:
            config.setdefault("rag", {})["kb_name"] = kb_name

        # Map frontend enabled_tools list → ResearchAgent config flags.
        # "RAG" covers rag_hybrid, rag_naive, query_item.
        # "Web" → enable_web_search, "Paper" → enable_paper_search.
        has_rag = "RAG" in tools_upper
        researching_cfg = config.setdefault("researching", {})
        researching_cfg["enable_rag_hybrid"] = has_rag
        researching_cfg["enable_rag_naive"] = has_rag
        researching_cfg["enable_web_search"] = "WEB" in tools_upper
        researching_cfg["enable_paper_search"] = "PAPER" in tools_upper
        logger.debug(
            "pipeline_registry: enabled_tools=%s → rag=%s, web=%s, paper=%s",
            enabled_tools, has_rag, "WEB" in tools_upper, "PAPER" in tools_upper,
        )

        llm_cfg = get_llm_config()

        pipeline = ResearchPipeline(
            config=config,
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            api_version=getattr(llm_cfg, "api_version", None),
            research_id=research_id,
            kb_name=kb_name or None,
        )
        # Enable frontend mode (suppress stdin prompts)
        pipeline.progress_callback = lambda e: None

        _PIPELINE_REGISTRY[research_id] = pipeline
        logger.debug("pipeline_registry: created pipeline for research_id=%s", research_id)
        return pipeline


def _release_pipeline(research_id: str) -> None:
    """Remove a pipeline from the registry after the run completes."""
    _PIPELINE_REGISTRY.pop(research_id, None)
    logger.debug("pipeline_registry: released pipeline for research_id=%s", research_id)


# ---------------------------------------------------------------------------
# Node: rephrase
# ---------------------------------------------------------------------------


async def rephrase_node(state: ResearchState) -> dict[str, Any]:
    """
    Phase 1a — Optimise/rephrase the raw research topic.

    Emits frontend events:
      planning_started  — signals start of planning phase
      rephrase_completed — carries the optimized topic string
    """
    topic: str = state.get("topic", "") or ""
    skip: bool = state.get("skip_rephrase", False)

    # Signal planning phase start
    planning_started_event = {
        "type": "planning_started",
        "user_topic": topic,
    }

    if skip:
        logger.debug("rephrase_node: skip_rephrase=True, passing topic through")
        return {
            "optimized_topic": topic,
            "streaming_events": [
                planning_started_event,
                {"type": "rephrase_completed", "optimized_topic": topic},
            ],
        }

    optimized = topic
    try:
        pipeline = await _get_or_create_pipeline(state)
        rephrase_result = await pipeline.agents["rephrase"].process(topic, iteration=0)
        optimized = rephrase_result.get("topic", topic) or topic
        logger.debug("rephrase_node: '%s' -> '%s'", topic[:60], optimized[:60])
    except Exception as exc:
        logger.warning("rephrase_node failed, using original topic: %s", exc)

    return {
        "optimized_topic": optimized,
        "streaming_events": [
            planning_started_event,
            {"type": "rephrase_completed", "optimized_topic": optimized},
        ],
    }


# ---------------------------------------------------------------------------
# Node: decompose
# ---------------------------------------------------------------------------


async def decompose_node(state: ResearchState) -> dict[str, Any]:
    """
    Phase 1b — Decompose the optimised topic into TopicBlock subtopics.
    Also seeds the shared pipeline queue so research_block_node can use it.

    Emits frontend events:
      decompose_completed  — with generated_subtopics count
      planning_completed   — with total_blocks count
    """
    topic: str = state.get("optimized_topic") or state.get("topic", "")
    initial_subtopics: int = state.get("initial_subtopics", 5)

    blocks: list[TopicBlockDict] = []
    try:
        pipeline = await _get_or_create_pipeline(state)
        config = pipeline.config

        pipeline.agents["decompose"].set_citation_manager(pipeline.citation_manager)

        decompose_config = config.get("planning", {}).get("decompose", {})
        mode = decompose_config.get("mode", "manual")

        decompose_result = await pipeline.agents["decompose"].process(
            topic=topic,
            num_subtopics=initial_subtopics,
            mode=mode,
        )

        # Seed the shared queue (used by research_block_node and report_node)
        pipeline.queue.blocks.clear()
        for sub_topic_data in decompose_result.get("sub_topics", []):
            title = (sub_topic_data.get("title") or "").strip()
            overview = sub_topic_data.get("overview", "")
            if not title:
                continue
            block = pipeline.queue.add_block(sub_topic=title, overview=overview)
            blocks.append(
                TopicBlockDict(
                    block_id=block.block_id,
                    sub_topic=block.sub_topic,
                    overview=block.overview,
                    status="pending",
                    tool_traces=[],
                    iteration_count=0,
                    metadata={},
                )
            )

        # Set manager primary topic
        pipeline.agents["manager"].set_primary_topic(topic)
        logger.debug("decompose_node: %d blocks for '%s'", len(blocks), topic[:60])

    except Exception as exc:
        logger.warning("decompose_node failed: %s", exc)
        # Fallback: single block for the whole topic
        blocks = [
            TopicBlockDict(
                block_id=str(uuid.uuid4()),
                sub_topic=topic,
                overview="",
                status="pending",
                tool_traces=[],
                iteration_count=0,
                metadata={"fallback": True},
            )
        ]

    return {
        "topic_blocks": blocks,
        "streaming_events": [
            {
                "type": "decompose_completed",
                "generated_subtopics": len(blocks),
            },
            {
                "type": "planning_completed",
                "total_blocks": len(blocks),
            },
            {
                # Also signal researching phase start (parallel mode)
                "type": "researching_started",
                "total_blocks": len(blocks),
                "execution_mode": "parallel",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Node: research_block
# ---------------------------------------------------------------------------


async def research_block_node(state: ResearchState) -> dict[str, Any]:
    """
    Phase 2 — Research a single TopicBlock using ResearchAgent + NoteAgent.

    Invoked in parallel for each pending block via LangGraph's Send API.
    Uses the shared pipeline instance (same queue, citation_manager, agents).

    Emits frontend events:
      block_started    — when research begins for this block
      block_completed  — on success, with tools_used list
      block_failed     — on error, with error message
    """
    current_block: TopicBlockDict | None = state.get("_current_block")  # type: ignore[assignment]
    if not current_block:
        all_blocks: list[TopicBlockDict] = state.get("topic_blocks", [])
        pending = [b for b in all_blocks if b["status"] == "pending"]
        if not pending:
            return {"streaming_events": []}
        current_block = pending[0]

    block_id: str = current_block["block_id"]
    sub_topic: str = current_block["sub_topic"]
    max_iterations: int = state.get("max_iterations", 4)

    logger.debug("research_block_node: block_id=%s sub_topic='%s'", block_id, sub_topic[:60])

    updated_block: TopicBlockDict = {**current_block, "status": "researching"}
    new_citations: list[dict[str, str]] = []
    detail_events: list[dict[str, Any]] = []

    # Emit block_started before the long-running research
    start_event: dict[str, Any] = {
        "type": "block_started",
        "block_id": block_id,
        "sub_topic": sub_topic,
    }

    # Collect per-iteration detail events for frontend ActiveTaskDetail display
    _FORWARDED_EVENT_TYPES = {
        "checking_sufficiency",
        "knowledge_sufficient",
        "generating_query",
        "tool_calling",
        "tool_completed",
        "processing_notes",
        "new_topic_added",
    }

    def _progress_callback(event_type: str, **data: Any) -> None:
        if event_type in _FORWARDED_EVENT_TYPES:
            detail_events.append({"type": event_type, "block_id": block_id, **data})

    try:
        pipeline = await _get_or_create_pipeline(state)

        # Find the corresponding TopicBlock in the shared queue
        block_obj = pipeline.queue.get_block_by_id(block_id)
        if block_obj is None:
            # Block not in queue (fallback path) — create a transient one
            from src.agents.research.data_structures import TopicBlock
            block_obj = TopicBlock(
                block_id=block_id,
                sub_topic=sub_topic,
                overview=current_block.get("overview", ""),
            )
            pipeline.queue.blocks.append(block_obj)

        # Override max_iterations from state
        pipeline.agents["research"].max_iterations = max_iterations
        pipeline.config.setdefault("researching", {})["max_iterations"] = max_iterations

        # Run the research loop for this block, collecting detail events via callback
        result = await pipeline.agents["research"].process(
            topic_block=block_obj,
            call_tool_callback=pipeline._call_tool,
            note_agent=pipeline.agents["note"],
            citation_manager=pipeline.citation_manager,
            queue=pipeline.queue,
            manager_agent=pipeline.agents["manager"],
            config=pipeline.config,
            progress_callback=_progress_callback,
        )

        # Mark the block as completed in the shared queue
        pipeline.queue.mark_completed(block_id)

        # Build ToolTraceDict list from result
        tool_traces_raw: list[dict] = result.get("tool_traces", [])
        tool_traces = [
            {
                "tool_id": t.get("tool_id", str(uuid.uuid4())),
                "citation_id": t.get("citation_id", ""),
                "tool_type": t.get("tool_type", ""),
                "query": t.get("query", ""),
                "raw_answer": t.get("raw_answer", ""),
                "summary": t.get("summary", ""),
            }
            for t in tool_traces_raw
        ]

        # Collect citations
        for t in tool_traces_raw:
            if t.get("citation_id"):
                new_citations.append(
                    {
                        "citation_id": t.get("citation_id", ""),
                        "tool_type": t.get("tool_type", ""),
                        "query": t.get("query", ""),
                        "content": t.get("raw_answer", "")[:500],
                    }
                )

        # Collect unique tools used for frontend display
        tools_used = list({t.get("tool_type", "") for t in tool_traces_raw if t.get("tool_type")})

        # Get note summary from the block's last trace
        note_summary = ""
        if block_obj.tool_traces:
            note_summary = getattr(block_obj.tool_traces[-1], "summary", "")

        updated_block = {
            **updated_block,
            "status": "completed",
            "tool_traces": tool_traces,
            "iteration_count": result.get("iterations", len(tool_traces)),
            "metadata": {
                **current_block.get("metadata", {}),
                "note_summary": note_summary,
            },
        }

        end_event: dict[str, Any] = {
            "type": "block_completed",
            "block_id": block_id,
            "sub_topic": sub_topic,
            "tools_used": tools_used,
            "iteration_count": updated_block["iteration_count"],
        }

    except Exception as exc:
        logger.warning("research_block_node failed for block_id=%s: %s", block_id, exc)
        updated_block = {
            **updated_block,
            "status": "failed",
            "metadata": {**current_block.get("metadata", {}), "error": str(exc)},
        }
        end_event = {
            "type": "block_failed",
            "block_id": block_id,
            "sub_topic": sub_topic,
            "error": str(exc),
        }

    return {
        "topic_blocks": [updated_block],
        "citations": new_citations,
        "streaming_events": [start_event, *detail_events, end_event],
    }


# ---------------------------------------------------------------------------
# Node: report
# ---------------------------------------------------------------------------


async def report_node(state: ResearchState) -> dict[str, Any]:
    """
    Phase 3 — Generate the final research report from completed TopicBlocks.
    Uses the shared pipeline's queue (populated by research_block_node).

    Emits frontend events:
      reporting_started    — signals report generation has begun
      reporting_completed  — carries word_count, sections, citations counts
    """
    import json
    from pathlib import Path

    topic: str = state.get("optimized_topic") or state.get("topic", "")
    research_id: str = state.get("research_id", str(uuid.uuid4()))
    citations: list[dict] = state.get("citations", [])

    final_report = ""
    report_path = ""
    try:
        pipeline = await _get_or_create_pipeline(state)

        pipeline.agents["reporting"].set_citation_manager(pipeline.citation_manager)

        result = await pipeline.agents["reporting"].process(
            queue=pipeline.queue,
            topic=topic,
        )
        final_report = result.get("report", "") or ""

        # Save report to disk
        output_dir = Path("data/user/research") / research_id
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path_obj = output_dir / "report.md"
        report_path_obj.write_text(final_report, encoding="utf-8")
        report_path = str(report_path_obj)

        # Save citations
        (output_dir / "citations.json").write_text(
            json.dumps(citations, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.debug("report_node: %d chars written to %s", len(final_report), report_path)

    except Exception as exc:
        logger.warning("report_node failed: %s", exc)
        final_report = f"[Report generation failed: {exc}]"
    finally:
        # Release the shared pipeline — this run is complete
        _release_pipeline(research_id)

    # Estimate word/section/citation counts for frontend
    word_count = len(final_report.split())
    section_count = final_report.count("\n## ")
    citation_count = len(citations)

    return {
        "final_report": final_report,
        "report_path": report_path,
        "streaming_events": [
            {"type": "reporting_started"},
            {
                "type": "reporting_completed",
                "word_count": word_count,
                "sections": section_count,
                "citations": citation_count,
                "report": final_report,
            },
        ],
    }


__all__ = [
    "rephrase_node",
    "decompose_node",
    "research_block_node",
    "report_node",
]
