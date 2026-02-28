# -*- coding: utf-8 -*-
"""
Solve Module — LangGraph Node Implementations
=============================================

Each function wraps the corresponding original Agent class, translating
between SolveState dicts and the Memory objects those agents expect.

Analysis Loop nodes:
  investigate_node      — InvestigateAgent: queries + knowledge gathering
  note_node             — NoteAgent: summarises new knowledge items

Solve Loop nodes:
  plan_node             — ManagerAgent: creates SolveChainStep list
  exec_tools_node       — ToolAgent: executes all pending tool calls in current step
  solve_step_node       — SolveAgent: decides next tool calls or marks step done
  response_node         — ResponseAgent: writes step_response
  finalize_node         — Compiles final_answer from all step responses
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from .lg_state import (
    CitationRecordDict,
    KnowledgeItemDict,
    SolveState,
    SolveStepDict,
    ToolCallRecordDict,
)

logger = logging.getLogger(__name__)


def _get_llm_cfg():
    from src.services.llm.config import get_llm_config
    return get_llm_config()


# ---------------------------------------------------------------------------
# Helpers: build Memory objects from state dicts (read-only adapters)
# ---------------------------------------------------------------------------


def _build_investigate_memory(state: SolveState):
    """Reconstruct InvestigateMemory from state dicts (no disk I/O)."""
    from src.agents.solve.memory.investigate_memory import InvestigateMemory, KnowledgeItem

    mem = InvestigateMemory(user_question=state["question"])
    for item in state.get("knowledge_chain", []):
        ki = KnowledgeItem(
            cite_id=item["cite_id"],
            tool_type=item["tool_type"],
            query=item["query"],
            raw_result=item["raw_result"],
            summary=item["summary"],
        )
        mem.knowledge_chain.append(ki)
    return mem


def _build_citation_memory(state: SolveState):
    """Reconstruct CitationMemory from state dicts."""
    from src.agents.solve.memory.citation_memory import CitationMemory, CitationItem

    # Pass output_dir so that agents can call cm.save() without error
    cm = CitationMemory(output_dir=state.get("output_dir") or None)
    for c in state.get("citations", []):
        # CitationItem is a dataclass — construct directly to avoid mis-mapping add_citation() args
        item = CitationItem(
            cite_id=c["cite_id"],
            tool_type=c.get("tool_type", "rag_naive"),
            query=c.get("query", ""),
            raw_result=c.get("raw_result", ""),
            source=c.get("source", ""),
            content=c.get("content", ""),
        )
        cm.citations.append(item)
        # Restore tool counter so new cite_ids don't collide
        prefix = cm._get_tool_prefix(item.tool_type)
        num = cm._extract_counter_from_cite_id(item.cite_id, prefix)
        if num is not None:
            cm.tool_counters[prefix] = max(cm.tool_counters.get(prefix, 0), num)

    # Also pre-register any cite_ids referenced in solve_steps tool_calls that are
    # not yet in citations. SolveAgent registers them when it creates tool calls,
    # but since we rebuild cm from state each time, we must ensure they exist so
    # that ToolAgent.update_citation() does not raise "cite_id not found".
    existing_ids = {item.cite_id for item in cm.citations}
    for step in state.get("solve_steps", []):
        for tc in step.get("tool_calls", []):
            cid = tc.get("cite_id", "")
            if cid and cid not in existing_ids:
                item = CitationItem(
                    cite_id=cid,
                    tool_type=tc.get("tool_type", "rag_naive"),
                    query=tc.get("query", ""),
                    raw_result=tc.get("raw_answer", ""),
                    content=tc.get("summary", ""),
                )
                cm.citations.append(item)
                existing_ids.add(cid)
                prefix = cm._get_tool_prefix(item.tool_type)
                num = cm._extract_counter_from_cite_id(cid, prefix)
                if num is not None:
                    cm.tool_counters[prefix] = max(cm.tool_counters.get(prefix, 0), num)
    return cm


def _build_solve_memory(state: SolveState):
    """Reconstruct SolveMemory from state dicts."""
    from src.agents.solve.memory.solve_memory import SolveMemory, SolveChainStep, ToolCallRecord

    sm = SolveMemory(output_dir=state.get("output_dir") or None)
    for step_d in state.get("solve_steps", []):
        step = SolveChainStep(
            step_id=step_d["step_id"],
            step_target=step_d["step_target"],
            available_cite=step_d.get("available_cite", []),
        )
        step.status = step_d.get("status", "undone")
        step.step_response = step_d.get("step_response", "")
        step.used_citations = step_d.get("used_citations", [])
        for tc_d in step_d.get("tool_calls", []):
            tc = ToolCallRecord(
                call_id=tc_d["call_id"],
                tool_type=tc_d["tool_type"],
                query=tc_d["query"],
            )
            tc.cite_id = tc_d.get("cite_id", "")
            tc.raw_answer = tc_d.get("raw_answer", "")
            tc.summary = tc_d.get("summary", "")
            tc.status = tc_d.get("status", "pending")
            step.tool_calls.append(tc)
        sm.solve_chains.append(step)
    return sm


def _dump_knowledge_chain(mem) -> list[KnowledgeItemDict]:
    return [
        KnowledgeItemDict(
            cite_id=ki.cite_id,
            tool_type=ki.tool_type,
            query=ki.query,
            raw_result=ki.raw_result,
            summary=ki.summary,
        )
        for ki in mem.knowledge_chain
    ]


def _dump_citations(cm) -> list[CitationRecordDict]:
    # cm.citations is list[CitationItem] (dataclass instances)
    result = []
    for item in cm.citations:
        result.append(CitationRecordDict(
            cite_id=item.cite_id,
            tool_type=item.tool_type,
            source=item.source,
            query=item.query,
            content=item.content,
        ))
    return result


def _dump_solve_steps(sm) -> list[SolveStepDict]:
    steps = []
    for s in sm.solve_chains:
        tcs = [
            ToolCallRecordDict(
                call_id=tc.call_id,
                tool_type=tc.tool_type,
                query=tc.query,
                cite_id=tc.cite_id or "",
                raw_answer=tc.raw_answer or "",
                summary=tc.summary or "",
                status=tc.status,
            )
            for tc in s.tool_calls
        ]
        steps.append(SolveStepDict(
            step_id=s.step_id,
            step_target=s.step_target,
            available_cite=s.available_cite or [],
            tool_calls=tcs,
            step_response=s.step_response or "",
            status=s.status,
            used_citations=s.used_citations or [],
        ))
    return steps


# ---------------------------------------------------------------------------
# Analysis Loop — Node: investigate
# ---------------------------------------------------------------------------


async def investigate_node(state: SolveState) -> dict[str, Any]:
    """
    Analysis Loop iteration: run InvestigateAgent to gather knowledge.

    Returns updates to knowledge_chain, analysis_iteration, analysis_should_stop,
    new_knowledge_ids, citations.
    """
    from src.agents.solve.analysis_loop.investigate_agent import InvestigateAgent

    cfg = _get_llm_cfg()
    mem = _build_investigate_memory(state)
    cm = _build_citation_memory(state)

    agent = InvestigateAgent(
        config={},
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        api_version=getattr(cfg, "api_version", None),
    )
    try:
        result = await agent.process(
            question=state["question"],
            memory=mem,
            citation_memory=cm,
            kb_name=state.get("kb_name", ""),
            output_dir=state.get("output_dir") or None,  # pass None not "" to avoid save() path error
            verbose=False,
        )
        should_stop: bool = result.get("should_stop", False)
        new_ids: list[str] = result.get("knowledge_item_ids", [])
        logger.debug(
            "investigate_node iter=%d, new_ids=%d, should_stop=%s",
            state.get("analysis_iteration", 0) + 1, len(new_ids), should_stop,
        )
    except Exception as exc:
        logger.warning("investigate_node failed: %s", exc)
        should_stop = True
        new_ids = []

    return {
        "knowledge_chain": _dump_knowledge_chain(mem),
        "citations": _dump_citations(cm),
        "analysis_iteration": state.get("analysis_iteration", 0) + 1,
        "analysis_should_stop": should_stop,
        "new_knowledge_ids": new_ids,
        "streaming_events": [
            {
                "type": "progress",
                "stage": "investigate",
                "progress": {
                    "round": state.get("analysis_iteration", 0) + 1,
                    "new_items": len(new_ids),
                },
            },
            {
                "type": "log",
                "level": "INFO",
                "content": f"[Investigate] Round {state.get('analysis_iteration', 0) + 1} — {len(new_ids)} knowledge item(s) gathered",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Analysis Loop — Node: note
# ---------------------------------------------------------------------------


async def note_node(state: SolveState) -> dict[str, Any]:
    """
    Analysis Loop: summarise newly discovered knowledge items via NoteAgent.
    """
    from src.agents.solve.analysis_loop.note_agent import NoteAgent

    new_ids: list[str] = state.get("new_knowledge_ids", [])
    if not new_ids:
        return {"streaming_events": []}

    cfg = _get_llm_cfg()
    mem = _build_investigate_memory(state)
    cm = _build_citation_memory(state)

    agent = NoteAgent(
        config={},
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        api_version=getattr(cfg, "api_version", None),
    )
    try:
        await agent.process(
            question=state["question"],
            memory=mem,
            citation_memory=cm,
            new_knowledge_ids=new_ids,
            output_dir=state.get("output_dir") or None,
            verbose=False,
        )
        logger.debug("note_node: summarised %d items", len(new_ids))
    except Exception as exc:
        logger.warning("note_node failed: %s", exc)

    return {
        "knowledge_chain": _dump_knowledge_chain(mem),
        "streaming_events": [
            {
                "type": "progress",
                "stage": "investigate",
                "progress": {"summarised": len(new_ids)},
            },
            {
                "type": "log",
                "level": "INFO",
                "content": f"[Note] Summarised {len(new_ids)} knowledge item(s)",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: plan
# ---------------------------------------------------------------------------


async def plan_node(state: SolveState) -> dict[str, Any]:
    """
    Solve Loop: ManagerAgent creates the SolveChainStep list.
    """
    from src.agents.solve.solve_loop.manager_agent import ManagerAgent

    cfg = _get_llm_cfg()
    mem = _build_investigate_memory(state)
    cm = _build_citation_memory(state)
    sm = _build_solve_memory(state)  # empty at this point

    agent = ManagerAgent(
        config={},
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        api_version=getattr(cfg, "api_version", None),
    )
    try:
        await agent.process(
            question=state["question"],
            investigate_memory=mem,
            solve_memory=sm,
            # ManagerAgent.process() only accepts: question, investigate_memory, solve_memory, verbose
            verbose=False,
        )
        logger.debug("plan_node: %d solve steps created", len(sm.solve_chains))
    except Exception as exc:
        logger.warning("plan_node failed: %s", exc)

    return {
        "solve_steps": _dump_solve_steps(sm),
        "current_step_index": 0,
        "solve_iteration": 0,
        "finish_requested": False,
        "streaming_events": [
            {
                "type": "progress",
                "stage": "solve",
                "progress": {"steps": len(sm.solve_chains)},
            },
            {
                "type": "log",
                "level": "INFO",
                "content": f"[Plan] Created {len(sm.solve_chains)} solve step(s)",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: exec_tools
# ---------------------------------------------------------------------------


async def exec_tools_node(state: SolveState) -> dict[str, Any]:
    """
    Solve Loop: ToolAgent executes all pending tool calls in the current step.
    """
    from src.agents.solve.solve_loop.tool_agent import ToolAgent

    idx: int = state.get("current_step_index", 0)
    steps = state.get("solve_steps", [])
    if idx >= len(steps):
        return {"streaming_events": []}

    cfg = _get_llm_cfg()
    sm = _build_solve_memory(state)
    cm = _build_citation_memory(state)

    agent = ToolAgent(
        config={},
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        api_version=getattr(cfg, "api_version", None),
    )
    try:
        # ToolAgent.process() accepts: step, solve_memory, citation_memory, kb_name, output_dir
        current_step = sm.solve_chains[idx] if idx < len(sm.solve_chains) else None
        if current_step is None:
            return {"streaming_events": []}
        await agent.process(
            step=current_step,
            solve_memory=sm,
            citation_memory=cm,
            kb_name=state.get("kb_name", ""),
            output_dir=state.get("output_dir") or None,
            verbose=False,
        )
        logger.debug("exec_tools_node: step %d tools executed", idx)
    except Exception as exc:
        logger.warning("exec_tools_node failed at step %d: %s", idx, exc)

    return {
        "solve_steps": _dump_solve_steps(sm),
        "citations": _dump_citations(cm),
        "streaming_events": [
            {
                "type": "progress",
                "stage": "solve",
                "progress": {"step_index": idx},
            },
            {
                "type": "log",
                "level": "INFO",
                "content": f"[Tool] Step {idx + 1} — tools executed",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: solve_step
# ---------------------------------------------------------------------------


async def solve_step_node(state: SolveState) -> dict[str, Any]:
    """
    Solve Loop: SolveAgent decides whether to request more tools or finish step.
    """
    from src.agents.solve.solve_loop.solve_agent import SolveAgent

    idx: int = state.get("current_step_index", 0)

    cfg = _get_llm_cfg()
    mem = _build_investigate_memory(state)
    sm = _build_solve_memory(state)
    cm = _build_citation_memory(state)

    agent = SolveAgent(
        config={},
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        api_version=getattr(cfg, "api_version", None),
    )
    finish_requested = False
    try:
        # SolveAgent.process() accepts: question, current_step, solve_memory, investigate_memory, citation_memory, kb_name, output_dir
        current_step = sm.solve_chains[idx] if idx < len(sm.solve_chains) else None
        if current_step is None:
            finish_requested = True
        else:
            result = await agent.process(
                question=state["question"],
                current_step=current_step,
                solve_memory=sm,
                investigate_memory=mem,
                citation_memory=cm,
                kb_name=state.get("kb_name", ""),
                output_dir=state.get("output_dir") or None,
                verbose=False,
            )
            finish_requested = result.get("finish_requested", False)
            logger.debug("solve_step_node: step %d finish_requested=%s", idx, finish_requested)
    except Exception as exc:
        logger.warning("solve_step_node failed at step %d: %s", idx, exc)
        finish_requested = True  # force progress on error

    return {
        "solve_steps": _dump_solve_steps(sm),
        "finish_requested": finish_requested,
        "solve_iteration": state.get("solve_iteration", 0) + 1,
        "streaming_events": [
            {
                "type": "progress",
                "stage": "solve",
                "progress": {
                    "step_index": idx,
                    "step_id": sm.solve_chains[idx].step_id if idx < len(sm.solve_chains) else "",
                    "step_target": sm.solve_chains[idx].step_target if idx < len(sm.solve_chains) else "",
                },
            },
            {
                "type": "log",
                "level": "INFO",
                "content": f"[Solve] Step {idx + 1}: {sm.solve_chains[idx].step_target[:80] if idx < len(sm.solve_chains) else ''}",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: response
# ---------------------------------------------------------------------------


async def response_node(state: SolveState) -> dict[str, Any]:
    """
    Solve Loop: ResponseAgent writes the step_response for the current step.
    """
    from src.agents.solve.solve_loop.response_agent import ResponseAgent

    idx: int = state.get("current_step_index", 0)

    cfg = _get_llm_cfg()
    mem = _build_investigate_memory(state)
    sm = _build_solve_memory(state)
    cm = _build_citation_memory(state)

    agent = ResponseAgent(
        config={},
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        api_version=getattr(cfg, "api_version", None),
    )
    try:
        # ResponseAgent.process() accepts: question, step, solve_memory, investigate_memory, citation_memory, output_dir
        current_step = sm.solve_chains[idx] if idx < len(sm.solve_chains) else None
        if current_step is None:
            return {"solve_steps": _dump_solve_steps(sm), "streaming_events": []}
        await agent.process(
            question=state["question"],
            step=current_step,
            solve_memory=sm,
            investigate_memory=mem,
            citation_memory=cm,
            output_dir=state.get("output_dir") or None,
            verbose=False,
        )
        logger.debug("response_node: step %d response written", idx)
    except Exception as exc:
        logger.warning("response_node failed at step %d: %s", idx, exc)

    return {
        "solve_steps": _dump_solve_steps(sm),
        "streaming_events": [
            {
                "type": "progress",
                "stage": "response",
                "progress": {
                    "step_index": idx,
                    "step_id": sm.solve_chains[idx].step_id if idx < len(sm.solve_chains) else "",
                    "step_target": sm.solve_chains[idx].step_target if idx < len(sm.solve_chains) else "",
                },
            },
            {
                "type": "log",
                "level": "INFO",
                "content": f"[Response] Step {idx + 1} response written",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: advance_step
# ---------------------------------------------------------------------------


async def advance_step_node(state: SolveState) -> dict[str, Any]:
    """Move to the next solve step and reset per-step iteration counter."""
    next_idx = state.get("current_step_index", 0) + 1
    logger.debug("advance_step_node: %d → %d", next_idx - 1, next_idx)
    return {
        "current_step_index": next_idx,
        "solve_iteration": 0,
        "finish_requested": False,
        "streaming_events": [
            {
                "type": "log",
                "level": "INFO",
                "content": f"[Advance] Moving to step {next_idx + 1}",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: finalize
# ---------------------------------------------------------------------------


async def finalize_node(state: SolveState) -> dict[str, Any]:
    """
    Compile the final answer from all completed step responses,
    optionally run PrecisionAnswerAgent for conciseness.
    """
    steps = state.get("solve_steps", [])
    parts = [
        s["step_response"]
        for s in steps
        if s.get("step_response", "").strip()
    ]
    raw_answer = "\n\n".join(parts) if parts else ""

    # Optional precision pass
    final_answer = raw_answer
    try:
        from src.agents.solve.solve_loop.precision_answer_agent import PrecisionAnswerAgent
        cfg = _get_llm_cfg()
        agent = PrecisionAnswerAgent(
            config={},
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            api_version=getattr(cfg, "api_version", None),
        )
        result = await agent.process(
            question=state["question"],
            # PrecisionAnswerAgent.process() accepts: question, detailed_answer, verbose
            detailed_answer=raw_answer,
            verbose=False,
        )
        candidate = result.get("final_answer", "") or ""
        # Reject obviously truncated/failed precision results (< 50 chars when raw is long)
        if candidate and (len(raw_answer) < 200 or len(candidate) > len(raw_answer) * 0.1):
            final_answer = candidate
        else:
            final_answer = raw_answer
    except Exception as exc:
        logger.warning("finalize_node precision pass failed: %s", exc)

    logger.debug("finalize_node: final_answer=%d chars", len(final_answer))
    return {
        "final_answer": final_answer,
        "streaming_events": [
            {
                "type": "progress",
                "stage": "response",
                "progress": {"step_index": len(steps) - 1},
            },
            {
                "type": "log",
                "level": "INFO",
                "content": f"[Finalize] Answer ready ({len(final_answer)} chars)",
            },
        ],
    }


__all__ = [
    "investigate_node",
    "note_node",
    "plan_node",
    "exec_tools_node",
    "solve_step_node",
    "response_node",
    "advance_step_node",
    "finalize_node",
]
