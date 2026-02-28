# -*- coding: utf-8 -*-
"""
Solve Module — LangGraph Node Implementations (refactored: pure LangChain, no BaseAgent)

All agent logic is inlined directly into node functions.
Memory objects (InvestigateMemory, CitationMemory, SolveMemory) are retained as-is
because they carry the core business logic (serialisation, cite_id management, etc.).

Analysis Loop nodes:
  investigate_node  — gather knowledge items via RAG / web search / query_item
  note_node         — summarise each knowledge item

Solve Loop nodes:
  plan_node         — ManagerAgent: create SolveChainStep list
  exec_tools_node   — ToolAgent: execute pending tool calls in current step
  solve_step_node   — SolveAgent: decide next tool calls or mark step done
  response_node     — ResponseAgent: write step_response
  advance_step_node — move to next step
  finalize_node     — compile final answer (optional PrecisionAnswer pass)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from .lg_state import (
    CitationRecordDict,
    KnowledgeItemDict,
    SolveState,
    SolveStepDict,
    ToolCallRecordDict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _llm():
    """Return the singleton LangChain chat model."""
    from src.services.llm.langchain_factory import get_chat_model_from_env
    return get_chat_model_from_env()


def _load_prompts(agent_name: str, language: str = "en") -> dict[str, str]:
    """Load prompt YAML for solve/<agent_name>."""
    from src.services.prompt import get_prompt_manager
    try:
        return get_prompt_manager().load_prompts("solve", agent_name, language) or {}
    except Exception:
        return {}


def _p(prompts: dict[str, str], key: str) -> str:
    return prompts.get(key, "") or ""


def _parse_json(text: str) -> dict | list | None:
    """Extract first JSON object/array from text (handles markdown fences)."""
    if not text:
        return None
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
    clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, clean)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


async def _call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Call LLM and return string response."""
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = _llm()
    if json_mode:
        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception:
            pass
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)



# ---------------------------------------------------------------------------
# Helpers: build Memory objects from state dicts
# ---------------------------------------------------------------------------


def _build_investigate_memory(state: SolveState):
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
    from src.agents.solve.memory.citation_memory import CitationMemory, CitationItem

    cm = CitationMemory(output_dir=state.get("output_dir") or None)
    for c in state.get("citations", []):
        item = CitationItem(
            cite_id=c["cite_id"],
            tool_type=c.get("tool_type", "rag_naive"),
            query=c.get("query", ""),
            raw_result=c.get("raw_result", ""),
            source=c.get("source", ""),
            content=c.get("content", ""),
        )
        cm.citations.append(item)
        prefix = cm._get_tool_prefix(item.tool_type)
        num = cm._extract_counter_from_cite_id(item.cite_id, prefix)
        if num is not None:
            cm.tool_counters[prefix] = max(cm.tool_counters.get(prefix, 0), num)

    # Pre-register cite_ids from solve_steps tool_calls not yet in citations
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
            cite_id=ki.cite_id, tool_type=ki.tool_type, query=ki.query,
            raw_result=ki.raw_result, summary=ki.summary,
        )
        for ki in mem.knowledge_chain
    ]


def _dump_citations(cm) -> list[CitationRecordDict]:
    return [
        CitationRecordDict(
            cite_id=item.cite_id, tool_type=item.tool_type, source=item.source,
            query=item.query, content=item.content,
        )
        for item in cm.citations
    ]


def _dump_solve_steps(sm) -> list[SolveStepDict]:
    steps = []
    for s in sm.solve_chains:
        tcs = [
            ToolCallRecordDict(
                call_id=tc.call_id, tool_type=tc.tool_type, query=tc.query,
                cite_id=tc.cite_id or "", raw_answer=tc.raw_answer or "",
                summary=tc.summary or "", status=tc.status,
            )
            for tc in s.tool_calls
        ]
        steps.append(SolveStepDict(
            step_id=s.step_id, step_target=s.step_target,
            available_cite=s.available_cite or [], tool_calls=tcs,
            step_response=s.step_response or "", status=s.status,
            used_citations=s.used_citations or [],
        ))
    return steps


# ---------------------------------------------------------------------------
# Analysis Loop — Node: investigate
# ---------------------------------------------------------------------------


async def investigate_node(state: SolveState) -> dict[str, Any]:
    """Gather knowledge via RAG / web search / query_item (InvestigateAgent logic inlined)."""
    from src.tools import query_numbered_item, rag_search, web_search

    language = state.get("language", "en")
    prompts = _load_prompts("investigate_agent", language)
    system_prompt = _p(prompts, "system")
    user_template = _p(prompts, "user_template")
    if not system_prompt or not user_template:
        logger.warning("investigate_node: missing prompts, stopping")
        return {
            "analysis_iteration": state.get("analysis_iteration", 0) + 1,
            "analysis_should_stop": True,
            "new_knowledge_ids": [],
            "streaming_events": [{"type": "log", "level": "WARNING",
                                   "content": "[Investigate] Missing prompts — stopping"}],
        }

    mem = _build_investigate_memory(state)
    cm = _build_citation_memory(state)

    knowledge_chain_summary = (
        "\n".join(
            f"- {ki.cite_id} ({ki.tool_type}): {ki.summary or ki.raw_result[:200]}"
            for ki in mem.knowledge_chain
        )
        if mem.knowledge_chain else "(none)"
    )
    remaining_questions = (
        mem.reflections.remaining_questions
        if mem.reflections and mem.reflections.remaining_questions else []
    )
    context = {
        "question": state["question"],
        "num_knowledge": len(mem.knowledge_chain),
        "knowledge_chain_full": [
            {"cite_id": ki.cite_id, "tool_type": ki.tool_type, "query": ki.query,
             "raw_result": ki.raw_result, "summary": ki.summary}
            for ki in mem.knowledge_chain
        ],
        "knowledge_chain_summary": knowledge_chain_summary,
        "reflections_summary": (
            "\n".join(f"- {q}" for q in remaining_questions) if remaining_questions
            else "(no remaining questions)"
        ),
        "remaining_questions": remaining_questions,
        "action_queue": "(no action history)",
    }
    user_prompt = user_template.format(**context)

    try:
        raw = await _call_llm(system_prompt, user_prompt, json_mode=True)
        parsed = _parse_json(raw)
    except Exception as exc:
        logger.warning("investigate_node LLM failed: %s", exc)
        parsed = None

    if not parsed or not isinstance(parsed, dict):
        return {
            "knowledge_chain": _dump_knowledge_chain(mem),
            "citations": _dump_citations(cm),
            "analysis_iteration": state.get("analysis_iteration", 0) + 1,
            "analysis_should_stop": True,
            "new_knowledge_ids": [],
            "streaming_events": [
                {"type": "log", "level": "WARNING",
                 "content": "[Investigate] LLM parse failed — stopping"},
            ],
        }

    tool_plans = parsed.get("plan", [])
    if isinstance(tool_plans, dict):
        tool_plans = [tool_plans]
    elif not isinstance(tool_plans, list):
        tool_plans = []

    should_stop = (not tool_plans) or any(p.get("tool") == "none" for p in tool_plans)
    if should_stop:
        new_round = state.get("analysis_iteration", 0) + 1
        return {
            "knowledge_chain": _dump_knowledge_chain(mem),
            "citations": _dump_citations(cm),
            "analysis_iteration": new_round,
            "analysis_should_stop": True,
            "new_knowledge_ids": [],
            "streaming_events": [
                {"type": "progress", "stage": "investigate",
                 "progress": {"round": new_round, "new_items": 0}},
                {"type": "log", "level": "INFO",
                 "content": f"[Investigate] Round {new_round} — sufficient, stopping"},
            ],
        }

    kb_name = state.get("kb_name", "")
    output_dir = state.get("output_dir") or None
    knowledge_ids: list[str] = []

    for plan in tool_plans[:1]:  # max_actions_per_round = 1
        tool_type = plan.get("tool", "")
        query = plan.get("query", "")
        identifier = plan.get("identifier")
        if not tool_type or tool_type == "none":
            continue
        try:
            if tool_type == "rag_naive":
                result = await rag_search(query=query, kb_name=kb_name, mode="naive")
                raw_result = result.get("answer", "")
            elif tool_type == "rag_hybrid":
                result = await rag_search(query=query, kb_name=kb_name, mode="hybrid")
                raw_result = result.get("answer", "")
            elif tool_type == "web_search":
                result = web_search(query=query, output_dir=output_dir or "./cache", verbose=False)
                raw_result = json.dumps(result, ensure_ascii=False)
            elif tool_type == "query_item":
                ident = identifier or query
                if not ident or not isinstance(ident, str) or not ident.strip():
                    continue
                result = await query_numbered_item(identifier=ident, kb_name=kb_name)
                raw_result = result.get("content", result.get("answer", ""))
            else:
                logger.warning("investigate_node: unknown tool %s", tool_type)
                continue

            cite_id = cm.add_citation(
                tool_type=tool_type, query=query, raw_result=raw_result,
                stage="analysis", metadata={"identifier": identifier},
            )
            from src.agents.solve.memory.investigate_memory import KnowledgeItem
            mem.add_knowledge(KnowledgeItem(
                cite_id=cite_id, tool_type=tool_type, query=query,
                raw_result=raw_result, summary="",
            ))
            knowledge_ids.append(cite_id)
        except Exception as exc:
            logger.warning("investigate_node tool %s failed: %s", tool_type, exc)

    new_round = state.get("analysis_iteration", 0) + 1
    return {
        "knowledge_chain": _dump_knowledge_chain(mem),
        "citations": _dump_citations(cm),
        "analysis_iteration": new_round,
        "analysis_should_stop": len(knowledge_ids) == 0,
        "new_knowledge_ids": knowledge_ids,
        "streaming_events": [
            {"type": "progress", "stage": "investigate",
             "progress": {"round": new_round, "new_items": len(knowledge_ids)}},
            {"type": "log", "level": "INFO",
             "content": f"[Investigate] Round {new_round} — {len(knowledge_ids)} knowledge item(s) gathered"},
        ],
    }


# ---------------------------------------------------------------------------
# Analysis Loop — Node: note
# ---------------------------------------------------------------------------


async def note_node(state: SolveState) -> dict[str, Any]:
    """Summarise newly discovered knowledge items (NoteAgent logic inlined)."""
    new_ids: list[str] = state.get("new_knowledge_ids", [])
    if not new_ids:
        return {"streaming_events": []}

    language = state.get("language", "en")
    prompts = _load_prompts("note_agent", language)
    system_prompt = _p(prompts, "system")
    user_template = _p(prompts, "user_template")
    if not system_prompt or not user_template:
        logger.warning("note_node: missing prompts, skipping")
        return {"streaming_events": [
            {"type": "log", "level": "WARNING", "content": "[Note] Missing prompts — skipped"},
        ]}

    mem = _build_investigate_memory(state)
    cm = _build_citation_memory(state)
    processed = 0

    for cite_id in new_ids:
        ki = next((k for k in mem.knowledge_chain if k.cite_id == cite_id), None)
        if not ki:
            continue
        context = {
            "question": state["question"],
            "tool_type": ki.tool_type,
            "query": ki.query,
            "raw_result": ki.raw_result,
        }
        try:
            user_prompt = user_template.format(**context)
            raw = await _call_llm(system_prompt, user_prompt, json_mode=True)
            parsed = _parse_json(raw)
            if not parsed or not isinstance(parsed, dict):
                continue
            summary = parsed.get("summary", "")
            if not summary:
                continue
            mem.update_knowledge_summary(cite_id=cite_id, summary=summary)
            citations_list = parsed.get("citations", [])
            sources = ", ".join(c.get("source", "") for c in citations_list if c.get("source"))
            try:
                cm.update_citation(
                    cite_id=cite_id, content=summary, source=sources or None,
                    metadata={"extracted_sources": citations_list} if citations_list else None,
                    stage="analysis",
                )
            except ValueError:
                pass
            processed += 1
        except Exception as exc:
            logger.warning("note_node failed for %s: %s", cite_id, exc)

    return {
        "knowledge_chain": _dump_knowledge_chain(mem),
        "streaming_events": [
            {"type": "progress", "stage": "investigate", "progress": {"summarised": processed}},
            {"type": "log", "level": "INFO",
             "content": f"[Note] Summarised {processed} knowledge item(s)"},
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: plan
# ---------------------------------------------------------------------------


async def plan_node(state: SolveState) -> dict[str, Any]:
    """Create SolveChainStep list (ManagerAgent logic inlined)."""
    from src.agents.solve.memory.solve_memory import SolveChainStep, SolveMemory

    language = state.get("language", "en")
    prompts = _load_prompts("manager_agent", language)
    system_prompt = _p(prompts, "system")
    user_template = _p(prompts, "user_template")
    if not system_prompt or not user_template:
        logger.warning("plan_node: missing prompts")
        return {
            "solve_steps": [],
            "current_step_index": 0,
            "solve_iteration": 0,
            "finish_requested": False,
            "streaming_events": [
                {"type": "log", "level": "WARNING", "content": "[Plan] Missing prompts — empty plan"},
            ],
        }

    mem = _build_investigate_memory(state)

    knowledge_info = [
        {"cite_id": ki.cite_id, "tool_type": ki.tool_type, "query": ki.query, "summary": ki.summary}
        for ki in mem.knowledge_chain if ki.summary
    ]
    knowledge_text = "".join(
        f"\n{info['cite_id']} [{info['tool_type']}]\n  Query: {info['query']}\n  Summary: {info['summary']}\n"
        for info in knowledge_info
    ) or "(No research information)"
    remaining_questions = (
        mem.reflections.remaining_questions
        if mem.reflections and getattr(mem.reflections, "remaining_questions", None) else []
    )
    context = {
        "question": state["question"],
        "knowledge_info": knowledge_info,
        "knowledge_text": knowledge_text,
        "knowledge_chain_summary": knowledge_text,
        "reflections_summary": (
            "\n".join(f"- {q}" for q in remaining_questions) if remaining_questions
            else "(No remaining questions)"
        ),
    }
    user_prompt = user_template.format(**context)

    try:
        raw = await _call_llm(system_prompt, user_prompt, json_mode=True)
        parsed = _parse_json(raw)
    except Exception as exc:
        logger.warning("plan_node LLM failed: %s", exc)
        parsed = None

    steps: list[SolveChainStep] = []
    knowledge_ids = {ki.cite_id for ki in mem.knowledge_chain}

    if isinstance(parsed, dict):
        steps_data = parsed.get("steps", [])
        if isinstance(steps_data, list):
            for idx, step_data in enumerate(steps_data, 1):
                if not isinstance(step_data, dict):
                    continue
                step_id = step_data.get("step_id", "").strip() or f"S{idx}"
                if not step_id.upper().startswith("S"):
                    step_id = f"S{step_id}"
                role = step_data.get("role", "").strip()
                target = step_data.get("target", "").strip()
                if not target:
                    continue
                if "：" in target or ":" in target:
                    step_target = target
                elif role:
                    step_target = f"{role}：{target}"
                else:
                    step_target = target
                cite_ids_raw = step_data.get("cite_ids", [])
                if isinstance(cite_ids_raw, str):
                    cite_ids_raw = [cite_ids_raw] if cite_ids_raw and cite_ids_raw != "none" else []
                elif not isinstance(cite_ids_raw, list):
                    cite_ids_raw = []
                filtered_cites = []
                for cite in cite_ids_raw:
                    if not cite or cite == "none":
                        continue
                    cleaned = str(cite).strip()
                    if not cleaned.startswith("["):
                        cleaned = f"[{cleaned.strip('[] ')}]"
                    if cleaned in knowledge_ids:
                        filtered_cites.append(cleaned)
                steps.append(SolveChainStep(
                    step_id=step_id, step_target=step_target,
                    available_cite=list(dict.fromkeys(filtered_cites)),
                ))

    if not steps:
        steps = [SolveChainStep(
            step_id="S1", step_target=state["question"],
            available_cite=list(knowledge_ids),
        )]

    sm = SolveMemory(output_dir=state.get("output_dir") or None)
    sm.create_chains(steps)

    return {
        "solve_steps": _dump_solve_steps(sm),
        "current_step_index": 0,
        "solve_iteration": 0,
        "finish_requested": False,
        "streaming_events": [
            {"type": "progress", "stage": "solve", "progress": {"steps": len(steps)}},
            {"type": "log", "level": "INFO",
             "content": f"[Plan] Created {len(steps)} solve step(s)"},
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: exec_tools
# ---------------------------------------------------------------------------

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp"}


def _snapshot_images(path: Path) -> set:
    if not path.exists():
        return set()
    return {f.resolve() for f in path.rglob("*") if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES}


def _new_image_rel_paths(artifacts_path: Path, before: set, output_dir: str | None) -> list[str]:
    after = _snapshot_images(artifacts_path)
    new_files = sorted(after - before)
    output_base = Path(output_dir).resolve() if output_dir else None
    rel_paths = []
    for fp in new_files:
        try:
            rel = str(fp.relative_to(output_base)).replace("\\", "/") if output_base else None
        except ValueError:
            rel = None
        if rel is None:
            try:
                rel = str(fp.relative_to(artifacts_path.parent)).replace("\\", "/")
            except ValueError:
                rel = str(Path("artifacts") / fp.name)
        rel_paths.append(rel)
    return rel_paths


def _infer_sources(text: str) -> tuple[str, list[str]]:
    if not text:
        return "", []
    matches = re.findall(r"(https?://[^\s\)\]]+)", text)
    cleaned = []
    for item in matches:
        n = item.strip().strip(".,;:()[]{}")
        if n and n not in cleaned:
            cleaned.append(n)
    return (", ".join(cleaned), cleaned)


async def _generate_code(intent: str) -> str:
    system = (
        "You are a Python code generator.\n"
        "Generate ONLY executable Python code.\n"
        "Do NOT include explanations.\n"
        "Do NOT include markdown fences.\n"
        "The code must be self-contained and runnable."
    )
    user = f"Task:\n{intent}\n\nRules:\n- Output only Python code\n- No ``` fences\n- No natural language"
    code = await _call_llm(system, user)
    if "```" in code:
        raise ValueError("LLM returned markdown code fences")
    if len(code) > 8000:
        raise ValueError("Generated code too large")
    return code.strip()


async def _summarize_tool_result(tool_type: str, query: str, raw_answer: str, language: str) -> str:
    prompts = _load_prompts("tool_agent", language)
    system_prompt = _p(prompts, "system")
    user_template = _p(prompts, "user_template")
    if not system_prompt or not user_template:
        return raw_answer[:500]
    user_prompt = user_template.format(tool_type=tool_type, query=query, raw_answer=raw_answer[:2000])
    try:
        return (await _call_llm(system_prompt, user_prompt)).strip()
    except Exception as exc:
        logger.warning("_summarize_tool_result failed: %s", exc)
        return raw_answer[:500]


async def exec_tools_node(state: SolveState) -> dict[str, Any]:
    """Execute pending tool calls in current step (ToolAgent logic inlined)."""
    from src.services.config import load_config_with_main as _lcm
    from src.tools.code_executor import run_code
    from src.tools.rag_tool import rag_search
    from src.tools.web_search import web_search

    _cfg = _lcm("main.yaml", Path(__file__).resolve().parents[3])
    code_execution_timeout: int = _cfg.get("solve", {}).get("code_execution_timeout", 60)

    idx: int = state.get("current_step_index", 0)
    if idx >= len(state.get("solve_steps", [])):
        return {"streaming_events": []}

    language = state.get("language", "en")
    sm = _build_solve_memory(state)
    cm = _build_citation_memory(state)

    if idx >= len(sm.solve_chains):
        return {"streaming_events": []}

    step = sm.solve_chains[idx]
    pending = [
        call for call in step.tool_calls
        if call.tool_type not in {"none", "finish"} and call.status in {"pending", "running"}
    ]
    if not pending:
        return {
            "solve_steps": _dump_solve_steps(sm),
            "citations": _dump_citations(cm),
            "streaming_events": [
                {"type": "progress", "stage": "solve", "progress": {"step_index": idx}},
                {"type": "log", "level": "INFO",
                 "content": f"[Tool] Step {idx + 1} — no pending tools"},
            ],
        }

    kb_name = state.get("kb_name", "")
    output_dir = state.get("output_dir") or None
    base_dir = Path(output_dir).resolve() if output_dir else Path().resolve()
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for record in pending:
        try:
            if record.tool_type == "rag_naive":
                result = await rag_search(query=record.query, kb_name=kb_name, mode="naive")
                raw_answer = result.get("answer", "")
                source, auto_sources = _infer_sources(raw_answer)
                metadata: dict[str, Any] = {"source": source, "auto_sources": auto_sources, "mode": "naive"}

            elif record.tool_type == "rag_hybrid":
                result = await rag_search(query=record.query, kb_name=kb_name, mode="hybrid")
                raw_answer = result.get("answer", "")
                source, auto_sources = _infer_sources(raw_answer)
                metadata = {"source": source, "auto_sources": auto_sources, "mode": "hybrid"}

            elif record.tool_type == "web_search":
                result = web_search(query=record.query, output_dir=output_dir, verbose=False)
                raw_answer = result.get("answer") or result.get("summary") or ""
                used_ids = re.findall(r"\[(\d+)\]", raw_answer)
                raw_citations = result.get("citations") or []
                search_map = {r.get("url"): r for r in (result.get("search_results") or []) if r.get("url")}
                selected = []
                for cid in used_ids:
                    for raw in raw_citations:
                        ref_id = str(raw.get("id")) if raw.get("id") is not None else ""
                        ref_token = (raw.get("reference") or "").strip().strip("[]")
                        if cid == ref_id or cid == ref_token:
                            matched = dict(raw)
                            url = matched.get("url")
                            if url and url in search_map:
                                fb = search_map[url]
                                matched.setdefault("title", fb.get("title", ""))
                                matched.setdefault("snippet", fb.get("snippet", ""))
                            selected.append({
                                "id": matched.get("id") or (int(cid) if cid.isdigit() else cid),
                                "reference": matched.get("reference") or f"[{cid}]",
                                "url": matched.get("url"),
                                "title": matched.get("title", ""),
                            })
                            break
                metadata = {"result_file": result.get("result_file"), "citations": selected}

            elif record.tool_type == "code_execution":
                before_snap = _snapshot_images(artifacts_dir)
                if not record.query or not record.query.strip():
                    raw_answer = "【⚠️ Code execution failed】\nError: No valid code input received."
                    metadata = {"exit_code": 1, "artifacts": [], "artifact_paths": [],
                                "artifact_rel_paths": [], "work_dir": str(artifacts_dir),
                                "execution_failed": True}
                else:
                    code = await _generate_code(record.query)
                    exec_result = await run_code(
                        language="python", code=code, timeout=code_execution_timeout, assets_dir=str(artifacts_dir)
                    )
                    stdout = exec_result.get("stdout", "")
                    stderr = exec_result.get("stderr", "")
                    exit_code = exec_result.get("exit_code", 0)
                    artifacts = exec_result.get("artifacts", [])
                    artifact_paths = exec_result.get("artifact_paths", [])
                    lines = [
                        "【Code Execution Result】",
                        f"Exit code: {exit_code}",
                        f"Elapsed time: {exec_result.get('elapsed_ms', 0):.2f} ms",
                        f"Working directory: {artifacts_dir}",
                        "", "stdout:", stdout or "(empty)", "", "stderr:", stderr or "(empty)",
                    ]
                    if artifacts:
                        lines += ["", "Artifacts:"]
                        for i, a in enumerate(artifacts):
                            abs_p = artifact_paths[i] if i < len(artifact_paths) else str(artifacts_dir / a)
                            lines.append(f"- {abs_p}")
                    raw_answer = "\n".join(lines)
                    is_failed = exit_code != 0
                    if is_failed:
                        prefix = "【⚠️ Code execution failed】\n"
                        if "FileNotFoundError" in stderr and "artifacts/" in stderr:
                            prefix += "Path error: use 'xxx.png' not 'artifacts/xxx.png'.\n\n"
                        raw_answer = prefix + raw_answer
                    new_image_paths = _new_image_rel_paths(artifacts_dir, before_snap, output_dir)
                    metadata = {
                        "exit_code": exit_code, "artifacts": artifacts,
                        "artifact_paths": artifact_paths, "artifact_rel_paths": new_image_paths,
                        "work_dir": str(artifacts_dir), "execution_failed": is_failed,
                    }
            else:
                raise ValueError(f"Unknown tool type: {record.tool_type}")

            summary = await _summarize_tool_result(record.tool_type, record.query, raw_answer, language)
            status = "failed" if metadata.get("execution_failed") else "success"
            sm.update_tool_call_result(
                step_id=step.step_id, call_id=record.call_id,
                raw_answer=raw_answer, summary=summary, status=status, metadata=metadata,
            )
            try:
                cm.update_citation(
                    cite_id=record.cite_id, raw_result=raw_answer,
                    content=summary, metadata=metadata, step_id=step.step_id,
                )
            except ValueError:
                pass

        except Exception as exc:
            error_msg = str(exc)
            logger.warning("exec_tools_node step=%s tool=%s: %s", step.step_id, record.tool_type, exc)
            sm.update_tool_call_result(
                step_id=step.step_id, call_id=record.call_id,
                raw_answer=error_msg, summary=error_msg[:200], status="failed",
                metadata={"error": True},
            )
            try:
                cm.update_citation(
                    cite_id=record.cite_id, raw_result=error_msg,
                    content=error_msg[:200], metadata={"error": True}, step_id=step.step_id,
                )
            except ValueError:
                pass

    return {
        "solve_steps": _dump_solve_steps(sm),
        "citations": _dump_citations(cm),
        "streaming_events": [
            {"type": "progress", "stage": "solve", "progress": {"step_index": idx}},
            {"type": "log", "level": "INFO",
             "content": f"[Tool] Step {idx + 1} — tools executed"},
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: solve_step
# ---------------------------------------------------------------------------


async def solve_step_node(state: SolveState) -> dict[str, Any]:
    """Decide next tool calls or mark step done (SolveAgent logic inlined)."""
    SUPPORTED_TOOL_TYPES = {"none", "rag_naive", "rag_hybrid", "web_search", "code_execution", "finish"}

    idx: int = state.get("current_step_index", 0)
    language = state.get("language", "en")
    prompts = _load_prompts("solve_agent", language)
    system_prompt = _p(prompts, "system")
    user_template = _p(prompts, "user_template")

    mem = _build_investigate_memory(state)
    sm = _build_solve_memory(state)
    cm = _build_citation_memory(state)
    finish_requested = False

    if idx >= len(sm.solve_chains) or not system_prompt or not user_template:
        finish_requested = True
    else:
        current_step = sm.solve_chains[idx]

        avail_lines = []
        for cite_id in current_step.available_cite:
            ki = next((k for k in mem.knowledge_chain if k.cite_id == cite_id), None)
            if not ki:
                continue
            summary = ki.summary or ki.raw_result[:300]
            avail_lines.append(
                f"{cite_id} | {ki.tool_type}\n  Query: {ki.query}\n"
                f"  Summary: {summary}\n  Raw: {ki.raw_result[:300].replace(chr(10), ' ')}"
            )
        available_cite_text = "\n".join(avail_lines) if avail_lines else "(No available knowledge chain)"

        prev_snippets = []
        for step in sm.solve_chains:
            if step.step_id == current_step.step_id:
                break
            if step.step_response:
                prev_snippets.append(f"[{step.step_id}] {step.step_target}\n{step.step_response[:300]}...")
        previous_steps = "\n\n".join(prev_snippets[-3:]) if prev_snippets else "(No completed steps yet)"

        if current_step.tool_calls:
            tool_lines = []
            for call in current_step.tool_calls:
                tool_lines.append(
                    f"{call.tool_type} | cite_id={call.cite_id or 'N/A'} | Status={call.status}\n"
                    f"Query: {call.query}\nSummary: {(call.summary or '(Pending)')[:200]}"
                )
            current_tool_history = "\n\n".join(tool_lines)
        else:
            current_tool_history = "(No tool calls have been made yet)"

        context = {
            "question": state["question"],
            "current_step_id": current_step.step_id,
            "step_target": current_step.step_target,
            "available_cite_text": available_cite_text,
            "previous_steps": previous_steps,
            "current_tool_history": current_tool_history,
        }
        user_prompt = user_template.format(**context)

        try:
            raw = await _call_llm(system_prompt, user_prompt, json_mode=True)
            parsed = _parse_json(raw)
        except Exception as exc:
            logger.warning("solve_step_node LLM failed at step %d: %s", idx, exc)
            parsed = None

        tool_plan: list[dict[str, str]] = []
        if isinstance(parsed, dict):
            for item in (parsed.get("tool_calls") or []):
                if not isinstance(item, dict):
                    continue
                tool_type = str(item.get("type", "")).strip().lower()
                query = str(item.get("intent", "")).strip()
                if tool_type and tool_type in SUPPORTED_TOOL_TYPES:
                    tool_plan.append({"type": tool_type, "query": query})

        if not tool_plan:
            finish_requested = True
        else:
            finish_requested = any(item["type"] == "finish" for item in tool_plan)
            kb_name = state.get("kb_name", "")
            existing_calls = len(current_step.tool_calls)
            for order, item in enumerate(tool_plan, start=1):
                tool_type = item["type"]
                query = item["query"].strip()
                if tool_type == "finish":
                    continue
                cite_id = None
                if tool_type != "none":
                    cite_id = cm.add_citation(
                        tool_type=tool_type, query=query, raw_result="", content="",
                        stage="solve", step_id=current_step.step_id,
                    )
                record = sm.append_tool_call(
                    step_id=current_step.step_id, tool_type=tool_type, query=query,
                    cite_id=cite_id, metadata={"plan_order": existing_calls + order, "kb_name": kb_name},
                )
                if tool_type == "none":
                    sm.update_tool_call_result(
                        step_id=current_step.step_id, call_id=record.call_id,
                        raw_answer=query, summary=query, status="none",
                    )
                    finish_requested = True
                    break

    return {
        "solve_steps": _dump_solve_steps(sm),
        "citations": _dump_citations(cm),
        "finish_requested": finish_requested,
        "solve_iteration": state.get("solve_iteration", 0) + 1,
        "streaming_events": [
            {
                "type": "progress", "stage": "solve",
                "progress": {
                    "step_index": idx,
                    "step_id": sm.solve_chains[idx].step_id if idx < len(sm.solve_chains) else "",
                    "step_target": sm.solve_chains[idx].step_target if idx < len(sm.solve_chains) else "",
                },
            },
            {
                "type": "log", "level": "INFO",
                "content": (
                    f"[Solve] Step {idx + 1}: "
                    f"{sm.solve_chains[idx].step_target[:80] if idx < len(sm.solve_chains) else ''}"
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: response
# ---------------------------------------------------------------------------


async def response_node(state: SolveState) -> dict[str, Any]:
    """Write step_response for current step (ResponseAgent logic inlined)."""
    idx: int = state.get("current_step_index", 0)
    language = state.get("language", "en")
    prompts = _load_prompts("response_agent", language)
    system_base = _p(prompts, "system")
    user_template = _p(prompts, "user_template")
    if not system_base or not user_template:
        logger.warning("response_node: missing prompts at step %d", idx)
        return {
            "streaming_events": [
                {"type": "log", "level": "WARNING",
                 "content": f"[Response] Missing prompts for step {idx + 1}"},
            ],
        }

    mem = _build_investigate_memory(state)
    sm = _build_solve_memory(state)
    cm = _build_citation_memory(state)
    output_dir = state.get("output_dir") or None

    if idx >= len(sm.solve_chains):
        return {"solve_steps": _dump_solve_steps(sm), "streaming_events": []}

    step = sm.solve_chains[idx]

    # available_cite_details
    avail_lines = []
    for cite in step.available_cite:
        ki = next((k for k in mem.knowledge_chain if k.cite_id == cite), None)
        if not ki:
            continue
        summary = ki.summary or ki.raw_result[:300]
        avail_lines.append(
            f"{cite} [{ki.tool_type}]\n  Query: {ki.query}\n  Summary: {summary}\n  Raw: {ki.raw_result[:300]}..."
        )
    available_cite_details = "\n".join(avail_lines) if avail_lines else "(No available knowledge chain)"

    # tool_materials & image_materials
    tool_lines = []
    images: list[str] = []
    seen_images: set[str] = set()

    def _add_image(path_str: str):
        norm = str(path_str).replace("\\", "/")
        if norm and norm not in seen_images:
            images.append(norm)
            seen_images.add(norm)

    for call in step.tool_calls:
        summary = call.summary or "(Summary pending)"
        raw_preview = (call.raw_answer or "")[:500]
        tool_lines.append(
            f"{call.tool_type} | cite_id={call.cite_id} | Status={call.status}\n"
            f"Query: {call.query}\nSummary: {summary}\nRaw excerpt: {raw_preview}"
        )
        m = call.metadata or {}
        for rel in (m.get("artifact_rel_paths") or []):
            _add_image(rel)
        for abs_p in (m.get("artifact_paths") or []):
            p = Path(abs_p)
            if p.suffix.lower() in _IMAGE_SUFFIXES:
                if output_dir:
                    try:
                        _add_image(str(p.relative_to(Path(output_dir))))
                    except ValueError:
                        _add_image(p.name)
                else:
                    _add_image(p.name)
        if not m.get("artifact_rel_paths") and not m.get("artifact_paths"):
            for a in (m.get("artifacts") or []):
                p = Path(a)
                if p.suffix.lower() in _IMAGE_SUFFIXES:
                    _add_image(str(Path("artifacts") / p.name))

    tool_materials = "\n\n".join(tool_lines) if tool_lines else "(No tool calls yet)"

    # citation_details
    cite_ids = list(dict.fromkeys(
        step.available_cite + [tc.cite_id for tc in step.tool_calls if tc.cite_id]
    ))
    cit_lines = []
    for cid in cite_ids:
        citation = cm.get_citation(cid)
        if not citation:
            continue
        s = citation.content or citation.raw_result[:200]
        cit_lines.append(f"- {cid} [{citation.tool_type}] Query: {citation.query}")
        if s:
            cit_lines.append(f"  Summary: {s[:300]}")
    citation_details = "\n".join(cit_lines) if cit_lines else "(No citations)"

    # previous context
    prev_parts = []
    for s in sm.solve_chains:
        if s.step_id == step.step_id:
            break
        if s.step_response and s.step_response.strip():
            prev_parts.append(s.step_response)
    accumulated_response = "\n\n".join(prev_parts) if prev_parts else ""

    image_text = "\n".join(f"- {img}" for img in images) if images else "(No image files)"

    context = {
        "question": state["question"],
        "step_id": step.step_id,
        "step_target": step.step_target,
        "available_cite_details": available_cite_details,
        "tool_materials": tool_materials,
        "citation_details": citation_details,
        "image_materials": image_text,
        "previous_context": accumulated_response or "(No previous content, this is the first step)",
    }

    # Build system prompt with optional image instruction
    try:
        system_prompt = system_base.format(step_target=step.step_target)
    except KeyError:
        system_prompt = system_base
    if images:
        image_list = "\n".join(f"  - {img}" for img in images)
        img_instr = _p(prompts, "image_instruction")
        if img_instr:
            try:
                system_prompt += img_instr.format(image_list=image_list)
            except KeyError:
                system_prompt += img_instr

    user_prompt = user_template.format(**context)

    try:
        raw = await _call_llm(system_prompt, user_prompt)
        step_response = raw.strip() if raw else ""
    except Exception as exc:
        logger.warning("response_node LLM failed at step %d: %s", idx, exc)
        step_response = ""

    # Extract used citations
    allowed = set(step.available_cite + [tc.cite_id for tc in step.tool_calls if tc.cite_id])
    used_citations = []
    for m in re.findall(r"\[([^\]\[]+)\](?!\()|【([^】\[]+)】", step_response):
        candidate = m[0] or m[1]
        if not candidate:
            continue
        cite = f"[{candidate.strip()}]"
        if cite in allowed and cite not in used_citations:
            used_citations.append(cite)

    sm.submit_step_response(step_id=step.step_id, response=step_response,
                             used_citations=used_citations)

    return {
        "solve_steps": _dump_solve_steps(sm),
        "streaming_events": [
            {
                "type": "progress", "stage": "response",
                "progress": {"step_index": idx, "step_id": step.step_id,
                             "step_target": step.step_target},
            },
            {"type": "log", "level": "INFO",
             "content": f"[Response] Step {idx + 1} response written"},
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: advance_step
# ---------------------------------------------------------------------------


async def advance_step_node(state: SolveState) -> dict[str, Any]:
    """Move to the next solve step and reset per-step counters."""
    next_idx = state.get("current_step_index", 0) + 1
    logger.debug("advance_step_node: %d → %d", next_idx - 1, next_idx)
    return {
        "current_step_index": next_idx,
        "solve_iteration": 0,
        "finish_requested": False,
        "streaming_events": [
            {"type": "log", "level": "INFO",
             "content": f"[Advance] Moving to step {next_idx + 1}"},
        ],
    }


# ---------------------------------------------------------------------------
# Solve Loop — Node: finalize
# ---------------------------------------------------------------------------


async def finalize_node(state: SolveState) -> dict[str, Any]:
    """Compile final answer with optional PrecisionAnswer pass (inlined)."""
    steps = state.get("solve_steps", [])
    parts = [s["step_response"] for s in steps if s.get("step_response", "").strip()]
    raw_answer = "\n\n".join(parts) if parts else ""

    language = state.get("language", "en")
    prompts = _load_prompts("precision_answer_agent", language)
    decision_system = _p(prompts, "decision_system")
    decision_user_template = _p(prompts, "decision_user_template")
    precision_system = _p(prompts, "precision_system")
    precision_user_template = _p(prompts, "precision_user_template")

    final_answer = raw_answer
    try:
        if decision_system and decision_user_template:
            decision_user = decision_user_template.format(question=state["question"])
            decision_raw = await _call_llm(decision_system, decision_user)
            if decision_raw.strip().upper().startswith("Y") and precision_system and precision_user_template:
                precision_user = precision_user_template.format(
                    question=state["question"], detailed_answer=raw_answer
                )
                candidate = (await _call_llm(precision_system, precision_user)).strip()
                if candidate and (len(raw_answer) < 200 or len(candidate) > len(raw_answer) * 0.1):
                    final_answer = candidate
    except Exception as exc:
        logger.warning("finalize_node precision pass failed: %s", exc)

    logger.debug("finalize_node: final_answer=%d chars", len(final_answer))
    return {
        "final_answer": final_answer,
        "streaming_events": [
            {"type": "progress", "stage": "response",
             "progress": {"step_index": len(steps) - 1}},
            {"type": "log", "level": "INFO",
             "content": f"[Finalize] Answer ready ({len(final_answer)} chars)"},
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
