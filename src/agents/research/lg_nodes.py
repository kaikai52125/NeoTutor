# -*- coding: utf-8 -*-
"""
Research Module — LangGraph Node Implementations
==================================================

Each function is a LangGraph node: receives ResearchState, returns a dict of
state updates.

Nodes (in pipeline order):
  rephrase_node         — Phase 1: optimise topic
  decompose_node        — Phase 1: break topic into TopicBlocks
  research_block_node   — Phase 2: research a single TopicBlock (runs in parallel
                          via LangGraph Send API, one invocation per pending block)
  report_node           — Phase 3: generate final report

Context sharing:
  A lightweight _ResearchContext is created once per research_id and cached in
  _CONTEXT_REGISTRY so all nodes within the same graph run share queue and
  citation_manager.  Agent classes have been removed; all LLM logic is inlined
  as free functions using get_chat_model_from_env() + get_prompt_manager().
  The registry entry is removed after report_node completes.

Streaming events emitted by each node are typed to match the frontend
useResearchReducer event types exactly:
  planning_started, rephrase_completed, decompose_completed, planning_completed,
  researching_started, block_started, block_completed, block_failed,
  reporting_started, reporting_completed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable

from .lg_state import ResearchState, TopicBlockDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers (LaTeX-safe formatting)
# ---------------------------------------------------------------------------


def _convert_to_template_format(template_str: str) -> str:
    """Convert {var} placeholders to $var for string.Template (avoids LaTeX brace conflicts)."""
    return re.sub(r"\{(\w+)\}", r"$\1", template_str)


def _safe_format(template_str: str, **kwargs: Any) -> str:
    """Format a prompt template safely using string.Template to avoid LaTeX brace conflicts."""
    converted = _convert_to_template_format(template_str)
    return Template(converted).safe_substitute(**kwargs)


# ---------------------------------------------------------------------------
# _ResearchContext — lightweight shared state replacing ResearchPipeline
# ---------------------------------------------------------------------------


@dataclass
class _ResearchContext:
    """Shared context for one research run (queue + citation_manager + config)."""

    research_id: str
    config: dict[str, Any]
    queue: Any  # DynamicTopicQueue
    citation_manager: Any  # CitationManager
    cache_dir: Path
    # paper tool lazily initialised
    _paper_tool: Any = field(default=None, repr=False)
    # asyncio lock for queue mutations in parallel research_block_node calls
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def call_tool(self, tool_type: str, query: str) -> str:
        """Dispatch a tool call and return raw JSON string result."""
        from src.tools.code_executor import run_code
        from src.tools.paper_search_tool import PaperSearchTool
        from src.tools.query_item_tool import query_numbered_item
        from src.tools.rag_tool import rag_search
        from src.tools.web_search import web_search

        tool_type = (tool_type or "").lower()
        tool_cfg = self.config.get("researching", {})
        default_timeout: float = float(tool_cfg.get("tool_timeout", 60))
        max_retries: int = int(tool_cfg.get("tool_max_retries", 2))

        async def _with_retry(coro_fn, *args, timeout=default_timeout, name="tool", **kw):
            last_err = None
            for attempt in range(max_retries + 1):
                try:
                    return await asyncio.wait_for(coro_fn(*args, **kw), timeout=timeout)
                except Exception as exc:
                    last_err = exc
                    if attempt < max_retries:
                        await asyncio.sleep(1)
            raise last_err or RuntimeError(f"{name} failed")

        try:
            if tool_type in ("rag_hybrid", "rag_naive", "rag"):
                rag_cfg = self.config.get("rag", {})
                kb_name = rag_cfg.get("kb_name", "ai_textbook")
                mode = (
                    "hybrid" if tool_type == "rag_hybrid"
                    else "naive" if tool_type == "rag_naive"
                    else rag_cfg.get("default_mode", "hybrid")
                )
                fallback = rag_cfg.get("fallback_mode", "naive")
                try:
                    res = await _with_retry(rag_search, query=query, kb_name=kb_name, mode=mode, name=f"rag({mode})")
                except Exception:
                    res = await _with_retry(rag_search, query=query, kb_name=kb_name, mode=fallback, name=f"rag({fallback})")
                return json.dumps(res, ensure_ascii=False)

            if tool_type == "web_search":
                res = await _with_retry(web_search, query=query, output_dir=str(self.cache_dir), name="web_search")
                return json.dumps(res, ensure_ascii=False)

            if tool_type == "query_item":
                kb_name = self.config.get("rag", {}).get("kb_name", "ai_textbook")
                res = await _with_retry(query_numbered_item, identifier=query, kb_name=kb_name, name="query_item")
                return json.dumps(res, ensure_ascii=False)

            if tool_type == "paper_search":
                if self._paper_tool is None:
                    self._paper_tool = PaperSearchTool()
                years_limit = self.config.get("researching", {}).get("paper_search_years_limit", 3)
                papers = await _with_retry(
                    self._paper_tool.search_papers,
                    query=query, max_results=3, years_limit=years_limit,
                    name="paper_search",
                )
                return json.dumps({"papers": papers}, ensure_ascii=False)

            if tool_type == "run_code":
                result = await _with_retry(run_code, language="python", code=query, timeout=30, name="run_code")
                return json.dumps(result, ensure_ascii=False)

            # Default: rag_hybrid
            rag_cfg = self.config.get("rag", {})
            kb_name = rag_cfg.get("kb_name", "ai_textbook")
            res = await _with_retry(rag_search, query=query, kb_name=kb_name, mode="hybrid", name="rag(hybrid)")
            return json.dumps(res, ensure_ascii=False)

        except Exception as exc:
            return json.dumps(
                {"status": "failed", "error": str(exc), "tool": tool_type, "query": query},
                ensure_ascii=False,
            )


# ---------------------------------------------------------------------------
# Context registry — keyed by research_id, shared across nodes in one run
# ---------------------------------------------------------------------------

_CONTEXT_REGISTRY: dict[str, _ResearchContext] = {}
_REGISTRY_LOCK = asyncio.Lock()


async def _get_or_create_context(state: ResearchState) -> _ResearchContext:
    """Return the _ResearchContext for this research_id, creating it on first call."""
    from src.agents.research.data_structures import DynamicTopicQueue
    from src.agents.research.utils.citation_manager import CitationManager
    from src.services.config import load_config_with_main

    research_id: str = state.get("research_id") or str(uuid.uuid4())

    async with _REGISTRY_LOCK:
        if research_id in _CONTEXT_REGISTRY:
            return _CONTEXT_REGISTRY[research_id]

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        config = load_config_with_main("research_config.yaml", project_root)

        language = state.get("language", "en")
        kb_name = state.get("kb_name", "") or ""
        enabled_tools: list[str] = state.get("enabled_tools", ["RAG"]) or ["RAG"]
        tools_upper = {t.upper() for t in enabled_tools}

        config.setdefault("system", {})["language"] = language
        if kb_name:
            config.setdefault("rag", {})["kb_name"] = kb_name

        has_rag = "RAG" in tools_upper
        researching_cfg = config.setdefault("researching", {})
        researching_cfg["enable_rag_hybrid"] = has_rag
        researching_cfg["enable_rag_naive"] = has_rag
        researching_cfg["enable_web_search"] = "WEB" in tools_upper
        researching_cfg["enable_paper_search"] = "PAPER" in tools_upper

        cache_dir = Path("data/user/research") / research_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        queue = DynamicTopicQueue(
            research_id,
            max_length=config.get("queue", {}).get("max_length"),
            state_file=str(cache_dir / "queue_progress.json"),
        )
        citation_manager = CitationManager(research_id, cache_dir)

        ctx = _ResearchContext(
            research_id=research_id,
            config=config,
            queue=queue,
            citation_manager=citation_manager,
            cache_dir=cache_dir,
        )
        _CONTEXT_REGISTRY[research_id] = ctx
        logger.debug("context_registry: created context for research_id=%s", research_id)
        return ctx


def _release_context(research_id: str) -> None:
    _CONTEXT_REGISTRY.pop(research_id, None)
    logger.debug("context_registry: released context for research_id=%s", research_id)


# ---------------------------------------------------------------------------
# LLM + prompt helpers (replaces BaseAgent.call_llm / BaseAgent.get_prompt)
# ---------------------------------------------------------------------------


def _get_llm(config: dict[str, Any]):
    """Return a LangChain chat model from environment config."""
    from src.services.llm.langchain_factory import get_chat_model_from_env
    return get_chat_model_from_env()


def _load_prompts(agent_name: str, language: str) -> dict[str, Any]:
    """Load prompt dict for a research agent."""
    from src.services.prompt import get_prompt_manager
    return get_prompt_manager().load_prompts("research", agent_name, language)


def _get_prompt(prompts: dict, *keys: str, default: str = "") -> str:
    """Navigate nested prompt dict by dot-path keys."""
    node = prompts
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, {})
    return node if isinstance(node, str) else default


async def _call_llm(llm, system_prompt: str, user_prompt: str) -> str:
    """Invoke a LangChain chat model and return the text content."""
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    resp = await llm.ainvoke(messages)
    return resp.content if hasattr(resp, "content") else str(resp)


# ---------------------------------------------------------------------------
# JSON parse helpers (shared)
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Any:
    from src.agents.research.utils.json_utils import extract_json_from_text
    return extract_json_from_text(text)


def _ensure_dict(data: Any) -> dict:
    from src.agents.research.utils.json_utils import ensure_json_dict
    return ensure_json_dict(data)


# ---------------------------------------------------------------------------
# Inlined rephrase logic
# ---------------------------------------------------------------------------


async def _rephrase(topic: str, language: str, config: dict[str, Any]) -> str:
    """Rephrase/optimise the user's raw topic. Returns optimised topic string."""
    prompts = _load_prompts("rephrase_agent", language)
    system_prompt = _get_prompt(prompts, "system", "role")
    user_tmpl = _get_prompt(prompts, "process", "rephrase")
    if not system_prompt or not user_tmpl:
        return topic  # Graceful fallback: no prompt → return as-is

    # Single-pass rephrase (no multi-turn needed in headless pipeline)
    user_prompt = user_tmpl.format(
        user_input=topic,
        iteration=0,
        conversation_history="",
        previous_result="",
    )
    llm = _get_llm(config)
    response = await _call_llm(llm, system_prompt, user_prompt)
    try:
        data = _extract_json(response)
        obj = _ensure_dict(data)
        return obj.get("topic", topic) or topic
    except Exception:
        return topic


# ---------------------------------------------------------------------------
# Inlined decompose logic
# ---------------------------------------------------------------------------


async def _decompose(
    topic: str,
    num_subtopics: int,
    language: str,
    config: dict[str, Any],
    citation_manager: Any,
    enabled_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Decompose topic into sub-topics. Returns dict with sub_topics list."""
    from src.agents.research.utils.json_utils import ensure_keys

    prompts = _load_prompts("decompose_agent", language)
    system_prompt = _get_prompt(prompts, "system", "role")
    llm = _get_llm(config)

    planning_cfg = config.get("planning", {}).get("decompose", {})
    mode = planning_cfg.get("mode", "manual")
    rag_cfg = config.get("rag", {})
    kb_name = rag_cfg.get("kb_name", "ai_textbook")

    # Step 1: Fetch RAG context for decomposition (only when RAG tool is enabled)
    rag_context = ""
    _rag_enabled = not enabled_tools or any(
        t.upper().startswith("RAG") for t in enabled_tools
    )
    if _rag_enabled:
        try:
            from src.tools.rag_tool import rag_search
            sub_queries_tmpl = _get_prompt(prompts, "process", "generate_queries")
            if sub_queries_tmpl and mode == "manual":
                sq_prompt = sub_queries_tmpl.format(
                    topic=topic, num_queries=num_subtopics
                )
                sq_resp = await _call_llm(llm, system_prompt, sq_prompt)
                sq_data = _extract_json(sq_resp)
                sq_obj = _ensure_dict(sq_data)
                # prompt returns "queries" key (not "sub_queries")
                sub_queries = sq_obj.get("queries") or sq_obj.get("sub_queries") or [topic]
            else:
                sub_queries = [topic]

            rag_results = []
            for q in (sub_queries or [topic])[:3]:
                try:
                    res = await asyncio.wait_for(
                        rag_search(query=str(q), kb_name=kb_name, mode="hybrid"),
                        timeout=30,
                    )
                    rag_results.append(str(res)[:1000])
                except Exception:
                    pass
            rag_context = "\n".join(rag_results)

            # Record planning citations
            if citation_manager and rag_context:
                from src.agents.research.data_structures import ToolTrace as _ToolTrace

                for q in (sub_queries or [topic])[:3]:
                    cid = citation_manager.get_next_citation_id(stage="planning")
                    _raw = json.dumps({"context": rag_context[:500]})
                    _trace = _ToolTrace(
                        tool_id=f"plan_{cid}",
                        citation_id=cid,
                        tool_type="rag_hybrid",
                        query=str(q),
                        raw_answer=_raw,
                        summary="",
                    )
                    citation_manager.add_citation(
                        citation_id=cid,
                        tool_type="rag_hybrid",
                        tool_trace=_trace,
                        raw_answer=_raw,
                    )
        except Exception as exc:
            logger.warning("decompose: RAG fetch failed: %s", exc)

    # Step 2: Generate sub-topics
    if mode == "manual" or not rag_context:
        decompose_tmpl = _get_prompt(prompts, "process", "decompose") or _get_prompt(
            prompts, "process", "decompose_without_rag"
        )
    else:
        decompose_tmpl = _get_prompt(prompts, "process", "decompose") or _get_prompt(
            prompts, "process", "decompose_without_rag"
        )

    if not decompose_tmpl:
        # Minimal fallback
        return {
            "main_topic": topic,
            "sub_topics": [{"title": topic, "overview": ""}],
            "total_subtopics": 1,
            "mode": "fallback",
        }

    decompose_prompt = _safe_format(
        decompose_tmpl,
        topic=topic,
        num_subtopics=num_subtopics,
        rag_context=rag_context or "(None)",
        decompose_requirement=(
            f"Requirements:\n"
            f"- Generate exactly {num_subtopics} subtopics\n"
            f"- Each subtopic should cover a distinct aspect\n"
            f"- Subtopics should not overlap"
        ),
    )
    resp = await _call_llm(llm, system_prompt, decompose_prompt)
    try:
        data = _extract_json(resp)
        obj = _ensure_dict(data)
        sub_topics = obj.get("sub_topics", [])
        if not sub_topics:
            sub_topics = [{"title": topic, "overview": ""}]
        return {
            "main_topic": obj.get("main_topic", topic),
            "sub_topics": sub_topics,
            "total_subtopics": len(sub_topics),
            "mode": mode,
        }
    except Exception:
        return {
            "main_topic": topic,
            "sub_topics": [{"title": topic, "overview": ""}],
            "total_subtopics": 1,
            "mode": "fallback",
        }


# ---------------------------------------------------------------------------
# Inlined note logic
# ---------------------------------------------------------------------------


async def _make_note(
    tool_type: str,
    query: str,
    raw_answer: str,
    citation_id: str,
    topic: str,
    context: str,
    language: str,
    config: dict[str, Any],
) -> Any:
    """Generate a ToolTrace with LLM summary (NoteAgent logic inlined)."""
    from src.agents.research.data_structures import ToolTrace

    prompts = _load_prompts("note_agent", language)
    system_prompt = _get_prompt(prompts, "system", "role")
    user_tmpl = _get_prompt(prompts, "process", "generate_summary")

    summary = ""
    if system_prompt and user_tmpl:
        tmpl_converted = _convert_to_template_format(user_tmpl)
        user_prompt = Template(tmpl_converted).safe_substitute(
            tool_type=tool_type,
            query=query,
            raw_answer=raw_answer,
            topic=topic,
            context=context,
        )
        llm = _get_llm(config)
        response = await _call_llm(llm, system_prompt, user_prompt)
        try:
            from src.agents.research.utils.json_utils import ensure_json_dict, ensure_keys
            data = _extract_json(response)
            obj = ensure_json_dict(data)
            ensure_keys(obj, ["summary"])
            summary = obj.get("summary", "") or ""
        except Exception:
            summary = (response or "")[:1000]
    else:
        summary = raw_answer[:500]

    tool_id = f"tool_{int(time.time() * 1000)}"
    return ToolTrace(
        tool_id=tool_id,
        citation_id=citation_id,
        tool_type=tool_type,
        query=query,
        raw_answer=raw_answer,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Inlined research agent helpers
# ---------------------------------------------------------------------------


def _build_available_tools_text(researching_cfg: dict, tools_cfg: dict) -> str:
    enable_rag = researching_cfg.get("enable_rag_hybrid", True) or researching_cfg.get("enable_rag_naive", True)
    tools_web_enabled = tools_cfg.get("web_search", {}).get("enabled", True)
    enable_web = tools_web_enabled and researching_cfg.get("enable_web_search", False)
    enable_paper = researching_cfg.get("enable_paper_search", False)
    enable_code = researching_cfg.get("enable_run_code", True)

    tools = []
    if enable_rag:
        tools.append("- rag_hybrid: Hybrid RAG retrieval (knowledge base) | Query format: Natural language")
        tools.append("- rag_naive: Basic RAG retrieval (knowledge base) | Query format: Natural language")
        tools.append("- query_item: Entity/item query (e.g., Theorem 3.1, Fig 2.1) | Query format: Entry number")
    if enable_paper:
        tools.append("- paper_search: Academic paper search | Query format: 3-5 English keywords, space-separated")
    if enable_web:
        tools.append("- web_search: Web search for latest information | Query format: Natural language")
    if enable_code:
        tools.append("- run_code: Code execution for calculation/visualization | Query format: Python code")
    if not tools:
        tools.append("- rag_hybrid: Hybrid RAG retrieval (default) | Query format: Natural language")
    return "\n".join(tools)


def _build_research_depth_guidance(
    iteration: int,
    max_iterations: int,
    iteration_mode: str,
    used_tools: list[str],
    enable_paper: bool,
    enable_web: bool,
) -> str:
    early_threshold = max(2, max_iterations // 3)
    middle_threshold = max(4, max_iterations * 2 // 3)

    if iteration <= early_threshold:
        phase_desc = f"Early Stage (Iteration 1-{early_threshold})"
        guidance = "Focus on building foundational knowledge using RAG/knowledge base tools."
    elif iteration <= middle_threshold:
        phase_desc = f"Middle Stage (Iteration {early_threshold + 1}-{middle_threshold})"
        guidance = (
            "Consider using Paper/Web search to add academic depth and real-time information."
            if enable_paper or enable_web
            else "Deepen knowledge coverage, explore different angles of the topic."
        )
    else:
        phase_desc = f"Late Stage (Iteration {middle_threshold + 1}+)"
        guidance = "Fill knowledge gaps, ensure completeness before concluding."

    unique_tools = set(used_tools)
    available_tools = []
    if (enable_rag := True) and not any(t in unique_tools for t in ["rag_hybrid", "rag_naive", "query_item"]):
        available_tools.append("RAG tools (rag_hybrid/rag_naive/query_item)")
    if enable_paper and "paper_search" not in unique_tools:
        available_tools.append("paper_search")
    if enable_web and "web_search" not in unique_tools:
        available_tools.append("web_search")

    diversity_hint = (
        f"\n**Tool Diversity Suggestion**: Consider using unexplored tools: {', '.join(available_tools)}"
        if available_tools and iteration > early_threshold
        else ""
    )

    if iteration_mode == "flexible":
        mode_guidance = (
            "\n**Iteration Mode: FLEXIBLE (Auto)**\n"
            "You have autonomy to decide when knowledge is sufficient. You may stop early if:\n"
            "- Core concepts are well covered from multiple angles\n"
            "- Key questions about the topic have been addressed\n"
            "- Further iterations would only add marginal value\n"
            "However, ensure you have made meaningful exploration before concluding."
        )
    else:
        mode_guidance = (
            "\n**Iteration Mode: FIXED**\n"
            "This mode requires thorough exploration. Be CONSERVATIVE about declaring knowledge sufficient:\n"
            "- In early iterations (first third), rarely conclude sufficiency\n"
            "- In middle iterations, require strong evidence of comprehensive coverage\n"
            "- Only in late iterations, conclude if truly comprehensive"
        )

    return (
        f"\n**Research Phase Guidance** ({phase_desc}):\n{guidance}\n\n"
        f"Current iteration: {iteration}/{max_iterations}\n"
        f"Tools already used: {', '.join(used_tools) if used_tools else 'None'}"
        f"{diversity_hint}{mode_guidance}\n"
    )


async def _research_block(
    topic_block: Any,
    ctx: _ResearchContext,
    max_iterations: int,
    language: str,
    progress_callback: Callable | None,
) -> dict[str, Any]:
    """
    Execute the research loop for a single TopicBlock (ResearchAgent logic inlined).
    Returns dict with block_id, iterations, tools_used, tool_traces.
    """
    researching_cfg = ctx.config.get("researching", {})
    tools_cfg = ctx.config.get("tools", {})
    iteration_mode = researching_cfg.get("iteration_mode", "fixed")
    tools_web_enabled = tools_cfg.get("web_search", {}).get("enabled", True)

    enable_rag = researching_cfg.get("enable_rag_hybrid", True) or researching_cfg.get("enable_rag_naive", True)
    enable_web = tools_web_enabled and researching_cfg.get("enable_web_search", False)
    enable_paper = researching_cfg.get("enable_paper_search", False)

    available_tools_text = _build_available_tools_text(researching_cfg, tools_cfg)

    prompts = _load_prompts("research_agent", language)
    system_prompt = _get_prompt(prompts, "system", "role")
    check_tmpl = _get_prompt(prompts, "process", "check_sufficiency")
    plan_tmpl = _get_prompt(prompts, "process", "generate_query_plan")

    llm = _get_llm(ctx.config)
    min_new_topic_score: float = float(researching_cfg.get("new_topic_min_score", 0.75))

    iteration = 0
    current_knowledge = ""
    tools_used: list[str] = []
    tool_traces_raw: list[dict] = []

    def send_progress(event_type: str, **data: Any):
        if progress_callback:
            try:
                progress_callback(event_type, **data)
            except Exception:
                pass

    while iteration < max_iterations:
        iteration += 1

        # --- Check sufficiency ---
        send_progress("checking_sufficiency", iteration=iteration, max_iterations=max_iterations)
        is_sufficient = False
        if check_tmpl and system_prompt:
            depth_guidance = _build_research_depth_guidance(
                iteration, max_iterations, iteration_mode, tools_used, enable_paper, enable_web
            )
            check_prompt = _safe_format(
                check_tmpl,
                topic=topic_block.sub_topic,
                overview=topic_block.overview,
                current_knowledge=current_knowledge if current_knowledge else "(None)",
                iteration=iteration,
                max_iterations=max_iterations,
                online_search_instruction="",
                research_depth_guidance=depth_guidance,
                iteration_mode_criteria="",
            )
            try:
                check_resp = await _call_llm(llm, system_prompt, check_prompt)
                check_obj = _ensure_dict(_extract_json(check_resp))
                is_sufficient = bool(check_obj.get("is_sufficient", False))
                if is_sufficient:
                    send_progress(
                        "knowledge_sufficient",
                        iteration=iteration,
                        max_iterations=max_iterations,
                        reason=check_obj.get("reason", ""),
                    )
                    break
            except Exception as exc:
                logger.debug("check_sufficiency parse error: %s", exc)

        # --- Generate query plan ---
        send_progress("generating_query", iteration=iteration, max_iterations=max_iterations)
        query = ""
        tool_type = "rag_hybrid"
        rationale = ""
        if plan_tmpl and system_prompt:
            existing_topics = ctx.queue.list_topics() if hasattr(ctx.queue, "list_topics") else []
            topics_text = "\n".join(f"- {t}" for t in existing_topics) if existing_topics else "(No other topics)"
            depth_guidance = _build_research_depth_guidance(
                iteration, max_iterations, iteration_mode, tools_used, enable_paper, enable_web
            )
            plan_prompt = _safe_format(
                plan_tmpl,
                topic=topic_block.sub_topic,
                overview=topic_block.overview,
                current_knowledge=current_knowledge[:2000] if current_knowledge else "(None)",
                iteration=iteration,
                max_iterations=max_iterations,
                existing_topics=topics_text,
                available_tools=available_tools_text,
                tool_phase_guidance="",
                research_depth_guidance=depth_guidance,
            )
            try:
                plan_resp = await _call_llm(llm, system_prompt, plan_prompt)
                plan_obj = _ensure_dict(_extract_json(plan_resp))
                query = (plan_obj.get("query") or "").strip()
                tool_type = plan_obj.get("tool_type", "rag_hybrid")
                rationale = plan_obj.get("rationale", "")

                # Dynamic topic splitting
                new_topic = plan_obj.get("new_sub_topic")
                new_overview = plan_obj.get("new_overview", "")
                new_topic_score = float(plan_obj.get("new_topic_score") or 0)
                should_add = plan_obj.get("should_add_new_topic")

                if isinstance(new_topic, str) and new_topic.strip():
                    trimmed = new_topic.strip()
                    if should_add is not False and new_topic_score >= min_new_topic_score:
                        async with ctx._lock:
                            if hasattr(ctx.queue, "has_topic") and not ctx.queue.has_topic(trimmed):
                                ctx.queue.add_block(trimmed, new_overview or "")
                        send_progress(
                            "new_topic_added",
                            iteration=iteration,
                            max_iterations=max_iterations,
                            new_topic=trimmed,
                            new_overview=new_overview,
                        )
            except Exception as exc:
                logger.debug("generate_query_plan parse error: %s", exc)

        if not query:
            continue

        # --- Call tool ---
        send_progress(
            "tool_calling",
            iteration=iteration,
            max_iterations=max_iterations,
            tool_type=tool_type,
            query=query,
            rationale=rationale,
        )
        raw_answer = await ctx.call_tool(tool_type, query)
        send_progress(
            "tool_completed",
            iteration=iteration,
            max_iterations=max_iterations,
            tool_type=tool_type,
            query=query,
        )

        # --- Get citation ID ---
        send_progress("processing_notes", iteration=iteration, max_iterations=max_iterations)
        citation_id = ctx.citation_manager.get_next_citation_id(
            stage="research", block_id=topic_block.block_id
        )

        # --- Make note (summarise) ---
        trace = await _make_note(
            tool_type=tool_type,
            query=query,
            raw_answer=raw_answer,
            citation_id=citation_id,
            topic=topic_block.sub_topic,
            context=current_knowledge,
            language=language,
            config=ctx.config,
        )
        topic_block.add_tool_trace(trace)
        ctx.citation_manager.add_citation(
            citation_id=citation_id,
            tool_type=tool_type,
            tool_trace=trace,
            raw_answer=raw_answer,
        )

        current_knowledge = (current_knowledge + "\n" + trace.summary).strip()
        topic_block.iteration_count = iteration
        tools_used.append(tool_type)
        tool_traces_raw.append(
            {
                "tool_id": trace.tool_id,
                "citation_id": citation_id,
                "tool_type": tool_type,
                "query": query,
                "raw_answer": raw_answer,
                "summary": trace.summary,
            }
        )

    return {
        "block_id": topic_block.block_id,
        "iterations": iteration,
        "tools_used": tools_used,
        "tool_traces": tool_traces_raw,
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Inlined reporting logic
# ---------------------------------------------------------------------------


async def _deduplicate_blocks(blocks: list, language: str, config: dict[str, Any]) -> list:
    if len(blocks) <= 1:
        return blocks
    prompts = _load_prompts("reporting_agent", language)
    system_prompt = _get_prompt(prompts, "system", "role")
    user_tmpl = _get_prompt(prompts, "process", "deduplicate")
    if not system_prompt or not user_tmpl:
        return blocks
    topics_text = "\n".join(f"{i + 1}. {b.sub_topic}: {b.overview[:200]}" for i, b in enumerate(blocks))
    filled = _safe_format(user_tmpl, topics=topics_text, total_topics=len(blocks))
    llm = _get_llm(config)
    resp = await _call_llm(llm, system_prompt, filled)
    try:
        obj = _ensure_dict(_extract_json(resp))
        keep_indices = obj.get("keep_indices", [])
        return [blocks[i] for i in keep_indices if isinstance(i, int) and i < len(blocks)] or blocks
    except Exception:
        return blocks


async def _generate_outline(
    topic: str, blocks: list, language: str, config: dict[str, Any]
) -> dict[str, Any]:
    prompts = _load_prompts("reporting_agent", language)
    system_prompt = _get_prompt(prompts, "system", "role")
    user_tmpl = _get_prompt(prompts, "process", "generate_outline")
    if not system_prompt or not user_tmpl:
        return _default_outline(topic, blocks)

    topics_data = [
        {
            "index": i,
            "block_id": b.block_id,
            "sub_topic": b.sub_topic,
            "overview": b.overview,
            "tool_summaries": [t.summary for t in b.tool_traces] if b.tool_traces else [],
        }
        for i, b in enumerate(blocks, 1)
    ]
    topics_json = json.dumps(topics_data, ensure_ascii=False, indent=2)
    filled = _safe_format(user_tmpl, topic=topic, topics_json=topics_json, total_topics=len(blocks))
    llm = _get_llm(config)
    resp = await _call_llm(llm, system_prompt, filled)
    try:
        obj = _ensure_dict(_extract_json(resp))
        if not obj.get("title", "").startswith("#"):
            obj["title"] = f"# {obj.get('title', topic)}"
        if obj.get("introduction") and not obj["introduction"].startswith("##"):
            obj["introduction"] = f"## {obj['introduction']}"
        if obj.get("conclusion") and not obj["conclusion"].startswith("##"):
            obj["conclusion"] = f"## {obj['conclusion']}"
        for section in obj.get("sections", []):
            if section.get("title") and not section["title"].startswith("##"):
                section["title"] = f"## {section['title']}"
            for sub in section.get("subsections", []):
                if sub.get("title") and not sub["title"].startswith("###"):
                    sub["title"] = f"### {sub['title']}"
        return obj
    except Exception:
        return _default_outline(topic, blocks)


def _default_outline(topic: str, blocks: list) -> dict[str, Any]:
    sections = [
        {
            "title": f"## {i}. {b.sub_topic}",
            "instruction": f"Provide detailed introduction to {b.sub_topic}.",
            "block_id": b.block_id,
            "subsections": [],
        }
        for i, b in enumerate(blocks, 1)
    ]
    return {
        "title": f"# {topic}",
        "introduction": "## Introduction",
        "introduction_instruction": "Present the research background, motivation, and objectives.",
        "sections": sections,
        "conclusion": "## Conclusion",
        "conclusion_instruction": "Summarize core findings and future directions.",
    }


def _ser_block(block: Any) -> dict[str, Any]:
    traces = [
        {
            "citation_id": getattr(t, "citation_id", None) or f"CIT-{block.block_id.split('_')[-1]}-01",
            "tool_type": t.tool_type,
            "query": t.query,
            "raw_answer": t.raw_answer,
            "summary": t.summary,
        }
        for t in block.tool_traces
    ]
    return {
        "block_id": block.block_id,
        "sub_topic": block.sub_topic,
        "overview": block.overview,
        "traces": traces,
    }


async def _write_section(
    topic: str,
    block: Any,
    section_outline: dict[str, Any],
    language: str,
    config: dict[str, Any],
    reporting_cfg: dict[str, Any],
) -> str:
    prompts = _load_prompts("reporting_agent", language)
    system_prompt = _get_prompt(
        prompts, "system", "role",
    ) or "You are an academic writing expert."
    tmpl = _get_prompt(prompts, "process", "write_section_body")
    if not tmpl:
        return f"{section_outline.get('title', block.sub_topic)}\n\n{block.overview}\n"

    citation_instruction = _get_prompt(prompts, "citation", "disabled_instruction") or ""
    block_data_json = json.dumps(_ser_block(block), ensure_ascii=False, indent=2)
    filled = _safe_format(
        tmpl,
        topic=topic,
        section_title=section_outline.get("title", block.sub_topic),
        section_instruction=section_outline.get("instruction", ""),
        block_data=block_data_json,
        min_section_length=reporting_cfg.get("min_section_length", 500),
        citation_instruction=citation_instruction,
        citation_output_hint="",
    )
    llm = _get_llm(config)
    resp = await _call_llm(llm, system_prompt, filled)
    try:
        obj = _ensure_dict(_extract_json(resp))
        content = obj.get("section_content", "")
        if isinstance(content, str) and content.strip():
            return content
    except Exception:
        pass
    return resp.strip() if resp else f"{block.sub_topic}\n\n{block.overview}"


async def _write_introduction(
    topic: str, blocks: list, outline: dict, language: str, config: dict[str, Any]
) -> str:
    prompts = _load_prompts("reporting_agent", language)
    system_prompt = _get_prompt(prompts, "system", "role") or "You are an academic writing expert."
    tmpl = _get_prompt(prompts, "process", "write_introduction")
    if not tmpl:
        return f"This report explores {topic}."

    topics_summary = json.dumps(
        [{"sub_topic": b.sub_topic, "overview": b.overview, "tool_count": len(b.tool_traces)} for b in blocks],
        ensure_ascii=False, indent=2,
    )
    intro_instruction = outline.get("introduction_instruction", "") or outline.get("introduction", "")
    filled = _safe_format(
        tmpl,
        topic=topic,
        introduction_instruction=intro_instruction,
        topics_summary=topics_summary,
        total_topics=len(blocks),
    )
    llm = _get_llm(config)
    resp = await _call_llm(llm, system_prompt, filled)
    try:
        obj = _ensure_dict(_extract_json(resp))
        intro = obj.get("introduction", "")
        if isinstance(intro, str) and intro.strip():
            return intro
    except Exception:
        pass
    return resp.strip() if resp else f"This report explores {topic}."


async def _write_conclusion(
    topic: str, blocks: list, outline: dict, language: str, config: dict[str, Any]
) -> str:
    prompts = _load_prompts("reporting_agent", language)
    system_prompt = _get_prompt(prompts, "system", "role") or "You are an academic writing expert."
    tmpl = _get_prompt(prompts, "process", "write_conclusion")
    if not tmpl:
        return f"This report has surveyed the key aspects of {topic}."

    topics_findings = json.dumps(
        [
            {
                "sub_topic": b.sub_topic,
                "overview": b.overview,
                "key_findings": [t.summary for t in b.tool_traces[:3]],
            }
            for b in blocks
        ],
        ensure_ascii=False, indent=2,
    )
    conclusion_instruction = outline.get("conclusion_instruction", "") or outline.get("conclusion", "")
    filled = _safe_format(
        tmpl,
        topic=topic,
        conclusion_instruction=conclusion_instruction,
        topics_findings=topics_findings,
        total_topics=len(blocks),
    )
    llm = _get_llm(config)
    resp = await _call_llm(llm, system_prompt, filled)
    try:
        obj = _ensure_dict(_extract_json(resp))
        conclusion = obj.get("conclusion", "")
        if isinstance(conclusion, str) and conclusion.strip():
            return conclusion
    except Exception:
        pass
    return resp.strip() if resp else f"This report has surveyed the key aspects of {topic}."


async def _generate_report(
    topic: str,
    queue: Any,
    citation_manager: Any,
    language: str,
    config: dict[str, Any],
) -> str:
    """Full ReportingAgent.process() logic inlined."""
    reporting_cfg = config.get("reporting", {})

    # 1) Deduplicate blocks
    cleaned_blocks = await _deduplicate_blocks(list(queue.blocks), language, config)

    # 2) Outline
    outline = await _generate_outline(topic, cleaned_blocks, language, config)

    # 3) Write report section by section
    parts: list[str] = []

    title = outline.get("title", f"# {topic}")
    if not title.startswith("#"):
        title = f"# {title}"
    parts.append(f"{title}\n\n")

    # Introduction
    intro_title = outline.get("introduction", "## Introduction")
    if not intro_title.startswith("##"):
        intro_title = f"## {intro_title}"
    introduction = await _write_introduction(topic, cleaned_blocks, outline, language, config)
    parts.append(f"{intro_title}\n\n{introduction}\n\n")

    # Sections
    for i, section in enumerate(outline.get("sections", []), 1):
        block_id = section.get("block_id")
        block = next((b for b in cleaned_blocks if b.block_id == block_id), None)
        if not block:
            continue
        section_content = await _write_section(topic, block, section, language, config, reporting_cfg)
        parts.append(f"{section_content}\n\n")

    # Conclusion
    conclusion_title = outline.get("conclusion", "## Conclusion")
    if not conclusion_title.startswith("##"):
        conclusion_title = f"## {conclusion_title}"
    conclusion = await _write_conclusion(topic, cleaned_blocks, outline, language, config)
    parts.append(f"{conclusion_title}\n\n{conclusion}\n\n")

    return "".join(parts)


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

    planning_started_event = {"type": "planning_started", "user_topic": topic}

    if skip:
        return {
            "optimized_topic": topic,
            "streaming_events": [
                planning_started_event,
                {"type": "rephrase_completed", "optimized_topic": topic},
            ],
        }

    optimized = topic
    try:
        ctx = await _get_or_create_context(state)
        language = ctx.config.get("system", {}).get("language", "en")
        optimized = await _rephrase(topic, language, ctx.config)
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

    Emits frontend events:
      decompose_completed  — with generated_subtopics count
      planning_completed   — with total_blocks count
    """
    topic: str = state.get("optimized_topic") or state.get("topic", "")
    initial_subtopics: int = state.get("initial_subtopics", 5)

    blocks: list[TopicBlockDict] = []
    try:
        ctx = await _get_or_create_context(state)
        language = ctx.config.get("system", {}).get("language", "en")

        decompose_result = await _decompose(
            topic=topic,
            num_subtopics=initial_subtopics,
            language=language,
            config=ctx.config,
            citation_manager=ctx.citation_manager,
            enabled_tools=state.get("enabled_tools"),
        )

        # Seed the shared queue
        ctx.queue.blocks.clear()
        for sub_topic_data in decompose_result.get("sub_topics", []):
            title = (sub_topic_data.get("title") or "").strip()
            overview = sub_topic_data.get("overview", "")
            if not title:
                continue
            block = ctx.queue.add_block(sub_topic=title, overview=overview)
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

        logger.debug("decompose_node: %d blocks for '%s'", len(blocks), topic[:60])

    except Exception as exc:
        logger.warning("decompose_node failed: %s", exc)
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
            {"type": "decompose_completed", "generated_subtopics": len(blocks)},
            {"type": "planning_completed", "total_blocks": len(blocks)},
            {"type": "researching_started", "total_blocks": len(blocks), "execution_mode": "parallel"},
        ],
    }


# ---------------------------------------------------------------------------
# Node: research_block
# ---------------------------------------------------------------------------


async def research_block_node(state: ResearchState) -> dict[str, Any]:
    """
    Phase 2 — Research a single TopicBlock.

    Invoked in parallel for each pending block via LangGraph's Send API.
    Uses the shared context (same queue, citation_manager).

    Emits frontend events:
      block_started    — when research begins for this block
      block_completed  — on success, with tools_used list
      block_failed     — on error, with error message
    """
    from src.agents.research.data_structures import TopicBlock

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

    updated_block: TopicBlockDict = {**current_block, "status": "researching"}
    new_citations: list[dict[str, str]] = []
    detail_events: list[dict[str, Any]] = []

    start_event: dict[str, Any] = {"type": "block_started", "block_id": block_id, "sub_topic": sub_topic}

    _FORWARDED_EVENT_TYPES = {
        "checking_sufficiency", "knowledge_sufficient", "generating_query",
        "tool_calling", "tool_completed", "processing_notes", "new_topic_added",
    }

    def _progress_callback(event_type: str, **data: Any) -> None:
        if event_type in _FORWARDED_EVENT_TYPES:
            detail_events.append({"type": event_type, "block_id": block_id, **data})

    try:
        ctx = await _get_or_create_context(state)
        language = ctx.config.get("system", {}).get("language", "en")

        # Find or create the TopicBlock in the shared queue
        block_obj = ctx.queue.get_block_by_id(block_id)
        if block_obj is None:
            block_obj = TopicBlock(
                block_id=block_id,
                sub_topic=sub_topic,
                overview=current_block.get("overview", ""),
            )
            ctx.queue.blocks.append(block_obj)

        result = await _research_block(
            topic_block=block_obj,
            ctx=ctx,
            max_iterations=max_iterations,
            language=language,
            progress_callback=_progress_callback,
        )

        ctx.queue.mark_completed(block_id)

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

        tools_used = list({t.get("tool_type", "") for t in tool_traces_raw if t.get("tool_type")})
        note_summary = ""
        if block_obj.tool_traces:
            note_summary = getattr(block_obj.tool_traces[-1], "summary", "")

        updated_block = {
            **updated_block,
            "status": "completed",
            "tool_traces": tool_traces,
            "iteration_count": result.get("iterations", len(tool_traces)),
            "metadata": {**current_block.get("metadata", {}), "note_summary": note_summary},
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

    Emits frontend events:
      reporting_started    — signals report generation has begun
      reporting_completed  — carries word_count, sections, citations counts
    """
    topic: str = state.get("optimized_topic") or state.get("topic", "")
    research_id: str = state.get("research_id", str(uuid.uuid4()))
    citations: list[dict] = state.get("citations", [])

    final_report = ""
    report_path = ""
    try:
        ctx = await _get_or_create_context(state)
        language = ctx.config.get("system", {}).get("language", "en")

        final_report = await _generate_report(
            topic=topic,
            queue=ctx.queue,
            citation_manager=ctx.citation_manager,
            language=language,
            config=ctx.config,
        )

        output_dir = Path("data/user/research") / research_id
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path_obj = output_dir / "report.md"
        report_path_obj.write_text(final_report, encoding="utf-8")
        report_path = str(report_path_obj)

        (output_dir / "citations.json").write_text(
            json.dumps(citations, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.debug("report_node: %d chars written to %s", len(final_report), report_path)

    except Exception as exc:
        logger.warning("report_node failed: %s", exc)
        final_report = f"[Report generation failed: {exc}]"
    finally:
        _release_context(research_id)

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
