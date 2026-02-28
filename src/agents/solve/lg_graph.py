# -*- coding: utf-8 -*-
"""
Solve Module — LangGraph Graph Definition
=========================================

Implements the dual-loop pipeline originally in MainSolver._run_dual_loop_pipeline():

  Analysis Loop (runs first, up to max_analysis_iterations):
    START → investigate_node → note_node ──(should_continue_analysis?)──► investigate_node
                                                                        ► plan_node

  Solve Loop (runs after analysis, step-by-step):
    plan_node → exec_tools_node ──(after_exec_tools?)──► solve_step_node
                                                      ► finalize_node

    solve_step_node ──(after_solve_step?)──► exec_tools_node   (more tools needed)
                                          ► response_node      (step ready for response)

    response_node ──(after_response?)──► advance_step_node → exec_tools_node
                                      ► finalize_node

    finalize_node → END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .lg_nodes import (
    advance_step_node,
    exec_tools_node,
    finalize_node,
    investigate_node,
    note_node,
    plan_node,
    response_node,
    solve_step_node,
)
from .lg_state import SolveState


# ---------------------------------------------------------------------------
# Condition functions (control-flow edges)
# ---------------------------------------------------------------------------


def should_continue_analysis(state: SolveState) -> str:
    """
    After note_node: decide whether to run another analysis iteration
    or move on to the Solve Loop.
    """
    if state.get("analysis_should_stop", False):
        return "plan"
    if state.get("analysis_iteration", 0) >= state.get("max_analysis_iterations", 5):
        return "plan"
    return "investigate"


def after_exec_tools(state: SolveState) -> str:
    """
    After exec_tools_node: are there still pending tool calls?
    Or is the step index already past the end?
    """
    idx: int = state.get("current_step_index", 0)
    steps: list = state.get("solve_steps", [])

    if idx >= len(steps):
        return "finalize"

    step = steps[idx]
    pending = [tc for tc in step.get("tool_calls", []) if tc.get("status") == "pending"]
    if pending:
        return "exec_tools"     # re-run exec_tools until all tools done
    return "solve_step"


def after_solve_step(state: SolveState) -> str:
    """
    After solve_step_node: does the step need more tool calls, or is it ready
    for a response (finish_requested), or max iterations reached?
    """
    idx: int = state.get("current_step_index", 0)
    steps: list = state.get("solve_steps", [])

    if idx >= len(steps):
        return "finalize"

    step = steps[idx]
    pending = [tc for tc in step.get("tool_calls", []) if tc.get("status") == "pending"]
    if pending:
        return "exec_tools"

    if state.get("finish_requested", False):
        return "response"

    if state.get("solve_iteration", 0) >= state.get("max_solve_iterations", 6):
        return "response"       # force response on iteration limit

    return "solve_step"         # keep iterating within the step


def after_response(state: SolveState) -> str:
    """
    After response_node: advance to next step or finalise if all steps done.
    """
    next_idx = state.get("current_step_index", 0) + 1
    if next_idx >= len(state.get("solve_steps", [])):
        return "finalize"
    return "advance_step"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_solve_graph():
    """
    Build and compile the Solve LangGraph (dual-loop).

    Returns:
        Compiled CompiledStateGraph (no checkpointer — each question is a
        fresh run; sessions are managed by the router layer).
    """
    builder = StateGraph(SolveState)

    # ── Register nodes ───────────────────────────────────────────────────────
    # Analysis Loop
    builder.add_node("investigate", investigate_node)
    builder.add_node("note", note_node)

    # Solve Loop
    builder.add_node("plan", plan_node)
    builder.add_node("exec_tools", exec_tools_node)
    builder.add_node("solve_step", solve_step_node)
    builder.add_node("response", response_node)
    builder.add_node("advance_step", advance_step_node)
    builder.add_node("finalize", finalize_node)

    # ── Analysis Loop edges ───────────────────────────────────────────────────
    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "note")
    builder.add_conditional_edges(
        "note",
        should_continue_analysis,
        {"investigate": "investigate", "plan": "plan"},
    )

    # ── Solve Loop edges ──────────────────────────────────────────────────────
    builder.add_edge("plan", "exec_tools")
    builder.add_conditional_edges(
        "exec_tools",
        after_exec_tools,
        {"exec_tools": "exec_tools", "solve_step": "solve_step", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "solve_step",
        after_solve_step,
        {"exec_tools": "exec_tools", "response": "response", "solve_step": "solve_step",
         "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "response",
        after_response,
        {"advance_step": "advance_step", "finalize": "finalize"},
    )
    builder.add_edge("advance_step", "exec_tools")
    builder.add_edge("finalize", END)

    # No checkpointer: each question is a one-shot run.
    return builder.compile()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_solve_graph: Any = None


def get_solve_graph():
    """Return the singleton compiled Solve graph."""
    global _solve_graph
    if _solve_graph is None:
        _solve_graph = build_solve_graph()
    return _solve_graph


__all__ = ["build_solve_graph", "get_solve_graph"]
