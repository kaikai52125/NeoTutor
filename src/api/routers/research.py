import asyncio
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

from fastapi import APIRouter, WebSocket
from pydantic import BaseModel

from src.api.utils.history import ActivityType, history_manager
from src.api.utils.task_id_manager import TaskIDManager
from src.logging import get_logger
from src.services.config import load_config_with_main
from src.services.settings.interface_settings import get_ui_language

# Force stdout to use utf-8 to prevent encoding errors with emojis on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

router = APIRouter()


# Helper to load config (with main.yaml merge)
def load_config():
    project_root = Path(__file__).parent.parent.parent.parent
    return load_config_with_main("research_config.yaml", project_root)


# Initialize logger with config
config = load_config()
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("ResearchAPI", log_dir=log_dir)


class OptimizeRequest(BaseModel):
    topic: str
    iteration: int = 0
    previous_result: dict[str, Any] | None = None
    kb_name: str | None = "ai_textbook"


@router.post("/optimize_topic")
async def optimize_topic(request: OptimizeRequest):
    try:
        cfg = load_config()
        cfg.setdefault("system", {})
        language = get_ui_language(default=cfg.get("system", {}).get("language", "en"))
        cfg["system"]["language"] = language

        from src.agents.research.lg_nodes import _rephrase

        optimized_topic = await _rephrase(request.topic, language, cfg)
        return {"topic": optimized_topic, "iteration": request.iteration}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@router.websocket("/run")
async def websocket_research_run(websocket: WebSocket):
    """Research WebSocket endpoint — powered by LangGraph 1.0.9."""
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        topic = data.get("topic")
        kb_name = data.get("kb_name", "ai_textbook")
        plan_mode = data.get("plan_mode", "medium")
        enabled_tools = data.get("enabled_tools", ["RAG"])
        skip_rephrase = data.get("skip_rephrase", False)

        if not topic:
            await websocket.send_json({"type": "error", "content": "Topic is required"})
            return

        import uuid as _uuid
        language = get_ui_language(default=config.get("system", {}).get("language", "en"))
        research_id = str(_uuid.uuid4())
        initial_subtopics, max_iterations = _PLAN_MODE_DEFAULTS.get(
            plan_mode, _PLAN_MODE_DEFAULTS["medium"]
        )

        await websocket.send_json({"type": "task_id", "task_id": research_id})
        await websocket.send_json({"type": "status", "content": "started",
                                   "research_id": research_id})

        logger.info(f"Research request: topic='{topic[:50]}', plan_mode={plan_mode}, id={research_id}")

        from src.agents.research.lg_graph import get_research_graph
        from src.api.utils.langgraph_ws_adapter import stream_graph_to_websocket

        graph = get_research_graph()
        initial_state = {
            "topic": topic,
            "kb_name": kb_name,
            "research_id": research_id,
            "language": language,
            "initial_subtopics": initial_subtopics,
            "max_iterations": max_iterations,
            "plan_mode": plan_mode,
            "skip_rephrase": skip_rephrase,
            "enabled_tools": enabled_tools,
            "optimized_topic": "",
            "topic_blocks": [],
            "citations": [],
            "final_report": "",
            "report_path": "",
            "streaming_events": [],
        }

        final_state = await stream_graph_to_websocket(
            graph=graph,
            initial_state=initial_state,
            websocket=websocket,
            config={},
        )

        report_content = final_state.get("final_report", "")

        try:
            history_manager.add_entry(
                activity_type=ActivityType.RESEARCH,
                title=topic,
                content={"topic": topic, "report": report_content, "kb_name": kb_name},
                summary=f"Research ID: {research_id}",
            )
        except Exception as hist_exc:
            logger.warning(f"Failed to save research history: {hist_exc}")

        await websocket.send_json({
            "type": "result",
            "report": report_content,
            "metadata": {"research_id": research_id, "plan_mode": plan_mode},
            "research_id": research_id,
        })

    except Exception as e:
        await websocket.send_json({"type": "error", "content": str(e)})
        logger.error(f"Research error: {e}", exc_info=True)


# =============================================================================
# LangGraph WebSocket Endpoint (parallel testing — replaces /run after validation)
# =============================================================================

# plan_mode → (initial_subtopics, max_iterations)
_PLAN_MODE_DEFAULTS: dict[str, tuple[int, int]] = {
    "quick":  (2, 2),
    "medium": (5, 4),
    "deep":   (8, 7),
    "auto":   (8, 6),
}


@router.websocket("/run/lg")
async def websocket_research_run_lg(websocket: WebSocket):
    """
    LangGraph-based Research endpoint.

    Drop-in replacement for /run using LangGraph 1.0.9 + LangChain 1.2.x.
    Uses Map-Reduce Send API for parallel block research instead of
    asyncio.Semaphore.

    Message format (same as /run):
    {
        "topic":         str,               # Required
        "kb_name":       str,               # Default: ai_textbook
        "plan_mode":     str,               # quick|medium|deep|auto (default: medium)
        "enabled_tools": list[str],         # RAG|Paper|Web subsets
        "skip_rephrase": bool               # Skip topic optimisation
    }
    """
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        topic: str = data.get("topic", "").strip()
        if not topic:
            await websocket.send_json({"type": "error", "content": "Topic is required"})
            return

        kb_name: str = data.get("kb_name", "ai_textbook") or "ai_textbook"
        plan_mode: str = data.get("plan_mode", "medium")
        enabled_tools: list[str] = data.get("enabled_tools", ["RAG"])
        skip_rephrase: bool = bool(data.get("skip_rephrase", False))
        language: str = get_ui_language(
            default=config.get("system", {}).get("language", "en")
        )

        initial_subtopics, max_iterations = _PLAN_MODE_DEFAULTS.get(
            plan_mode, _PLAN_MODE_DEFAULTS["medium"]
        )

        import uuid as _uuid
        research_id = str(_uuid.uuid4())

        # Notify frontend
        await websocket.send_json({"type": "task_id", "task_id": research_id})
        await websocket.send_json(
            {"type": "status", "content": "started", "research_id": research_id}
        )

        logger.info(
            f"LangGraph Research request: topic='{topic[:60]}', plan_mode={plan_mode}, research_id={research_id}"
        )

        from src.agents.research.lg_graph import get_research_graph
        from src.api.utils.langgraph_ws_adapter import stream_graph_to_websocket

        graph = get_research_graph()
        initial_state = {
            "topic": topic,
            "kb_name": kb_name,
            "research_id": research_id,
            "language": language,
            "initial_subtopics": initial_subtopics,
            "max_iterations": max_iterations,
            "plan_mode": plan_mode,
            "skip_rephrase": skip_rephrase,
            "enabled_tools": enabled_tools,
            "optimized_topic": "",
            "topic_blocks": [],
            "citations": [],
            "final_report": "",
            "report_path": "",
            "streaming_events": [],
        }

        final_state = await stream_graph_to_websocket(
            graph=graph,
            initial_state=initial_state,
            websocket=websocket,
            config={},
        )

        report_content: str = final_state.get("final_report", "")
        report_path: str = final_state.get("report_path", "")

        # Save to history
        try:
            history_manager.add_entry(
                activity_type=ActivityType.RESEARCH,
                title=topic,
                content={
                    "topic": topic,
                    "report": report_content,
                    "kb_name": kb_name,
                },
                summary=f"Research ID: {research_id}",
            )
        except Exception as hist_exc:
            logger.warning(f"Failed to save research history: {hist_exc}")

        await websocket.send_json(
            {
                "type": "result",
                "report": report_content,
                "metadata": {
                    "research_id": research_id,
                    "report_path": report_path,
                    "topic": topic,
                    "plan_mode": plan_mode,
                },
                "research_id": research_id,
            }
        )

        logger.info(
            f"LangGraph Research completed: research_id={research_id}, report={len(report_content)} chars"
        )

    except Exception as exc:
        logger.error(f"LangGraph Research error: {exc}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": str(exc)})
        except Exception:
            pass
