"""
IdeaGen API Router
==================

WebSocket endpoint for research idea generation, powered by LangGraph.
"""

import uuid as _uuid
from datetime import datetime
from pathlib import Path
import sys

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.api.utils.notebook_manager import notebook_manager
from src.logging import get_logger
from src.services.config import load_config_with_main
from src.services.settings.interface_settings import get_ui_language

router = APIRouter()

project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("solve_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("IdeaGen", level="INFO", log_dir=log_dir)


class IdeaGenStage:
    """IdeaGen status stages"""

    INIT = "init"
    EXTRACTING = "extracting"
    KNOWLEDGE_EXTRACTED = "knowledge_extracted"
    FILTERING = "filtering"
    FILTERED = "filtered"
    EXPLORING = "exploring"
    EXPLORED = "explored"
    STRICT_FILTERING = "strict_filtering"
    GENERATING = "generating"
    IDEA_READY = "idea_ready"
    COMPLETE = "complete"
    ERROR = "error"


async def send_status(
    websocket: WebSocket, stage: str, message: str, data: dict = None, task_id: str = None
):
    """Unified status sending function"""
    payload = {
        "type": "status",
        "stage": stage,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }
    if data:
        payload["data"] = data
    await websocket.send_json(payload)
    log_msg = f"[{stage}] {message}"
    if task_id:
        log_msg = f"[{task_id}] {log_msg}"
    logger.info(log_msg)


@router.websocket("/generate")
async def websocket_ideagen(websocket: WebSocket):
    """
    IdeaGen WebSocket — powered by LangGraph.

    Message format:
    {
        "notebook_id":   str | null,  # Optional notebook ID
        "record_ids":    list | null, # Optional record filter
        "records":       list | null, # Direct records (cross-notebook mode)
        "user_thoughts": str          # Optional free-text research topic
    }
    """
    await websocket.accept()
    logger.info("IdeaGen WebSocket connected")

    task_id = str(_uuid.uuid4())[:8]

    try:
        data = await websocket.receive_json()
        notebook_id = data.get("notebook_id")
        record_ids = data.get("record_ids")
        direct_records = data.get("records")
        user_thoughts: str = data.get("user_thoughts", "") or ""

        await websocket.send_json({"type": "task_id", "task_id": task_id})

        language: str = get_ui_language(
            default=config.get("system", {}).get("language", "en")
        )

        # Resolve records
        records: list = []
        if direct_records and isinstance(direct_records, list):
            records = direct_records
        elif notebook_id:
            notebook = notebook_manager.get_notebook(notebook_id)
            if not notebook:
                await send_status(websocket, IdeaGenStage.ERROR, "Notebook not found",
                                  task_id=task_id)
                return
            records = notebook.get("records", [])
            if record_ids:
                records = [r for r in records if r.get("id") in record_ids]

        if not records and not user_thoughts:
            await send_status(
                websocket, IdeaGenStage.ERROR,
                "Please provide notebook records or describe your research topic",
                task_id=task_id,
            )
            return

        await send_status(websocket, IdeaGenStage.INIT,
                          "Initializing idea generation workflow...", task_id=task_id)

        from src.agents.ideagen.lg_graph import get_ideagen_graph
        from src.api.utils.langgraph_ws_adapter import stream_graph_to_websocket

        graph = get_ideagen_graph()
        initial_state = {
            "notebook_records": records,
            "user_thoughts": user_thoughts,
            "kb_name": "",
            "language": language,
            "run_id": task_id,
            "knowledge_points": [],
            "filtered_points": [],
            "explored_ideas": [],
            "strict_filtered_ideas": [],
            "idea_results": [],
            "streaming_events": [],
        }

        final_state = await stream_graph_to_websocket(
            graph=graph,
            initial_state=initial_state,
            websocket=websocket,
            config={},
        )

        idea_results: list = final_state.get("idea_results", [])

        # Stream each idea to frontend
        for idea in idea_results:
            await websocket.send_json({"type": "idea", "data": idea})

        await send_status(
            websocket, IdeaGenStage.COMPLETE,
            f"Successfully generated {len(idea_results)} research ideas",
            {"ideas": idea_results, "count": len(idea_results)},
            task_id=task_id,
        )
        logger.info(f"IdeaGen completed: task_id={task_id}, {len(idea_results)} ideas")

    except WebSocketDisconnect:
        logger.info(f"IdeaGen WebSocket disconnected (task_id={task_id})")
    except Exception as exc:
        logger.error(f"IdeaGen error: {exc}")
        try:
            await send_status(websocket, IdeaGenStage.ERROR, f"Error: {exc!s}",
                              {"error": str(exc)}, task_id=task_id)
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("IdeaGen WebSocket closed")


@router.get("/test")
async def test_ideagen():
    return {"status": "ok", "message": "IdeaGen API is working"}
