"""
Guided Learning API Router
==========================

All endpoints are powered by LangGraph (no GuideManager / BaseAgent).
"""

import uuid
from pathlib import Path
import sys

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.api.utils.history import history_manager, ActivityType
from src.api.utils.notebook_manager import notebook_manager
from src.logging import get_logger
from src.services.config import load_config_with_main
from src.services.settings.interface_settings import get_ui_language

router = APIRouter()

project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("guide_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("Guide", level="INFO", log_dir=log_dir)


# === Request / Response Models ===


class CreateSessionRequest(BaseModel):
    notebook_id: str | None = None
    records: list[dict] | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class FixHtmlRequest(BaseModel):
    session_id: str
    bug_description: str


class NextKnowledgeRequest(BaseModel):
    session_id: str


# === Shared helpers ===


def _language() -> str:
    return get_ui_language(default=config.get("system", {}).get("language", "en"))


def _get_graph():
    from src.agents.guide.lg_graph import get_guide_graph
    return get_guide_graph()


def _lg_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


# === REST Endpoints ===


@router.post("/create_session")
async def create_session(request: CreateSessionRequest):
    """Create a guide session — locate knowledge points via LangGraph."""
    session_id = str(uuid.uuid4())
    language = _language()

    records: list = []
    notebook_name: str = ""
    if request.records:
        records = request.records
    elif request.notebook_id:
        notebook = notebook_manager.get_notebook(request.notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        records = notebook.get("records", [])
        notebook_name = notebook.get("name", "")

    if not records:
        raise HTTPException(status_code=400, detail="No records available")

    graph = _get_graph()
    initial_state = {
        "session_id": session_id,
        "notebook_id": request.notebook_id or "",
        "notebook_name": notebook_name,
        "language": language,
        "action": "create",
        "user_message": "",
        "notebook_records": records,
        "knowledge_points": [],
        "current_index": 0,
        "current_html": "",
        "chat_history": [],
        "status": "initialized",
        "summary": "",
        "streaming_events": [],
    }

    try:
        final_state = await graph.ainvoke(initial_state, config=_lg_config(session_id))
        kps = final_state.get("knowledge_points", [])
        return {
            "success": True,
            "session_id": session_id,
            "knowledge_points": kps,
            "total_points": len(kps),
            "status": final_state.get("status", "initialized"),
        }
    except Exception as exc:
        logger.error("create_session error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/start")
async def start_learning(request: NextKnowledgeRequest):
    """Generate the first HTML page for the session."""
    session_id = request.session_id
    graph = _get_graph()
    try:
        final_state = await graph.ainvoke(
            {"action": "start", "user_message": "", "notebook_records": []},
            config=_lg_config(session_id),
        )
        kps = final_state.get("knowledge_points", [])
        idx = final_state.get("current_index", 0)
        return {
            "success": True,
            "session_id": session_id,
            "html": final_state.get("current_html", ""),
            "current_index": idx,
            "status": final_state.get("status", "learning"),
            "progress": round((idx + 1) / len(kps) * 100) if kps else 0,
            "message": f"Starting knowledge point {idx + 1}/{len(kps)}",
            "knowledge_point": kps[idx] if kps and idx < len(kps) else {},
        }
    except Exception as exc:
        logger.error("start_learning error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/next")
async def next_knowledge(request: NextKnowledgeRequest):
    """Advance to the next knowledge point (or finish with a summary)."""
    session_id = request.session_id
    graph = _get_graph()
    try:
        final_state = await graph.ainvoke(
            {"action": "next", "user_message": "", "notebook_records": []},
            config=_lg_config(session_id),
        )
        kps = final_state.get("knowledge_points", [])
        idx = final_state.get("current_index", 0)
        status = final_state.get("status", "learning")

        if status == "completed":
            notebook_name = final_state.get("notebook_name", "")
            summary = final_state.get("summary", "")
            history_manager.add_entry(
                activity_type=ActivityType.GUIDE,
                title=notebook_name or "Guided Learning",
                content={
                    "notebook_name": notebook_name,
                    "knowledge_points": [
                        {
                            "knowledge_title": kp.get("knowledge_title", ""),
                            "knowledge_summary": kp.get("knowledge_summary", ""),
                            "user_difficulty": kp.get("user_difficulty", ""),
                        }
                        for kp in kps
                    ],
                    "summary": summary,
                    "session_id": session_id,
                },
                summary=summary[:200] + "..." if len(summary) > 200 else summary,
            )
            return {
                "success": True,
                "session_id": session_id,
                "status": "completed",
                "summary": summary,
                "message": "Congratulations on completing all knowledge points!",
                "progress": 100,
            }

        return {
            "success": True,
            "session_id": session_id,
            "html": final_state.get("current_html", ""),
            "summary": final_state.get("summary", ""),
            "status": status,
            "current_index": idx,
            "progress": round((idx + 1) / len(kps) * 100) if kps else 0,
            "message": f"Moving to knowledge point {idx + 1}/{len(kps)}",
        }
    except Exception as exc:
        logger.error("next_knowledge error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat")
async def chat(request: ChatRequest):
    """Answer a user question about the current knowledge point."""
    graph = _get_graph()
    try:
        final_state = await graph.ainvoke(
            {"action": "chat", "user_message": request.message, "notebook_records": []},
            config=_lg_config(request.session_id),
        )
        history = final_state.get("chat_history", [])
        answer = history[-1]["content"] if history else ""
        return {"success": True, "session_id": request.session_id, "answer": answer}
    except Exception as exc:
        logger.error("chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fix_html")
async def fix_html(request: FixHtmlRequest):
    """Regenerate the current HTML page after a reported bug."""
    graph = _get_graph()
    try:
        final_state = await graph.ainvoke(
            {
                "action": "fix_html",
                "user_message": request.bug_description,
                "notebook_records": [],
            },
            config=_lg_config(request.session_id),
        )
        return {
            "success": True,
            "session_id": request.session_id,
            "html": final_state.get("current_html", ""),
        }
    except Exception as exc:
        logger.error("fix_html error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Read the current session state from the LangGraph MemorySaver checkpointer.
    """
    from src.agents.guide.lg_graph import get_guide_graph
    try:
        graph = get_guide_graph()
        snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
        if not snapshot or not snapshot.values:
            raise HTTPException(status_code=404, detail="Session not found")
        state = snapshot.values
        return {
            "session_id": session_id,
            "status": state.get("status", ""),
            "current_index": state.get("current_index", 0),
            "knowledge_points": state.get("knowledge_points", []),
            "chat_history": state.get("chat_history", []),
            "summary": state.get("summary", ""),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_session error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/session/{session_id}/html")
async def get_current_html(session_id: str):
    """Read the current HTML page from the LangGraph MemorySaver checkpointer."""
    from src.agents.guide.lg_graph import get_guide_graph
    try:
        graph = get_guide_graph()
        snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
        if not snapshot or not snapshot.values:
            raise HTTPException(status_code=404, detail="Session not found")
        html = snapshot.values.get("current_html", "")
        if not html:
            raise HTTPException(status_code=404, detail="No HTML content yet")
        return {"html": html}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_current_html error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# === WebSocket Endpoint ===


@router.websocket("/ws/{session_id}")
async def websocket_guide(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time interaction — backed by LangGraph.

    Message types:
    - "start"     → generate first HTML page
    - "next"      → advance to next knowledge point
    - "chat"      → answer a question (send "message" field)
    - "fix_html"  → regenerate HTML (send "bug_description" field)
    - "get_session" → return current session state
    """
    await websocket.accept()

    graph = _get_graph()

    try:
        # Verify session exists
        snapshot = await graph.aget_state(_lg_config(session_id))
        if not snapshot or not snapshot.values:
            await websocket.send_json({"type": "error", "content": "Session not found"})
            await websocket.close()
            return

        await websocket.send_json({"type": "session_info", "data": snapshot.values})

        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "start":
                    final_state = await graph.ainvoke(
                        {"action": "start", "user_message": "", "notebook_records": []},
                        config=_lg_config(session_id),
                    )
                    kps = final_state.get("knowledge_points", [])
                    idx = final_state.get("current_index", 0)
                    await websocket.send_json({
                        "type": "start_result",
                        "data": {
                            "success": True,
                            "html": final_state.get("current_html", ""),
                            "current_index": idx,
                            "knowledge_point": kps[idx] if kps and idx < len(kps) else {},
                        },
                    })

                elif msg_type == "next":
                    final_state = await graph.ainvoke(
                        {"action": "next", "user_message": "", "notebook_records": []},
                        config=_lg_config(session_id),
                    )
                    status = final_state.get("status", "learning")
                    if status == "completed":
                        await websocket.send_json({
                            "type": "next_result",
                            "data": {
                                "success": True,
                                "status": "completed",
                                "summary": final_state.get("summary", ""),
                            },
                        })
                    else:
                        kps = final_state.get("knowledge_points", [])
                        idx = final_state.get("current_index", 0)
                        await websocket.send_json({
                            "type": "next_result",
                            "data": {
                                "success": True,
                                "html": final_state.get("current_html", ""),
                                "current_index": idx,
                                "knowledge_point": kps[idx] if kps and idx < len(kps) else {},
                            },
                        })

                elif msg_type == "chat":
                    message = data.get("message", "")
                    if message:
                        final_state = await graph.ainvoke(
                            {"action": "chat", "user_message": message, "notebook_records": []},
                            config=_lg_config(session_id),
                        )
                        history = final_state.get("chat_history", [])
                        answer = history[-1]["content"] if history else ""
                        await websocket.send_json({
                            "type": "chat_result",
                            "data": {"success": True, "answer": answer},
                        })

                elif msg_type == "fix_html":
                    bug_desc = data.get("bug_description", "")
                    final_state = await graph.ainvoke(
                        {
                            "action": "fix_html",
                            "user_message": bug_desc,
                            "notebook_records": [],
                        },
                        config=_lg_config(session_id),
                    )
                    await websocket.send_json({
                        "type": "fix_result",
                        "data": {
                            "success": True,
                            "html": final_state.get("current_html", ""),
                        },
                    })

                elif msg_type == "get_session":
                    snap = await graph.aget_state(_lg_config(session_id))
                    await websocket.send_json({
                        "type": "session_info",
                        "data": snap.values if snap else {},
                    })

                else:
                    await websocket.send_json(
                        {"type": "error", "content": f"Unknown message type: {msg_type}"}
                    )

            except WebSocketDisconnect:
                logger.debug("WebSocket disconnected: %s", session_id)
                break
            except Exception as exc:
                logger.error("WebSocket error: %s", exc)
                await websocket.send_json({"type": "error", "content": str(exc)})

    except Exception as exc:
        logger.error("WebSocket connection error: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "guide"}
