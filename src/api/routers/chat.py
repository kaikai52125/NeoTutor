"""
Chat API Router
================

WebSocket endpoint for lightweight chat with session management.
REST endpoints for session operations.

Powered by LangGraph + MemorySaver checkpointer for per-session history.
"""

from pathlib import Path
import sys
import uuid

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.agents.chat.session_manager import SessionManager
from src.logging import get_logger
from src.services.config import load_config_with_main
from src.services.settings.interface_settings import get_ui_language

project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("solve_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("ChatAPI", level="INFO", log_dir=log_dir)

router = APIRouter()

session_manager = SessionManager()


# =============================================================================
# REST Endpoints for Session Management
# =============================================================================


@router.get("/chat/sessions")
async def list_sessions(limit: int = 20):
    """List recent chat sessions."""
    return session_manager.list_sessions(limit=limit, include_messages=False)


@router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific chat session with full message history."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_manager.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


# =============================================================================
# WebSocket Endpoint for Chat
# =============================================================================


@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for chat — powered by LangGraph.

    Message format:
    {
        "message": str,              # User message
        "session_id": str | null,    # Session ID (null for new session)
        "kb_name": str,              # Knowledge base name (for RAG)
        "enable_rag": bool,          # Enable RAG retrieval
        "enable_web_search": bool    # Enable Web Search
    }

    Response format:
    - {"type": "session", "session_id": str}
    - {"type": "status", "stage": str, "message": str}
    - {"type": "stream", "content": str}
    - {"type": "sources", "rag": list, "web": list}
    - {"type": "result", "content": str}
    - {"type": "error", "message": str}
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            message = data.get("message", "").strip()
            if not message:
                await websocket.send_json({"type": "error", "message": "Message is required"})
                continue

            session_id: str = data.get("session_id") or str(uuid.uuid4())
            kb_name: str = data.get("kb_name", "") or ""
            enable_rag: bool = bool(data.get("enable_rag", False))
            enable_web_search: bool = bool(data.get("enable_web_search", False))
            language: str = data.get("language") or get_ui_language(
                default=config.get("system", {}).get("language", "en")
            )

            logger.info(
                f"Chat request: session={session_id}, message={message[:50]!r}, "
                f"rag={enable_rag}, web={enable_web_search}"
            )

            try:
                from langchain_core.messages import HumanMessage
                from src.agents.chat.lg_graph import get_chat_graph
                from src.api.utils.langgraph_ws_adapter import stream_graph_to_websocket

                await websocket.send_json({"type": "session", "session_id": session_id})

                # Create session in SessionManager on first message
                if not session_manager.get_session(session_id):
                    import time as _time
                    now = _time.time()
                    title = message[:50] + ("..." if len(message) > 50 else "")
                    new_session = {
                        "session_id": session_id,
                        "title": title,
                        "messages": [],
                        "settings": {
                            "kb_name": kb_name,
                            "enable_rag": enable_rag,
                            "enable_web_search": enable_web_search,
                        },
                        "created_at": now,
                        "updated_at": now,
                    }
                    sessions = session_manager._get_sessions()
                    sessions.insert(0, new_session)
                    sessions = sessions[:100]
                    session_manager._save_sessions(sessions)

                graph = get_chat_graph()
                config_lg = {"configurable": {"thread_id": session_id}}
                initial_state = {
                    "messages": [HumanMessage(content=message)],
                    "kb_name": kb_name,
                    "enable_rag": enable_rag,
                    "enable_web_search": enable_web_search,
                    "language": language,
                    "session_id": session_id,
                }

                final_state = await stream_graph_to_websocket(
                    graph=graph,
                    initial_state=initial_state,
                    websocket=websocket,
                    config=config_lg,
                )

                sources = final_state.get("sources", {})
                if sources.get("rag") or sources.get("web"):
                    await websocket.send_json({"type": "sources", **sources})

                msgs = final_state.get("messages", [])
                last_content = msgs[-1].content if msgs else ""
                await websocket.send_json({
                    "type": "result",
                    "content": last_content,
                    "session_id": session_id,
                })

                # Persist this turn to SessionManager
                session_manager.add_message(session_id, role="user", content=message)
                session_manager.add_message(
                    session_id,
                    role="assistant",
                    content=last_content,
                    sources=sources if (sources.get("rag") or sources.get("web")) else None,
                )

                logger.info(
                    f"Chat completed: session={session_id}, {len(str(last_content))} chars"
                )

            except Exception as e:
                err_str = str(e)
                logger.error(f"Chat processing error: {err_str}")
                # 内容审核拦截：阿里云/其他平台触发的安全过滤
                if "inappropriate content" in err_str or "content_filter" in err_str or "sensitive" in err_str.lower():
                    user_msg = "抱歉，该内容触发了平台安全审核，无法生成回复。请尝试换一种表达方式。"
                else:
                    user_msg = err_str
                await websocket.send_json({"type": "error", "message": user_msg})

    except WebSocketDisconnect:
        logger.debug("Client disconnected from chat")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
