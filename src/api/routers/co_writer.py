from pathlib import Path
import sys
import traceback
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json

from src.agents.co_writer.utils import TOOL_CALLS_DIR, load_history
from src.logging import get_logger
from src.services.config import load_config_with_main
from src.services.settings.interface_settings import get_ui_language
from src.services.tts import get_tts_config

router = APIRouter()

project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("solve_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("CoWriter", level="INFO", log_dir=log_dir)


def _current_language() -> str:
    return get_ui_language(default=config.get("system", {}).get("language", "en"))


# =============================================================================
# Request / Response Models
# =============================================================================


class EditRequest(BaseModel):
    text: str
    instruction: str
    action: Literal["rewrite", "shorten", "expand"] = "rewrite"
    source: Literal["rag", "web"] | None = None
    kb_name: str | None = None


class EditResponse(BaseModel):
    edited_text: str
    operation_id: str


class AutoMarkRequest(BaseModel):
    text: str


class AutoMarkResponse(BaseModel):
    marked_text: str
    operation_id: str


class NarrateRequest(BaseModel):
    content: str
    style: Literal["friendly", "academic", "concise"] = "friendly"
    voice: str | None = None
    skip_audio: bool = False


class NarrateResponse(BaseModel):
    script: str
    key_points: list[str]
    style: str
    original_length: int
    script_length: int
    has_audio: bool
    audio_url: str | None = None
    audio_id: str | None = None
    voice: str | None = None
    audio_error: str | None = None


class ScriptOnlyRequest(BaseModel):
    content: str
    style: Literal["friendly", "academic", "concise"] = "friendly"


# =============================================================================
# Edit Endpoints
# =============================================================================


@router.post("/edit", response_model=EditResponse)
async def edit_text(request: EditRequest):
    """Co-writer text editing — powered by LangGraph."""
    from src.agents.co_writer.lg_graph import get_edit_graph

    language = _current_language()
    operation_id = str(uuid.uuid4())

    try:
        graph = get_edit_graph()
        initial_state = {
            "text": request.text,
            "instruction": request.instruction,
            "action": request.action or "rewrite",
            "source": request.source or "none",
            "kb_name": request.kb_name or "",
            "language": language,
            "operation_id": operation_id,
            "content": "",
            "style": "friendly",
            "voice": "",
            "skip_audio": True,
            "context": "",
            "edited_text": "",
            "script": "",
            "key_points": [],
            "has_audio": False,
            "audio_url": "",
            "streaming_events": [],
        }
        final_state = await graph.ainvoke(initial_state)
        return {
            "edited_text": final_state.get("edited_text", request.text),
            "operation_id": final_state.get("operation_id", operation_id),
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/automark", response_model=AutoMarkResponse)
async def auto_mark_text(request: AutoMarkRequest):
    """AI auto-mark text."""
    from datetime import datetime

    from langchain_core.messages import HumanMessage, SystemMessage

    from src.agents.co_writer.utils import load_history, save_history
    from src.services.llm.langchain_factory import get_chat_model_from_env
    from src.services.prompt import get_prompt_manager

    try:
        language = _current_language()
        text = request.text
        operation_id = str(uuid.uuid4())

        prompts = get_prompt_manager().load_prompts(
            module_name="co_writer", agent_name="edit_agent", language=language
        ) or {}
        system_text = (prompts.get("auto_mark_system") or "").strip()
        user_tpl = (prompts.get("auto_mark_user_template") or "Process the following text:\n{text}").strip()

        llm = get_chat_model_from_env()
        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_tpl.format(text=text)),
        ])
        marked_text = response.content if hasattr(response, "content") else str(response)

        history = load_history()
        history.append({
            "id": operation_id,
            "timestamp": datetime.now().isoformat(),
            "action": "automark",
            "source": None,
            "kb_name": None,
            "input": {"original_text": text, "instruction": "AI Auto Mark"},
            "output": {"edited_text": marked_text},
            "tool_call_file": None,
        })
        save_history(history)

        return {"marked_text": marked_text, "operation_id": operation_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_history():
    """Get all operation history."""
    try:
        history = load_history()
        return {"history": history, "total": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{operation_id}")
async def get_operation(operation_id: str):
    """Get single operation details."""
    try:
        history = load_history()
        for op in history:
            if op.get("id") == operation_id:
                return op
        raise HTTPException(status_code=404, detail="Operation not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tool_calls/{operation_id}")
async def get_tool_call(operation_id: str):
    """Get tool call details."""
    try:
        for filepath in TOOL_CALLS_DIR.glob(f"{operation_id}_*.json"):
            with open(filepath, encoding="utf-8") as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="Tool call not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/markdown")
async def export_markdown(content: dict):
    """Export as Markdown file."""
    try:
        markdown_content = content.get("content", "")
        filename = content.get("filename", "document.md")
        return Response(
            content=markdown_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Narrate Endpoints
# =============================================================================


@router.post("/narrate", response_model=NarrateResponse)
async def narrate_content(request: NarrateRequest):
    """Co-writer narration — powered by LangGraph."""
    from src.agents.co_writer.lg_graph import get_narrate_graph

    try:
        graph = get_narrate_graph()
        initial_state = {
            "content": request.content,
            "style": request.style or "friendly",
            "voice": request.voice or "",
            "skip_audio": request.skip_audio,
            "language": _current_language(),
            "text": "",
            "instruction": "",
            "action": "rewrite",
            "source": "none",
            "kb_name": "",
            "operation_id": str(uuid.uuid4()),
            "context": "",
            "edited_text": "",
            "script": "",
            "key_points": [],
            "has_audio": False,
            "audio_url": "",
            "streaming_events": [],
        }
        final_state = await graph.ainvoke(initial_state)
        return {
            "script": final_state.get("script", ""),
            "key_points": final_state.get("key_points", []),
            "style": request.style,
            "original_length": len(request.content),
            "script_length": len(final_state.get("script", "")),
            "has_audio": final_state.get("has_audio", False),
            "audio_url": final_state.get("audio_url") or None,
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/narrate/script")
async def generate_script_only(request: ScriptOnlyRequest):
    """Generate script only (no audio)."""
    import json
    import re

    from langchain_core.messages import HumanMessage, SystemMessage

    from src.services.llm.langchain_factory import get_chat_model_from_env
    from src.services.prompt import get_prompt_manager

    try:
        language = _current_language()
        content = request.content
        style = request.style

        prompts = get_prompt_manager().load_prompts(
            module_name="co_writer", agent_name="narrator_agent", language=language
        ) or {}

        def _p(key: str, default: str = "") -> str:
            return (prompts.get(key) or default).strip()

        is_long = len(content) > 5000
        style_prompts = {
            "friendly": _p("style_friendly"),
            "academic": _p("style_academic"),
            "concise":  _p("style_concise"),
        }
        length_instruction = _p("length_instruction_long" if is_long else "length_instruction_short")
        system_tpl = _p("generate_script_system_template")
        system_text = system_tpl.format(
            style_prompt=style_prompts.get(style, style_prompts["friendly"]),
            length_instruction=length_instruction,
        )
        if is_long:
            user_tpl = _p("generate_script_user_long", "Convert to narration script:\n\n{content}")
            user_text = user_tpl.format(content=content[:8000] + "...")
        else:
            user_tpl = _p("generate_script_user_short", "Convert to narration script:\n\n{content}")
            user_text = user_tpl.format(content=content)

        llm = get_chat_model_from_env()
        script_resp = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw_script = script_resp.content if hasattr(script_resp, "content") else str(script_resp)
        script = raw_script.strip()
        if len(script) > 4000:
            truncated = script[:3997]
            last_period = max(
                truncated.rfind("。"), truncated.rfind("！"), truncated.rfind("？"),
                truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"),
            )
            script = truncated[:last_period + 1] if last_period > 3500 else truncated + "..."

        kp_resp = await llm.ainvoke([
            SystemMessage(content=_p("extract_key_points_system", "Extract 3-5 key points as a JSON array.")),
            HumanMessage(content=_p(
                "extract_key_points_user",
                "Please extract key points from the following notes:\n\n{content}",
            ).format(content=content[:4000])),
        ])
        kp_text = kp_resp.content if hasattr(kp_resp, "content") else str(kp_resp)
        key_points = []
        json_match = re.search(r"\[.*\]", kp_text, re.DOTALL)
        if json_match:
            try:
                key_points = json.loads(json_match.group())
            except Exception:
                pass

        return {
            "script": script,
            "key_points": key_points,
            "style": style,
            "original_length": len(content),
            "script_length": len(script),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tts/status")
async def get_tts_status():
    """Check TTS service status."""
    try:
        tts_config = get_tts_config()
        return {
            "available": True,
            "model": tts_config.get("model"),
            "default_voice": tts_config.get("voice", "alloy"),
        }
    except ValueError as e:
        return {
            "available": False,
            "error": str(e),
            "hint": "Please configure TTS_MODEL, TTS_API_KEY, TTS_URL in .env file",
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@router.get("/tts/voices")
async def get_available_voices():
    """Get available TTS voice list."""
    voices = [
        {"id": "alloy", "name": "Alloy", "description": "Neutral and balanced voice"},
        {"id": "echo", "name": "Echo", "description": "Warm and conversational voice"},
        {"id": "fable", "name": "Fable", "description": "Expressive and dramatic voice"},
        {"id": "onyx", "name": "Onyx", "description": "Deep and authoritative voice"},
        {"id": "nova", "name": "Nova", "description": "Friendly and upbeat voice"},
        {"id": "shimmer", "name": "Shimmer", "description": "Clear and pleasant voice"},
    ]
    return {"voices": voices}
