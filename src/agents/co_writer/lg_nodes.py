# -*- coding: utf-8 -*-
"""
CoWriter Module — LangGraph Node Implementations
=================================================

All LLM calls use LangChain's BaseChatModel via get_chat_model_from_env().
No BaseAgent / custom agent wrapper is used.

Edit pipeline nodes:
  retrieve_context_node  — fetch RAG/web context (if source != "none")
  edit_node              — LLM text editing

Narrate pipeline node:
  narrate_node           — script generation + optional TTS audio
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage

from .lg_state import CoWriterState
from .utils import load_history, save_history

logger = logging.getLogger(__name__)

_AUDIO_DIR = Path(__file__).parent.parent.parent.parent / "data" / "user" / "co-writer" / "audio"


def _ensure_audio_dir() -> None:
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _load_prompts(agent_name: str, language: str) -> dict[str, Any]:
    """Load co_writer prompt YAML for the given agent and language."""
    from src.services.prompt import get_prompt_manager
    try:
        return get_prompt_manager().load_prompts(
            module_name="co_writer",
            agent_name=agent_name,
            language=language,
        ) or {}
    except Exception as exc:
        logger.warning("Failed to load prompts %s/%s: %s", agent_name, language, exc)
        return {}


def _get(prompts: dict, key: str, default: str = "") -> str:
    return (prompts.get(key) or default).strip()


# ---------------------------------------------------------------------------
# Edit pipeline
# ---------------------------------------------------------------------------


async def retrieve_context_node(state: CoWriterState) -> dict[str, Any]:
    """
    Optionally retrieve RAG or web context before editing.

    Returns:
        {"context": str, "streaming_events": list}
    """
    source: str = state.get("source", "none") or "none"
    kb_name: str = state.get("kb_name", "") or ""
    text: str = state.get("text", "") or ""
    instruction: str = state.get("instruction", "") or ""

    context = ""
    if source == "rag" and kb_name:
        try:
            from src.tools.rag_tool import rag_search
            result = await rag_search(
                query=f"{instruction}\n\n{text[:200]}",
                kb_name=kb_name,
                mode="hybrid",
            )
            context = result.get("answer", "") or result.get("content", "")
            logger.debug("retrieve_context_node (rag): %d chars", len(context))
        except Exception as exc:
            logger.warning("retrieve_context_node RAG failed: %s", exc)

    elif source == "web":
        try:
            from src.tools.web_search import web_search
            # web_search is a synchronous function — call directly, no await
            result = web_search(query=f"{instruction} {text[:100]}")
            if isinstance(result, dict):
                context = result.get("answer", "") or result.get("content", "")
            elif isinstance(result, list):
                context = "\n\n".join(
                    r.get("content", r.get("snippet", "")) for r in result[:3]
                )
            elif isinstance(result, str):
                context = result
            logger.debug("retrieve_context_node (web): %d chars", len(context))
        except Exception as exc:
            logger.warning("retrieve_context_node web failed: %s", exc)

    return {
        "context": context,
        "streaming_events": [
            {"type": "status", "stage": "retrieve_context",
             "message": f"retrieving_{source}_context"}
        ] if source != "none" else [],
    }


async def edit_node(state: CoWriterState) -> dict[str, Any]:
    """
    Edit text using a LangChain LLM call.

    Returns:
        {"edited_text": str, "operation_id": str, "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env

    text: str = state.get("text", "") or ""
    instruction: str = state.get("instruction", "") or ""
    action: str = state.get("action", "rewrite") or "rewrite"
    source: str = state.get("source", "none") or "none"
    kb_name: str = state.get("kb_name", "") or ""
    context: str = state.get("context", "") or ""
    language: str = state.get("language", "en")
    operation_id: str = state.get("operation_id", "") or str(uuid.uuid4())

    edited_text = text
    try:
        prompts = _load_prompts("edit_agent", language)

        action_verbs = {"rewrite": "Rewrite", "shorten": "Shorten", "expand": "Expand"}
        action_verb = action_verbs.get(action, "Rewrite")

        system_text = _get(prompts, "system", "You are an expert editor and writing assistant.")

        action_tpl = _get(
            prompts, "action_template",
            "{action_verb} the following text based on the user's instruction.\n\nUser Instruction: {instruction}\n\n",
        )
        user_text = action_tpl.format(action_verb=action_verb, instruction=instruction)

        if context:
            ctx_tpl = _get(prompts, "context_template", "Reference Context:\n{context}\n\n")
            user_text += ctx_tpl.format(context=context)

        text_tpl = _get(
            prompts, "user_template",
            "Target Text to Edit:\n{text}\n\nOutput only the edited text, without quotes or explanations.",
        )
        user_text += text_tpl.format(text=text)

        llm = get_chat_model_from_env()
        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        edited_text = response.content if hasattr(response, "content") else str(response)
        logger.debug("edit_node: %d → %d chars", len(text), len(edited_text))

        # Persist history
        history = load_history()
        history.append({
            "id": operation_id,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "source": source if source != "none" else None,
            "kb_name": kb_name or None,
            "input": {"original_text": text, "instruction": instruction},
            "output": {"edited_text": edited_text},
            "tool_call_file": None,
        })
        save_history(history)

    except Exception as exc:
        logger.warning("edit_node failed: %s", exc)

    return {
        "edited_text": edited_text,
        "operation_id": operation_id,
        "streaming_events": [
            {"type": "result", "stage": "edit",
             "edited_text": edited_text, "operation_id": operation_id}
        ],
    }


# ---------------------------------------------------------------------------
# Narrate pipeline helpers
# ---------------------------------------------------------------------------


def _truncate_script(script: str, max_len: int = 4000) -> str:
    if len(script) <= max_len:
        return script
    truncated = script[:max_len - 3]
    last_period = max(
        truncated.rfind("。"), truncated.rfind("！"), truncated.rfind("？"),
        truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"),
    )
    if last_period > max_len // 2:
        return truncated[:last_period + 1]
    return truncated + "..."


def _split_script(text: str, max_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        segment = remaining[:max_len]
        cut = max(
            segment.rfind("。"), segment.rfind("！"), segment.rfind("？"),
            segment.rfind("…"), segment.rfind("."), segment.rfind("!"),
            segment.rfind("?"), segment.rfind("\n"),
        )
        if cut > max_len // 2:
            chunks.append(remaining[:cut + 1])
            remaining = remaining[cut + 1:]
        else:
            chunks.append(segment)
            remaining = remaining[max_len:]
    return [c for c in chunks if c.strip()]


async def _generate_audio_dashscope(
    script: str, voice: str, api_key: str, model: str, base_url: str,
    audio_path: Path, audio_filename: str, audio_id: str,
) -> dict[str, Any]:
    import aiohttp

    if "dashscope-intl" in base_url:
        api_base = "https://dashscope-intl.aliyuncs.com/api/v1"
    elif "dashscope-us" in base_url:
        api_base = "https://dashscope-us.aliyuncs.com/api/v1"
    else:
        api_base = "https://dashscope.aliyuncs.com/api/v1"

    url = f"{api_base}/services/aigc/multimodal-generation/generation"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    chunks = _split_script(script, 512)
    logger.info("DashScope TTS: %d chunk(s), %d chars", len(chunks), len(script))

    all_audio: list[bytes] = []
    async with aiohttp.ClientSession() as session:
        for i, chunk in enumerate(chunks, 1):
            async with session.post(url, json={"model": model, "input": {"text": chunk, "voice": voice}}, headers=headers) as resp:
                if resp.status != 200:
                    raise ValueError(f"DashScope TTS error {resp.status} chunk {i}: {await resp.text()}")
                result = await resp.json()
            remote_url = result.get("output", {}).get("audio", {}).get("url", "")
            if not remote_url:
                raise ValueError(f"DashScope TTS missing audio URL chunk {i}: {result}")
            async with session.get(remote_url) as dl:
                if dl.status != 200:
                    raise ValueError(f"Failed to download TTS chunk {i}: HTTP {dl.status}")
                all_audio.append(await dl.read())

    with open(audio_path, "wb") as f:
        for b in all_audio:
            f.write(b)
    logger.info("DashScope TTS saved: %s (%d bytes)", audio_path, sum(len(b) for b in all_audio))
    return {"audio_path": str(audio_path), "audio_url": f"/api/outputs/co-writer/audio/{audio_filename}", "audio_id": audio_id, "voice": voice}


async def _generate_audio_openai(
    script: str, voice: str, tts_config: dict[str, Any],
    audio_path: Path, audio_filename: str, audio_id: str,
) -> dict[str, Any]:
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    binding = os.getenv("TTS_BINDING", "openai")
    api_version = tts_config.get("api_version")
    if binding == "azure_openai" or (binding == "openai" and api_version):
        client = AsyncAzureOpenAI(
            api_key=tts_config["api_key"], azure_endpoint=tts_config["base_url"], api_version=api_version,
        )
    else:
        client = AsyncOpenAI(base_url=tts_config["base_url"], api_key=tts_config["api_key"])

    response = await client.audio.speech.create(model=tts_config["model"], voice=voice, input=script)
    await response.stream_to_file(audio_path)
    logger.info("OpenAI TTS saved: %s", audio_path)
    return {"audio_path": str(audio_path), "audio_url": f"/api/outputs/co-writer/audio/{audio_filename}", "audio_id": audio_id, "voice": voice}


# ---------------------------------------------------------------------------
# Narrate pipeline
# ---------------------------------------------------------------------------


async def narrate_node(state: CoWriterState) -> dict[str, Any]:
    """
    Generate narration script and optionally TTS audio using LangChain LLM.

    Returns:
        {"script": str, "key_points": list, "has_audio": bool, "audio_url": str,
         "streaming_events": list}
    """
    from src.services.llm.langchain_factory import get_chat_model_from_env
    from src.services.tts import get_tts_config

    content: str = state.get("content", "") or ""
    style: str = state.get("style", "friendly") or "friendly"
    voice: Optional[str] = state.get("voice", "") or None
    skip_audio: bool = state.get("skip_audio", True)
    language: str = state.get("language", "en")

    script = ""
    key_points: list[str] = []
    has_audio = False
    audio_url = ""
    audio_error: Optional[str] = None

    try:
        prompts = _load_prompts("narrator_agent", language)
        llm = get_chat_model_from_env()

        # ── Generate script ──────────────────────────────────────────────────
        is_long = len(content) > 5000
        style_prompts = {
            "friendly": _get(prompts, "style_friendly"),
            "academic": _get(prompts, "style_academic"),
            "concise":  _get(prompts, "style_concise"),
        }
        length_instruction = _get(
            prompts, "length_instruction_long" if is_long else "length_instruction_short"
        )
        system_tpl = _get(prompts, "generate_script_system_template", "")
        system_text = system_tpl.format(
            style_prompt=style_prompts.get(style, style_prompts["friendly"]),
            length_instruction=length_instruction,
        )

        if is_long:
            user_tpl = _get(prompts, "generate_script_user_long", "Convert to narration script:\n\n{content}")
            user_text = user_tpl.format(content=content[:8000] + "...")
        else:
            user_tpl = _get(prompts, "generate_script_user_short", "Convert to narration script:\n\n{content}")
            user_text = user_tpl.format(content=content)

        script_resp = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw_script = script_resp.content if hasattr(script_resp, "content") else str(script_resp)
        script = _truncate_script(raw_script.strip(), max_len=4000)
        logger.debug("narrate_node: script=%d chars", len(script))

        # ── Extract key points ───────────────────────────────────────────────
        kp_system = _get(prompts, "extract_key_points_system", "Extract 3-5 key points as a JSON array.")
        kp_user_tpl = _get(
            prompts, "extract_key_points_user",
            "Please extract key points from the following notes:\n\n{content}",
        )
        kp_resp = await llm.ainvoke([
            SystemMessage(content=kp_system),
            HumanMessage(content=kp_user_tpl.format(content=content[:4000])),
        ])
        kp_text = kp_resp.content if hasattr(kp_resp, "content") else str(kp_resp)
        json_match = re.search(r"\[.*\]", kp_text, re.DOTALL)
        if json_match:
            try:
                key_points = json.loads(json_match.group())
            except Exception:
                key_points = []

        # ── TTS audio ────────────────────────────────────────────────────────
        if not skip_audio:
            try:
                tts_config = get_tts_config()
                for key in ("model", "api_key", "base_url"):
                    if not tts_config.get(key):
                        raise ValueError(f"TTS config missing '{key}'")
                base_url = tts_config["base_url"]
                if not base_url.startswith(("http://", "https://")):
                    raise ValueError(f"TTS base_url invalid: {base_url}")
                if not urlparse(base_url).netloc:
                    raise ValueError(f"TTS base_url has no netloc: {base_url}")

                effective_voice = voice or tts_config.get("voice", "alloy")
                model = tts_config["model"]
                _ensure_audio_dir()
                audio_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
                audio_filename = f"narration_{audio_id}.mp3"
                audio_path = _AUDIO_DIR / audio_filename

                dashscope_prefixes = {"qwen-tts", "qwen-tts-latest", "qwen3-tts-flash", "qwen3-tts-flash-latest"}
                if any(model.startswith(m) for m in dashscope_prefixes):
                    audio_result = await _generate_audio_dashscope(
                        script=script, voice=effective_voice, api_key=tts_config["api_key"],
                        model=model, base_url=base_url, audio_path=audio_path,
                        audio_filename=audio_filename, audio_id=audio_id,
                    )
                else:
                    audio_result = await _generate_audio_openai(
                        script=_truncate_script(script, max_len=4096),
                        voice=effective_voice, tts_config=tts_config,
                        audio_path=audio_path, audio_filename=audio_filename, audio_id=audio_id,
                    )
                audio_url = audio_result["audio_url"]
                has_audio = True
            except Exception as exc:
                logger.error("narrate_node TTS failed: %s", exc)
                audio_error = str(exc)

    except Exception as exc:
        logger.warning("narrate_node failed: %s", exc)

    return {
        "script": script,
        "key_points": key_points,
        "has_audio": has_audio,
        "audio_url": audio_url,
        "streaming_events": [
            {
                "type": "result", "stage": "narrate",
                "script": script, "key_points": key_points,
                "has_audio": has_audio, "audio_url": audio_url,
                **({"audio_error": audio_error} if audio_error else {}),
            }
        ],
    }


__all__ = [
    "retrieve_context_node",
    "edit_node",
    "narrate_node",
]
