# -*- coding: utf-8 -*-
"""
LangChain ChatModel Factory
============================

Builds LangChain 1.x BaseChatModel instances from the project's LLM configuration.
This is the entry point for all LangGraph nodes to obtain a ChatModel.

Supports the same `binding` identifiers as the existing factory.py:
  openai, azure_openai, anthropic, deepseek, openrouter, groq, together,
  mistral, ollama, lm_studio, vllm, llama_cpp

NOTE: The original src/services/llm/factory.py is preserved and still used by the
RAG pipeline (LightRAG). This file adds LangChain-native support on top.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel


def build_chat_model(
    binding: str,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    api_version: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> BaseChatModel:
    """
    Build a LangChain 1.x BaseChatModel based on the provider binding.

    Args:
        binding:     Provider type string (same identifiers as original factory.py).
        model:       Model name/identifier.
        api_key:     API key (None for local models).
        base_url:    API endpoint URL.
        api_version: API version string (Azure OpenAI only).
        temperature: Sampling temperature.
        max_tokens:  Maximum tokens to generate.

    Returns:
        A configured BaseChatModel instance with streaming=True.
    """
    if binding == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore[import]

        return ChatAnthropic(
            model=model,
            api_key=api_key,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif binding == "azure_openai":
        from langchain_openai import AzureChatOpenAI  # type: ignore[import]

        return AzureChatOpenAI(
            azure_deployment=model,
            azure_endpoint=base_url,
            api_key=api_key,  # type: ignore[arg-type]
            api_version=api_version or "2024-02-01",
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )

    else:
        # Covers: openai / deepseek / openrouter / groq / together / mistral
        #         / ollama / lm_studio / vllm / llama_cpp
        # All expose an OpenAI-compatible /v1/chat/completions endpoint.
        from langchain_openai import ChatOpenAI  # type: ignore[import]

        return ChatOpenAI(
            model=model,
            api_key=api_key or "no-key-required",  # type: ignore[arg-type]
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )


def get_chat_model_from_env(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Build a BaseChatModel from the project's active LLM configuration (.env or
    unified config service). This is the primary entry point for LangGraph nodes.

    Args:
        temperature: Override temperature (falls back to config value if None).
        max_tokens:  Override max_tokens (falls back to config value if None).

    Returns:
        A configured BaseChatModel ready for use in LangGraph nodes.
    """
    from src.services.llm.config import get_llm_config

    cfg = get_llm_config()

    return build_chat_model(
        binding=cfg.binding,
        model=cfg.model,
        api_key=cfg.api_key if cfg.api_key else None,
        base_url=cfg.base_url,
        api_version=cfg.api_version,
        temperature=temperature if temperature is not None else cfg.temperature,
        max_tokens=max_tokens if max_tokens is not None else cfg.max_tokens,
    )


__all__ = [
    "build_chat_model",
    "get_chat_model_from_env",
]
