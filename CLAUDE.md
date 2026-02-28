# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepTutor is an AI-powered personalized learning assistant with a Python FastAPI backend and Next.js frontend. It uses a multi-agent architecture with RAG (Retrieval-Augmented Generation) for document-based learning, problem solving, research, and question generation.

## Setup

### Environment Configuration

Copy `.env.example` to `.env` and configure:

- **LLM**: `LLM_BINDING`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_HOST`
- **Embedding**: `EMBEDDING_BINDING`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_HOST`, `EMBEDDING_DIMENSION`
- **Optional**: `TTS_*` for text-to-speech, `SEARCH_PROVIDER` + `SEARCH_API_KEY` for web search
- **Ports**: `BACKEND_PORT` (default 8001), `FRONTEND_PORT` (default 3782)

Supported LLM providers: `openai`, `azure_openai`, `anthropic`, `deepseek`, `openrouter`, `groq`, `together`, `mistral`, `ollama`, `lm_studio`, `vllm`, `llama_cpp`

### Installation

```bash
# One-click install (backend + frontend)
python scripts/install_all.py

# Or separately
pip install -r requirements.txt
cd web && npm install
```

## Development Commands

### Backend

```bash
# Start API server (from project root)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload

# Or via CLI launcher (interactive menu)
python scripts/start.py

# Run tests
pytest tests/

# Run a single test file
pytest tests/agents/test_solve.py -v

# Lint and format
ruff check .
ruff format .

# Type check
mypy .

# Pre-commit hooks
pre-commit run --all-files
```

### Frontend

```bash
cd web
npm run dev          # Development server (port 3000 by default)
npm run build        # Production build
npm run lint         # ESLint
npm run i18n:check   # Validate i18n parity between en/zh locales
```

### Docker

```bash
docker compose up -d                                                          # Production
docker compose -f docker-compose.yml -f docker-compose.dev.yml up            # Development
```

## Architecture

### Directory Structure

- `src/agents/` — Seven specialized agents, each in its own subdirectory
- `src/api/` — FastAPI application; `main.py` registers all routers; `routers/` maps to agents; `utils/` handles history, progress, notebooks
- `src/services/` — Shared services: `llm/` (provider abstraction), `embedding/`, `rag/`, `config/`, `prompt/`, `search/`, `tts/`, `settings/`
- `src/config/settings.py` — Global settings singleton
- `src/logging/` — Unified logger (use `get_logger(name)`)
- `config/main.yaml` — System paths, tool configuration, logging, per-module agent params
- `config/agents.yaml` — Per-agent temperature, max_tokens, valid_tools
- `data/` — Runtime data: `knowledge_bases/` (vector DBs), `user/` (outputs per module)
- `web/` — Next.js frontend with App Router, i18n (en/zh), TailwindCSS

### Agent Architecture

All agents extend `src/agents/base_agent.py:BaseAgent`, which provides:
- LLM calls via `src/services/llm` (unified provider interface)
- Agent parameters loaded from `config/agents.yaml` via `get_agent_params()`
- Prompt loading via `src/services/prompt:PromptManager`
- Token tracking via `LLMStats`

The seven agent modules:
- **solve** — Dual-loop: Investigation Loop (InvestigateAgent → NoteAgent) then Solving Loop (ManagerAgent → SolveAgent → ToolAgent → ResponseAgent → PrecisionAgent)
- **research** — RAG + web search driven deep research pipeline
- **question** — Question generation with validation
- **guide** — Step-by-step guided learning
- **ideagen** — Research idea generation from knowledge points
- **co_writer** — Writing assistance
- **chat** — Conversational Q&A

### API Layer

`src/api/main.py` initializes FastAPI with:
- Lifespan handler that validates tool config consistency between `main.yaml` and `agents.yaml` at startup (startup fails if drift detected)
- Static file serving at `/api/outputs` → `data/user/`
- All routers prefixed `/api/v1/`
- WebSocket endpoints for streaming agent responses

### LLM Service

`src/services/llm/` abstracts multiple providers. Configuration is read from `.env` at runtime. The LLM client is a singleton initialized at API startup; it sets `OPENAI_API_KEY` in `os.environ` for LightRAG compatibility.

### RAG Pipeline

Two implementations available (selectable via config): Docling-based (default) and MinerU alternative. Knowledge bases are stored in `data/knowledge_bases/<kb_name>/`. The default KB is `ai_textbook`.

### Configuration System

`src/services/config.py:load_config_with_main(filename, project_root)` merges `config/main.yaml` with any module-specific YAML. Agent parameters (temperature, max_tokens) come from `config/agents.yaml` and are accessed via `get_agent_params(module, agent)`.

### Frontend

Next.js App Router with:
- i18n via `react-i18next`; translation files in `web/locales/en/` and `web/locales/zh/`
- React Context for global state
- WebSocket connections to backend for streaming
- Math rendering via `rehype-katex`; diagram rendering via `mermaid`

## Code Style

- Line length: 100 characters (Black + Ruff)
- Python: double quotes, space indentation
- Ruff rules enabled: E (pycodestyle), F (pyflakes), I (isort)
- McCabe complexity limit: 10
- `__init__.py` unused imports are allowed (re-exports)
