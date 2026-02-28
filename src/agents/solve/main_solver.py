#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Solver - Problem-Solving System Controller

Based on Dual-Loop Architecture: Analysis Loop + Solve Loop
"""

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import traceback
from typing import Any

import yaml

from ...services.config import parse_language

# Memory (SolveChainStep used in _has_pending_tool_calls)
from .memory import SolveChainStep
from .utils import ConfigValidator, PerformanceMonitor, SolveAgentLogger
from .utils.display_manager import get_display_manager
from .utils.token_tracker import TokenTracker


class MainSolver:
    """Problem-Solving System Controller"""

    def __init__(
        self,
        config_path: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        language: str | None = None,
        kb_name: str = "ai_textbook",
        output_base_dir: str | None = None,
    ):
        """
        Initialize MainSolver with lightweight setup.
        Call ainit() to complete async initialization.

        Args:
            config_path: Config file path (default: config.yaml in current directory)
            api_key: API key (if not provided, read from environment)
            base_url: API URL (if not provided, read from environment)
            api_version: API version (if not provided, read from environment)
            language: Preferred language for prompts ("en"/"zh"/"cn")
            kb_name: Knowledge base name
            output_base_dir: Output base directory (optional, overrides config)
        """
        # Store initialization parameters
        self._config_path = config_path
        self._api_key = api_key
        self._base_url = base_url
        self._api_version = api_version
        self._language = language
        self._kb_name = kb_name
        self._output_base_dir = output_base_dir

        # Initialize with None - will be set in ainit()
        self.config = None
        self.api_key = None
        self.base_url = None
        self.api_version = None
        self.kb_name = kb_name
        self.logger = None
        self.monitor = None
        self.token_tracker = None

    async def ainit(self) -> None:
        """
        Complete the asynchronous second phase of MainSolver initialization.

        This class uses a two-phase initialization pattern:

        1. ``__init__`` performs only lightweight, synchronous setup and stores
           constructor arguments. Attributes such as ``config``, ``api_key``,
           ``base_url``, ``api_version``, ``logger``, ``monitor``, and
           ``token_tracker`` are intentionally left as ``None``.
        2. :meth:`ainit` performs all I/O-bound and asynchronous work required to
           make the instance fully usable (e.g., loading configuration, wiring up
           logging/monitoring, and preparing external-service clients).

        You **must** call and await this method exactly once after constructing
        ``MainSolver`` and **before** invoking any other methods that rely on
        configuration, logging, metrics, or API access. Using the object prior
        to calling :meth:`ainit` may result in attributes still being ``None``,
        which can lead to confusing runtime errors such as ``AttributeError``,
        misconfigured API calls, missing logs/metrics, or incorrect output paths.

        This async initialization pattern is used instead of performing all setup
        in ``__init__`` so that object construction remains fast and synchronous,
        while allowing potentially slow operations (disk I/O, network requests,
        validation) to be awaited explicitly by the caller in an async context.
        """
        config_path = self._config_path
        api_key = self._api_key
        base_url = self._base_url
        api_version = self._api_version
        kb_name = self._kb_name
        output_base_dir = self._output_base_dir
        language = self._language

        # Load config from config directory (main.yaml unified config)
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            # Load main.yaml (solve_config.yaml is optional and will be merged if exists)
            from ...services.config.loader import load_config_with_main_async

            full_config = await load_config_with_main_async("main.yaml", project_root)

            # Extract solve-specific config and build validator-compatible structure
            solve_config = full_config.get("solve", {})
            paths_config = full_config.get("paths", {})

            # Build config structure expected by ConfigValidator
            self.config = {
                "system": {
                    "output_base_dir": paths_config.get("solve_output_dir", "./data/user/solve"),
                    "save_intermediate_results": solve_config.get(
                        "save_intermediate_results", True
                    ),
                    "language": full_config.get("system", {}).get("language", "en"),
                },
                "agents": solve_config.get("agents", {}),
                "logging": full_config.get("logging", {}),
                "tools": full_config.get("tools", {}),
                "paths": paths_config,
                # Keep solve-specific settings accessible
                "solve": solve_config,
            }
        else:
            # If custom config path provided, load it directly (for backward compatibility)
            local_config = {}
            if Path(config_path).exists():
                try:

                    def load_local_config(path: str) -> dict:
                        with open(path, encoding="utf-8") as f:
                            return yaml.safe_load(f) or {}

                    local_config = await asyncio.to_thread(load_local_config, config_path)
                except Exception:
                    # Config loading warning will be handled by config_loader
                    pass
            self.config = local_config if isinstance(local_config, dict) else {}

        if self.config is None or not isinstance(self.config, dict):
            self.config = {}

        # Override system language from UI if provided
        if language:
            self.config.setdefault("system", {})
            self.config["system"]["language"] = parse_language(language)

        # Override output directory config
        if output_base_dir:
            if "system" not in self.config:
                self.config["system"] = {}
            self.config["system"]["output_base_dir"] = str(output_base_dir)

            # Note: log_dir and performance_log_dir are now in paths section from main.yaml
            # Only override if explicitly needed

        # Validate config
        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate(self.config)
        if not is_valid:
            raise ValueError(f"Config validation failed: {errors}")

        # API config
        if api_key is None or base_url is None or "llm" not in self.config:
            try:
                from ...services.llm.config import get_llm_config_async

                llm_config = await get_llm_config_async()
                if api_key is None:
                    api_key = llm_config.api_key
                if base_url is None:
                    base_url = llm_config.base_url
                if api_version is None:
                    api_version = getattr(llm_config, "api_version", None)

                # Ensure LLM config is populated in self.config for agents
                if "llm" not in self.config:
                    self.config["llm"] = {}

                # Update config with complete details (binding, model, etc.)
                from dataclasses import asdict

                self.config["llm"].update(asdict(llm_config))

            except ValueError as e:
                raise ValueError(f"LLM config error: {e!s}")

        # Check if API key is required
        # Local LLM servers (Ollama, LM Studio, etc.) don't need API keys
        from src.services.llm import is_local_llm_server

        if not api_key and not is_local_llm_server(base_url):
            raise ValueError("API key not set. Provide api_key param or set LLM_API_KEY in .env")

        # For local servers, use a placeholder key if none provided
        if not api_key and is_local_llm_server(base_url):
            api_key = "sk-no-key-required"

        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.kb_name = kb_name

        # Initialize logging system
        logging_config = self.config.get("logging", {})
        # Get log_dir from paths (user_log_dir from main.yaml) or logging config
        log_dir = (
            self.config.get("paths", {}).get("user_log_dir")
            or self.config.get("paths", {}).get("log_dir")
            or logging_config.get("log_dir")
        )
        self.logger = SolveAgentLogger(
            name="Solver",
            level=logging_config.get("level", "INFO"),
            log_dir=log_dir,
            console_output=logging_config.get("console_output", True),
            file_output=logging_config.get("save_to_file", True),
        )

        # Attach display manager for TUI and frontend status updates
        self.logger.display_manager = get_display_manager()

        # Initialize performance monitor (disabled by default - performance logging is deprecated)
        monitoring_config = self.config.get("monitoring", {})
        # Disable performance monitor by default to avoid creating performance directory
        self.monitor = PerformanceMonitor(
            enabled=False,
            save_dir=None,  # Disabled - performance logging is deprecated
        )

        # Initialize Token tracker
        self.token_tracker = TokenTracker(prefer_tiktoken=True)

        # Connect token_tracker to display_manager for real-time updates
        if self.logger.display_manager:
            self.token_tracker.set_on_usage_added_callback(
                self.logger.display_manager.update_token_stats
            )

        self.logger.section("Dual-Loop Solver Initializing")
        self.logger.info(f"Knowledge Base: {kb_name}")

        # Initialize Agents
        self._init_agents()

        self.logger.success("Solver ready")

    def _deep_merge(self, base: dict, update: dict) -> dict:
        """Deep merge two dictionaries"""
        if base is None:
            base = {}
        if update is None:
            update = {}

        result = base.copy() if base else {}
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _init_agents(self):
        """Agent classes removed — logic inlined into lg_nodes.py (LangGraph pipeline)."""
        self.logger.progress("Initializing agents (LangGraph mode)...")
        # All agent logic is now inlined in lg_nodes.py.
        # MainSolver.solve() is a legacy path; use the LangGraph WebSocket endpoint instead.
        self.investigate_agent = None
        self.note_agent = None
        self.manager_agent = None
        self.solve_agent = None
        self.tool_agent = None
        self.response_agent = None
        self.precision_answer_agent = None
        self.logger.info("  LangGraph nodes ready (no individual agent objects)")

    async def solve(self, question: str, verbose: bool = True) -> dict[str, Any]:
        """
        Main solving process - Dual-Loop Architecture

        Args:
            question: User question
            verbose: Whether to print detailed info

        Returns:
            dict: Solving result
        """
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = self.config.get("system", {}).get("output_base_dir", "./user/solve")
        output_dir = os.path.join(output_base_dir, f"solve_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Add task log file handler
        task_log_file = os.path.join(output_dir, "task.log")
        self.logger.add_task_log_handler(task_log_file)

        self.logger.section("Problem Solving Started")
        self.logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        self.logger.info(f"Output: {output_dir}")

        try:
            # Execute dual-loop pipeline
            result = await self._run_dual_loop_pipeline(question, output_dir)

            # Add metadata
            result["metadata"] = {
                "mode": "dual_loop",
                "timestamp": timestamp,
                "output_dir": output_dir,
            }

            # Save performance report
            if self.config.get("monitoring", {}).get("enabled", True):
                perf_report = self.monitor.generate_report()
                perf_file = os.path.join(output_dir, "performance_report.json")
                with open(perf_file, "w", encoding="utf-8") as f:
                    json.dump(perf_report, f, ensure_ascii=False, indent=2)
                self.logger.debug(f"Performance report saved: {perf_file}")

            # Output cost report
            if self.token_tracker:
                cost_summary = self.token_tracker.get_summary()
                if cost_summary["total_calls"] > 0:
                    cost_text = self.token_tracker.format_summary()
                    self.logger.info(f"\n{cost_text}")

                    cost_file = os.path.join(output_dir, "cost_report.json")
                    self.token_tracker.save(cost_file)
                    self.logger.debug(f"Cost report saved: {cost_file}")

                    self.token_tracker.reset()

            self.logger.success("Problem solving completed")
            self.logger.remove_task_log_handlers()

            return result

        except Exception as e:
            self.logger.error(f"Solving failed: {e!s}")
            self.logger.error(traceback.format_exc())
            self.logger.remove_task_log_handlers()
            raise

        finally:
            if hasattr(self, "logger"):
                self.logger.shutdown()

    async def _run_dual_loop_pipeline(self, question: str, output_dir: str) -> dict[str, Any]:
        """Delegate to LangGraph pipeline (agent classes removed — logic inlined in lg_nodes.py)."""
        from src.agents.solve.lg_graph import get_solve_graph
        from src.services.config import parse_language as _parse_language

        language = self.config.get("system", {}).get("language", "en")
        lang_code = _parse_language(language)

        graph = get_solve_graph()
        initial_state = {
            "question": question,
            "kb_name": self.kb_name,
            "language": lang_code,
            "output_dir": output_dir,
            "knowledge_chain": [],
            "analysis_iteration": 0,
            "max_analysis_iterations": 5,
            "analysis_should_stop": False,
            "new_knowledge_ids": [],
            "solve_steps": [],
            "current_step_index": 0,
            "solve_iteration": 0,
            "max_solve_iterations": 6,
            "finish_requested": False,
            "citations": [],
            "final_answer": "",
            "streaming_events": [],
        }

        self.logger.info("Pipeline: LangGraph dual-loop")
        final_state = await graph.ainvoke(initial_state)

        final_answer = final_state.get("final_answer", "")
        final_answer_file = Path(output_dir) / "final_answer.md"
        with open(final_answer_file, "w", encoding="utf-8") as f:
            f.write(final_answer)

        citations = [c.get("cite_id", "") for c in final_state.get("citations", [])]
        solve_steps = final_state.get("solve_steps", [])

        return {
            "question": question,
            "output_dir": output_dir,
            "final_answer": final_answer,
            "output_md": str(final_answer_file),
            "output_json": str(Path(output_dir) / "solve_chain.json"),
            "formatted_solution": final_answer,
            "citations": citations,
            "pipeline": "langgraph",
            "total_steps": len(solve_steps),
            "analysis_iterations": final_state.get("analysis_iteration", 0),
            "solve_steps": len(solve_steps),
            "metadata": {
                "coverage_rate": 1.0,
                "avg_confidence": 0.9,
                "total_steps": len(solve_steps),
            },
        }

    @staticmethod
    def _has_pending_tool_calls(step: SolveChainStep) -> bool:
        return any(call.status in {"pending", "running"} for call in step.tool_calls)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    async def test():
        solver = MainSolver(kb_name="ai_textbook")
        result = await solver.solve(question="What is linear convolution?", verbose=True)
        print(f"Output file: {result['output_md']}")

    asyncio.run(test())
