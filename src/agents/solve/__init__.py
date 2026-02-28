#!/usr/bin/env python
"""
Solve Agent System - Dual-Loop Architecture
Analysis Loop + Solve Loop

Agent classes removed — logic inlined into lg_nodes.py.
"""

from pathlib import Path
import sys

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.logging import Logger, get_logger

from .utils import (
    ConfigValidator,
    PerformanceMonitor,
)

# Backwards compatibility
SolveAgentLogger = Logger

# Memory system
from .memory import (
    InvestigateMemory,
    KnowledgeItem,
    Reflections,
    SolveChainStep,
    SolveMemory,
    ToolCallRecord,
)

# Main controller
from .main_solver import MainSolver

# Session management
from .session_manager import SolverSessionManager, get_solver_session_manager

__all__ = [
    # Logging
    "Logger",
    "get_logger",
    "SolveAgentLogger",
    "PerformanceMonitor",
    "ConfigValidator",
    # Memory system
    "InvestigateMemory",
    "KnowledgeItem",
    "Reflections",
    "SolveMemory",
    "SolveChainStep",
    "ToolCallRecord",
    # Main Controller
    "MainSolver",
    # Session Management
    "SolverSessionManager",
    "get_solver_session_manager",
]
