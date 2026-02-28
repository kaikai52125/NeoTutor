"""
Question Generation System

LangGraph-based question generation pipeline (no BaseAgent):
- retrieve_node: Knowledge base retrieval via LangChain LLM + RAG
- generate_node: Question generation via LangChain LLM
- relevance_node: Question-KB relevance analysis via LangChain LLM
- AgentCoordinator: Removed — logic lives in lg_nodes.py and exam_mimic.py
"""
