"""
Guide Module

LangGraph-based guided learning pipeline (no BaseAgent):
- locate_node:      Extract knowledge points via LangChain LLM
- interactive_node: Generate interactive HTML via LangChain LLM
- chat_node:        Answer user questions via LangChain LLM
- summary_node:     Generate learning summary via LangChain LLM
- fix_html_node:    Regenerate HTML (delegates to interactive_node)

All agent classes removed — logic inlined into lg_nodes.py.
"""
