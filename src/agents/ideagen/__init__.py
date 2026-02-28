"""
IdeaGen Module

LangGraph-based research idea generation pipeline (no BaseAgent):
- extract_node:       Extract knowledge points from notebook records
- loose_filter_node:  Remove obviously unsuitable knowledge points
- explore_node:       Generate research ideas per knowledge point
- strict_filter_node: Keep only high-quality idea candidates
- statement_node:     Generate final markdown research statements

All agent classes removed — logic inlined into lg_nodes.py.
"""
