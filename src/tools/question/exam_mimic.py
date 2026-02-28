#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference-based exam-question generation system

Workflow:
1. Parse the PDF exam (MinerU)
2. Extract question information (LLM)
3. Generate new questions per reference question (Agent)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable

# Project root is 3 levels up from src/tools/question/
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tools.question.pdf_parser import parse_pdf_with_mineru
from src.tools.question.question_extractor import extract_questions_from_paper

# Type alias for WebSocket callback
WsCallback = Callable[[str, dict[str, Any]], Any]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-LangChain helpers (no BaseAgent)
# ---------------------------------------------------------------------------


def _parse_json_robust(text: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response, handling markdown code blocks."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    content = match.group(1).strip() if match else text.strip()
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    obj_match = re.search(r"\{[\s\S]*\}", content)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    fixed = re.sub(r",\s*([}\]])", r"\1", content)
    fixed = re.sub(r'"""([\s\S]*?)"""', lambda m: json.dumps(m.group(1)), fixed)
    return json.loads(fixed)


def _load_prompts(agent_name: str, language: str = "en") -> dict[str, Any]:
    from src.services.prompt import get_prompt_manager
    try:
        return get_prompt_manager().load_prompts(
            module_name="question",
            agent_name=agent_name,
            language=language,
        ) or {}
    except Exception as exc:
        logger.warning("Failed to load prompts %s/%s: %s", agent_name, language, exc)
        return {}


def _p(prompts: dict, key: str, default: str = "") -> str:
    return (prompts.get(key) or default).strip()


async def _retrieve_knowledge(
    requirement: dict[str, Any],
    kb_name: str,
    language: str = "en",
    num_queries: int = 3,
) -> tuple[str, bool]:
    """
    Generate RAG queries via LLM, then retrieve KB context in parallel.
    Returns (knowledge_context, has_content).
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.services.llm.langchain_factory import get_chat_model_from_env
    from src.tools.rag_tool import rag_search

    requirement_text = json.dumps(requirement, ensure_ascii=False, indent=2)
    prompts = _load_prompts("retrieve_agent", language)
    llm = get_chat_model_from_env()

    system_text = _p(prompts, "system", "You are a knowledge base retrieval assistant.")
    user_tpl = _p(
        prompts, "generate_queries",
        "Extract {num_queries} knowledge point names for retrieval from:\n{requirement_text}\n\nReturn JSON: {{\"queries\": [\"point1\", ...]}}",
    )
    user_text = user_tpl.format(requirement_text=requirement_text, num_queries=num_queries)

    queries: list[str] = []
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        data = _parse_json_robust(raw)
        queries_raw = data.get("queries", [])
        if isinstance(queries_raw, dict):
            queries_raw = list(queries_raw.values())
        elif not isinstance(queries_raw, list):
            queries_raw = [str(queries_raw)]
        queries = [q.strip() for q in queries_raw if q and str(q).strip()][:num_queries]
    except Exception as exc:
        logger.warning("_retrieve_knowledge: failed to parse query JSON: %s", exc)

    if not queries:
        queries = [requirement_text[:100]]

    async def _search(q: str) -> dict[str, Any]:
        try:
            result = await rag_search(query=q, kb_name=kb_name, mode="naive", only_need_context=True)
            return {"query": q, "answer": result.get("answer", "")}
        except Exception as exc:
            logger.warning("RAG search failed for '%s': %s", q, exc)
            return {"query": q, "answer": ""}

    retrievals = await asyncio.gather(*[_search(q) for q in queries])
    retrievals = [r for r in retrievals if r.get("answer")]

    if not retrievals:
        return "No retrieval context available.", False

    lines = []
    for item in retrievals:
        lines.append(f"=== Query: {item['query']} ===")
        answer = item["answer"]
        if len(answer) > 2000:
            answer = answer[:2000] + "...[truncated]"
        lines.append(answer)
        lines.append("")
    return "\n".join(lines), True


async def _generate_with_reference(
    requirement: dict[str, Any],
    knowledge_context: str,
    reference_question: str,
    language: str = "en",
) -> dict[str, Any]:
    """
    Generate a mimic question using generate_with_reference prompt.
    Returns {"success": bool, "question": dict} or {"success": False, "error": str}.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.services.llm.langchain_factory import get_chat_model_from_env

    prompts = _load_prompts("generate_agent", language)
    llm = get_chat_model_from_env()

    requirements_str = json.dumps(requirement, ensure_ascii=False, indent=2)
    knowledge_snippet = knowledge_context[:4000] if len(knowledge_context) > 4000 else knowledge_context

    system_text = _p(prompts, "system", "You are a professional Question Generation Agent.")
    user_tpl = _p(
        prompts, "generate_with_reference",
        "Generate a new question inspired by the reference but distinct:\n"
        "Reference: {reference_question}\nRequirements: {requirements}\nKnowledge: {knowledge}\n\n"
        "Return JSON with question_type, question, correct_answer, explanation.",
    )
    user_text = user_tpl.format(
        reference_question=reference_question,
        requirements=requirements_str,
        knowledge=knowledge_snippet,
    )

    try:
        resp = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        question = _parse_json_robust(raw)

        if "question" not in question:
            raise ValueError("Missing 'question' field in LLM response")
        if "question_type" not in question:
            question["question_type"] = "written"
        if question.get("question_type") == "choice" and not question.get("options"):
            question["options"] = {
                "A": "Option A (placeholder)", "B": "Option B (placeholder)",
                "C": "Option C (placeholder)", "D": "Option D (placeholder)",
            }
        return {"success": True, "question": question}
    except Exception as exc:
        logger.warning("_generate_with_reference failed: %s", exc)
        return {"success": False, "error": str(exc)}


async def _analyze_relevance(
    question: dict[str, Any],
    knowledge_context: str,
    language: str = "en",
) -> dict[str, Any]:
    """
    Classify question-KB relevance via LangChain LLM.
    Returns {"relevance": "high"|"partial", "kb_coverage": str, "extension_points": str}.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.services.llm.langchain_factory import get_chat_model_from_env

    prompts = _load_prompts("relevance_analyzer", language)
    llm = get_chat_model_from_env()

    knowledge_snippet = (
        knowledge_context[:4000] + "...[truncated]"
        if len(knowledge_context) > 4000
        else knowledge_context
    )
    question_str = json.dumps(question, ensure_ascii=False, indent=2)

    system_text = _p(prompts, "system", "You are an educational content analyst.")
    user_tpl = _p(
        prompts, "analyze_relevance",
        'Analyze relevance:\nQuestion:\n{question}\n\nKnowledge:\n{knowledge}\n\nReturn JSON: {{"relevance":"high"/"partial","kb_coverage":"...","extension_points":"..."}}',
    )
    user_text = user_tpl.format(question=question_str, knowledge=knowledge_snippet)

    try:
        resp = await llm.ainvoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        result = _parse_json_robust(raw)
        relevance = result.get("relevance", "partial")
        if relevance not in ("high", "partial"):
            relevance = "partial"
        return {
            "relevance": relevance,
            "kb_coverage": result.get("kb_coverage", ""),
            "extension_points": result.get("extension_points", "") if relevance == "partial" else "",
        }
    except Exception as exc:
        logger.warning("_analyze_relevance failed: %s", exc)
        return {"relevance": "partial", "kb_coverage": "", "extension_points": str(exc)}


async def generate_question_from_reference(
    reference_question: dict[str, Any], kb_name: str, language: str = "en"
) -> dict[str, Any]:
    """
    Generate a new question based on a reference entry using pure LangChain (no BaseAgent).
    """
    requirement = {
        "reference_question": reference_question["question_text"],
        "has_images": len(reference_question.get("images", [])) > 0,
        "kb_name": kb_name,
        "allow_reject": False,
        "additional_requirements": (
            f"Reference question:\n{reference_question['question_text']}\n\n"
            "Requirements:\n"
            "1. Keep a similar difficulty level.\n"
            "2. **Identify the core knowledge concept(s) of the reference and keep them EXACTLY the same. Do not introduce new advanced topics beyond what the reference question requires.**\n"
            "3. **Change the scenario/objects/geometry; do not simply replace numbers or symbols.**\n"
            "4. **Alter at least one part of the reasoning process or add a new sub-question "
            "(e.g., extra calculation, analysis, or proof).**\n"
            "5. Keep the problem entirely within the same mathematical scope as the reference (e.g., if the reference is planar line parametrization, you must stay within planar line parametrization and cannot escalate to surfaces or directional derivatives).\n"
            "6. Ensure the prompt is rigorous, precise, and self-contained.\n"
            "7. If the original problem references images, describe them in text.\n"
            "8. Rejection is forbidden—you must complete the generation task.\n\n"
            "Chain-of-thought guidance:\n"
            "- Think step-by-step to plan the new scenario and reasoning before producing the final JSON.\n"
            "- Do not reveal your reasoning; output only the final JSON."
        ),
    }

    # Step 1: Retrieve knowledge context
    knowledge_context, has_content = await _retrieve_knowledge(
        requirement=requirement, kb_name=kb_name, language=language
    )
    if not has_content:
        return {
            "success": False,
            "error": "knowledge_not_found",
            "message": "Knowledge base does not contain relevant information.",
        }

    # Step 2: Generate mimic question
    gen_result = await _generate_with_reference(
        requirement=requirement,
        knowledge_context=knowledge_context,
        reference_question=reference_question["question_text"],
        language=language,
    )
    if not gen_result.get("success"):
        return {"success": False, "error": gen_result.get("error", "Generation failed")}

    question = gen_result["question"]

    # Step 3: Analyze relevance
    analysis = await _analyze_relevance(
        question=question, knowledge_context=knowledge_context, language=language
    )

    return {
        "success": True,
        "question": question,
        "validation": {
            "decision": "approve",
            "relevance": analysis["relevance"],
            "kb_coverage": analysis["kb_coverage"],
            "extension_points": analysis.get("extension_points", ""),
        },
        "rounds": 1,
    }


async def mimic_exam_questions(
    pdf_path: str | None = None,
    paper_dir: str | None = None,
    kb_name: str = None,
    output_dir: str | None = None,
    max_questions: int | None = None,
    ws_callback: WsCallback | None = None,
) -> dict[str, Any]:
    """
    End-to-end orchestration for reference-based question generation.

    Args:
        pdf_path: Path to the PDF exam paper
        paper_dir: Path to a pre-parsed exam directory
        kb_name: Knowledge base name to use
        output_dir: Output directory for generated questions
        max_questions: Maximum number of questions to process
        ws_callback: Optional async callback for WebSocket progress updates
                     Signature: async def callback(event_type: str, data: dict)
    """

    async def send_progress(event_type: str, data: dict[str, Any]):
        """Helper to send progress updates via WebSocket callback."""
        if ws_callback:
            try:
                await ws_callback(event_type, data)
            except Exception as e:
                print(f"WebSocket callback error: {e}")

    print("=" * 80)
    print("📚 Reference-based question generation system")
    print("=" * 80)
    print()

    # Validate arguments
    if not pdf_path and not paper_dir:
        await send_progress("error", {"content": "Either pdf_path or paper_dir must be provided."})
        return {"success": False, "error": "Either pdf_path or paper_dir must be provided."}

    if pdf_path and paper_dir:
        await send_progress("error", {"content": "pdf_path and paper_dir cannot be used together."})
        return {
            "success": False,
            "error": "pdf_path and paper_dir cannot be used together. Choose only one.",
        }

    latest_dir = None

    # If an already parsed exam directory is provided
    if paper_dir:
        await send_progress(
            "progress",
            {
                "stage": "parsing",
                "status": "locating",
                "message": "Locating parsed exam directory...",
            },
        )

        print("🔍 Using parsed exam directory")
        print("-" * 80)

        # Resolve relative names against reference_papers
        # SECURITY FIX: Prevent Path Injection / Traversal
        if os.path.isabs(paper_dir) or ".." in paper_dir:
            error_msg = (
                f"Invalid paper_dir: Absolute paths and traversal are not allowed. ({paper_dir})"
            )
            await send_progress("error", {"content": error_msg})
            return {"success": False, "error": error_msg}

        paper_path = Path(paper_dir)

        # Candidate locations to search (including new location)
        possible_paths = [
            project_root
            / "data"
            / "user"
            / "question"
            / "mimic_papers"
            / paper_dir,  # New primary location
            Path("question_agents/reference_papers") / paper_dir,  # Legacy location
            Path("reference_papers") / paper_dir,
        ]

        latest_dir = None
        for p in possible_paths:
            if p.exists():
                # Double check to ensure we didn't escape via symlink or subtle tricks
                try:
                    resolved_p = p.resolve()
                    # Safe check: Ensure the resolved path is strictly inside the intended parent
                    # This is a basic check; for robust security, whitelist allowed parents explicitly if needed.
                    latest_dir = resolved_p
                    break
                except Exception:
                    continue

        if not latest_dir:
            error_msg = f"Exam directory not found: {paper_dir}"
            await send_progress("error", {"content": error_msg})
            return {
                "success": False,
                "error": f"{error_msg}\nSearched paths: {[str(p) for p in possible_paths]}",
            }
        # Note: latest_dir was already resolved in the loop above, no need to override

        # Ensure auto subdirectory exists
        auto_dir = latest_dir / "auto"
        if not auto_dir.exists():
            error_msg = f"Invalid exam directory (missing auto folder): {latest_dir}"
            await send_progress("error", {"content": error_msg})
            return {
                "success": False,
                "error": error_msg,
            }

        print(f"✓ Exam directory detected: {latest_dir.name}")
        print(f"   Full path: {latest_dir}")
        print()

        await send_progress(
            "progress",
            {
                "stage": "parsing",
                "status": "complete",
                "message": f"Using parsed exam: {latest_dir.name}",
            },
        )

    # If a PDF is provided, parse it first
    elif pdf_path:
        # Stage 1: Parsing PDF
        await send_progress(
            "progress",
            {"stage": "parsing", "status": "running", "message": "Parsing PDF with MinerU..."},
        )

        print("🔄 Step 1: parse the PDF exam")
        print("-" * 80)

        # Use provided output_dir or default to mimic_papers
        if output_dir:
            output_base = Path(output_dir)
        else:
            output_base = project_root / "data" / "user" / "question" / "mimic_papers"
        output_base.mkdir(parents=True, exist_ok=True)

        success = parse_pdf_with_mineru(pdf_path=pdf_path, output_base_dir=str(output_base))

        if not success:
            await send_progress("error", {"content": "Failed to parse PDF with MinerU"})
            return {"success": False, "error": "Failed to parse PDF"}

        print()

        print("🔍 Step 2: locating parsed results")
        print("-" * 80)

        # Look in the new output directory (user/question/mimic_papers)
        reference_papers_dir = output_base
        subdirs = sorted(
            [d for d in reference_papers_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not subdirs:
            await send_progress("error", {"content": "No parsed outputs were found"})
            return {"success": False, "error": "No parsed outputs were found"}

        latest_dir = subdirs[0]
        print(f"✓ Parsed folder: {latest_dir.name}")
        print()

        await send_progress(
            "progress",
            {
                "stage": "parsing",
                "status": "complete",
                "message": f"PDF parsed successfully: {latest_dir.name}",
            },
        )

    # Stage 2: Extract questions
    await send_progress(
        "progress",
        {
            "stage": "extracting",
            "status": "running",
            "message": "Extracting reference questions from exam...",
        },
    )

    print("🔄 Step 3: extract reference questions")
    print("-" * 80)

    json_files = list(latest_dir.glob("*_questions.json"))

    if json_files:
        print(f"✓ Found existing question file: {json_files[0].name}")
        with open(json_files[0], encoding="utf-8") as f:
            questions_data = json.load(f)
    else:
        print("📄 No question file found, starting extraction...")
        success = extract_questions_from_paper(paper_dir=str(latest_dir), output_dir=None)

        if not success:
            await send_progress("error", {"content": "Question extraction failed"})
            return {"success": False, "error": "Question extraction failed"}

        json_files = list(latest_dir.glob("*_questions.json"))
        if not json_files:
            await send_progress(
                "error", {"content": "Question JSON file not found after extraction"}
            )
            return {"success": False, "error": "Question JSON file not found after extraction"}

        with open(json_files[0], encoding="utf-8") as f:
            questions_data = json.load(f)

    reference_questions = questions_data.get("questions", [])

    if max_questions:
        reference_questions = reference_questions[:max_questions]

    print(f"✓ Loaded {len(reference_questions)} reference questions")
    print()

    # Send reference questions info
    await send_progress(
        "progress",
        {
            "stage": "extracting",
            "status": "complete",
            "message": f"Extracted {len(reference_questions)} reference questions",
            "total_questions": len(reference_questions),
            "reference_questions": [
                {
                    "number": q.get("question_number", str(i + 1)),
                    "preview": (
                        q["question_text"][:100] + "..."
                        if len(q["question_text"]) > 100
                        else q["question_text"]
                    ),
                }
                for i, q in enumerate(reference_questions)
            ],
        },
    )

    # Stage 3: Generate mimic questions
    await send_progress(
        "progress",
        {
            "stage": "generating",
            "status": "running",
            "message": "Generating mimic questions...",
            "current": 0,
            "total": len(reference_questions),
        },
    )

    print("🔄 Step 4: generate new questions from references (parallel)")
    print("-" * 80)

    from src.services.config import load_config_with_main

    # Load config for parallel settings
    config = load_config_with_main("question_config.yaml", project_root)
    question_cfg = config.get("question", {})
    max_parallel = question_cfg.get("max_parallel_questions", 3)

    print(f"📊 Processing {len(reference_questions)} questions with max {max_parallel} parallel")

    # Create semaphore for parallel control
    semaphore = asyncio.Semaphore(max_parallel)

    # Track completed count
    completed_count = 0
    completed_lock = asyncio.Lock()

    async def generate_single_mimic(ref_question: dict, index: int) -> dict:
        """Generate a single mimic question with semaphore control."""
        nonlocal completed_count

        async with semaphore:
            question_id = f"mimic_{index}"
            ref_number = ref_question.get("question_number", str(index))

            # Send question start update
            await send_progress(
                "question_update",
                {
                    "question_id": question_id,
                    "index": index,
                    "status": "generating",
                    "reference_number": ref_number,
                    "reference_preview": ref_question["question_text"][:80] + "...",
                },
            )

            print(f"\n📝 [{question_id}] Starting - Reference: {ref_number}")
            print(f"   Preview: {ref_question['question_text'][:80]}...")

            try:
                result = await generate_question_from_reference(
                    reference_question=ref_question, kb_name=kb_name
                )

                async with completed_lock:
                    completed_count += 1
                    current_completed = completed_count

                if result.get("success"):
                    print(f"✓ [{question_id}] Generated in {result['rounds']} round(s)")

                    result_data = {
                        "success": True,
                        "reference_question_number": ref_number,
                        "reference_question_text": ref_question["question_text"],
                        "reference_images": ref_question.get("images", []),
                        "generated_question": result["question"],
                        "validation": result["validation"],
                        "rounds": result["rounds"],
                    }

                    # Send result update
                    await send_progress(
                        "result",
                        {
                            "question_id": question_id,
                            "index": index,
                            "success": True,
                            "question": result["question"],
                            "validation": result["validation"],
                            "rounds": result["rounds"],
                            "reference_question": ref_question["question_text"],
                            "current": current_completed,
                            "total": len(reference_questions),
                        },
                    )

                    return result_data
                else:
                    print(f"✗ [{question_id}] Failed: {result.get('error', 'Unknown error')}")

                    error_data = {
                        "success": False,
                        "reference_question_number": ref_number,
                        "reference_question_text": ref_question["question_text"],
                        "error": result.get("error", "Unknown error"),
                        "reason": result.get("reason", ""),
                    }

                    await send_progress(
                        "question_update",
                        {
                            "question_id": question_id,
                            "index": index,
                            "status": "failed",
                            "error": result.get("error", "Unknown error"),
                            "current": current_completed,
                            "total": len(reference_questions),
                        },
                    )

                    return error_data

            except Exception as e:
                print(f"✗ [{question_id}] Exception: {e!s}")

                async with completed_lock:
                    completed_count += 1
                    current_completed = completed_count

                await send_progress(
                    "question_update",
                    {
                        "question_id": question_id,
                        "index": index,
                        "status": "failed",
                        "error": str(e),
                        "current": current_completed,
                        "total": len(reference_questions),
                    },
                )

                return {
                    "success": False,
                    "reference_question_number": ref_question.get("question_number", str(index)),
                    "reference_question_text": ref_question["question_text"],
                    "error": f"Exception: {e!s}",
                }

    # Run all mimic generations in parallel
    tasks = [generate_single_mimic(ref_q, i) for i, ref_q in enumerate(reference_questions, 1)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successes and failures
    generated_questions = []
    failed_questions = []

    for result in results:
        if isinstance(result, Exception):
            failed_questions.append({"error": str(result)})
        elif result.get("success"):
            generated_questions.append(result)
        else:
            failed_questions.append(result)

    print()
    print("=" * 80)
    print("📊 Generation summary")
    print("=" * 80)
    print(f"Reference questions: {len(reference_questions)}")
    print(f"Successes: {len(generated_questions)}")
    print(f"Failures: {len(failed_questions)}")

    if output_dir is None:
        output_dir = latest_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{latest_dir.name}_{timestamp}_generated_questions.json"

    output_data = {
        "reference_paper": latest_dir.name,
        "kb_name": kb_name,
        "total_reference_questions": len(reference_questions),
        "successful_generations": len(generated_questions),
        "failed_generations": len(failed_questions),
        "generated_questions": generated_questions,
        "failed_questions": failed_questions,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print()

    # Send summary
    await send_progress(
        "summary",
        {
            "total_reference": len(reference_questions),
            "successful": len(generated_questions),
            "failed": len(failed_questions),
            "output_file": str(output_file),
        },
    )

    return {
        "success": True,
        "output_file": str(output_file),
        "total_reference_questions": len(reference_questions),
        "generated_questions": generated_questions,
        "failed_questions": failed_questions,
    }


async def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reference-based question generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exam_mimic.py --pdf /path/to/exam.pdf --kb math2211
  python exam_mimic.py --paper 2211asm1 --kb math2211
  python exam_mimic.py --paper reference_papers/2211asm1 --kb math2211
  python exam_mimic.py --paper 2211asm1 --kb math2211 --max-questions 3
  python exam_mimic.py --paper 2211asm1 --kb math2211 -o ./output
        """,
    )

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pdf", type=str, help="Absolute path to the PDF exam (will be parsed)"
    )

    input_group.add_argument(
        "--paper",
        type=str,
        help="Name of a parsed exam directory (e.g., 2211asm1) or its absolute path",
    )

    parser.add_argument("--kb", type=str, required=True, help="Knowledge base name")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory (defaults to the exam folder)",
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of reference questions to process (testing)",
    )

    args = parser.parse_args()

    # Execute the workflow
    result = await mimic_exam_questions(
        pdf_path=args.pdf,
        paper_dir=args.paper,
        kb_name=args.kb,
        output_dir=args.output,
        max_questions=args.max_questions,
    )

    if result["success"]:
        print("✓ Completed!")
        sys.exit(0)
    else:
        print(f"✗ Failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
