import asyncio
import base64
from datetime import datetime
from pathlib import Path
import re
import sys
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agents.question import AgentCoordinator
from src.api.utils.history import ActivityType, history_manager
from src.api.utils.log_interceptor import LogInterceptor
from src.api.utils.task_id_manager import TaskIDManager
from src.tools.question import mimic_exam_questions
from src.utils.document_validator import DocumentValidator
from src.utils.error_utils import format_exception_message

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.logging import get_logger
from src.services.config import load_config_with_main
from src.services.llm.config import get_llm_config
from src.services.settings.interface_settings import get_ui_language

# Setup module logger with unified logging system (from config)
project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("question_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("QuestionAPI", log_dir=log_dir)

router = APIRouter()

# Output directory for mimic mode - use data/user/question
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MIMIC_OUTPUT_DIR = PROJECT_ROOT / "data" / "user" / "question" / "mimic_papers"


@router.websocket("/mimic")
async def websocket_mimic_generate(websocket: WebSocket):
    """
    WebSocket endpoint for mimic exam paper question generation.

    Supports two modes:
    1. Upload PDF directly via WebSocket (base64 encoded)
    2. Use a pre-parsed paper directory path

    Message format for PDF upload:
    {
        "mode": "upload",
        "pdf_data": "base64_encoded_pdf_content",
        "pdf_name": "exam.pdf",
        "kb_name": "knowledge_base_name",
        "max_questions": 5  // optional
    }

    Message format for pre-parsed:
    {
        "mode": "parsed",
        "paper_path": "directory_name",
        "kb_name": "knowledge_base_name",
        "max_questions": 5  // optional
    }
    """
    await websocket.accept()

    pusher_task = None
    original_stdout = sys.stdout

    try:
        # 1. Wait for config
        data = await websocket.receive_json()
        mode = data.get("mode", "parsed")  # "upload" or "parsed"
        kb_name = data.get("kb_name", "ai_textbook")
        max_questions = data.get("max_questions")

        logger.info(f"Starting mimic generation (mode: {mode}, kb: {kb_name})")

        # 2. Setup Log Queue
        log_queue = asyncio.Queue()

        async def log_pusher():
            while True:
                entry = await log_queue.get()
                try:
                    await websocket.send_json(entry)
                except Exception:
                    break
                log_queue.task_done()

        pusher_task = asyncio.create_task(log_pusher())

        # 3. Stdout interceptor for capturing prints
        # ANSI escape sequence pattern for stripping color codes
        ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

        class StdoutInterceptor:
            def __init__(self, queue, original):
                self.queue = queue
                self.original_stdout = original
                self._closed = False

            def write(self, message):
                if self._closed:
                    return
                # Write to terminal first (with ANSI codes for color)
                try:
                    self.original_stdout.write(message)
                except Exception:
                    pass
                # Strip ANSI escape codes before sending to frontend
                clean_message = ANSI_ESCAPE_PATTERN.sub("", message).strip()
                # Then send to frontend (non-blocking)
                if clean_message:
                    try:
                        self.queue.put_nowait(
                            {
                                "type": "log",
                                "content": clean_message,
                                "timestamp": asyncio.get_event_loop().time(),
                            }
                        )
                    except (asyncio.QueueFull, RuntimeError):
                        pass

            def flush(self):
                if not self._closed:
                    try:
                        self.original_stdout.flush()
                    except Exception:
                        pass

            def close(self):
                """Mark interceptor as closed to prevent further writes."""
                self._closed = True

        interceptor = StdoutInterceptor(log_queue, original_stdout)
        sys.stdout = interceptor

        try:
            await websocket.send_json(
                {"type": "status", "stage": "init", "content": "Initializing..."}
            )

            pdf_path = None
            paper_dir = None

            # Handle PDF upload mode
            if mode == "upload":
                pdf_data = data.get("pdf_data")
                pdf_name = data.get("pdf_name", "exam.pdf")

                if not pdf_data:
                    await websocket.send_json(
                        {"type": "error", "content": "PDF data is required for upload mode"}
                    )
                    return

                # Decode PDF data first to check size
                try:
                    pdf_bytes = base64.b64decode(pdf_data)
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "content": f"Invalid base64 PDF data: {e}"}
                    )
                    return

                # Pre-validate filename and file size before writing
                try:
                    safe_name = DocumentValidator.validate_upload_safety(
                        pdf_name, len(pdf_bytes), {".pdf"}
                    )
                except ValueError as e:
                    await websocket.send_json({"type": "error", "content": str(e)})
                    return

                # Create batch directory for this mimic session
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_stem = Path(safe_name).stem
                batch_dir = MIMIC_OUTPUT_DIR / f"mimic_{timestamp}_{pdf_stem}"
                batch_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded PDF in batch directory
                pdf_path = batch_dir / safe_name

                await websocket.send_json(
                    {"type": "status", "stage": "upload", "content": f"Saving PDF: {safe_name}"}
                )

                # Write the validated PDF bytes
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)

                # Additional validation (file readability, etc.)
                try:
                    DocumentValidator.validate_file(pdf_path)
                except (ValueError, FileNotFoundError, PermissionError) as e:
                    # Clean up invalid or inaccessible file
                    pdf_path.unlink(missing_ok=True)
                    await websocket.send_json({"type": "error", "content": str(e)})
                    return

                await websocket.send_json(
                    {
                        "type": "status",
                        "stage": "parsing",
                        "content": "Parsing PDF exam paper (MinerU)...",
                    }
                )
                logger.info(f"Saved and validated uploaded PDF to: {pdf_path}")

                # Pass batch_dir as output directory
                pdf_path = str(pdf_path)
                output_dir = str(batch_dir)

            elif mode == "parsed":
                paper_path = data.get("paper_path")
                if not paper_path:
                    await websocket.send_json(
                        {"type": "error", "content": "paper_path is required for parsed mode"}
                    )
                    return
                paper_dir = paper_path

                # Create batch directory for parsed mode too
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_dir = MIMIC_OUTPUT_DIR / f"mimic_{timestamp}_{Path(paper_path).name}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                output_dir = str(batch_dir)

            else:
                await websocket.send_json({"type": "error", "content": f"Unknown mode: {mode}"})
                return

            # Create WebSocket callback for real-time progress updates
            async def ws_callback(event_type: str, data: dict):
                """Send progress updates to the frontend via WebSocket."""
                try:
                    message = {"type": event_type, **data}
                    await websocket.send_json(message)
                except Exception as e:
                    logger.debug(f"WebSocket send failed: {e}")

            # Run the complete mimic workflow with callback
            await websocket.send_json(
                {
                    "type": "status",
                    "stage": "processing",
                    "content": "Executing question generation workflow...",
                }
            )

            result = await mimic_exam_questions(
                pdf_path=pdf_path,
                paper_dir=paper_dir,
                kb_name=kb_name,
                output_dir=output_dir,
                max_questions=max_questions,
                ws_callback=ws_callback,
            )

            if result.get("success"):
                # Results are already sent via ws_callback during generation
                # Just send the final complete signal
                total_ref = result.get("total_reference_questions", 0)
                generated = result.get("generated_questions", [])
                failed = result.get("failed_questions", [])

                logger.success(
                    f"Mimic generation complete: {len(generated)} succeeded, {len(failed)} failed"
                )

                try:
                    await websocket.send_json({"type": "complete"})
                except (RuntimeError, WebSocketDisconnect):
                    logger.debug("WebSocket closed before complete signal could be sent")
            else:
                error_msg = result.get("error", "Unknown error")
                try:
                    await websocket.send_json({"type": "error", "content": error_msg})
                except (RuntimeError, WebSocketDisconnect):
                    pass
                logger.error(f"Mimic generation failed: {error_msg}")

        finally:
            # Close interceptor and restore stdout
            if "interceptor" in locals():
                interceptor.close()
            sys.stdout = original_stdout

    except WebSocketDisconnect:
        logger.debug("Client disconnected during mimic generation")
    except Exception as e:
        logger.exception("Mimic generation error")
        error_msg = format_exception_message(e)
        try:
            await websocket.send_json({"type": "error", "content": error_msg})
        except Exception:
            pass
    finally:
        # Ensure stdout is always restored
        sys.stdout = original_stdout

        # Clean up pusher task
        if pusher_task:
            try:
                pusher_task.cancel()
                await pusher_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
            except Exception:
                pass

        # Drain any remaining items in the queue
        try:
            while not log_queue.empty():
                log_queue.get_nowait()
        except Exception:
            pass

        # Close WebSocket
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/generate")
async def websocket_question_generate(websocket: WebSocket):
    """Question generation WebSocket — powered by LangGraph 1.0.9."""
    # Delegate directly to the LangGraph implementation
    await websocket_question_generate_lg(websocket)

# (original implementation kept below as _websocket_question_generate_legacy)
async def _websocket_question_generate_legacy(websocket: WebSocket):
    await websocket.accept()

    # Get task ID manager
    task_manager = TaskIDManager.get_instance()

    try:
        # 1. Wait for config
        data = await websocket.receive_json()
        requirement = data.get("requirement")
        kb_name = data.get("kb_name", "ai_textbook")
        count = data.get("count", 1)

        if not requirement:
            try:
                await websocket.send_json({"type": "error", "content": "Requirement is required"})
            except (RuntimeError, WebSocketDisconnect):
                pass
            return

        # Generate task ID
        task_key = f"question_{kb_name}_{hash(str(requirement))}"
        task_id = task_manager.generate_task_id("question_gen", task_key)

        # Send task ID to frontend
        try:
            await websocket.send_json({"type": "task_id", "task_id": task_id})
        except (RuntimeError, WebSocketDisconnect):
            logger.debug("WebSocket closed, cannot send task_id")
            return

        logger.info(
            f"[{task_id}] Starting question generation: {requirement.get('knowledge_point', 'Unknown')}"
        )

        # 2. Initialize Coordinator
        # Define unified output directory (DeepTutor/data/user/question)
        root_dir = Path(__file__).parent.parent.parent.parent
        output_base = root_dir / "data" / "user" / "question"

        try:
            llm_config = get_llm_config()
            api_key = llm_config.api_key
            base_url = llm_config.base_url
            api_version = getattr(llm_config, "api_version", None)
        except Exception:
            api_key = None
            base_url = None
            api_version = None

        coordinator = AgentCoordinator(
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            kb_name=kb_name,
            language=get_ui_language(default=config.get("system", {}).get("language", "en")),
            max_rounds=10,
            output_dir=str(output_base),
        )

        # 3. Setup Log Queue for WebSocket streaming
        log_queue = asyncio.Queue()

        # WebSocket callback for coordinator to send structured updates
        async def ws_callback(data: dict):
            try:
                await log_queue.put(data)
            except Exception:
                pass

        coordinator.set_ws_callback(ws_callback)

        # 4. Define background pusher for logs
        async def log_pusher():
            while True:
                entry = await log_queue.get()
                try:
                    await websocket.send_json(entry)
                except Exception:
                    break
                log_queue.task_done()

        pusher_task = asyncio.create_task(log_pusher())

        # 5. Setup LogInterceptor for capturing logger output (same as solve.py)
        # Get the coordinator's logger to intercept
        target_logger = coordinator.logger.logger
        interceptor = LogInterceptor(target_logger, log_queue)

        # 6. Run Generation with LogInterceptor
        try:
            with interceptor:
                try:
                    await websocket.send_json({"type": "status", "content": "started"})
                except (RuntimeError, WebSocketDisconnect):
                    logger.debug("WebSocket closed, stopping question generation")
                    return

                # Use custom mode generation (new streamlined flow)
                logger.info(f"Starting custom mode generation for {count} question(s)")

                # Use the new custom generation method
                batch_result = await coordinator.generate_questions_custom(
                    requirement=requirement,
                    num_questions=count,
                )

                # Results are already sent via WebSocket callbacks in the coordinator
                # Just need to save to history for successful results
                for result in batch_result.get("results", []):
                    # Save to history
                    history_manager.add_entry(
                        activity_type=ActivityType.QUESTION,
                        title=f"{requirement.get('knowledge_point', 'Question')} ({requirement.get('question_type')})",
                        content={
                            "requirement": requirement,
                            "question": result.get("question", {}),
                            "validation": result.get("validation", {}),
                            "kb_name": kb_name,
                        },
                        summary=result.get("question", {}).get("question", "")[:100],
                    )

                # Send final token stats
                try:
                    await websocket.send_json(
                        {"type": "token_stats", "stats": coordinator.token_stats}
                    )
                except (RuntimeError, WebSocketDisconnect):
                    logger.debug("WebSocket closed, stopping question generation")

                # Send batch summary
                try:
                    await websocket.send_json(
                        {
                            "type": "batch_summary",
                            "requested": batch_result.get("requested", count),
                            "completed": batch_result.get("completed", 0),
                            "failed": batch_result.get("failed", 0),
                            "plan": batch_result.get("plan", {}),
                        }
                    )
                except (RuntimeError, WebSocketDisconnect):
                    pass

                if not batch_result.get("success"):
                    logger.warning(
                        f"Question generation had failures: {batch_result.get('failed', 0)} failed"
                    )

                # Wait for any pending messages in the queue to be sent
                # Give the pusher a moment to process remaining messages
                await asyncio.sleep(0.1)
                while not log_queue.empty():
                    await asyncio.sleep(0.05)

                # Send complete signal
                try:
                    await websocket.send_json({"type": "complete"})
                    logger.info(f"[{task_id}] Question generation completed")
                    task_manager.update_task_status(task_id, "completed")
                except (RuntimeError, WebSocketDisconnect):
                    logger.debug("WebSocket closed, cannot send complete signal")

        except Exception as e:
            error_msg = format_exception_message(e)
            error_traceback = traceback.format_exc()
            logger.error(f"Question generation error: {error_msg}")
            logger.error(f"Error traceback:\n{error_traceback}")

            # Log additional context if available
            try:
                if "result" in locals():
                    logger.error(
                        f"Result type: {type(result)}, result keys: {result.keys() if isinstance(result, dict) else 'N/A'}"
                    )
                    if isinstance(result, dict) and "validation" in result:
                        validation = result["validation"]
                        logger.error(f"Validation type: {type(validation)}")
                        if isinstance(validation, dict):
                            logger.error(f"Validation keys: {validation.keys()}")
                            logger.error(
                                f"Issues type: {type(validation.get('issues'))}, value: {validation.get('issues')}"
                            )
                            logger.error(
                                f"Suggestions type: {type(validation.get('suggestions'))}, value: {validation.get('suggestions')}"
                            )
            except Exception as context_error:
                logger.warning(f"Failed to log error context: {context_error}")

            try:
                await websocket.send_json({"type": "error", "content": error_msg})
            except (RuntimeError, WebSocketDisconnect):
                logger.debug("WebSocket closed, cannot send error message")
            task_manager.update_task_status(task_id, "error", error=error_msg)

        finally:
            pusher_task.cancel()
            try:
                await pusher_task
            except asyncio.CancelledError:
                pass
            await websocket.close()

    except WebSocketDisconnect:
        logger.debug("Client disconnected")
    except Exception as e:
        error_msg = format_exception_message(e)
        logger.error(f"WebSocket error: {error_msg}")


# =============================================================================
# LangGraph WebSocket Endpoint (parallel testing — replaces /generate after validation)
# =============================================================================


@router.websocket("/generate/lg")
async def websocket_question_generate_lg(websocket: WebSocket):
    """
    LangGraph-based Question Generation endpoint.

    Drop-in replacement for /generate using LangGraph 1.0.9 + LangChain 1.2.x.
    Graph: START → retrieve_node → generate_node → relevance_node → END

    Message format (same as /generate):
    {
        "requirement": dict,     # Required: {knowledge_point, question_type, difficulty, ...}
        "kb_name": str,          # Default: ai_textbook
        "count": int,            # Number of questions to generate (default: 1)
        "focus": str             # Optional focus hint
    }
    """
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        requirement = data.get("requirement")
        if not requirement:
            await websocket.send_json({"type": "error", "content": "Requirement is required"})
            await websocket.close()
            return

        kb_name: str = data.get("kb_name", "ai_textbook") or "ai_textbook"
        count: int = int(data.get("count", 1))
        focus: str = data.get("focus", "") or ""
        language: str = get_ui_language(
            default=config.get("system", {}).get("language", "en")
        )

        import uuid as _uuid
        run_id = str(_uuid.uuid4())
        await websocket.send_json({"type": "task_id", "task_id": run_id})
        await websocket.send_json({"type": "status", "content": "started"})

        logger.info(
            f"LangGraph Question request: kb={kb_name}, count={count}, run_id={run_id}"
        )

        from src.agents.question.lg_graph import get_question_graph
        from src.api.utils.langgraph_ws_adapter import stream_graph_to_websocket

        graph = get_question_graph()
        initial_state = {
            "kb_name": kb_name,
            "language": language,
            "num_questions": count,
            "requirement": requirement,
            "focus": focus,
            "knowledge_context": "",
            "retrieval_queries": [],
            "generated_questions": [],
            "relevance_results": [],
            "streaming_events": [],
        }

        final_state = await stream_graph_to_websocket(
            graph=graph,
            initial_state=initial_state,
            websocket=websocket,
            config={},
        )

        questions = final_state.get("relevance_results", []) or \
                    final_state.get("generated_questions", [])

        # Send each question as a "result" message (frontend adds questions via "result" type)
        for i, q in enumerate(questions):
            await websocket.send_json({
                "type": "result",
                "question_id": f"q_{i + 1}",
                "index": i,
                "question": q,
                "validation": {
                    "relevance": q.get("relevance", "high"),
                    "kb_coverage": q.get("kb_coverage", ""),
                },
                "rounds": 1,
                "extended": q.get("relevance") == "partial",
            })
            # Save each question to history
            try:
                history_manager.add_entry(
                    activity_type=ActivityType.QUESTION,
                    title=f"{requirement.get('knowledge_point', 'Question')} ({requirement.get('question_type', '')})",
                    content={"requirement": requirement, "question": q, "kb_name": kb_name},
                    summary=str(q.get("question", ""))[:100],
                )
            except Exception as hist_exc:
                logger.warning(f"Failed to save question history: {hist_exc}")

        await websocket.send_json({
            "type": "batch_summary",
            "requested": count,
            "completed": len(questions),
            "failed": max(0, count - len(questions)),
        })
        await websocket.send_json({"type": "complete"})

        logger.info(
            f"LangGraph Question completed: run_id={run_id}, {len(questions)} questions"
        )

    except WebSocketDisconnect:
        logger.debug("Client disconnected from LangGraph question")
    except Exception as exc:
        logger.error(f"LangGraph Question error: {exc}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
