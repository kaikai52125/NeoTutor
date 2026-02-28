import asyncio
import base64
from datetime import datetime
from pathlib import Path
import re
import sys
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.utils.history import ActivityType, history_manager
from src.tools.question import mimic_exam_questions
from src.utils.document_validator import DocumentValidator
from src.utils.error_utils import format_exception_message

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.logging import get_logger
from src.services.config import load_config_with_main
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




# =============================================================================
# LangGraph WebSocket Endpoint
# =============================================================================


@router.websocket("/generate")
async def websocket_question_generate(websocket: WebSocket):
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
