# DeepTutor → LangChain/LangGraph 全量重构计划

## 实施进度（Implementation Status）

| 阶段 | 状态 | 新建文件 | 修改文件 |
|------|------|---------|---------|
| 阶段1：基础设施 | ✅ 完成 | `langchain_factory.py`, `langchain_tools.py`, `langgraph_ws_adapter.py` | `requirements.txt` |
| 阶段2：Chat | ✅ 完成 | `chat/lg_state.py`, `lg_nodes.py`, `lg_graph.py` | `routers/chat.py` (+`/chat/lg`) |
| 阶段3：Research | ✅ 完成 | `research/lg_state.py`, `lg_nodes.py`, `lg_graph.py` | `routers/research.py` (+`/run/lg`) |
| 阶段4：Question/IdeaGen/CoWriter | ✅ 完成 | 各模块 `lg_*.py` × 3 | 各 router (+`/generate/lg`, `/edit/lg`, `/narrate/lg`) |
| 阶段5：Guide | ✅ 完成 | `guide/lg_state.py`, `lg_nodes.py`, `lg_graph.py` | `routers/guide.py` (+多个 `/lg` 端点) |
| 阶段6：Solve | ✅ 完成 | `solve/lg_state.py`, `lg_nodes.py`, `lg_graph.py` | `routers/solve.py` (+`/solve/lg`) |
| 阶段7：切换主端点 | ⏳ 待验证 | — | 验证通过后替换原端点 |

**共新建文件：** 21 个 `lg_*.py` + 3 个基础设施文件 = **24 个新文件**
**共修改文件：** `requirements.txt` + 7 个 API router = **8 个修改文件**
**原有代码：** 全部保留，零删除

---


## Context

DeepTutor 是一个 AI 驱动的个性化学习助手，当前使用自定义 BaseAgent 继承体系 + 手动 for 循环控制流 + 自定义 Memory 状态管理。重构目标：

- 使用 **LangChain 1.2.x + LangGraph 1.0.9**（项目已声明依赖，但未使用）替换核心架构
- 全量重构 7 个 Agent 模块（solve、research、guide、question、ideagen、co_writer、chat）
- 保留原有 RAG 后端（LightRAG/RagAnything）不变，仅做接口适配
- 保持 FastAPI API 完全兼容（路由路径、WebSocket 消息格式不变，前端无需改动）

## 已确认的版本（来自 requirements.txt）

```
langchain-core==1.2.15
langchain==1.2.10
langgraph==1.0.9
langchain-openai>=0.3.0
langchain-community==0.3.0
langchain-anthropic（需新增）
langgraph-checkpoint-sqlite（需新增）
```

---

## 架构对比

| 原始组件 | 新 LangGraph 1.x 组件 |
|---------|----------------------|
| `BaseAgent.call_llm()` | `ChatOpenAI.ainvoke(messages)` |
| `BaseAgent.stream_llm()` | `graph.astream_events(state, version="v2")` |
| `InvestigateMemory` | `SolveState.knowledge_chain: list[KnowledgeItem]` |
| `SolveMemory` | `SolveState.solve_steps: list[SolveStep]` |
| `CitationMemory` | `SolveState.citations: list[CitationRecord]` |
| `DynamicTopicQueue` | `ResearchState.topic_blocks` + 自定义 Reducer |
| `SessionManager` (Chat) | LangGraph `MemorySaver/SqliteSaver` checkpointer |
| `MainSolver._run_dual_loop_pipeline` | `SolveGraph` StateGraph（节点+条件边）|
| `ResearchPipeline.run` | `ResearchGraph` StateGraph（三阶段节点）|
| `log_queue + pusher_task` | `astream_events(version="v2")` + WebSocket 适配器 |

---

## LangGraph 1.0.9 / LangChain 1.2.x 关键 API

### LangGraph 1.x StateGraph 写法

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver   # 内置内存 checkpointer

# 1. 定义状态（TypedDict）
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class MyState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 内置消息 reducer
    custom_field: str

# 2. 定义节点函数（接收 state，返回更新字典）
async def my_node(state: MyState) -> dict:
    return {"custom_field": "updated"}

# 3. 构建图
builder = StateGraph(MyState)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)

# 4. 条件边
def router(state: MyState) -> str:
    if state["custom_field"] == "done":
        return "end_node"
    return "continue_node"

builder.add_conditional_edges(
    "my_node",
    router,
    {"end_node": END, "continue_node": "my_node"},
)

# 5. 编译（可选 checkpointer）
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 6. 执行（流式）
config = {"configurable": {"thread_id": "session_123"}}
async for event in graph.astream_events(
    {"messages": [HumanMessage(content="hello")]},
    config=config,
    version="v2",  # 推荐使用 v2
):
    event_type = event["event"]   # on_chat_model_stream / on_chain_start 等
```

### LangGraph 1.x Send API（Map-Reduce 并行）

```python
from langgraph.types import Send

def distribute_work(state: MyState) -> list[Send]:
    """返回 Send 列表实现并行节点执行"""
    return [
        Send("worker_node", {"item": item})
        for item in state["items"]
        if item["status"] == "pending"
    ]

builder.add_conditional_edges("dispatch_node", distribute_work, ["worker_node"])
builder.add_edge("worker_node", "aggregate_node")
```

### LangChain 1.x ChatModel 写法

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# ChatModel 初始化（OpenAI 兼容接口，涵盖 DeepSeek/Ollama 等）
llm = ChatOpenAI(
    model="gpt-4o",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",   # 可覆盖用于 DeepSeek/本地模型
    temperature=0.3,
    max_tokens=8192,
    streaming=True,
)

# 工具定义
@tool
async def search_knowledge_base(query: str, kb_name: str = "ai_textbook") -> str:
    """Search the knowledge base for relevant information."""
    from src.tools.rag_tool import rag_search
    result = await rag_search(query=query, kb_name=kb_name, mode="hybrid")
    return result.get("answer", "")

# 工具绑定
llm_with_tools = llm.bind_tools([search_knowledge_base])

# 调用
response = await llm.ainvoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is convolution?"),
])

# 结构化输出（替代 response_format={"type": "json_object"}）
from pydantic import BaseModel

class AnalysisPlan(BaseModel):
    reasoning: str
    tool: str
    query: str

structured_llm = llm.with_structured_output(AnalysisPlan)
result: AnalysisPlan = await structured_llm.ainvoke(messages)
```

---

## 实施策略

**原则：最小破坏性变更**
1. `src/tools/rag_tool.py` 等原有工具函数不改动，新工具层只是 `@tool` 包装
2. `src/services/llm/factory.py` 保留，RAG pipeline 内部依赖它
3. 所有路由路径保持 `/api/v1/*`，WebSocket 消息格式兼容
4. 各原始 Agent 类（InvestigateAgent 等）保留不删除，LangGraph 节点通过包装调用它们

---

## 阶段 1：基础设施（新建文件，不改动现有代码）

### 1.1 新建：`src/services/llm/langchain_factory.py`

```python
from langchain_core.language_models import BaseChatModel

def build_chat_model(
    binding: str,       # openai/anthropic/azure_openai/ollama/lm_studio/vllm/...
    model: str,
    api_key: str | None,
    base_url: str | None,
    api_version: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> BaseChatModel:
    """
    根据 binding 类型返回对应的 LangChain 1.x ChatModel。
    与原 factory.py 的 binding 字符串保持一致。
    """
    if binding == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key,
                             temperature=temperature, max_tokens=max_tokens)

    elif binding == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model, azure_endpoint=base_url,
            api_key=api_key, api_version=api_version or "2024-02-01",
            temperature=temperature, max_tokens=max_tokens, streaming=True,
        )

    else:
        # 涵盖：openai / deepseek / openrouter / groq / together / mistral
        #       / ollama / lm_studio / vllm / llama_cpp（均为 OpenAI 兼容接口）
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key or "no-key-required",
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )

def get_chat_model_from_env(temperature: float | None = None,
                             max_tokens: int | None = None) -> BaseChatModel:
    """从项目 .env 配置构建 ChatModel（LangGraph 节点调用入口）"""
    from src.services.llm.config import get_llm_config
    cfg = get_llm_config()
    return build_chat_model(
        binding=cfg.binding, model=cfg.model,
        api_key=cfg.api_key, base_url=cfg.base_url,
        api_version=getattr(cfg, "api_version", None),
        temperature=temperature or cfg.temperature,
        max_tokens=max_tokens or cfg.max_tokens,
    )
```

### 1.2 新建：`src/tools/langchain_tools.py`

```python
from langchain_core.tools import tool

@tool
async def rag_search_tool(query: str, kb_name: str = "ai_textbook",
                           mode: str = "hybrid") -> str:
    """搜索知识库。mode: hybrid（混合）或 naive（简单向量）"""
    from src.tools.rag_tool import rag_search
    result = await rag_search(query=query, kb_name=kb_name, mode=mode)
    return result.get("answer", "") or result.get("content", "")

@tool
async def web_search_tool(query: str) -> str:
    """执行网络搜索获取最新信息。"""
    from src.tools.web_search import web_search
    result = await web_search(query=query, verbose=False)
    return result.get("answer", "")

@tool
async def paper_search_tool(query: str, max_results: int = 3) -> str:
    """搜索学术论文。"""
    import json
    from src.tools.paper_search_tool import PaperSearchTool
    papers = await PaperSearchTool().search_papers(query=query, max_results=max_results)
    return json.dumps({"papers": papers}, ensure_ascii=False)

@tool
async def code_execution_tool(code: str, language: str = "python") -> str:
    """执行代码并返回结果。"""
    import json
    from src.tools.code_executor import run_code
    result = await run_code(language=language, code=code)
    return json.dumps(result, ensure_ascii=False)

@tool
async def query_item_tool(identifier: str, kb_name: str = "ai_textbook") -> str:
    """按编号查询知识库特定条目（如公式3.1、定理2.3）。"""
    import json
    from src.tools.query_item_tool import query_numbered_item
    result = await query_numbered_item(identifier=identifier, kb_name=kb_name)
    return json.dumps(result, ensure_ascii=False)

# 按模块分组的工具集
SOLVE_TOOLS    = [rag_search_tool, web_search_tool, code_execution_tool, query_item_tool]
RESEARCH_TOOLS = [rag_search_tool, web_search_tool, paper_search_tool,
                  code_execution_tool, query_item_tool]
CHAT_TOOLS     = [rag_search_tool, web_search_tool]
```

### 1.3 新建：`src/api/utils/langgraph_ws_adapter.py`

```python
from fastapi import WebSocket
from langgraph.graph.state import CompiledStateGraph

async def stream_graph_to_websocket(
    graph: CompiledStateGraph,
    initial_state: dict,
    websocket: WebSocket,
    config: dict | None = None,
) -> dict:
    """
    将 LangGraph astream_events(version="v2") 桥接到 FastAPI WebSocket。

    事件映射（保持前端消息格式兼容）：
      on_chat_model_stream → {"type":"stream",  "content": token}
      on_chain_start(节点)  → {"type":"status", "stage": node_name}
      on_tool_start         → {"type":"progress","stage":"tool_call","tool":...}
      streaming_events字段  → 各节点推送的自定义进度事件
    """
    final_state = {}
    NODE_STATUS_MAP = {
        "investigate": "analyzing", "note": "taking_notes",
        "plan": "planning", "solve_step": "solving",
        "response": "generating_response", "finalize": "finalizing",
        "rephrase": "rephrasing_topic", "decompose": "decomposing_topic",
        "research_block": "researching", "report": "generating_report",
        "retrieve": "retrieving_context", "chat": "responding",
    }

    async def safe_send(data: dict) -> None:
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    async for event in graph.astream_events(
        initial_state, config=config or {}, version="v2"
    ):
        etype = event["event"]
        ename = event.get("name", "")
        edata = event.get("data", {})

        if etype == "on_chat_model_stream":
            chunk = edata.get("chunk")
            if chunk and chunk.content:
                await safe_send({"type": "stream", "content": chunk.content})

        elif etype == "on_chain_start" and ename in NODE_STATUS_MAP:
            await safe_send({"type": "status", "stage": ename,
                              "message": NODE_STATUS_MAP[ename]})

        elif etype == "on_chain_end":
            output = edata.get("output", {})
            for pe in output.get("streaming_events", []):
                await safe_send(pe)
            if output:
                final_state = output

        elif etype == "on_tool_start":
            tool_input = edata.get("input", {})
            await safe_send({"type": "progress", "stage": "tool_call",
                              "tool": ename, "query": tool_input.get("query", "")})

    return final_state
```

### 1.4 更新 `requirements.txt`（新增两行）

```
langchain-anthropic>=0.3.0
langgraph-checkpoint-sqlite>=2.0.0
```

---

## 阶段 2：Chat 模块（最简单，先验证端到端流程）

**关键原始文件：**
- `src/agents/chat/chat_agent.py`
- `src/api/routers/chat.py`

### 新建 `src/agents/chat/lg_state.py`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages   # 内置 reducer，自动追加消息

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    kb_name: str
    enable_rag: bool
    enable_web_search: bool
    language: str
    session_id: str
    rag_context: str
    streaming_events: Annotated[list[dict], lambda a, b: a + b]
```

### 新建 `src/agents/chat/lg_nodes.py`

```python
async def retrieve_context_node(state: ChatState) -> dict:
    """可选 RAG/Web 检索，结果注入 rag_context"""
    context = ""
    if state.get("enable_rag") and state.get("kb_name"):
        from src.tools.rag_tool import rag_search
        result = await rag_search(
            query=state["messages"][-1].content,
            kb_name=state["kb_name"], mode="hybrid"
        )
        context = result.get("answer", "")
    return {"rag_context": context}

async def chat_node(state: ChatState) -> dict:
    """主对话节点：调用 LangChain ChatModel"""
    from src.services.llm.langchain_factory import get_chat_model_from_env
    from langchain_core.messages import SystemMessage

    system_prompt = "You are a helpful learning assistant."
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    if state.get("rag_context"):
        context_msg = SystemMessage(content=f"Reference:\n{state['rag_context']}")
        messages.insert(-1, context_msg)

    llm = get_chat_model_from_env()
    response = await llm.ainvoke(messages)
    return {"messages": [response]}  # add_messages reducer 自动追加
```

### 新建 `src/agents/chat/lg_graph.py`

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .lg_state import ChatState
from .lg_nodes import retrieve_context_node, chat_node

def build_chat_graph():
    builder = StateGraph(ChatState)
    builder.add_node("retrieve", retrieve_context_node)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "chat")
    builder.add_edge("chat", END)
    # MemorySaver：thread_id=session_id 实现多会话隔离
    # 持久化版本替换为：SqliteSaver.from_conn_string("data/user/chat/checkpoints.db")
    return builder.compile(checkpointer=MemorySaver())

_chat_graph = None
def get_chat_graph():
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = build_chat_graph()
    return _chat_graph
```

### 修改 `src/api/routers/chat.py`

保持原 `/chat` WebSocket 端点不变，**新增** `/chat/lg` 端点用于并行测试：

```python
@router.websocket("/chat/lg")
async def websocket_chat_lg(websocket: WebSocket):
    """LangGraph 实现的 Chat（测试用，验证通过后替换 /chat）"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message    = data.get("message", "").strip()
            session_id = data.get("session_id") or str(uuid.uuid4())

            from src.agents.chat.lg_graph import get_chat_graph
            from langchain_core.messages import HumanMessage
            from src.api.utils.langgraph_ws_adapter import stream_graph_to_websocket

            await websocket.send_json({"type": "session", "session_id": session_id})

            graph = get_chat_graph()
            config = {"configurable": {"thread_id": session_id}}
            initial = {
                "messages": [HumanMessage(content=message)],
                "kb_name": data.get("kb_name", ""),
                "enable_rag": data.get("enable_rag", False),
                "enable_web_search": data.get("enable_web_search", False),
            }

            final_state = await stream_graph_to_websocket(graph, initial, websocket, config)
            last_msg = final_state.get("messages", [])[-1] if final_state.get("messages") else None
            await websocket.send_json({
                "type": "result",
                "content": last_msg.content if last_msg else "",
                "session_id": session_id,
            })
    except WebSocketDisconnect:
        pass
```

---

## 阶段 3：Research 模块

**关键原始文件：**
- `src/agents/research/research_pipeline.py`（三阶段管道）
- `src/agents/research/data_structures.py`（DynamicTopicQueue）

### 新建 `src/agents/research/lg_state.py`

```python
from typing import Annotated
from typing_extensions import TypedDict

class TopicBlock(TypedDict):
    block_id: str
    sub_topic: str
    overview: str
    status: str          # pending | researching | completed | failed
    notes: list[str]
    citations: list[str]

def merge_topic_blocks(existing: list, updates: list) -> list:
    """自定义 Reducer：按 block_id 合并，对应 DynamicTopicQueue 的动态添加"""
    existing_map = {b["block_id"]: b for b in existing}
    for b in updates:
        existing_map[b["block_id"]] = b
    return list(existing_map.values())

class ResearchState(TypedDict):
    topic: str
    kb_name: str
    research_id: str
    language: str
    optimized_topic: str
    topic_blocks: Annotated[list[TopicBlock], merge_topic_blocks]
    completed_block_ids: list[str]
    final_report: str
    streaming_events: Annotated[list[dict], lambda a, b: a + b]
```

### 新建 `src/agents/research/lg_graph.py`

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send   # Map-Reduce 并行

def distribute_blocks(state: ResearchState) -> list[Send]:
    """并行研究所有 pending blocks，对应原 _phase2_researching_parallel"""
    pending = [b for b in state["topic_blocks"] if b["status"] == "pending"]
    return [Send("research_block", {"block": b, **state}) for b in pending]

def build_research_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("rephrase", rephrase_node)
    builder.add_node("decompose", decompose_node)
    builder.add_node("research_block", research_block_node)  # 并行节点
    builder.add_node("report", report_node)

    builder.add_edge(START, "rephrase")
    builder.add_edge("rephrase", "decompose")
    # decompose 后用 Send API 并行分发
    builder.add_conditional_edges("decompose", distribute_blocks, ["research_block"])
    # 所有 research_block 完成后聚合到 report
    builder.add_edge("research_block", "report")
    builder.add_edge("report", END)
    return builder.compile()
```

### 新建 `src/agents/research/lg_nodes.py`

- `rephrase_node`：包装 `RephraseAgent.process()`
- `decompose_node`：包装 `DecomposeAgent.process()` → 填充 `topic_blocks`
- `research_block_node`：包装 `ResearchAgent.process()`（工具调用循环）
- `report_node`：包装 `ReportingAgent.process()`

---

## 阶段 4：简单模块（Question / IdeaGen / CoWriter）

三个模块均为**线性图**，无循环，同时并行实现：

### Question 模块

```
START → retrieve_node → generate_node → analyze_node → END
```

状态 `QuestionState`：`input_text, knowledge_points, generated_questions, validated_questions, language`

对应原 `AgentCoordinator`（`src/agents/question/coordinator.py`）的三阶段流程。

### IdeaGen 模块

```
START → organize_material_node → generate_ideas_node → END
```

状态 `IdeaGenState`：`topic, kb_name, knowledge_context, generated_ideas, language`

对应原 `IdeaGenerationWorkflow`（`src/agents/ideagen/idea_generation_workflow.py`）。

### CoWriter 模块

```
START → edit_node → [条件] narrate_node → END
```

状态 `CoWriterState`：`messages（add_messages）, current_document, edited_document, need_narration`

---

## 阶段 5：Guide 模块（有状态多轮交互）

```
START → locate_node → interactive_node [用户驱动循环] → summary_node → END
```

**条件边（interactive_node 后）**：
- 用户请求下一个知识点 → `locate_node`
- 用户继续交互 → `interactive_node`
- 全部完成 → `summary_node`

使用 `MemorySaver` checkpointer 管理多轮会话，`thread_id=session_id`。

对应原 `GuideManager`（`src/agents/guide/guide_manager.py`）的 `GuidedSession` 状态管理。

---

## 阶段 6：Solve 模块（最复杂，双循环）

**关键原始文件：**
- `src/agents/solve/main_solver.py`（`_run_dual_loop_pipeline` 约 450 行手动循环）
- `src/agents/solve/memory/investigate_memory.py`（InvestigateMemory）
- `src/agents/solve/memory/solve_memory.py`（SolveMemory）
- `src/agents/solve/memory/citation_memory.py`（CitationMemory）

### 新建 `src/agents/solve/lg_state.py`

将三个 Memory 类合并为统一 TypedDict：

```python
from typing import Annotated
from typing_extensions import TypedDict

class KnowledgeItem(TypedDict):
    cite_id: str
    tool_type: str
    query: str
    raw_result: str
    summary: str

class ToolCallRecord(TypedDict):
    call_id: str
    tool_type: str
    query: str
    cite_id: str | None
    raw_answer: str | None
    summary: str | None
    status: str   # pending | success | failed

class SolveStep(TypedDict):
    step_id: str
    step_target: str
    available_cite: list[str]
    tool_calls: list[ToolCallRecord]
    step_response: str | None
    status: str   # undone | in_progress | waiting_response | done

class CitationRecord(TypedDict):
    cite_id: str
    source: str
    query: str
    content: str

class SolveState(TypedDict):
    # 输入
    question: str
    kb_name: str
    language: str

    # Analysis Loop（原 InvestigateMemory）
    knowledge_chain: list[KnowledgeItem]
    analysis_iteration: int
    max_analysis_iterations: int
    analysis_should_stop: bool
    new_knowledge_ids: list[str]

    # Solve Loop（原 SolveMemory）
    solve_steps: list[SolveStep]
    current_step_index: int
    solve_iteration: int
    max_solve_iterations: int

    # 引用（原 CitationMemory）
    citations: list[CitationRecord]

    # 输出 & 流式推送
    final_answer: str
    streaming_events: Annotated[list[dict], lambda a, b: a + b]
```

### 新建 `src/agents/solve/lg_graph.py`

完整 StateGraph，对应 `_run_dual_loop_pipeline` 的控制流：

```python
from langgraph.graph import StateGraph, START, END
from .lg_state import SolveState
from .lg_nodes import (
    investigate_node, exec_analysis_tools_node, note_node,
    plan_node, exec_step_tools_node, solve_step_node,
    response_node, advance_step_node, finalize_node,
)

def should_continue_analysis(state: SolveState) -> str:
    if state["analysis_should_stop"]: return "plan"
    if state["analysis_iteration"] >= state["max_analysis_iterations"]: return "plan"
    return "investigate"

def after_exec_tools(state: SolveState) -> str:
    if state["current_step_index"] >= len(state["solve_steps"]): return "finalize"
    step = state["solve_steps"][state["current_step_index"]]
    if any(tc["status"] == "pending" for tc in step.get("tool_calls", [])): return "exec_step_tools"
    return "solve_step"

def after_solve_step(state: SolveState) -> str:
    step = state["solve_steps"][state["current_step_index"]]
    if any(tc["status"] == "pending" for tc in step.get("tool_calls", [])): return "exec_step_tools"
    if step.get("status") == "waiting_response": return "response"
    if state["solve_iteration"] < state["max_solve_iterations"]: return "solve_step"
    return "response"

def after_response(state: SolveState) -> str:
    next_idx = state["current_step_index"] + 1
    return "advance_step" if next_idx < len(state["solve_steps"]) else "finalize"

def build_solve_graph():
    builder = StateGraph(SolveState)

    # Analysis Loop 节点
    builder.add_node("investigate", investigate_node)
    builder.add_node("exec_analysis_tools", exec_analysis_tools_node)
    builder.add_node("note", note_node)

    # Solve Loop 节点
    builder.add_node("plan", plan_node)
    builder.add_node("exec_step_tools", exec_step_tools_node)
    builder.add_node("solve_step", solve_step_node)
    builder.add_node("response", response_node)
    builder.add_node("advance_step", advance_step_node)
    builder.add_node("finalize", finalize_node)

    # Analysis Loop 边
    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "exec_analysis_tools")
    builder.add_edge("exec_analysis_tools", "note")
    builder.add_conditional_edges("note", should_continue_analysis,
        {"investigate": "investigate", "plan": "plan"})

    # Solve Loop 边
    builder.add_edge("plan", "exec_step_tools")
    builder.add_conditional_edges("exec_step_tools", after_exec_tools,
        {"exec_step_tools": "exec_step_tools",
         "solve_step": "solve_step", "finalize": "finalize"})
    builder.add_conditional_edges("solve_step", after_solve_step,
        {"exec_step_tools": "exec_step_tools",
         "solve_step": "solve_step", "response": "response"})
    builder.add_conditional_edges("response", after_response,
        {"advance_step": "advance_step", "finalize": "finalize"})
    builder.add_edge("advance_step", "exec_step_tools")
    builder.add_edge("finalize", END)

    return builder.compile()
```

### 新建 `src/agents/solve/lg_nodes.py`

每个节点函数**包装原有 Agent 类**（原始 Agent 代码不删除）：

```python
async def investigate_node(state: SolveState) -> dict:
    """包装 InvestigateAgent.process()，将结果写回 SolveState"""
    from .analysis_loop.investigate_agent import InvestigateAgent
    from src.services.llm.config import get_llm_config
    from .memory.investigate_memory import InvestigateMemory
    from .memory.citation_memory import CitationMemory

    # 从 state 重建临时 Memory（内存中，不写磁盘）
    tmp_mem = InvestigateMemory(user_question=state["question"])
    tmp_cite = CitationMemory()
    # 填充已有知识链
    for ki in state["knowledge_chain"]:
        tmp_mem.knowledge_chain.append(ki)

    cfg = get_llm_config()
    agent = InvestigateAgent(config={}, api_key=cfg.api_key, base_url=cfg.base_url)
    result = await agent.process(
        question=state["question"], memory=tmp_mem,
        citation_memory=tmp_cite, kb_name=state["kb_name"], verbose=False,
    )

    return {
        "analysis_iteration": state["analysis_iteration"] + 1,
        "analysis_should_stop": result.get("should_stop", False),
        "new_knowledge_ids": result.get("knowledge_item_ids", []),
        # 新增知识项到 knowledge_chain
        "knowledge_chain": state["knowledge_chain"] + [
            ki for ki in tmp_mem.knowledge_chain
            if ki["cite_id"] not in {k["cite_id"] for k in state["knowledge_chain"]}
        ],
        "streaming_events": [{"type": "progress", "stage": "investigate",
                               "round": state["analysis_iteration"] + 1}],
    }

# 其余节点：note_node / plan_node / exec_step_tools_node / solve_step_node
# / response_node / advance_step_node / finalize_node
# 均采用相同的"从 state 重建临时 Memory → 调用原 Agent.process() → 写回 state"模式
```

---

## 阶段 7：切换与清理

验证所有 `/xxx/lg` 测试端点行为与原端点一致后：

1. 将 `src/api/routers/*.py` WebSocket 处理切换到 `stream_graph_to_websocket`
2. 删除 `/xxx/lg` 测试端点
3. 可选清理（不影响功能）：
   - `src/agents/solve/memory/*.py` → 被 `SolveState` TypedDict 替代
   - `src/agents/chat/session_manager.py` → 被 LangGraph checkpointer 替代
   - `src/agents/solve/main_solver.py` → 被 `lg_graph.py` 替代

**永久保留（RAG 和 LightRAG 依赖）**：
- `src/services/llm/factory.py` / `cloud_provider.py` / `local_provider.py`
- `src/tools/rag_tool.py`, `web_search.py`, `code_executor.py`
- `src/agents/base_agent.py`（保留兼容性）
- `src/api/main.py`（路由注册不变）
- `web/`（前端完全不改动）

---

## 关键风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| LightRAG 需要 `OPENAI_API_KEY` 环境变量 | `langchain_factory.py` 用显式 `api_key` 参数传递，不依赖环境变量 |
| LangGraph checkpointer 需要 JSON 可序列化状态 | TypedDict 只使用 JSON 基础类型（str/int/list/dict），不使用 dataclass |
| Research 并行研究（原用 asyncio.Semaphore） | LangGraph 1.x `Send` API 实现 Map-Reduce 自动并行 |
| WebSocket 消息格式兼容性 | 适配器保持原有 `type` 字段枚举（stream/status/progress/result/error）不变 |
| `langchain-community==0.3.0` 版本约束 | 仅用 `langchain-openai` + `langchain-anthropic`，不依赖 community 包的 ChatModel |

---

## 验证方案

每个阶段完成后：

1. 安装新增依赖：`pip install langchain-anthropic langgraph-checkpoint-sqlite`
2. 启动后端：`python -m uvicorn src.api.main:app --port 8001 --reload`
3. 测试新端点（如 `/api/v1/chat/lg` WebSocket）与原端点行为一致
4. 运行现有测试：`pytest tests/ -v`
5. 完成所有阶段后，用前端做端到端测试（前端代码无需修改）

---

## 完整文件清单

### 新建文件（不破坏任何现有代码）

```
src/services/llm/langchain_factory.py
src/tools/langchain_tools.py
src/api/utils/langgraph_ws_adapter.py
src/agents/chat/lg_state.py
src/agents/chat/lg_graph.py
src/agents/chat/lg_nodes.py
src/agents/research/lg_state.py
src/agents/research/lg_graph.py
src/agents/research/lg_nodes.py
src/agents/question/lg_state.py
src/agents/question/lg_graph.py
src/agents/question/lg_nodes.py
src/agents/ideagen/lg_state.py
src/agents/ideagen/lg_graph.py
src/agents/ideagen/lg_nodes.py
src/agents/co_writer/lg_state.py
src/agents/co_writer/lg_graph.py
src/agents/co_writer/lg_nodes.py
src/agents/guide/lg_state.py
src/agents/guide/lg_graph.py
src/agents/guide/lg_nodes.py
src/agents/solve/lg_state.py
src/agents/solve/lg_graph.py
src/agents/solve/lg_nodes.py
```

### 修改文件

```
requirements.txt                    新增 langchain-anthropic, langgraph-checkpoint-sqlite
src/api/routers/chat.py             新增 /chat/lg 测试端点，后续切换主端点
src/api/routers/solve.py            新增 /solve/lg 测试端点，后续切换主端点
src/api/routers/research.py         同上
src/api/routers/question.py         同上
src/api/routers/guide.py            同上
src/api/routers/ideagen.py          同上
src/api/routers/co_writer.py        同上
```

### 不改动文件（永久保留）

```
src/services/llm/factory.py         RAG 内部依赖
src/services/llm/cloud_provider.py
src/services/llm/local_provider.py
src/tools/rag_tool.py
src/tools/web_search.py
src/tools/code_executor.py
src/agents/base_agent.py
src/api/main.py
web/                                前端无需修改
```
