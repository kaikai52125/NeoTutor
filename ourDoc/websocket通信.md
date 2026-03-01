# WebSocket 通信机制

## 一、整体架构

NeoTutor 的 Agent 输出全部通过 WebSocket 实时流式推送到前端，没有普通 HTTP 轮询。

```
前端 React Context
    ↓ new WebSocket(wsUrl(...))
FastAPI WebSocket 端点
    ↓ graph.astream_events(version="v2")
LangGraph 图执行
    ↓ 每个节点输出 streaming_events
stream_graph_to_websocket() 转换
    ↓ ws.send_json(...)
前端收到 → 更新 UI 状态
```

---

## 二、后端 WebSocket 端点列表

| 模块 | 端点 | 入口文件 |
|------|------|---------|
| solve | `/api/v1/solve` | `src/api/routers/solve.py` |
| research | `/api/v1/research/run` | `src/api/routers/research.py` |
| chat | `/api/v1/chat` | `src/api/routers/chat.py` |
| guide | `/api/v1/guide/ws/{session_id}` | `src/api/routers/guide.py` |
| question | `/api/v1/question/generate` | `src/api/routers/question.py` |
| question | `/api/v1/question/mimic` | `src/api/routers/question.py` |
| ideagen | `/api/v1/ideagen/generate` | `src/api/routers/ideagen.py` |

---

## 三、消息协议

### 前端 → 后端（连接后发送一次）

```json
// Solve 模块
{ "question": "...", "kb_name": "ai_textbook", "session_id": null }

// Research 模块
{ "topic": "...", "kb_name": "...", "enabled_tools": ["web", "paper"], "plan_mode": false }

// Chat 模块
{ "message": "...", "session_id": "...", "kb_name": "...", "enable_rag": true }

// IdeaGen 模块
{ "notebook_id": "...", "record_ids": [...], "user_thoughts": "..." }
```

### 后端 → 前端（持续推送）

所有模块统一的消息格式：

```json
// LLM 流式 token（打字机效果）
{ "type": "stream", "content": "下一个 token 字符串" }

// 节点状态变化
{ "type": "status", "stage": "analyzing", "message": "正在分析..." }

// Agent 运行日志（显示在 Activity Log 面板）
{ "type": "log", "content": "...", "timestamp": 1234567890 }

// 进度信息（显示在 Progress 面板）
{ "type": "progress", "stage": "investigate", "progress": { "round": 1, "new_items": 3 } }

// Agent 状态（多 Agent 协同时的状态图）
{ "type": "agent_status", "agent": "solve", "status": "running", "all_agents": {...} }

// Token 统计
{ "type": "token_stats", "stats": { "model": "...", "calls": 3, "cost": 0.002 } }

// 最终结果
{ "type": "result", "final_answer": "...", "session_id": "..." }

// 错误
{ "type": "error", "content": "错误描述" }

// IdeaGen 专用：逐条推送 idea
{ "type": "idea", "data": { "title": "...", "description": "...", ... } }
```

---

## 四、后端流式实现

### 核心适配器

**文件：`src/api/utils/langgraph_ws_adapter.py`**

```python
async def stream_graph_to_websocket(graph, state, websocket):
    async for event in graph.astream_events(state, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # LLM 输出 token → stream 消息
            chunk = event["data"]["chunk"].content
            await websocket.send_json({"type": "stream", "content": chunk})

        elif kind == "on_chain_start":
            # 节点开始 → status 消息
            node = event["name"]
            await websocket.send_json({
                "type": "status",
                "stage": NODE_STATUS_MAP.get(node, node),
                "message": NODE_MSG_MAP.get(node, "")
            })

        elif kind == "on_chain_end":
            # 节点结束 → 转发节点自定义的 streaming_events
            output = event["data"].get("output", {})
            for ev in output.get("streaming_events", []):
                await websocket.send_json(ev)
```

### 节点自定义事件

每个 LangGraph 节点在返回 state 时附带 `streaming_events` 列表：

```python
# 节点函数示例（lg_nodes.py 中的模式）
async def some_node(state):
    # ... 处理逻辑 ...
    return {
        **state,
        "streaming_events": [
            {"type": "progress", "stage": "investigate", "progress": {"round": 1}},
            {"type": "log", "content": "完成第一轮调查"}
        ]
    }
```

---

## 五、前端 WebSocket 处理

### URL 构建

**文件：`web/lib/api.ts`**

```typescript
export function wsUrl(path: string): string {
    // 自动将 http → ws，https → wss
    const base = API_BASE_URL
        .replace(/^http:/, "ws:")
        .replace(/^https:/, "wss:");
    return `${base}${path}`;
}
```

### 连接与消息处理（以 Solve 为例）

**文件：`web/context/solver/SolverContext.tsx`**

```typescript
const ws = new WebSocket(wsUrl("/api/v1/solve"));

ws.onopen = () => {
    ws.send(JSON.stringify({ question, kb_name: selectedKb }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch (data.type) {
        case "stream":
            // 追加 token 到当前消息
            appendStreamToken(data.content);
            break;
        case "log":
            // 写入 Activity Log 面板
            addSolverLog(data);
            break;
        case "progress":
            // 更新 Progress 面板
            setSolverState(prev => ({
                ...prev,
                progress: { stage: data.stage, progress: data.progress }
            }));
            break;
        case "agent_status":
            setSolverState(prev => ({
                ...prev,
                agentStatus: data.all_agents
            }));
            break;
        case "result":
            // 写入最终答案，关闭连接
            addAssistantMessage(data.final_answer);
            ws.close();
            break;
        case "error":
            setError(data.content);
            ws.close();
            break;
    }
};

ws.onerror = () => setIsSolving(false);
ws.onclose = () => { wsRef.current = null; };
```

---

## 六、各模块前端 Context 文件

| 模块 | Context 文件 |
|------|-------------|
| Solve | `web/context/solver/SolverContext.tsx` |
| Research | `web/context/research/ResearchContext.tsx` |
| Chat | `web/context/chat/ChatContext.tsx` |
| Guide | `web/context/guide/GuideContext.tsx` |
| Question | `web/context/question/QuestionContext.tsx` |
| IdeaGen | `web/context/ideagen/IdeaGenContext.tsx` |

---

## 七、注意事项

1. **连接复用**：每次提问新建连接，`result` 或 `error` 后由后端主动关闭
2. **session_id**：Chat 和 Solve 支持 session_id，传 null 则后端自动生成新 session
3. **`type: log` vs `type: progress`**：log 写日志面板；progress 写进度面板（`data.progress` 是嵌套对象）
4. **IdeaGen**：逐条推送 `type: idea`，不是最后一次性发送 result
