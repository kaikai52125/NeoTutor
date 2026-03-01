# RAG 模块

## 一、整体架构

NeoTutor 的 RAG 模块基于 **LightRAG / RAGAnything** 构建知识图谱，支持多种 Provider，统一通过 `RAGService` 对外暴露。

```
Agent 调用
    ↓
src/tools/rag_tool.py: rag_search()
    ↓
src/services/rag/service.py: RAGService.search()
    ↓
src/services/rag/factory.py: 根据 kb metadata 选 Provider
    ↓
Pipeline（raganything / lightrag / llamaindex）
    ├─ Indexer：文档 → 知识图谱
    └─ Retriever：查询 → 检索结果
    ↓
SearchResult { query, answer, content, chunks, metadata }
```

---

## 二、Provider 对比

| Provider | 解析器 | 特点 | 适用场景 |
|----------|--------|------|---------|
| **raganything** | MinerU | 多模态（图片/表格/公式）；速度慢 | 学术 PDF |
| **raganything_docling** | Docling | 多模态；支持 Office 文档 | .docx / .pptx |
| **lightrag** | PDFParser | 纯文本；构建知识图谱；速度快 | 文本型文档 |
| **llamaindex** | 简单分块 | 最快；向量检索 | 快速原型 |

默认使用 `raganything`，可通过环境变量 `RAG_PROVIDER` 修改。

---

## 三、目录结构

```
src/services/rag/
├── service.py                      # 统一入口 RAGService
├── factory.py                      # Provider 工厂（懒加载）
├── types.py                        # Chunk / Document / SearchResult 数据类
├── pipelines/
│   ├── raganything.py              # RAGAnything 完整 pipeline
│   ├── raganything_docling.py      # RAGAnything + Docling
│   └── lightrag.py                 # LightRAG 简化 pipeline
└── components/
    ├── indexers/
    │   ├── lightrag.py             # 文档插入 LightRAG
    │   └── graph.py                # 知识图谱构建
    └── retrievers/
        ├── lightrag.py             # 图谱检索
        └── hybrid.py               # 混合检索

src/tools/rag_tool.py               # 对 Agent 暴露的工具函数
src/knowledge/
├── initializer.py                  # 知识库初始化（首次建库）
└── add_documents.py                # 增量添加文档
```

---

## 四、核心 API

### 4.1 Agent 调用入口

**文件：`src/tools/rag_tool.py`**

```python
# 查询
result = await rag_search(
    query="transformer attention mechanism",
    kb_name="ai_textbook",   # 默认从 config 读取
    mode="hybrid",           # hybrid / local / global / naive
)
# result["answer"]   — LLM 综合回答
# result["content"]  — 原始检索片段
# result["chunks"]   — 结构化 Chunk 列表

# 建库
await initialize_rag(kb_name="my_kb", documents=["/path/to/doc.pdf"])

# 删库
await delete_rag(kb_name="my_kb")

# 查询可用 provider
providers = get_available_providers()
```

### 4.2 RAGService

**文件：`src/services/rag/service.py`**

```python
class RAGService:
    async def initialize(kb_name, file_paths, **kwargs) -> bool
    async def search(query, kb_name, mode="hybrid", **kwargs) -> dict
    async def delete(kb_name) -> bool
    @staticmethod list_providers() -> List[Dict]
```

---

## 五、知识库初始化流程

**文件：`src/knowledge/initializer.py`**

```
① create_directory_structure()
   data/knowledge_bases/<kb_name>/
   ├── raw/          # 原始文档
   ├── images/       # 提取的图片
   ├── rag_storage/  # LightRAG 向量/图谱数据
   ├── content_list/ # MinerU 解析结果（中间文件）
   └── metadata.json # 元信息（provider、文件哈希等）

② copy_documents(files)
   → 复制到 raw/

③ process_documents()
   → 调用 RAGService.initialize() 触发解析 + 入库

④ 写入 metadata.json
{
  "name": "kb_name",
  "created_at": "2025-01-01 10:00:00",
  "rag_provider": "raganything",
  "file_hashes": { "doc.pdf": "sha256..." },
  "last_updated": "2025-01-02 15:00:00"
}
```

---

## 六、增量添加文档

**文件：`src/knowledge/add_documents.py`**

```python
adder = DocumentAdder(kb_name="ai_textbook")

# 同步阶段：哈希去重
new_files = adder.add_documents(
    files=["/path/to/new.pdf"],
    allow_duplicates=False      # 相同内容跳过
)

# 异步阶段：解析 + 入库
processed = await adder.process_new_documents(new_files)
```

**去重逻辑：** SHA-256（64KB 分块读取），与 `metadata.file_hashes` 比对，内容相同无论文件名是否一致均跳过。

---

## 七、RAGAnything Pipeline 详解

**文件：`src/services/rag/pipelines/raganything.py`**

```
① 解析（MinerU）
   parse_document(pdf_path)
   → content_list: List[Dict]  # 包含 text / image / table / equation

② 图片迁移
   migrate_images_and_update_paths(
       content_list,
       source_base_dir=content_list_dir,
       target_images_dir=kb/images/
   )
   → 更新 content_list 中的图片路径

③ 插入 RAG
   rag.insert_content_list(updated_content_list)
   → LightRAG 构建知识图谱

④ 清理
   删除 content_list_dir 临时目录
```

---

## 八、检索模式

通过 `mode` 参数控制检索策略：

| 模式 | 说明 |
|------|------|
| `hybrid` | 向量检索 + 图谱检索组合（默认，推荐） |
| `local` | 基于实体的图谱局部检索（精确概念） |
| `global` | 基于关系的图谱全局检索（宏观主题） |
| `naive` | 关键词检索（速度最快，效果最弱） |

---

## 九、与 LightRAG 的关系

`raganything` 包内部封装了 LightRAG：

- **建库时**：调用 `rag.insert()` / `rag.insert_content_list()` → LightRAG 抽取实体关系构建图谱，同时建向量索引
- **检索时**：调用 `rag.query(query, param=QueryParam(mode=mode))` → LightRAG 融合图谱 + 向量返回答案
- **LLM Key**：`src/api/main.py` 启动时将 `LLM_API_KEY` 写入 `os.environ["OPENAI_API_KEY"]`，LightRAG 以此初始化其内部 OpenAI 客户端

---

## 十、配置

**`config/main.yaml`（关键部分）：**

```yaml
tools:
  rag_tool:
    kb_base_dir: ./data/knowledge_bases
    default_kb: ai_textbook

research:
  rag:
    kb_name: DE-all
    default_mode: hybrid
    fallback_mode: naive
```

**环境变量：**

| 变量 | 说明 |
|------|------|
| `RAG_PROVIDER` | 默认 provider（raganything） |
| `LLM_API_KEY` | LightRAG 内部 LLM 调用 |
| `EMBEDDING_API_KEY` | 向量化 |
| `EMBEDDING_MODEL` | 嵌入模型 |

---

## 十一、数据类型

**文件：`src/services/rag/types.py`**

```python
@dataclass
class Chunk:
    content: str
    chunk_type: str   # "text" / "definition" / "equation" / "figure" / "table"
    metadata: Dict
    embedding: Optional[List[float]]

@dataclass
class SearchResult:
    query: str
    answer: str       # LLM 综合回答
    content: str      # 原始检索文本
    mode: str         # 使用的检索模式
    provider: str     # 使用的 provider
    chunks: List[Chunk]
    metadata: Dict
```
