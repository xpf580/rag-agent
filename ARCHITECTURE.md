# Architecture

## Domains

| Domain | Path | Responsibility |
|---|---|---|
| UI | `app.py` | Streamlit 输入、回答和来源展示 |
| Configuration | `src/rag_agent/config.py` | 环境变量和路径配置 |
| Ingestion | `rag.py` | 扫描文档、切分、构建持久化索引 |
| Retrieval | `src/rag_agent/retriever.py` | FAISS dense retrieval、词法检索、RRF 融合 |
| Orchestration | `src/rag_agent/workflow.py` | LangGraph retrieve → generate 状态图 |
| Model adapter | `src/rag_agent/agent.py` | OpenAI-compatible / Ollama 模型适配 |
| Tool boundary | `mcp_server.py` | MCP `search_knowledge` 与 `answer_question` 工具 |
| API | `api.py`, `src/rag_agent/api.py` | FastAPI `/health` 和 `/query` 边界 |
| Evaluation | `src/rag_agent/evaluation.py`, `data/evaluation/` | 无 LLM 的检索回归评测 |

## Data Flow

```text
data/*.md, data/*.txt
  → heading-aware chunking
  → chunks.json + index.faiss + manifest.json
  → dense similarity search
  → lexical BM25-like scoring
  → reciprocal-rank fusion
  → LangGraph state
  → constrained LLM answer + source metadata
```

## Important Contracts

- `RAGAgent.answer(question) -> str` 保持向后兼容。
- `RAGAgent.answer_with_sources(question) -> dict` 返回 `answer`、`sources`、`error`
  和检索器运行信息。
- LangGraph 在生成前执行 query 长度校验，并限制上下文字符预算。
- `LocalKnowledgeRetriever.retrieve(query, k) -> list[Document]` 返回包含
  `source`、`section`、`chunk_id`、`hybrid_score` 的 Document。
- `data/faiss_index/manifest.json` 由知识库内容哈希和切分配置决定；
  内容或配置变化时自动重建索引。

## Extension Points

1. 在 `retrieve` 前增加 query rewrite。
2. 在 RRF 后增加 Cross-Encoder reranker。
3. 在 `workflow.py` 增加事实性检查、人工审核和多轮记忆节点。
4. 将 FAISS 替换为 Qdrant、pgvector 或 OpenSearch。
