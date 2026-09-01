# Hybrid RAG 工程化计划（2026-09-01）

## 目标

将单文档 Demo 改造为可复用的本地混合 RAG 核心，并预留 LangGraph、
MCP 和生产运维扩展点。

## 已完成

- Markdown/TXT 多文件扫描；
- 标题感知切分和 overlap；
- 内容哈希 manifest；
- FAISS 持久化与自动重建；
- 词法 BM25-like 召回；
- dense + lexical 的 RRF 融合；
- LangGraph retrieve → generate 工作流；
- Streamlit 来源展示；
- MCP `search_knowledge` / `answer_question` 工具；
- AGENTS、架构、安全、可靠性、质量和技术债文档。

## 验收

```powershell
python -m compileall -q app.py rag.py mcp_server.py src
pytest -q
```

## 后续垂直切片

1. 评测集和离线指标；
2. reranker；
3. 多轮对话和 checkpoint；
4. HTTP API、认证、限流和 tracing；
5. Qdrant/pgvector/OpenSearch 适配。
