# Hybrid RAG Agent

一个用于学习和实践 RAG/LLM 应用工程的本地知识库问答项目。

## 能力

- Markdown/TXT 多文档导入与标题感知切分；
- Hugging Face Embedding + FAISS 持久化索引；
- Dense Retrieval + BM25-like Lexical Retrieval；
- Reciprocal Rank Fusion（RRF）混合排序；
- LangGraph `validate → retrieve → generate` 工作流；
- Streamlit UI、FastAPI API 和 MCP 工具；
- 来源引用、输入边界、降级策略、检索评测和 CI。

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python rag.py
streamlit run app.py
```

API：

```powershell
uvicorn api:app --reload
```

- `GET /health`
- `POST /query`，请求体：`{"query":"最大回撤是什么？"}`

MCP：

```powershell
python mcp_server.py
```

评测：

```powershell
python scripts/evaluate_retrieval.py --mode hybrid
python scripts/evaluate_retrieval.py --mode dense
python scripts/evaluate_retrieval.py --mode lexical
```

## 阅读顺序

```text
app.py
  → src/rag_agent/agent.py
  → src/rag_agent/workflow.py
  → src/rag_agent/retriever.py
  → src/rag_agent/config.py
  → rag.py / api.py / mcp_server.py
```

## 质量检查

```powershell
python -m compileall -q app.py api.py rag.py mcp_server.py src scripts
pytest -q
```

## 学习材料

- [10 周学习计划](learning/README.md)
- [面试题与项目表达](learning/interview_questions.md)
- [工程架构](docs/ARCHITECTURE.md)
- [检索评测数据](data/evaluation/questions.jsonl)
- [开源项目研究](docs/RESEARCH.md)
- [安全要求](docs/SECURITY.md)
