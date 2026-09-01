# Hybrid RAG Agent

## 项目定位

这是一个面向本地 Markdown/TXT 知识库的可落地 RAG 问答服务，使用
LangChain 负责模型与文档抽象，LangGraph 负责可观测工作流，FAISS 负责
向量检索，并通过词法检索 + 向量检索实现混合召回。

## 开发入口

| 需求 | 位置 |
|---|---|
| Streamlit 页面 | `app.py` |
| Agent 门面 | `src/rag_agent/agent.py` |
| LangGraph 工作流 | `src/rag_agent/workflow.py` |
| 混合检索与索引 | `src/rag_agent/retriever.py` |
| 配置 | `src/rag_agent/config.py` |
| 索引构建命令 | `rag.py` |
| MCP 工具服务 | `mcp_server.py` |
| 知识库 | `data/` |
| 研究材料 | `reference_sources/` |

## 工作约束

1. 修改检索行为时，同时更新测试和 `docs/ARCHITECTURE.md`。
2. 不提交 `.env`、API Key、FAISS 索引和外部参考仓库。
3. 所有外部输入必须有长度、路径和类型边界。
4. 回答必须能返回来源；没有足够上下文时必须明确拒答或降级。
5. 优先写行为测试，避免仅测试内部调用顺序。
6. 运行质量检查：

```powershell
python -m compileall -q app.py rag.py mcp_server.py src
pytest -q
```
