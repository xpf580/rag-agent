# Labs

## Week 1: trace a query

```powershell
streamlit run app.py
```

在 `agent.py`、`workflow.py`、`retriever.py` 设置断点，记录一次问题的输入、Chunk、来源和答案。

## Week 2: prompt comparison

复制 `RAGAgent._build_prompt`，分别测试“普通回答”“必须引用”“允许拒答”三个版本，记录幻觉和拒答差异。

## Week 3-4: retrieval comparison

```powershell
python learning/labs/compare_retrieval.py
```

比较 dense、lexical、hybrid 的 `hit_rate_at_k`、`MRR` 和失败问题。

## Week 5: minimal LangChain RAG

不使用 `RAGAgent`，独立完成 Document → Embedding → VectorStore → Retriever → Prompt → Model。

## Week 6: LangGraph extension

将当前工作流增加一个 `validate` 或 `query_rewrite` 节点，并为节点写行为测试。

## Week 7: MCP boundary

运行 `python mcp_server.py`，审查每个工具的输入、输出和权限；不要加入任意文件读取或 shell 工具。

## Week 8: service boundary

```powershell
uvicorn api:app --reload
```

使用 `/health` 和 `/query` 验证成功、空输入、超长输入、索引失败和模型失败。

## Week 9: evaluation regression

```powershell
python scripts/evaluate_retrieval.py --mode hybrid
```

修改 Chunk 或权重前后保存 JSON 报告，禁止只凭主观感受判断效果。

## Week 10: interview rehearsal

按照 `learning/interview_questions.md`，完成一次 5 分钟项目介绍和一次 15 分钟系统设计。
