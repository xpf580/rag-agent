# 开源项目研究记录

研究时间：2026-09-01。Stars 会随时间变化，以下是本次通过 GitHub API
核验时的数量；源码快照保存在本项目的 `reference_sources/`，该目录被
`.gitignore` 忽略。

## 按相关性和热度分类

| 项目 | Stars（本次核验） | 分类 | 重点借鉴 |
|---|---:|---|---|
| `modelcontextprotocol/servers` | 89,990 | MCP 官方服务器 | `fetch`、`filesystem`、`memory`、测试和安全边界 |
| `langchain-ai/langchain` | 145,390 | LLM 应用基础层 | Runnable、Document、Retriever、模型适配 |
| `HKUDS/LightRAG` | 39,299 | 工程化/混合 RAG | local/global/hybrid/mix 查询、reranker、引用、API、文档处理 |
| `langchain-ai/langgraph` | 40,813 | 状态化编排 | StateGraph、持久化、人工介入、RAG 示例 |
| `langchain-ai/deepagents` | 28,780 | Agent Harness | 文件系统、子 Agent、上下文管理、MCP 工具、持久化 |
| `langchain-ai/open_deep_research` | 12,680 | 研究型 Agent | clarify → brief → supervisor → researcher → compression → report |
| `lattice-technologies-inc/harness-engineering` | 2 | 工程规范/Harness | AGENTS、架构文档、计划流水线、质量分数、技术债和 hooks |

## 分门别类结论

### LangChain

适合作为组件层：模型、Prompt、Document、Embedding、Retriever、工具调用。
本项目保留 LangChain 的组件抽象，但把检索策略独立到 `retriever.py`，
避免所有逻辑堆在 Chain 中。

### LangGraph

适合作为运行时编排层：每个节点输入输出明确，后续可以加入 query rewrite、
rerank、guardrail、人工审核、重试和 checkpoint。本项目已经落地最小
`retrieve → generate` 状态图。

### 混合 RAG / LightRAG

重点是不要把“向量库”误认为完整 RAG。可落地系统还需要文档切分、索引
生命周期、精确词法召回、重排序、上下文预算、来源引用、文档删除和 API
输入限制。本项目先落地 dense + lexical + RRF、manifest 重建和 source 引用。

### MCP

MCP 适合作为工具边界，而不是替代 RAG。`mcp_server.py` 将本地知识库能力
暴露为两个工具，让 Claude、ChatGPT 或其他 MCP 客户端能够调用：

- `search_knowledge`：只检索，适合 Agent 自己组织答案；
- `answer_question`：检索并生成，适合直接问答。

### Harness Engineering

重点不是增加一个 Agent，而是把“如何正确修改项目”的知识写进仓库：

- `AGENTS.md` 做导航；
- `ARCHITECTURE.md` 写边界和数据流；
- `docs/RELIABILITY.md`、`docs/SECURITY.md`、质量评分和技术债可持续维护；
- 计划、测试、文档和代码一起交付。

本项目已经采用这一套最小闭环。

## 源码研究材料

已下载的快照：

```text
reference_sources/langgraph
reference_sources/LightRAG
reference_sources/open_deep_research
reference_sources/deepagents
reference_sources/harness-engineering
reference_sources/langchain-source
reference_sources/mcp-servers-source
```

以上快照均用于本地分类研究，不参与运行时依赖。外部源码可能持续变化，
升级前应重新核验 API、许可证和安全公告。
