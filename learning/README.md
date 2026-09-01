# 10 周 RAG/LLM 应用工程师学习计划

适用人群：有 Python/后端基础、RAG 基础不足、目标为 RAG/LLM 应用工程师岗位。
基准投入：每周 15 小时以上；主线周期：10 周。

## 学习方法

每周固定完成：

- 3 小时原理：只学习本周代码需要的概念；
- 8 小时编码：改当前项目或写可运行实验；
- 2 小时源码：阅读 LangChain、LangGraph、LightRAG、MCP 或 Harness 的对应部分；
- 2 小时面试：写答案并进行 5 分钟口述。

不要先通读全部框架源码。先沿项目调用链，再回到框架抽象；每周保留一个可运行结果。

## 10 周路线

| 周次 | 重点 | 项目产出 | 通过标准 |
|---|---|---|---|
| 1 | 项目调用链、Document、FAISS | 代码地图和 5 次查询记录 | 5 分钟讲清请求链路 |
| 2 | Token、Prompt、上下文、幻觉 | 3 版 Prompt 对比 | 能区分模型/检索/Prompt 错误 |
| 3 | Embedding、Chunk、向量距离 | chunk 参数实验表 | 能解释 Chunk 取舍 |
| 4 | BM25、混合检索、RRF | dense/lexical/hybrid 对比 | 能解释为什么混合检索有效 |
| 5 | LangChain 组件 | 手写最小 RAG 和结构化输出实验 | 能脱离项目重写最小 RAG |
| 6 | LangGraph StateGraph | `validate → retrieve → generate` | 能画状态图并说明失败路径 |
| 7 | MCP 工具边界 | 安全的 `get_knowledge_base_info` | 能区分 MCP、REST、Tool Calling |
| 8 | API、日志、安全、Docker、CI | `/health`、`/query`、CI | 异常可诊断、密钥不入库 |
| 9 | RAG 评测 | 35 条 JSONL 评测集和指标报告 | 能用 Recall/MRR 证明改动效果 |
| 10 | 项目表达和系统设计 | 一页架构图、20 道面试题 | 能讲 5/15 分钟项目方案 |

## P0/P1/P2

### P0：必须掌握

LLM 基础、Embedding、Chunking、Dense/Lexical/RRF、LangChain、LangGraph、
评测、API 化、测试、安全和项目表达。

### P1：完成主线后加分

Reranker、Query Rewrite、多轮对话、Checkpoint、MCP、FastAPI、Tracing、Docker、CI。

### P2：暂缓

微调、多 Agent、知识图谱 RAG、分布式向量库、自己训练 Embedding。

## 每周自测模板

```text
本周问题：
我修改了什么：
我测量了什么：
失败案例：
性能/质量变化：
我能否不用看代码解释：
下周要验证的假设：
```

## 推荐阶段标签

```text
v1-baseline   基础 RAG
v2-retrieval  Chunk + 混合检索
v3-workflow   LangGraph + MCP + 来源
v4-production  API + 评测 + 日志 + CI + 安全
```
