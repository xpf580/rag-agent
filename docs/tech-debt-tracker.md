# Tech Debt Tracker

| 优先级 | 债务 | 影响 | 下一步 |
|---|---|---|---|
| P0 | 轮换 `.env` 中曾暴露的 API Key | 账户和数据安全 | 在服务商控制台撤销并重新注入 |
| P1 | 尚未接入 reranker | 复杂问题召回精度有限 | 增加 Cross-Encoder 或托管 rerank |
| P1 | 尚无真实评测集 | 无法量化质量回归 | 建立问题、标准答案、引用集 |
| P1 | LLM 调用无超时/重试 | 网络抖动时体验不稳定 | 在模型 Runnable 外包裹 retry/timeout |
| P2 | MCP 服务启动时全局初始化 Agent | 启动慢，模型失败影响工具发现 | 增加 lazy initialization 和 health tool |
| P2 | LangGraph 目前只有两节点 | 尚未利用持久化和人工审核 | 增加 rewrite/rerank/guardrail 节点 |
