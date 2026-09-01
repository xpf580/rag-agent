# Reliability

## 当前可靠性策略

- 知识库为空或路径不存在时快速失败，不生成无依据答案。
- 索引通过内容签名和切分配置自动失效。
- FAISS 加载失败时自动重建。
- LLM 不可用时返回检索到的本地内容，并明确提示为降级结果。
- 问题为空时拒绝执行。

## 生产化待办

1. 增加超时、指数退避和限流。
2. 使用 LangGraph checkpoint 支持中断恢复。
3. 记录检索耗时、LLM 耗时、命中文档和 token 用量。
4. 建立离线评测集，跟踪 Recall@K、MRR、nDCG、faithfulness 和 answer
   relevance。
5. 为索引构建和查询建立健康检查接口。
