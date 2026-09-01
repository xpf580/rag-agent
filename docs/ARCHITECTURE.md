# 工程化 RAG 架构说明

## 为什么选择混合检索

向量检索擅长同义表达和语义匹配，词法检索擅长专有名词、指标名、模型名和
精确数字。金融量化知识中同时存在“最大回撤”“Sharpe Ratio”“Amihud”
等精确术语，因此采用 dense + lexical 双路召回，再用 Reciprocal Rank
Fusion（RRF）统一排序。

当前默认权重为：

```text
dense = 0.6
lexical = 0.4
```

权重通过 `.env` 中的 `DENSE_WEIGHT` 和 `LEXICAL_WEIGHT` 调整。

## 索引生命周期

启动时系统对知识库文件内容计算 SHA-256：

1. manifest 与文件指纹、Embedding、切分参数一致：加载已有 FAISS。
2. 任一条件变化：重新切分并构建 FAISS。
3. `chunks.json` 保存可审计的文本和元数据，避免依赖 FAISS pickle
   反推原始文档。

## 上下文工程

每个 chunk 保留：

- 原文件路径；
- Markdown 章节；
- 稳定的 chunk id；
- dense/lexical 排名；
- 混合分数。

模型 Prompt 明确要求只依据上下文回答，检索结果不足时说明知识库信息不足。
