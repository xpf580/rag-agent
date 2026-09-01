from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from rag_agent import retriever


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), float(text.count("最大回撤"))] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), float(text.count("最大回撤"))]


def test_retriever_supports_dense_lexical_and_hybrid_modes(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "_build_embeddings", lambda _: FakeEmbeddings())
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text(
        "# 风险管理\n最大回撤是从峰值到谷底的跌幅。\n\n# 动量\n动量关注价格趋势。",
        encoding="utf-8",
    )
    instance = retriever.LocalKnowledgeRetriever(
        knowledge,
        index_dir=tmp_path / "index",
        chunk_size=80,
        chunk_overlap=10,
    )

    assert instance.retrieve("最大回撤", mode="lexical")[0].metadata["section"] == "风险管理"
    assert instance.retrieve("最大回撤", mode="dense")
    assert instance.retrieve("最大回撤", mode="hybrid")[0].metadata["hybrid_score"] > 0

    with pytest.raises(ValueError, match="mode"):
        instance.retrieve("最大回撤", mode="invalid")
    with pytest.raises(ValueError, match="characters"):
        instance.retrieve("最大回撤" * 500)
