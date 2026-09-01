from rag_agent.retriever import _split_text, _tokenize


def test_tokenize_supports_english_and_chinese():
    tokens = _tokenize("Sharpe Ratio 与最大回撤")

    assert "sharpe" in tokens
    assert "ratio" in tokens
    assert "最" in tokens
    assert "大" in tokens


def test_split_text_preserves_markdown_heading_and_overlap():
    text = "# 风险管理\n" + ("最大回撤用于衡量风险。 " * 80)

    chunks = _split_text(text, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(heading == "风险管理" for heading, _ in chunks)
    assert all(len(content) <= 120 for _, content in chunks)
    assert chunks[0][1][-20:] in chunks[1][1]
