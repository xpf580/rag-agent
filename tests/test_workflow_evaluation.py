from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from rag_agent.evaluation import EvaluationCase, evaluate_retrieval
from rag_agent.workflow import RAGWorkflow


class StubRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, k=4, mode="hybrid"):
        self.calls.append((query, k, mode))
        if "未知" in query:
            return []
        return [
            Document(
                page_content="最大回撤是峰值到谷底的跌幅。",
                metadata={"source": "knowledge.md", "section": "6.1 最大回撤", "chunk_id": "1"},
            )
        ]


class StubLLM:
    def invoke(self, prompt):
        return type("Response", (), {"content": "基于上下文：最大回撤是峰值到谷底的跌幅。"})()


def test_workflow_validates_and_returns_sources():
    retriever = StubRetriever()
    workflow = RAGWorkflow(
        retriever,
        StubLLM(),
        PromptTemplate.from_template("Context={context}\nQuestion={question}"),
        max_query_chars=20,
    )

    result = workflow.invoke("  最大回撤是什么？  ")

    assert result["answer"].startswith("基于上下文")
    assert result["sources"][0]["section"] == "6.1 最大回撤"
    assert retriever.calls == [("最大回撤是什么？", 4, "hybrid")]


def test_workflow_rejects_overlong_query_without_retrieval():
    retriever = StubRetriever()
    workflow = RAGWorkflow(retriever, None, None, max_query_chars=4)

    result = workflow.invoke("这是一个过长的问题")

    assert "不能超过" in result["answer"]
    assert retriever.calls == []


def test_retrieval_report_supports_positive_and_negative_cases():
    report = evaluate_retrieval(
        StubRetriever(),
        [
            EvaluationCase("最大回撤", expected_sections=("6.1 最大回撤",)),
            EvaluationCase("未知问题", expect_no_match=True),
        ],
        k=1,
    )

    assert report.total == 2
    assert report.hit_rate_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
