from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # Allows lightweight unit tests before optional deps are installed.
    END = "__end__"
    START = "__start__"
    StateGraph = None


class RAGState(TypedDict, total=False):
    question: str
    documents: list[Document]
    context: str
    answer: str
    sources: list[dict[str, Any]]
    error: str


class RAGWorkflow:
    """Stateful RAG workflow with validation, retrieval and generation nodes."""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        prompt: PromptTemplate | None,
        retrieval_k: int = 4,
        max_query_chars: int = 400,
        max_context_chars: int = 12000,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.retrieval_k = retrieval_k
        self.max_query_chars = max_query_chars
        self.max_context_chars = max_context_chars
        self.graph = self._build_graph() if StateGraph is not None else None

    def _build_graph(self) -> Any:
        builder = StateGraph(RAGState)
        builder.add_node("validate", self._validate)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_edge(START, "validate")
        builder.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {"retrieve": "retrieve", "generate": "generate"},
        )
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def _validate(self, state: RAGState) -> dict[str, Any]:
        question = state.get("question", "").strip()
        if not question:
            return {"error": "问题不能为空。", "question": question}
        if len(question) > self.max_query_chars:
            return {
                "error": f"问题长度不能超过 {self.max_query_chars} 个字符。",
                "question": question[: self.max_query_chars],
            }
        return {"question": question, "error": ""}

    @staticmethod
    def _route_after_validation(state: RAGState) -> str:
        return "generate" if state.get("error") else "retrieve"

    def _retrieve(self, state: RAGState) -> dict[str, Any]:
        documents = self.retriever.retrieve(state["question"], k=self.retrieval_k)
        context = "\n\n".join(
            f"[{index}] {doc.page_content}"
            for index, doc in enumerate(documents, start=1)
        )[: self.max_context_chars]
        sources = [
            {
                "source": doc.metadata.get("source", ""),
                "section": doc.metadata.get("section", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "hybrid_score": doc.metadata.get("hybrid_score"),
            }
            for doc in documents
        ]
        return {"documents": documents, "context": context, "sources": sources}

    def _generate(self, state: RAGState) -> dict[str, str]:
        if state.get("error"):
            return {"answer": state["error"]}
        context = state.get("context", "").strip()
        if not context:
            return {"answer": "知识库中没有检索到足够相关的信息。"}
        if self.llm is None or self.prompt is None:
            return {
                "answer": "未配置可用的大语言模型。以下是本地知识库中检索到的内容：\n\n"
                + context[:3000]
            }
        try:
            prompt_text = self.prompt.format(
                context=context,
                question=state["question"],
            )
            response = self.llm.invoke(prompt_text)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "".join(getattr(item, "text", str(item)) for item in content)
            return {"answer": str(content).strip()}
        except Exception as exc:
            return {"answer": f"模型调用失败，请稍后重试：{exc}"}

    def invoke(self, question: str) -> RAGState:
        """Execute the workflow and return answer, sources and diagnostic state."""
        if self.graph is None:
            state: RAGState = {"question": question}
            state.update(self._validate(state))
            if not state.get("error"):
                state.update(self._retrieve(state))
            state.update(self._generate(state))
            return state
        return self.graph.invoke({"question": question})
