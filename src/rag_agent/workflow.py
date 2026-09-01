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


class RAGWorkflow:
    """Small LangGraph workflow for observable, extensible RAG execution."""

    def __init__(self, retriever: Any, llm: Any, prompt: PromptTemplate | None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        if StateGraph is None:
            self.graph = None
            return
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        self.graph = builder.compile()

    def _retrieve(self, state: RAGState) -> dict[str, Any]:
        documents = self.retriever.retrieve(state["question"], k=4)
        context = "\n\n".join(
            f"[{index}] {doc.page_content}"
            for index, doc in enumerate(documents, start=1)
        )
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
        if self.llm is None or self.prompt is None:
            answer = (
                "未配置可用的大语言模型。以下是本地知识库中检索到的内容：\n\n"
                + state.get("context", "")[:3000]
            )
        else:
            prompt_text = self.prompt.format(
                context=state.get("context", "无相关上下文"),
                question=state["question"],
            )
            response = self.llm.invoke(prompt_text)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "".join(getattr(item, "text", str(item)) for item in content)
            answer = str(content).strip()
        return {"answer": answer}

    def invoke(self, question: str) -> RAGState:
        """Execute the graph and return the answer together with source metadata."""
        if self.graph is None:
            state = {"question": question}
            state.update(self._retrieve(state))
            state.update(self._generate(state))
            return state
        return self.graph.invoke({"question": question})
