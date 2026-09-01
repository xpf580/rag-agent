from __future__ import annotations

import logging
from typing import Any

from langchain_core.prompts import PromptTemplate

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    INDEX_DIR,
    KNOWLEDGE_PATH,
    LEXICAL_WEIGHT,
    LLM_TIMEOUT_SECONDS,
    MAX_CONTEXT_CHARS,
    MAX_QUERY_CHARS,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    RETRIEVAL_K,
)
from .retriever import LocalKnowledgeRetriever
from .workflow import RAGWorkflow

logger = logging.getLogger(__name__)


class RAGAgent:
    """Application facade for hybrid retrieval and LangGraph answer generation."""

    def __init__(
        self,
        knowledge_path: str | None = None,
        retriever: LocalKnowledgeRetriever | None = None,
        llm: Any | None = None,
    ) -> None:
        self.knowledge_path = knowledge_path or str(KNOWLEDGE_PATH)
        self.retriever = retriever or LocalKnowledgeRetriever(
            knowledge_path=self.knowledge_path,
            embedding_model=EMBEDDING_MODEL,
            index_dir=INDEX_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            dense_weight=DENSE_WEIGHT,
            lexical_weight=LEXICAL_WEIGHT,
            max_query_chars=MAX_QUERY_CHARS,
        )
        self.llm = self._build_llm() if llm is None else llm
        self.prompt = self._build_prompt()
        self.workflow = RAGWorkflow(
            self.retriever,
            self.llm,
            self.prompt,
            retrieval_k=RETRIEVAL_K,
            max_query_chars=MAX_QUERY_CHARS,
            max_context_chars=MAX_CONTEXT_CHARS,
        )

    def _build_llm(self) -> Any:
        if OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI

                kwargs: dict[str, Any] = {
                    "model": OPENAI_MODEL,
                    "temperature": 0,
                    "api_key": OPENAI_API_KEY,
                    "timeout": LLM_TIMEOUT_SECONDS,
                    "max_retries": 2,
                }
                if OPENAI_BASE_URL:
                    kwargs["base_url"] = OPENAI_BASE_URL
                return ChatOpenAI(**kwargs)
            except Exception as exc:
                logger.exception("OpenAI-compatible LLM initialization failed: %s", exc)

        try:
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                model="llama3.2",
                base_url=OLLAMA_BASE_URL,
                temperature=0,
            )
        except Exception as exc:
            logger.warning("Ollama initialization failed: %s", exc)
            return None

    def _build_prompt(self) -> PromptTemplate | None:
        if self.llm is None:
            return None
        template = (
            "你是一个严谨的知识库问答助手。\n"
            "只能依据 Context 中的信息回答，不要编造事实。\n"
            "如果 Context 不足以回答，请明确说明知识库中没有足够信息。\n"
            "回答应简洁、结构清晰；必要时使用列表。\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )
        return PromptTemplate.from_template(template)

    def answer_with_sources(self, question: str) -> dict[str, Any]:
        """Answer a question and return traceable source metadata."""
        result = self.workflow.invoke(question)
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "error": result.get("error", ""),
            "retriever": self.retriever.describe(),
        }

    def answer(self, question: str) -> str:
        """Backward-compatible answer-only API used by the Streamlit app."""
        return str(self.answer_with_sources(question)["answer"])

    def health(self) -> dict[str, Any]:
        """Return a safe health payload without exposing secrets."""
        return {
            "status": "ok",
            "llm_configured": self.llm is not None,
            "workflow": "langgraph" if self.workflow.graph is not None else "fallback",
            "retriever": self.retriever.describe(),
        }
