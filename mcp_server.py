from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_agent.agent import RAGAgent  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - depends on optional runtime
    raise RuntimeError("Install the `mcp` package to run the MCP server.") from exc


mcp = FastMCP("local-hybrid-rag")


@lru_cache(maxsize=1)
def get_agent() -> RAGAgent:
    """Lazily initialize the expensive RAG agent once per MCP process."""
    return RAGAgent()


@mcp.tool()
def search_knowledge(query: str, k: int = 4) -> list[dict[str, Any]]:
    """Search the local knowledge base and return ranked chunks with sources."""
    documents = get_agent().retriever.retrieve(query, k=max(1, min(k, 10)))
    return [
        {
            "content": document.page_content,
            "source": document.metadata.get("source", ""),
            "section": document.metadata.get("section", ""),
            "chunk_id": document.metadata.get("chunk_id", ""),
            "hybrid_score": document.metadata.get("hybrid_score"),
        }
        for document in documents
    ]


@mcp.tool()
def answer_question(question: str) -> dict[str, Any]:
    """Answer a question with the local RAG agent and include citations."""
    return get_agent().answer_with_sources(question)


@mcp.tool()
def get_knowledge_base_info() -> dict[str, Any]:
    """Return safe index metadata without exposing documents or credentials."""
    return get_agent().health()


if __name__ == "__main__":
    mcp.run()
