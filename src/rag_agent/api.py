from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import RAGAgent
from .config import MAX_QUERY_CHARS


class SourceResponse(BaseModel):
    source: str
    section: str = ""
    chunk_id: str = ""
    hybrid_score: float | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    error: str = ""


app = FastAPI(title="Hybrid RAG Agent", version="1.0.0")


@lru_cache(maxsize=1)
def get_agent() -> RAGAgent:
    """Create the expensive RAG agent lazily and reuse it per process."""
    return RAGAgent()


@app.get("/health")
def health() -> dict[str, Any]:
    """Return model, workflow and index health without exposing credentials."""
    try:
        return get_agent().health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG agent unavailable: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer one bounded query and return traceable source metadata."""
    try:
        result = get_agent().answer_with_sources(request.query)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG query failed: {exc}") from exc
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])
    return QueryResponse(
        answer=result["answer"],
        sources=[SourceResponse(**source) for source in result.get("sources", [])],
    )
