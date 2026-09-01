from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")


def _build_embeddings(model_name: str) -> Any:
    """Build local embeddings, falling back to an OpenAI-compatible backend."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as exc:
        logger.warning("langchain-huggingface unavailable: %s", exc)

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as exc:
        logger.warning("local Hugging Face embeddings unavailable: %s", exc)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.lower() not in {"replace_me", "your_api_key_here"}:
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as exc:
            raise RuntimeError(
                "Install langchain-openai or install sentence-transformers "
                "for local embeddings."
            ) from exc
        return OpenAIEmbeddings()

    raise RuntimeError(
        "No embedding backend found. Install langchain-huggingface and "
        "sentence-transformers, or configure OPENAI_API_KEY."
    )


def _tokenize(text: str) -> list[str]:
    """Tokenize English words and individual CJK characters."""
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, str]]:
    """Split Markdown into bounded chunks while retaining the current heading."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")

    matches = list(_HEADING_PATTERN.finditer(text))
    sections: list[tuple[str, str]] = []
    if not matches:
        sections.append(("", text))
    else:
        if matches[0].start() > 0:
            sections.append(("", text[: matches[0].start()]))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections.append((match.group(2).strip(), text[match.start() : end]))

    chunks: list[tuple[str, str]] = []
    for heading, section in sections:
        normalized = section.strip()
        if not normalized:
            continue
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            piece = normalized[start:end].strip()
            if piece:
                chunks.append((heading, piece))
            if end == len(normalized):
                break
            start = end - chunk_overlap
    return chunks


class LocalKnowledgeRetriever:
    """Hybrid dense + lexical retriever backed by a persistent FAISS index."""

    def __init__(
        self,
        knowledge_path: str | Path | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: str | Path = "data/faiss_index",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        dense_weight: float = 0.6,
        lexical_weight: float = 0.4,
        force_rebuild: bool = False,
        max_query_chars: int = 400,
    ) -> None:
        self.knowledge_path = Path(knowledge_path) if knowledge_path else None
        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.max_query_chars = max_query_chars
        self.embeddings = _build_embeddings(embedding_model)
        self.documents: list[Document] = []
        self.vectorstore = self._load_or_build(force_rebuild)
        self._prepare_lexical_index()

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    def _source_files(self) -> list[Path]:
        if self.knowledge_path and self.knowledge_path.is_file():
            return [self.knowledge_path]
        root = self.knowledge_path or Path("data")
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".md", ".txt"}
            and "faiss_index" not in path.parts
        )

    def _corpus_signature(self, files: list[Path]) -> str:
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path.resolve()).encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _load_documents_from_sources(self, files: list[Path]) -> list[Document]:
        documents: list[Document] = []
        for source in files:
            content = source.read_text(encoding="utf-8")
            for index, (heading, chunk) in enumerate(
                _split_text(content, self.chunk_size, self.chunk_overlap)
            ):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": str(source),
                            "section": heading,
                            "chunk_id": f"{source.name}:{index}",
                        },
                    )
                )
        if not documents:
            raise FileNotFoundError("No .md or .txt files were found in the knowledge base")
        return documents

    def _manifest_matches(self, signature: str) -> bool:
        if not self.manifest_path.exists() or not (self.index_dir / "index.faiss").exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest == {
            "signature": signature,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _load_cached_documents(self) -> list[Document]:
        payload = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        return [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in payload
        ]

    def _save_documents(self, documents: list[Document], signature: str) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        self.chunks_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "signature": signature,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_or_build(self, force_rebuild: bool) -> FAISS:
        files = self._source_files()
        signature = self._corpus_signature(files)
        if (
            not force_rebuild
            and self._manifest_matches(signature)
            and self.chunks_path.exists()
        ):
            try:
                self.documents = self._load_cached_documents()
                return FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception as exc:
                logger.warning("Cached FAISS index is unusable; rebuilding: %s", exc)

        self.documents = self._load_documents_from_sources(files)
        vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        vectorstore.save_local(str(self.index_dir))
        self._save_documents(self.documents, signature)
        logger.info("Built FAISS index with %d chunks", len(self.documents))
        return vectorstore

    def _prepare_lexical_index(self) -> None:
        self.tokenized_documents = [_tokenize(doc.page_content) for doc in self.documents]
        self.document_lengths = [len(tokens) for tokens in self.tokenized_documents]
        self.average_length = sum(self.document_lengths) / max(len(self.documents), 1)
        self.document_frequency = Counter(
            token
            for tokens in self.tokenized_documents
            for token in set(tokens)
        )

    def _lexical_scores(self, query: str) -> dict[int, float]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}
        scores: dict[int, float] = {}
        total_documents = len(self.documents)
        for index, tokens in enumerate(self.tokenized_documents):
            term_counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                document_frequency = self.document_frequency.get(token, 0)
                if not document_frequency:
                    continue
                idf = math.log(
                    1 + (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                term_frequency = term_counts.get(token, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + 1.5 * (
                    0.25 + 0.75 * self.document_lengths[index] / max(self.average_length, 1)
                )
                score += idf * (term_frequency * 2.5) / denominator
            if score > 0:
                scores[index] = score
        return scores

    def retrieve(
        self,
        query: str,
        k: int = 4,
        mode: Literal["hybrid", "dense", "lexical"] = "hybrid",
    ) -> list[Document]:
        """Retrieve chunks with dense, lexical, or fused ranking."""
        query = query.strip()
        if not query:
            return []
        if len(query) > self.max_query_chars:
            raise ValueError(f"query must not exceed {self.max_query_chars} characters")
        if mode not in {"hybrid", "dense", "lexical"}:
            raise ValueError("mode must be hybrid, dense, or lexical")
        if k < 1 or k > 50:
            raise ValueError("k must be between 1 and 50")

        lexical_scores = self._lexical_scores(query)
        lexical_rank = {
            index: rank
            for rank, (index, _) in enumerate(
                sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True),
                start=1,
            )
        }
        if mode == "lexical":
            return [self.documents[index] for index in list(lexical_rank)[:k]]

        candidate_count = min(max(k * 4, 12), len(self.documents))
        dense_results = self.vectorstore.similarity_search_with_score(
            query, k=candidate_count
        )
        if mode == "dense":
            return [doc for doc, _ in dense_results[:k]]
        dense_rank = {
            doc.metadata["chunk_id"]: rank
            for rank, (doc, _) in enumerate(dense_results, start=1)
        }

        fused: list[tuple[float, Document]] = []
        for index, doc in enumerate(self.documents):
            dense_position = dense_rank.get(doc.metadata["chunk_id"])
            lexical_position = lexical_rank.get(index)
            score = 0.0
            if dense_position:
                score += self.dense_weight / (60 + dense_position)
            if lexical_position:
                score += self.lexical_weight / (60 + lexical_position)
            if score:
                fused.append(
                    (
                        score,
                        Document(
                            page_content=doc.page_content,
                            metadata={
                                **doc.metadata,
                                "hybrid_score": round(score, 6),
                                "dense_rank": dense_position,
                                "lexical_rank": lexical_position,
                            },
                        ),
                    )
                )
        fused.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in fused[:k]]

    def describe(self) -> dict[str, Any]:
        """Return safe operational information for the UI and health checks."""
        return {
            "chunks": len(self.documents),
            "index_dir": str(self.index_dir),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "dense_weight": self.dense_weight,
            "lexical_weight": self.lexical_weight,
            "max_query_chars": self.max_query_chars,
        }
