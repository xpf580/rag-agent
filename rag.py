from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_agent.config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    INDEX_DIR,
    KNOWLEDGE_PATH,
    LEXICAL_WEIGHT,
)
from rag_agent.retriever import LocalKnowledgeRetriever  # noqa: E402


class KnowledgeBaseBuilder:
    """Build or refresh the persistent hybrid-RAG index."""

    def __init__(
        self,
        data_dir: str = "data",
        embedding_model: str = EMBEDDING_MODEL,
        index_dir: str | Path = INDEX_DIR,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)

    def build_from_directory(self, force_rebuild: bool = True) -> dict[str, object]:
        retriever = LocalKnowledgeRetriever(
            knowledge_path=self.data_dir,
            embedding_model=self.embedding_model,
            index_dir=self.index_dir,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            dense_weight=DENSE_WEIGHT,
            lexical_weight=LEXICAL_WEIGHT,
            force_rebuild=force_rebuild,
        )
        info = retriever.describe()
        print(f"Built persistent hybrid index: {info}")
        return info


if __name__ == "__main__":
    KnowledgeBaseBuilder(
        data_dir=str(KNOWLEDGE_PATH.parent),
        index_dir=INDEX_DIR,
    ).build_from_directory()
