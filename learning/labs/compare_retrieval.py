from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_agent.config import INDEX_DIR, KNOWLEDGE_PATH  # noqa: E402
from rag_agent.evaluation import evaluate_retrieval, load_cases  # noqa: E402
from rag_agent.retriever import LocalKnowledgeRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG retrieval strategies")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    retriever = LocalKnowledgeRetriever(KNOWLEDGE_PATH, index_dir=INDEX_DIR)
    cases = load_cases(ROOT / "data/evaluation/questions.jsonl")
    reports = {
        mode: evaluate_retrieval(retriever, cases, k=args.k, mode=mode).to_dict()
        for mode in ("dense", "lexical", "hybrid")
    }
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
