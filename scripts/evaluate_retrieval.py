from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_agent.config import INDEX_DIR, KNOWLEDGE_PATH  # noqa: E402
from rag_agent.evaluation import evaluate_retrieval, load_cases  # noqa: E402
from rag_agent.retriever import LocalKnowledgeRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval")
    parser.add_argument("--dataset", default=str(ROOT / "data/evaluation/questions.jsonl"))
    parser.add_argument("--mode", choices=["hybrid", "dense", "lexical"], default="hybrid")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    retriever = LocalKnowledgeRetriever(KNOWLEDGE_PATH, index_dir=INDEX_DIR)
    report = evaluate_retrieval(retriever, load_cases(args.dataset), k=args.k, mode=args.mode)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
