from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvaluationCase:
    """A retrieval-only evaluation example."""

    question: str
    expected_sections: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()
    expected_answer: str = ""
    expect_no_match: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationCase":
        return cls(
            question=str(payload["question"]),
            expected_sections=tuple(payload.get("expected_sections", [])),
            expected_sources=tuple(payload.get("expected_sources", [])),
            expected_answer=str(payload.get("expected_answer", "")),
            expect_no_match=bool(payload.get("expect_no_match", False)),
        )


@dataclass(frozen=True)
class RetrievalReport:
    """Aggregate retrieval metrics for a fixed evaluation set."""

    total: int
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    mean_precision_at_k: float
    k: int
    mode: str
    rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases(path: str | Path) -> list[EvaluationCase]:
    """Load JSONL evaluation cases."""
    cases: list[EvaluationCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            cases.append(EvaluationCase.from_dict(json.loads(line)))
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def _matches(document: Any, case: EvaluationCase) -> bool:
    section = str(document.metadata.get("section", ""))
    source = str(document.metadata.get("source", ""))
    return (
        any(expected in section for expected in case.expected_sections)
        or any(expected in source for expected in case.expected_sources)
    )


def evaluate_retrieval(
    retriever: Any,
    cases: Iterable[EvaluationCase],
    k: int = 4,
    mode: str = "hybrid",
) -> RetrievalReport:
    """Evaluate hit rate, MRR and precision without calling an LLM."""
    rows: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    hits = 0
    cases = list(cases)
    for case in cases:
        documents = retriever.retrieve(case.question, k=k, mode=mode)
        matched_ranks = [
            rank for rank, document in enumerate(documents, start=1) if _matches(document, case)
        ]
        first_rank = matched_ranks[0] if matched_ranks else None
        success = (first_rank is None) if case.expect_no_match else (first_rank is not None)
        if success:
            hits += 1
        reciprocal_ranks.append(1.0 if success and case.expect_no_match else (1 / first_rank if first_rank else 0.0))
        precisions.append(
            (1.0 if not matched_ranks else 0.0)
            if case.expect_no_match
            else len(matched_ranks) / max(len(documents), 1)
        )
        rows.append(
            {
                "question": case.question,
                "hit": success,
                "first_rank": first_rank,
                "expect_no_match": case.expect_no_match,
                "sections": [document.metadata.get("section", "") for document in documents],
            }
        )
    total = len(cases)
    return RetrievalReport(
        total=total,
        hit_rate_at_k=round(hits / max(total, 1), 4),
        mean_reciprocal_rank=round(sum(reciprocal_ranks) / max(total, 1), 4),
        mean_precision_at_k=round(sum(precisions) / max(total, 1), 4),
        k=k,
        mode=mode,
        rows=rows,
    )
