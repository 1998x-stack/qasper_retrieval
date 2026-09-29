"""Dependency-free retrieval metric primitives."""

from typing import Iterable, Sequence, Set


def average_precision_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Iterable[int],
    k: int,
) -> float:
    """Compute AP@K with missed relevant documents correctly penalized."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    relevant: Set[int] = set(relevant_ids)
    if not relevant:
        return 0.0

    score_sum = 0.0
    hits = 0
    seen = set()

    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)

        if doc_id in relevant:
            hits += 1
            score_sum += hits / rank

    denominator = min(len(relevant), k)
    return score_sum / denominator
