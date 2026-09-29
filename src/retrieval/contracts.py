"""Pure retrieval contracts shared by runtime implementations.

These helpers intentionally avoid heavyweight ML dependencies so device, score,
and BM25 semantics can be verified in fast CI.
"""

from typing import Mapping, Sequence


_L2_INDEX_TYPES = {"IndexFlatL2", "IndexIVFFlat"}
_IP_INDEX_TYPES = {"IndexFlatIP"}


def resolve_device_name(requested: str, cuda_available: bool) -> str:
    """Resolve configured device into a runtime device name."""
    normalized = (requested or "auto").strip().lower()

    if normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda" or normalized.startswith("cuda:"):
        return normalized if cuda_available else "cpu"

    raise ValueError(
        f"Unsupported device {requested!r}; expected auto, cpu, cuda, "
        "or a CUDA device such as cuda:0"
    )


def faiss_raw_score_to_similarity(index_type: str, raw_score: float) -> float:
    """Expose a consistent higher-is-better score for supported FAISS indexes."""
    if index_type in _IP_INDEX_TYPES:
        return float(raw_score)
    if index_type in _L2_INDEX_TYPES:
        return 1.0 - (float(raw_score) / 2.0)

    raise ValueError(f"Unsupported FAISS index type: {index_type}")


def document_frequency(
    per_document_frequencies: Sequence[Mapping[str, int]],
    term: str,
) -> int:
    """Count how many BM25 corpus documents contain term."""
    return sum(1 for frequencies in per_document_frequencies if term in frequencies)
