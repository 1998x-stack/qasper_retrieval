"""Pure validation helpers for HybridRetriever cache composition."""

from typing import Sequence

from .errors import CacheCorruptError


def validate_child_alignment(
    *,
    bm25_passage_ids: Sequence[int],
    embedding_passage_ids: Sequence[int],
    bm25_document_ids: Sequence[str],
    embedding_document_ids: Sequence[str],
) -> None:
    """Require both child indexes to address the same ordered corpus."""
    if list(bm25_passage_ids) != list(embedding_passage_ids):
        raise CacheCorruptError(
            "Hybrid child passage IDs are not aligned in the same order"
        )
    if list(bm25_document_ids) != list(embedding_document_ids):
        raise CacheCorruptError(
            "Hybrid child document IDs are not aligned in the same order"
        )


def validate_child_source_identity(
    *,
    expected_passage_ids: Sequence[int],
    expected_document_ids: Sequence[str],
    actual_passage_ids: Sequence[int],
    actual_document_ids: Sequence[str],
) -> None:
    """Require a child index to address the current ordered source corpus."""
    if list(actual_passage_ids) != list(expected_passage_ids):
        raise CacheCorruptError(
            "Retriever passage IDs do not match the current source corpus"
        )
    if list(actual_document_ids) != list(expected_document_ids):
        raise CacheCorruptError(
            "Retriever document IDs do not match the current source corpus"
        )
