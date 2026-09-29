import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache.bm25 import bm25_build_config_fingerprint
from cache.embedding import embedding_build_config_fingerprint
from cache.errors import CacheCorruptError
from cache.hybrid import validate_child_alignment


def test_aligned_hybrid_children_are_valid():
    validate_child_alignment(
        bm25_passage_ids=[0, 1],
        embedding_passage_ids=[0, 1],
        bm25_document_ids=["p1", "p2"],
        embedding_document_ids=["p1", "p2"],
    )


def test_hybrid_rejects_child_passage_order_mismatch():
    with pytest.raises(CacheCorruptError):
        validate_child_alignment(
            bm25_passage_ids=[0, 1],
            embedding_passage_ids=[1, 0],
            bm25_document_ids=["p1", "p2"],
            embedding_document_ids=["p2", "p1"],
        )


def test_hybrid_rejects_child_document_mismatch():
    with pytest.raises(CacheCorruptError):
        validate_child_alignment(
            bm25_passage_ids=[0, 1],
            embedding_passage_ids=[0, 1],
            bm25_document_ids=["p1", "p2"],
            embedding_document_ids=["p1", "wrong"],
        )


def test_hybrid_weights_do_not_invalidate_child_build_caches():
    base = {
        "bm25": {"k1": 1.2, "b": 0.75, "epsilon": 0.25, "tokenizer": "jieba"},
        "models": {
            "embedding_model": "moka-ai/m3e-base",
            "embedding_dim": 768,
            "max_seq_length": 512,
            "device": "cpu",
            "batch_size": 32,
            "trust_remote_code": False,
        },
        "faiss": {"index_type": "IndexFlatIP", "nlist": 100, "nprobe": 10},
        "hybrid": {"bm25_weight": 0.5, "embedding_weight": 0.5},
    }
    changed = {**base, "hybrid": {"bm25_weight": 0.8, "embedding_weight": 0.2}}
    assert bm25_build_config_fingerprint(base) == bm25_build_config_fingerprint(changed)
    assert embedding_build_config_fingerprint(base) == embedding_build_config_fingerprint(changed)
