import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache.bm25 import (
    bm25_build_config_fingerprint,
    bm25_source_fingerprint,
    load_bm25_cache,
    save_bm25_cache,
)
from cache.errors import CacheStaleError


def _dataset():
    return {
        "corpus": {
            "passage_ids": [0, 1],
            "document_ids": ["p1", "p2"],
        },
        "passages": {
            0: {
                "paper_id": "p1",
                "section_name": "Intro",
                "section_idx": 0,
                "paragraph_idx": 0,
                "original_text": "Alpha",
                "cleaned_text": "alpha",
                "tokens": ["alpha"],
                "length": 5,
            },
            1: {
                "paper_id": "p2",
                "section_name": "Methods",
                "section_idx": 1,
                "paragraph_idx": 0,
                "original_text": "Beta",
                "cleaned_text": "beta",
                "tokens": ["beta"],
                "length": 4,
            },
        },
        "questions": {
            0: {"question": "unrelated"},
        },
    }


def _config(k1=1.2, tokenizer="jieba"):
    return {
        "bm25": {
            "k1": k1,
            "b": 0.75,
            "epsilon": 0.25,
            "tokenizer": tokenizer,
        }
    }


def test_question_only_changes_do_not_invalidate_bm25_source():
    left = _dataset()
    right = _dataset()
    right["questions"][0]["question"] = "changed"
    assert bm25_source_fingerprint(left) == bm25_source_fingerprint(right)


def test_passage_tokens_invalidate_bm25_source():
    left = _dataset()
    right = _dataset()
    right["passages"][0]["tokens"] = ["changed"]
    assert bm25_source_fingerprint(left) != bm25_source_fingerprint(right)


def test_materialized_metadata_invalidates_bm25_source():
    left = _dataset()
    right = _dataset()
    right["passages"][0]["section_name"] = "Changed"
    assert bm25_source_fingerprint(left) != bm25_source_fingerprint(right)


def test_bm25_build_config_tracks_ranking_parameters():
    assert bm25_build_config_fingerprint(_config(k1=1.2)) != (
        bm25_build_config_fingerprint(_config(k1=2.0))
    )


def test_bm25_cache_round_trip(tmp_path):
    path = tmp_path / "bm25_index.pkl"
    payload = {"corpus": [["alpha"]], "statistics": {"total_docs": 1}}
    save_bm25_cache(
        payload,
        index_path=path,
        preprocessed_dataset=_dataset(),
        config=_config(),
    )
    assert load_bm25_cache(
        index_path=path,
        preprocessed_dataset=_dataset(),
        config=_config(),
    ) == payload


def test_bm25_parameter_change_marks_cache_stale(tmp_path):
    path = tmp_path / "bm25_index.pkl"
    save_bm25_cache(
        {"value": 1},
        index_path=path,
        preprocessed_dataset=_dataset(),
        config=_config(k1=1.2),
    )
    with pytest.raises(CacheStaleError):
        load_bm25_cache(
            index_path=path,
            preprocessed_dataset=_dataset(),
            config=_config(k1=1.8),
        )


def test_bm25_tokenizer_change_marks_cache_stale(tmp_path):
    path = tmp_path / "bm25_index.pkl"
    save_bm25_cache(
        {"value": 1},
        index_path=path,
        preprocessed_dataset=_dataset(),
        config=_config(tokenizer="jieba"),
    )
    with pytest.raises(CacheStaleError):
        load_bm25_cache(
            index_path=path,
            preprocessed_dataset=_dataset(),
            config=_config(tokenizer="nltk"),
        )
