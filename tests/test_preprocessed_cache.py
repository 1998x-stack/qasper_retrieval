import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache.errors import CacheCorruptError, CacheLegacyError, CacheStaleError
from cache.preprocessed import (
    load_preprocessed_cache,
    manifest_path_for,
    preprocessing_build_config_fingerprint,
    processed_dataset_fingerprint,
    save_preprocessed_cache,
)


def _processed():
    return {
        "train": [{"paper_id": "p1", "title": "A", "passages": [1]}],
        "validation": [{"paper_id": "p2", "title": "B", "passages": [2]}],
    }


def _config(max_length=1000, overlap=0.8):
    return {
        "retrieval": {
            "max_passage_length": max_length,
            "overlap_threshold": overlap,
        }
    }


def test_processed_fingerprint_preserves_split_iteration_order():
    original = _processed()
    reordered = {
        "validation": original["validation"],
        "train": original["train"],
    }
    assert processed_dataset_fingerprint(original) != processed_dataset_fingerprint(reordered)


def test_preprocessing_fingerprint_ignores_unused_overlap_threshold():
    assert preprocessing_build_config_fingerprint(_config(overlap=0.8)) == (
        preprocessing_build_config_fingerprint(_config(overlap=0.2))
    )


def test_preprocessing_fingerprint_tracks_max_passage_length():
    assert preprocessing_build_config_fingerprint(_config(max_length=1000)) != (
        preprocessing_build_config_fingerprint(_config(max_length=600))
    )


def test_preprocessed_cache_round_trip_with_manifest(tmp_path):
    path = tmp_path / "preprocessed_dataset.json"
    data = {"passages": {"0": {"text": "hello"}}, "corpus": {"passage_ids": [0]}}
    save_preprocessed_cache(
        data, cache_path=path, processed_dataset=_processed(), config=_config()
    )

    assert manifest_path_for(path).is_file()
    assert load_preprocessed_cache(
        cache_path=path, processed_dataset=_processed(), config=_config()
    ) == data


def test_source_change_marks_preprocessed_cache_stale(tmp_path):
    path = tmp_path / "preprocessed_dataset.json"
    save_preprocessed_cache(
        {"value": 1}, cache_path=path, processed_dataset=_processed(), config=_config()
    )
    changed = _processed()
    changed["train"][0]["title"] = "changed"

    with pytest.raises(CacheStaleError):
        load_preprocessed_cache(
            cache_path=path, processed_dataset=changed, config=_config()
        )


def test_semantic_config_change_marks_preprocessed_cache_stale(tmp_path):
    path = tmp_path / "preprocessed_dataset.json"
    save_preprocessed_cache(
        {"value": 1}, cache_path=path, processed_dataset=_processed(), config=_config()
    )

    with pytest.raises(CacheStaleError):
        load_preprocessed_cache(
            cache_path=path,
            processed_dataset=_processed(),
            config=_config(max_length=600),
        )


def test_unused_config_change_keeps_preprocessed_cache_valid(tmp_path):
    path = tmp_path / "preprocessed_dataset.json"
    save_preprocessed_cache(
        {"value": 1}, cache_path=path, processed_dataset=_processed(), config=_config(overlap=0.8)
    )

    assert load_preprocessed_cache(
        cache_path=path,
        processed_dataset=_processed(),
        config=_config(overlap=0.1),
    ) == {"value": 1}


def test_tampered_preprocessed_cache_is_corrupt(tmp_path):
    path = tmp_path / "preprocessed_dataset.json"
    save_preprocessed_cache(
        {"value": 1}, cache_path=path, processed_dataset=_processed(), config=_config()
    )
    path.write_text('{"value": 2}\n', encoding="utf-8")

    with pytest.raises(CacheCorruptError):
        load_preprocessed_cache(
            cache_path=path, processed_dataset=_processed(), config=_config()
        )


def test_legacy_preprocessed_cache_requires_one_time_rebuild(tmp_path):
    path = tmp_path / "preprocessed_dataset.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")

    with pytest.raises(CacheLegacyError):
        load_preprocessed_cache(
            cache_path=path, processed_dataset=_processed(), config=_config()
        )
