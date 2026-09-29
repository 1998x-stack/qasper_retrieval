import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache.embedding import (
    commit_embedding_manifest,
    embedding_build_config_fingerprint,
    embedding_source_fingerprint,
    validate_embedding_cache,
)
from cache.errors import CacheCorruptError, CacheStaleError


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
                "length": 5,
            },
            1: {
                "paper_id": "p2",
                "section_name": "Methods",
                "section_idx": 1,
                "paragraph_idx": 0,
                "original_text": "Beta",
                "cleaned_text": "beta",
                "length": 4,
            },
        },
        "questions": {0: {"question": "unrelated"}},
    }


def _config(
    *,
    model="moka-ai/m3e-base",
    max_length=512,
    device="cpu",
    batch_size=32,
    index_type="IndexFlatIP",
    nlist=100,
    nprobe=10,
):
    return {
        "models": {
            "embedding_model": model,
            "embedding_dim": 768,
            "max_seq_length": max_length,
            "batch_size": batch_size,
            "device": device,
            "trust_remote_code": False,
        },
        "faiss": {
            "index_type": index_type,
            "nlist": nlist,
            "nprobe": nprobe,
        },
    }


def test_question_only_change_does_not_invalidate_embedding_source():
    left = _dataset()
    right = _dataset()
    right["questions"][0]["question"] = "changed"
    assert embedding_source_fingerprint(left) == embedding_source_fingerprint(right)


def test_cleaned_text_and_materialized_metadata_invalidate_source():
    left = _dataset()
    text_changed = _dataset()
    text_changed["passages"][0]["cleaned_text"] = "changed"
    metadata_changed = _dataset()
    metadata_changed["passages"][0]["section_name"] = "Changed"
    assert embedding_source_fingerprint(left) != embedding_source_fingerprint(text_changed)
    assert embedding_source_fingerprint(left) != embedding_source_fingerprint(metadata_changed)


@pytest.mark.parametrize(
    ("field", "left", "right"),
    [
        ("device", "cpu", "cuda"),
        ("batch_size", 16, 64),
        ("nprobe", 5, 30),
    ],
)
def test_runtime_only_config_does_not_invalidate_embedding_cache(field, left, right):
    kwargs_left = {field: left}
    kwargs_right = {field: right}
    assert embedding_build_config_fingerprint(_config(**kwargs_left)) == (
        embedding_build_config_fingerprint(_config(**kwargs_right))
    )


def test_model_and_max_length_are_build_semantic():
    assert embedding_build_config_fingerprint(_config(model="model-a")) != (
        embedding_build_config_fingerprint(_config(model="model-b"))
    )
    assert embedding_build_config_fingerprint(_config(max_length=256)) != (
        embedding_build_config_fingerprint(_config(max_length=512))
    )


def test_nlist_only_matters_for_ivf():
    assert embedding_build_config_fingerprint(_config(nlist=10)) == (
        embedding_build_config_fingerprint(_config(nlist=1000))
    )
    assert embedding_build_config_fingerprint(
        _config(index_type="IndexIVFFlat", nlist=10)
    ) != embedding_build_config_fingerprint(
        _config(index_type="IndexIVFFlat", nlist=1000)
    )


def test_embedding_manifest_validates_both_artifacts(tmp_path):
    (tmp_path / "faiss_index.bin").write_bytes(b"faiss")
    (tmp_path / "metadata.pkl").write_bytes(b"metadata")
    commit_embedding_manifest(
        index_dir=tmp_path,
        preprocessed_dataset=_dataset(),
        config=_config(),
    )
    validate_embedding_cache(
        index_dir=tmp_path,
        preprocessed_dataset=_dataset(),
        config=_config(),
    )


def test_embedding_config_change_marks_cache_stale(tmp_path):
    (tmp_path / "faiss_index.bin").write_bytes(b"faiss")
    (tmp_path / "metadata.pkl").write_bytes(b"metadata")
    commit_embedding_manifest(
        index_dir=tmp_path,
        preprocessed_dataset=_dataset(),
        config=_config(index_type="IndexFlatIP"),
    )
    with pytest.raises(CacheStaleError):
        validate_embedding_cache(
            index_dir=tmp_path,
            preprocessed_dataset=_dataset(),
            config=_config(index_type="IndexFlatL2"),
        )


def test_tampered_embedding_artifact_is_corrupt(tmp_path):
    faiss_path = tmp_path / "faiss_index.bin"
    faiss_path.write_bytes(b"faiss")
    (tmp_path / "metadata.pkl").write_bytes(b"metadata")
    commit_embedding_manifest(
        index_dir=tmp_path,
        preprocessed_dataset=_dataset(),
        config=_config(),
    )
    faiss_path.write_bytes(b"tampered")
    with pytest.raises(CacheCorruptError):
        validate_embedding_cache(
            index_dir=tmp_path,
            preprocessed_dataset=_dataset(),
            config=_config(),
        )
