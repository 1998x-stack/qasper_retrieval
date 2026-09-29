"""Lineage-aware persistence and fingerprints for BM25 caches."""

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union

from .atomic import atomic_write_json, atomic_write_with
from .errors import CacheCorruptError
from .fingerprint import fingerprint_records, fingerprint_value
from .manifest import build_manifest, load_manifest, validate_manifest


BM25_ARTIFACT_TYPE = "bm25_index"
BM25_BUILD_CONTRACT_VERSION = 1
PathLike = Union[str, Path]


def _bm25_source_records(preprocessed_dataset: Mapping[str, Any]) -> Iterable[Any]:
    corpus = preprocessed_dataset["corpus"]
    passages = preprocessed_dataset["passages"]
    passage_ids = corpus["passage_ids"]
    document_ids = corpus["document_ids"]

    yield ["corpus", len(passage_ids)]
    for position, passage_id in enumerate(passage_ids):
        passage = passages[passage_id]
        index_input = (
            ["tokens", passage["tokens"]]
            if passage.get("tokens")
            else ["cleaned_text", passage.get("cleaned_text", "")]
        )
        yield [
            "passage",
            position,
            passage_id,
            document_ids[position],
            index_input,
            {
                "paper_id": passage.get("paper_id"),
                "section_name": passage.get("section_name"),
                "section_idx": passage.get("section_idx"),
                "paragraph_idx": passage.get("paragraph_idx"),
                "original_text": passage.get("original_text"),
                "cleaned_text": passage.get("cleaned_text"),
                "length": passage.get("length"),
            },
        ]


def bm25_source_fingerprint(preprocessed_dataset: Mapping[str, Any]) -> str:
    """Fingerprint only corpus inputs and metadata materialized in the BM25 cache."""
    return fingerprint_records(_bm25_source_records(preprocessed_dataset))


def bm25_build_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    bm25 = config.get("bm25", {})
    return {
        "k1": bm25.get("k1", 1.5),
        "b": bm25.get("b", 0.75),
        "epsilon": bm25.get("epsilon", 0.25),
        "tokenizer": bm25.get("tokenizer", "nltk"),
    }


def bm25_build_config_fingerprint(config: Mapping[str, Any]) -> str:
    return fingerprint_value(bm25_build_config(config))


def manifest_path_for(index_path: PathLike) -> Path:
    path = Path(index_path)
    return path.with_name(f"{path.stem}.manifest.json")


def save_bm25_cache(
    payload: Any,
    *,
    index_path: PathLike,
    preprocessed_dataset: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Atomically write the pickle and commit its manifest last."""
    path = Path(index_path)

    def _write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle)

    atomic_write_with(path, _write)
    manifest = build_manifest(
        artifact_type=BM25_ARTIFACT_TYPE,
        build_contract_version=BM25_BUILD_CONTRACT_VERSION,
        source_fingerprint=bm25_source_fingerprint(preprocessed_dataset),
        build_config_fingerprint=bm25_build_config_fingerprint(config),
        artifacts={path.name: path},
        metadata={"build_config": bm25_build_config(config)},
    )
    atomic_write_json(manifest_path_for(path), manifest)


def load_bm25_cache(
    *,
    index_path: PathLike,
    preprocessed_dataset: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    """Validate BM25 lineage/integrity before unpickling."""
    path = Path(index_path)
    manifest = load_manifest(
        manifest_path_for(path),
        legacy_artifacts=[path],
    )
    validate_manifest(
        manifest,
        artifact_root=path.parent,
        expected_artifact_type=BM25_ARTIFACT_TYPE,
        expected_build_contract_version=BM25_BUILD_CONTRACT_VERSION,
        expected_source_fingerprint=bm25_source_fingerprint(preprocessed_dataset),
        expected_build_config_fingerprint=bm25_build_config_fingerprint(config),
    )

    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError) as error:
        raise CacheCorruptError(f"Cannot load BM25 cache {path}: {error}") from error
