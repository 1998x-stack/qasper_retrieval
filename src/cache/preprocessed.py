"""Lineage-aware persistence for the preprocessed dataset cache."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union

from .atomic import atomic_write_json
from .errors import CacheCorruptError
from .fingerprint import fingerprint_records, fingerprint_value
from .manifest import build_manifest, load_manifest, validate_manifest


PREPROCESSED_ARTIFACT_TYPE = "preprocessed_dataset"
PREPROCESSING_BUILD_CONTRACT_VERSION = 1
PathLike = Union[str, Path]


def _processed_dataset_records(
    processed_dataset: Mapping[str, Any],
) -> Iterable[Any]:
    """Yield records in the same split/document order that assigns global IDs."""
    for split_name, split_data in processed_dataset.items():
        yield ["split", split_name, len(split_data)]
        for document_index, document in enumerate(split_data):
            yield ["document", split_name, document_index, document]


def processed_dataset_fingerprint(processed_dataset: Mapping[str, Any]) -> str:
    """Fingerprint the actual ordered input consumed by preprocessing."""
    return fingerprint_records(_processed_dataset_records(processed_dataset))


def preprocessing_build_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only configuration that currently changes preprocessing output."""
    retrieval = config.get("retrieval", {})
    return {
        "max_passage_length": retrieval.get("max_passage_length", 1000),
    }


def preprocessing_build_config_fingerprint(config: Mapping[str, Any]) -> str:
    return fingerprint_value(preprocessing_build_config(config))


def manifest_path_for(cache_path: PathLike) -> Path:
    path = Path(cache_path)
    return path.with_name(f"{path.stem}.manifest.json")


def save_preprocessed_cache(
    data: Any,
    *,
    cache_path: PathLike,
    processed_dataset: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Atomically replace the artifact, then commit its manifest last."""
    path = Path(cache_path)
    atomic_write_json(path, data)

    manifest = build_manifest(
        artifact_type=PREPROCESSED_ARTIFACT_TYPE,
        build_contract_version=PREPROCESSING_BUILD_CONTRACT_VERSION,
        source_fingerprint=processed_dataset_fingerprint(processed_dataset),
        build_config_fingerprint=preprocessing_build_config_fingerprint(config),
        artifacts={path.name: path},
        metadata={
            "build_config": preprocessing_build_config(config),
        },
    )
    atomic_write_json(manifest_path_for(path), manifest)


def load_preprocessed_cache(
    *,
    cache_path: PathLike,
    processed_dataset: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    """Validate lineage and integrity before deserializing the cache artifact."""
    path = Path(cache_path)
    manifest = load_manifest(
        manifest_path_for(path),
        legacy_artifacts=[path],
    )
    validate_manifest(
        manifest,
        artifact_root=path.parent,
        expected_artifact_type=PREPROCESSED_ARTIFACT_TYPE,
        expected_build_contract_version=PREPROCESSING_BUILD_CONTRACT_VERSION,
        expected_source_fingerprint=processed_dataset_fingerprint(processed_dataset),
        expected_build_config_fingerprint=preprocessing_build_config_fingerprint(config),
    )

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(f"Cannot read preprocessed cache {path}: {error}") from error
