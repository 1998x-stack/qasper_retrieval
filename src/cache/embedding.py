"""Lineage and integrity contracts for embedding/FAISS cache generations."""

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union

from .atomic import atomic_write_json
from .errors import CacheCorruptError
from .fingerprint import fingerprint_records, fingerprint_value
from .manifest import build_manifest, load_manifest, validate_manifest


EMBEDDING_ARTIFACT_TYPE = "embedding_index"
EMBEDDING_BUILD_CONTRACT_VERSION = 1
EMBEDDING_MANIFEST_NAME = "manifest.json"
EMBEDDING_ARTIFACT_NAMES = {"faiss_index.bin", "metadata.pkl"}
PathLike = Union[str, Path]


def embedding_backend(model_name: str) -> str:
    """Mirror the runtime backend choice without importing ML dependencies."""
    lowered = model_name.lower()
    if "sentence-transformers" in lowered or "m3e" in lowered:
        return "sentence_transformer"
    return "automodel"


def _embedding_source_records(
    preprocessed_dataset: Mapping[str, Any],
) -> Iterable[Any]:
    corpus = preprocessed_dataset["corpus"]
    passages = preprocessed_dataset["passages"]
    passage_ids = corpus["passage_ids"]
    document_ids = corpus["document_ids"]

    yield ["corpus", len(passage_ids)]
    for position, passage_id in enumerate(passage_ids):
        passage = passages[passage_id]
        yield [
            "passage",
            position,
            passage_id,
            document_ids[position],
            passage.get("cleaned_text", ""),
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


def embedding_source_fingerprint(preprocessed_dataset: Mapping[str, Any]) -> str:
    """Fingerprint only inputs materialized into the embedding cache."""
    return fingerprint_records(_embedding_source_records(preprocessed_dataset))


def embedding_build_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return configuration that changes built vectors or FAISS structure."""
    models = config.get("models", {})
    faiss = config.get("faiss", {})
    model_name = models.get("embedding_model", "moka-ai/m3e-base")
    backend = embedding_backend(model_name)
    index_type = faiss.get("index_type", "IndexFlatIP")

    build_config: Dict[str, Any] = {
        "model_name": model_name,
        "backend": backend,
        "embedding_dim": models.get("embedding_dim", 768),
        "max_seq_length": models.get("max_seq_length", 512),
        "index_type": index_type,
    }
    if backend == "automodel":
        build_config["trust_remote_code"] = bool(
            models.get("trust_remote_code", False)
        )
    if index_type == "IndexIVFFlat":
        build_config["nlist"] = faiss.get("nlist", 100)
    return build_config


def embedding_build_config_fingerprint(config: Mapping[str, Any]) -> str:
    return fingerprint_value(embedding_build_config(config))


def manifest_path_for(index_dir: PathLike) -> Path:
    return Path(index_dir) / EMBEDDING_MANIFEST_NAME


def commit_embedding_manifest(
    *,
    index_dir: PathLike,
    preprocessed_dataset: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Commit the already-written FAISS+metadata generation via manifest last."""
    directory = Path(index_dir)
    artifacts = {
        name: directory / name
        for name in sorted(EMBEDDING_ARTIFACT_NAMES)
    }
    manifest = build_manifest(
        artifact_type=EMBEDDING_ARTIFACT_TYPE,
        build_contract_version=EMBEDDING_BUILD_CONTRACT_VERSION,
        source_fingerprint=embedding_source_fingerprint(preprocessed_dataset),
        build_config_fingerprint=embedding_build_config_fingerprint(config),
        artifacts=artifacts,
        metadata={"build_config": embedding_build_config(config)},
    )
    atomic_write_json(manifest_path_for(directory), manifest)


def validate_embedding_cache(
    *,
    index_dir: PathLike,
    preprocessed_dataset: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Validate lineage and both committed artifacts before deserialization."""
    directory = Path(index_dir)
    manifest = load_manifest(
        manifest_path_for(directory),
        legacy_artifacts=[
            directory / "faiss_index.bin",
            directory / "metadata.pkl",
        ],
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CacheCorruptError("Embedding manifest artifacts are invalid")
    missing_descriptors = EMBEDDING_ARTIFACT_NAMES.difference(artifacts)
    if missing_descriptors:
        raise CacheCorruptError(
            "Embedding manifest is missing artifact descriptors: "
            f"{sorted(missing_descriptors)}"
        )

    validate_manifest(
        manifest,
        artifact_root=directory,
        expected_artifact_type=EMBEDDING_ARTIFACT_TYPE,
        expected_build_contract_version=EMBEDDING_BUILD_CONTRACT_VERSION,
        expected_source_fingerprint=embedding_source_fingerprint(preprocessed_dataset),
        expected_build_config_fingerprint=embedding_build_config_fingerprint(config),
    )
