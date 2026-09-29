"""Versioned cache manifests and artifact integrity validation."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from .errors import (
    CacheCompatibilityError,
    CacheCorruptError,
    CacheIncompleteError,
    CacheLegacyError,
    CacheNotFoundError,
    CacheStaleError,
)
from .fingerprint import hash_file


MANIFEST_SCHEMA_VERSION = 1
PathLike = Union[str, Path]


def _artifact_descriptor(path: Path) -> Dict[str, Any]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": hash_file(path),
    }


def build_manifest(
    *,
    artifact_type: str,
    build_contract_version: int,
    source_fingerprint: str,
    build_config_fingerprint: str,
    artifacts: Mapping[str, PathLike],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a manifest from already-written artifacts."""
    descriptors: Dict[str, Any] = {}
    for name, path in sorted(artifacts.items()):
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise CacheIncompleteError(f"Artifact missing before commit: {artifact_path}")
        descriptors[name] = _artifact_descriptor(artifact_path)

    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "build_contract_version": build_contract_version,
        "source": {"fingerprint": source_fingerprint},
        "build_config": {"fingerprint": build_config_fingerprint},
        "artifacts": descriptors,
    }
    if metadata:
        manifest["metadata"] = dict(metadata)
    return manifest


def load_manifest(
    manifest_path: PathLike,
    *,
    legacy_artifacts: Iterable[PathLike] = (),
) -> Dict[str, Any]:
    """Load a manifest, distinguishing missing from legacy cache state."""
    path = Path(manifest_path)
    if not path.is_file():
        if any(Path(candidate).exists() for candidate in legacy_artifacts):
            raise CacheLegacyError(f"Legacy cache has no manifest: {path}")
        raise CacheNotFoundError(f"Cache manifest not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(f"Cannot read cache manifest {path}: {error}") from error

    if not isinstance(manifest, dict):
        raise CacheCorruptError(f"Cache manifest must be a JSON object: {path}")
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    artifact_root: PathLike,
    expected_artifact_type: str,
    expected_build_contract_version: int,
    expected_source_fingerprint: str,
    expected_build_config_fingerprint: str,
) -> None:
    """Validate format, semantic lineage, artifact presence, and checksums."""
    required = {
        "schema_version",
        "artifact_type",
        "build_contract_version",
        "source",
        "build_config",
        "artifacts",
    }
    missing = required.difference(manifest)
    if missing:
        raise CacheCorruptError(f"Manifest missing required keys: {sorted(missing)}")

    schema_version = manifest["schema_version"]
    if not isinstance(schema_version, int):
        raise CacheCorruptError("Manifest schema_version must be an integer")
    if schema_version > MANIFEST_SCHEMA_VERSION:
        raise CacheCompatibilityError(
            f"Manifest schema {schema_version} is newer than supported "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    if schema_version < MANIFEST_SCHEMA_VERSION:
        raise CacheStaleError(
            f"Manifest schema {schema_version} is older than supported "
            f"{MANIFEST_SCHEMA_VERSION}"
        )

    if manifest["artifact_type"] != expected_artifact_type:
        raise CacheCorruptError(
            f"Wrong artifact type: expected {expected_artifact_type!r}, "
            f"found {manifest['artifact_type']!r}"
        )
    if manifest["build_contract_version"] != expected_build_contract_version:
        raise CacheStaleError("Cache build contract version changed")

    source = manifest["source"]
    build_config = manifest["build_config"]
    if not isinstance(source, Mapping) or "fingerprint" not in source:
        raise CacheCorruptError("Manifest source fingerprint is missing")
    if not isinstance(build_config, Mapping) or "fingerprint" not in build_config:
        raise CacheCorruptError("Manifest build-config fingerprint is missing")

    if source["fingerprint"] != expected_source_fingerprint:
        raise CacheStaleError("Cache source fingerprint does not match current input")
    if build_config["fingerprint"] != expected_build_config_fingerprint:
        raise CacheStaleError("Cache build configuration has changed")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise CacheCorruptError("Manifest artifacts must be a non-empty object")

    root = Path(artifact_root)
    for relative_name, descriptor in artifacts.items():
        if not isinstance(relative_name, str) or not relative_name:
            raise CacheCorruptError("Artifact names must be non-empty strings")
        if Path(relative_name).is_absolute() or ".." in Path(relative_name).parts:
            raise CacheCorruptError(f"Unsafe artifact path in manifest: {relative_name}")
        if not isinstance(descriptor, Mapping):
            raise CacheCorruptError(f"Invalid descriptor for {relative_name}")
        if "size_bytes" not in descriptor or "sha256" not in descriptor:
            raise CacheCorruptError(f"Incomplete descriptor for {relative_name}")

        artifact_path = root / relative_name
        if not artifact_path.is_file():
            raise CacheIncompleteError(f"Committed artifact missing: {artifact_path}")
        if artifact_path.stat().st_size != descriptor["size_bytes"]:
            raise CacheCorruptError(f"Artifact size mismatch: {artifact_path}")
        if hash_file(artifact_path) != descriptor["sha256"]:
            raise CacheCorruptError(f"Artifact checksum mismatch: {artifact_path}")
