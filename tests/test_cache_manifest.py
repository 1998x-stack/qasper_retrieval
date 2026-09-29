import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache.errors import (
    CacheCompatibilityError,
    CacheCorruptError,
    CacheIncompleteError,
    CacheLegacyError,
    CacheNotFoundError,
    CacheStaleError,
)
from cache.fingerprint import fingerprint_value
from cache.manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    load_manifest,
    validate_manifest,
)


def _manifest(tmp_path):
    artifact = tmp_path / "index.bin"
    artifact.write_bytes(b"index-bytes")
    source = fingerprint_value({"passages": [1, 2]})
    config = fingerprint_value({"index_type": "flat"})
    manifest = build_manifest(
        artifact_type="test-index",
        build_contract_version=1,
        source_fingerprint=source,
        build_config_fingerprint=config,
        artifacts={"index.bin": artifact},
    )
    return artifact, source, config, manifest


def _validate(tmp_path, source, config, manifest):
    validate_manifest(
        manifest,
        artifact_root=tmp_path,
        expected_artifact_type="test-index",
        expected_build_contract_version=1,
        expected_source_fingerprint=source,
        expected_build_config_fingerprint=config,
    )


def test_valid_manifest_accepts_committed_artifact(tmp_path):
    _, source, config, manifest = _manifest(tmp_path)
    _validate(tmp_path, source, config, manifest)


def test_changed_source_is_stale(tmp_path):
    _, _, config, manifest = _manifest(tmp_path)
    with pytest.raises(CacheStaleError):
        _validate(tmp_path, fingerprint_value({"passages": [3]}), config, manifest)


def test_changed_build_config_is_stale(tmp_path):
    _, source, _, manifest = _manifest(tmp_path)
    with pytest.raises(CacheStaleError):
        _validate(tmp_path, source, fingerprint_value({"index_type": "ivf"}), manifest)


def test_newer_manifest_schema_fails_closed(tmp_path):
    _, source, config, manifest = _manifest(tmp_path)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    with pytest.raises(CacheCompatibilityError):
        _validate(tmp_path, source, config, manifest)


def test_older_manifest_schema_is_stale(tmp_path):
    _, source, config, manifest = _manifest(tmp_path)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION - 1
    with pytest.raises(CacheStaleError):
        _validate(tmp_path, source, config, manifest)


def test_missing_committed_artifact_is_incomplete(tmp_path):
    artifact, source, config, manifest = _manifest(tmp_path)
    artifact.unlink()
    with pytest.raises(CacheIncompleteError):
        _validate(tmp_path, source, config, manifest)


def test_checksum_mismatch_is_corrupt(tmp_path):
    artifact, source, config, manifest = _manifest(tmp_path)
    artifact.write_bytes(b"tampered")
    with pytest.raises(CacheCorruptError):
        _validate(tmp_path, source, config, manifest)


def test_missing_manifest_distinguishes_empty_and_legacy_cache(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    with pytest.raises(CacheNotFoundError):
        load_manifest(manifest_path)

    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(b"legacy")
    with pytest.raises(CacheLegacyError):
        load_manifest(manifest_path, legacy_artifacts=[legacy])


def test_invalid_json_manifest_is_corrupt(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CacheCorruptError):
        load_manifest(manifest_path)


def test_manifest_rejects_path_traversal(tmp_path):
    _, source, config, manifest = _manifest(tmp_path)
    descriptor = manifest["artifacts"].pop("index.bin")
    manifest["artifacts"]["../index.bin"] = descriptor
    with pytest.raises(CacheCorruptError):
        _validate(tmp_path, source, config, manifest)
