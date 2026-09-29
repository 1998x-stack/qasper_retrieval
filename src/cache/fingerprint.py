"""Deterministic fingerprints for cache build inputs."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Union


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def fingerprint_value(value: Any) -> str:
    """Return a namespaced SHA-256 fingerprint for JSON-compatible data."""
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"sha256:{digest}"


def fingerprint_records(records: Iterable[Any]) -> str:
    """Hash an ordered stream of JSON-compatible records without buffering it all."""
    digest = hashlib.sha256()
    digest.update(b"qasper-cache-records-v1\0")
    for record in records:
        payload = canonical_json_bytes(record)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def hash_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:    """Return a namespaced SHA-256 digest for a file without loading it all."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
