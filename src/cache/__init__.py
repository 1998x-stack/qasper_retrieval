"""Cache lineage, integrity, and persistence primitives."""

from .errors import (
    CacheCompatibilityError,
    CacheCorruptError,
    CacheError,
    CacheIncompleteError,
    CacheLegacyError,
    CacheNotFoundError,
    CacheStaleError,
)
from .fingerprint import fingerprint_records, fingerprint_value, hash_file
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    load_manifest,
    validate_manifest,
)

__all__ = [
    "CacheError",
    "CacheNotFoundError",
    "CacheLegacyError",
    "CacheStaleError",
    "CacheIncompleteError",
    "CacheCorruptError",
    "CacheCompatibilityError",
    "fingerprint_records",
    "fingerprint_value",
    "hash_file",
    "MANIFEST_SCHEMA_VERSION",
    "build_manifest",
    "load_manifest",
    "validate_manifest",
]
