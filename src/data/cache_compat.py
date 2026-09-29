"""Compatibility helpers for JSON-backed preprocessed caches."""

from typing import Any, Dict, Mapping


def restore_numeric_mapping_keys(mapping: Mapping[Any, Any]) -> Dict[Any, Any]:
    """Restore integer-like JSON object keys while preserving other keys."""
    restored: Dict[Any, Any] = {}
    for key, value in mapping.items():
        normalized_key: Any = key
        if isinstance(key, str):
            try:
                normalized_key = int(key)
            except ValueError:
                normalized_key = key
        restored[normalized_key] = value
    return restored


def restore_preprocessed_cache_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Restore ID mappings that JSON serializes with string object keys."""
    for field in ("passages", "questions"):
        mapping = data.get(field)
        if isinstance(mapping, Mapping):
            data[field] = restore_numeric_mapping_keys(mapping)
    return data
