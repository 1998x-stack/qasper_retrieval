import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache.fingerprint import canonical_json_bytes, fingerprint_value, hash_file


def test_fingerprint_is_stable_across_mapping_order():
    left = {"b": 2, "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": 2}
    assert fingerprint_value(left) == fingerprint_value(right)


def test_fingerprint_preserves_sequence_order():
    assert fingerprint_value([1, 2, 3]) != fingerprint_value([3, 2, 1])


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": float("nan")})


def test_hash_file_changes_with_content(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    first = hash_file(path)
    path.write_bytes(b"second")
    second = hash_file(path)
    assert first.startswith("sha256:")
    assert first != second
