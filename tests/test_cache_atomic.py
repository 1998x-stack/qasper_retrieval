import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache.atomic import atomic_write_json, atomic_write_text, atomic_write_with


def test_atomic_write_replaces_existing_file(tmp_path):
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_failed_writer_preserves_previous_generation(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"old-generation")

    def fail_writer(temporary):
        temporary.write_bytes(b"partial-new-generation")
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        atomic_write_with(target, fail_writer)

    assert target.read_bytes() == b"old-generation"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_json_writes_complete_json(tmp_path):
    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert __import__("json").loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
