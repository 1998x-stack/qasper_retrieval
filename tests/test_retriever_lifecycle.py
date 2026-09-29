import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.lifecycle import resolve_resource


def test_injected_resource_is_reused_without_calling_factory():
    resource = object()
    calls = []

    def factory():
        calls.append(True)
        return object()

    resolved, owned = resolve_resource(resource, factory)
    assert resolved is resource
    assert owned is False
    assert calls == []


def test_missing_resource_is_created_and_owned():
    resource = object()
    calls = []

    def factory():
        calls.append(True)
        return resource

    resolved, owned = resolve_resource(None, factory)
    assert resolved is resource
    assert owned is True
    assert calls == [True]
