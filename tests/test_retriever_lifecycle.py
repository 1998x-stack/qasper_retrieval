import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "retrieval" / "lifecycle.py"
SPEC = importlib.util.spec_from_file_location("retrieval_lifecycle", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
resolve_resource = MODULE.resolve_resource


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
