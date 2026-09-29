from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load(relative_path: str, name: str):
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location(name, root / relative_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


failures = _load("src/evaluation/failures.py", "evaluation_failures")


def test_no_failures_does_not_raise():
    failures.raise_for_method_failures({})


def test_failures_raise_after_aggregation():
    with pytest.raises(failures.MethodEvaluationError) as exc_info:
        failures.raise_for_method_failures(
            {
                "embedding": RuntimeError("model unavailable"),
                "hybrid": ValueError("index invalid"),
            }
        )

    message = str(exc_info.value)
    assert "2 requested retrieval method(s) failed" in message
    assert "embedding: RuntimeError: model unavailable" in message
    assert "hybrid: ValueError: index invalid" in message
