from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_module(relative_path: str, name: str):
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location(name, root / relative_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


metrics = _load_module("src/evaluation/retrieval_metrics.py", "retrieval_metrics")


def test_average_precision_penalizes_missed_relevant_documents():
    assert metrics.average_precision_at_k([1, 9], {1, 2}, 5) == pytest.approx(0.5)


def test_average_precision_uses_ranked_precision():
    expected = (1.0 + (2 / 3)) / 2
    assert metrics.average_precision_at_k([1, 9, 2], {1, 2}, 5) == pytest.approx(expected)


def test_average_precision_does_not_double_count_duplicates():
    expected = (1.0 + (2 / 3)) / 2
    assert metrics.average_precision_at_k([1, 1, 2], {1, 2}, 5) == pytest.approx(expected)


def test_average_precision_validates_k():
    with pytest.raises(ValueError):
        metrics.average_precision_at_k([1], {1}, 0)
