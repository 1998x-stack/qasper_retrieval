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


contracts = _load("src/retrieval/contracts.py", "retrieval_contracts")


@pytest.mark.parametrize(
    ("requested", "cuda_available", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
        ("cuda", True, "cuda"),
        ("cuda", False, "cpu"),
        ("cuda:1", True, "cuda:1"),
        ("cuda:1", False, "cpu"),
    ],
)
def test_resolve_device_name(requested, cuda_available, expected):
    assert contracts.resolve_device_name(requested, cuda_available) == expected


def test_resolve_device_rejects_unknown_values():
    with pytest.raises(ValueError):
        contracts.resolve_device_name("mps", cuda_available=False)


def test_faiss_ip_score_is_already_similarity():
    assert contracts.faiss_raw_score_to_similarity("IndexFlatIP", 0.75) == pytest.approx(0.75)


@pytest.mark.parametrize("index_type", ["IndexFlatL2", "IndexIVFFlat"])
def test_faiss_l2_distance_is_converted_to_higher_is_better_similarity(index_type):
    assert contracts.faiss_raw_score_to_similarity(index_type, 0.0) == pytest.approx(1.0)
    assert contracts.faiss_raw_score_to_similarity(index_type, 2.0) == pytest.approx(0.0)
    assert contracts.faiss_raw_score_to_similarity(index_type, 4.0) == pytest.approx(-1.0)


def test_document_frequency_counts_documents_not_term_occurrences():
    doc_freqs = [
        {"alpha": 3, "beta": 1},
        {"alpha": 1},
        {"beta": 2},
    ]
    assert contracts.document_frequency(doc_freqs, "alpha") == 2
    assert contracts.document_frequency(doc_freqs, "missing") == 0
