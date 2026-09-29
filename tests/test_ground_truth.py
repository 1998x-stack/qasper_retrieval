from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(relative_path: str, name: str):
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location(name, root / relative_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ground_truth = _load_module("src/evaluation/ground_truth.py", "ground_truth")


def test_prepare_evaluation_data_uses_global_passage_ids_and_textual_evidence():
    processed = {
        "validation": [
            {
                "paper_id": "paper-a",
                "qa_pairs": [
                    {
                        "question": "A?",
                        "answers": [{"evidence": [" alpha   evidence "]}],
                    }
                ],
            },
            {
                "paper_id": "paper-b",
                "qa_pairs": [
                    {
                        "question": "B?",
                        "answers": [{"evidence": ["beta evidence"]}],
                    }
                ],
            },
        ]
    }
    preprocessed = {
        "passages": {
            "10": {
                "paper_id": "paper-a",
                "split": "validation",
                "original_text": "alpha evidence",
            },
            "27": {
                "paper_id": "paper-b",
                "split": "validation",
                "original_text": "beta evidence",
            },
            "99": {
                "paper_id": "paper-b",
                "split": "train",
                "original_text": "beta evidence",
            },
        }
    }

    result = ground_truth.prepare_evaluation_data(
        processed, preprocessed, "validation"
    )

    assert result["queries"] == ["A?", "B?"]
    assert result["ground_truth"] == [[10], [27]]
    assert set(result["passage_texts"]) == {10, 27}


def test_prepare_evaluation_data_skips_non_text_and_unmatched_evidence():
    processed = {
        "validation": [
            {
                "paper_id": "paper-a",
                "qa_pairs": [
                    {
                        "question": "figure",
                        "answers": [{"evidence": ["FLOAT SELECTED: Figure 1"]}],
                    },
                    {
                        "question": "unmatched",
                        "answers": [{"evidence": ["missing paragraph"]}],
                    },
                    {
                        "question": "matched",
                        "answers": [{"evidence": ["kept paragraph"]}],
                    },
                ],
            }
        ]
    }
    preprocessed = {
        "passages": {
            5: {
                "paper_id": "paper-a",
                "split": "validation",
                "original_text": "kept paragraph",
            }
        }
    }

    result = ground_truth.prepare_evaluation_data(
        processed, preprocessed, "validation"
    )

    assert result["queries"] == ["matched"]
    assert result["ground_truth"] == [[5]]
    assert result["skipped_no_text_evidence"] == 1
    assert result["skipped_unmatched_evidence"] == 1


def test_max_queries_counts_only_evaluable_queries():
    processed = {
        "validation": [
            {
                "paper_id": "paper-a",
                "qa_pairs": [
                    {"question": "skip", "answers": [{"evidence": []}]},
                    {"question": "one", "answers": [{"evidence": ["p1"]}]},
                    {"question": "two", "answers": [{"evidence": ["p2"]}]},
                ],
            }
        ]
    }
    preprocessed = {
        "passages": {
            1: {"paper_id": "paper-a", "split": "validation", "original_text": "p1"},
            2: {"paper_id": "paper-a", "split": "validation", "original_text": "p2"},
        }
    }

    result = ground_truth.prepare_evaluation_data(
        processed, preprocessed, "validation", max_queries=1
    )

    assert result["queries"] == ["one"]
    assert result["ground_truth"] == [[1]]
