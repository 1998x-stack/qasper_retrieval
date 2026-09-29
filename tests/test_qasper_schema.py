from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(relative_path: str, name: str):
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location(name, root / relative_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


schema = _load_module("src/data/qasper_schema.py", "qasper_schema")


def test_normalize_answer_payloads_supports_legacy_dict_of_lists():
    raw_answers = {
        "annotation_id": ["a1", "a2"],
        "answer": {
            "unanswerable": [False, False],
            "free_form_answer": ["first", "second"],
            "extractive_spans": [[], []],
            "yes_no": [None, None],
            "evidence": [["paragraph one"], ["paragraph two"]],
        },
        "worker_id": ["w1", "w2"],
    }

    payloads = schema.normalize_answer_payloads(raw_answers)

    assert [item["free_form_answer"] for item in payloads] == ["first", "second"]
    assert payloads[1]["evidence"] == ["paragraph two"]


def test_normalize_answer_payloads_supports_list_of_dicts():
    raw_answers = [
        {
            "annotation_id": "a1",
            "answer": {
                "unanswerable": False,
                "free_form_answer": "answer",
                "extractive_spans": [],
                "yes_no": None,
                "evidence": ["paragraph"],
            },
        }
    ]

    payloads = schema.normalize_answer_payloads(raw_answers)

    assert payloads == [
        {
            "unanswerable": False,
            "free_form_answer": "answer",
            "extractive_spans": [],
            "yes_no": None,
            "evidence": ["paragraph"],
        }
    ]
