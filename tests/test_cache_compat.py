from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(relative_path: str, name: str):
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location(name, root / relative_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cache_compat = _load_module("src/data/cache_compat.py", "cache_compat")


def test_restore_preprocessed_cache_keys_converts_json_numeric_keys():
    data = {
        "passages": {"0": {"text": "a"}, "12": {"text": "b"}},
        "questions": {"7": {"question": "q"}},
        "statistics": {"num_passages": 2},
    }

    restored = cache_compat.restore_preprocessed_cache_keys(data)

    assert set(restored["passages"]) == {0, 12}
    assert set(restored["questions"]) == {7}
    assert restored["statistics"] == {"num_passages": 2}


def test_restore_numeric_mapping_keys_preserves_non_numeric_keys():
    restored = cache_compat.restore_numeric_mapping_keys({"abc": 1, "-2": 2})
    assert restored == {"abc": 1, -2: 2}
