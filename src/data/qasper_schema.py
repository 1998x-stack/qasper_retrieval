"""Compatibility helpers for QASPER nested Hugging Face row shapes.

Older Hugging Face datasets versions represent Sequence<dict> values as
dict-of-lists. Newer/materialized data may instead look like list-of-dicts.
These helpers normalize both forms into record lists.
"""

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _records_from_dict_of_lists(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    lengths = [
        len(value)
        for value in data.values()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ]
    if not lengths:
        return [dict(data)] if data else []

    record_count = max(lengths)
    records: List[Dict[str, Any]] = []

    for index in range(record_count):
        record: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                record[key] = value[index] if index < len(value) else None
            else:
                record[key] = value
        records.append(record)

    return records


def normalize_records(value: Any) -> List[Dict[str, Any]]:
    """Normalize list-of-dicts or dict-of-lists into list-of-dicts."""
    if value is None:
        return []

    if isinstance(value, Mapping):
        return _records_from_dict_of_lists(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]

    return []


def normalize_answer_payloads(raw_answers: Any) -> List[Dict[str, Any]]:
    """Extract answer payload dictionaries from QASPER annotation containers."""
    annotations = normalize_records(raw_answers)
    payloads: List[Dict[str, Any]] = []

    for annotation in annotations:
        answer = annotation.get("answer")

        if isinstance(answer, Mapping):
            # Nested Sequence<dict> can itself arrive as dict-of-lists.
            nested_records = normalize_records(answer)
            if nested_records:
                payloads.extend(nested_records)
            else:
                payloads.append(dict(answer))
        elif isinstance(answer, Sequence) and not isinstance(answer, (str, bytes)):
            payloads.extend(
                dict(item) for item in answer if isinstance(item, Mapping)
            )

    return payloads
