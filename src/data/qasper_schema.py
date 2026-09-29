"""Compatibility helpers for QASPER nested Hugging Face row shapes.

Older Hugging Face datasets versions represent Sequence<dict> values as
dict-of-lists. Materialized data may instead look like list-of-dicts.
These helpers normalize both forms without flattening list-valued answer fields
such as evidence and extractive_spans.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _records_from_parallel_fields(
    data: Mapping[str, Any],
    record_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert parallel field lists into records while preserving nested lists."""
    if record_count is None:
        preferred_fields = ("free_form_answer", "unanswerable", "yes_no")
        lengths = [
            len(data[field])
            for field in preferred_fields
            if field in data and _is_sequence(data[field])
        ]
        if not lengths:
            lengths = [
                len(value)
                for value in data.values()
                if _is_sequence(value)
            ]
        record_count = max(lengths) if lengths else 1

    records: List[Dict[str, Any]] = []
    for index in range(record_count):
        record: Dict[str, Any] = {}
        for key, value in data.items():
            if _is_sequence(value) and len(value) == record_count:
                record[key] = value[index]
            else:
                record[key] = value
        records.append(record)
    return records


def normalize_answer_payloads(raw_answers: Any) -> List[Dict[str, Any]]:
    """Extract answer payload dictionaries from supported QASPER row shapes."""
    payloads: List[Dict[str, Any]] = []

    if isinstance(raw_answers, Mapping):
        answer_field = raw_answers.get("answer")

        # Typical datasets<4 Sequence<dict> shape:
        # {"annotation_id": [...], "answer": [{...}, {...}], "worker_id": [...]}
        if _is_sequence(answer_field):
            payloads.extend(
                dict(item) for item in answer_field if isinstance(item, Mapping)
            )
            return payloads

        # Defensive fallback for already-inverted nested answer fields.
        if isinstance(answer_field, Mapping):
            annotation_ids = raw_answers.get("annotation_id")
            record_count = (
                len(annotation_ids) if _is_sequence(annotation_ids) else None
            )
            return _records_from_parallel_fields(answer_field, record_count)

        return payloads

    # Materialized / newer shape:
    # [{"annotation_id": "...", "answer": {...}}, ...]
    if _is_sequence(raw_answers):
        for annotation in raw_answers:
            if not isinstance(annotation, Mapping):
                continue

            answer = annotation.get("answer")
            if isinstance(answer, Mapping):
                payloads.append(dict(answer))
            elif _is_sequence(answer):
                payloads.extend(
                    dict(item) for item in answer if isinstance(item, Mapping)
                )

    return payloads
