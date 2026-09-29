"""QASPER evidence-to-passage ground-truth construction.

This module is deliberately dependency-free so the most important evaluation
semantics can be unit-tested without downloading datasets or embedding models.
"""

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple


_FLOAT_EVIDENCE_PREFIX = "FLOAT SELECTED"


def normalize_evidence_text(text: Any) -> str:
    """Normalize paragraph text for stable QASPER evidence matching."""
    if text is None:
        return ""
    return " ".join(str(text).split())


def _numeric_id(raw_id: Any) -> int:
    if isinstance(raw_id, bool):
        raise ValueError(f"Invalid passage id: {raw_id!r}")
    try:
        return int(raw_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Passage id must be an integer-compatible value: {raw_id!r}") from exc


def build_passage_lookup(
    preprocessed_dataset: Mapping[str, Any],
    split: str,
) -> Tuple[Dict[int, str], Dict[str, List[Tuple[int, str]]]]:
    """Build global passage-id lookup and paper-local normalized paragraph lists."""
    passages = preprocessed_dataset.get("passages", {})
    if not isinstance(passages, Mapping):
        raise ValueError("preprocessed_dataset['passages'] must be a mapping")

    passage_texts: Dict[int, str] = {}
    passages_by_paper: DefaultDict[str, List[Tuple[int, str]]] = defaultdict(list)

    for raw_id, passage in passages.items():
        if not isinstance(passage, Mapping) or passage.get("split") != split:
            continue

        passage_id = _numeric_id(raw_id)
        paper_id = str(passage.get("paper_id", ""))
        original_text = str(
            passage.get("original_text")
            or passage.get("cleaned_text")
            or ""
        )
        normalized_text = normalize_evidence_text(original_text)

        passage_texts[passage_id] = original_text
        passages_by_paper[paper_id].append((passage_id, normalized_text))

    return passage_texts, dict(passages_by_paper)


def collect_textual_evidence(qa_pair: Mapping[str, Any]) -> List[str]:
    """Return unique textual evidence paragraphs across all QASPER annotations."""
    evidence_texts: List[str] = []
    seen = set()

    answers = qa_pair.get("answers", [])
    if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
        return evidence_texts

    for answer in answers:
        if not isinstance(answer, Mapping):
            continue
        evidence_items = answer.get("evidence", [])
        if not isinstance(evidence_items, Sequence) or isinstance(evidence_items, (str, bytes)):
            continue

        for evidence in evidence_items:
            normalized = normalize_evidence_text(evidence)
            if not normalized or normalized.startswith(_FLOAT_EVIDENCE_PREFIX):
                continue
            if normalized not in seen:
                seen.add(normalized)
                evidence_texts.append(normalized)

    return evidence_texts


def match_relevant_passage_ids(
    qa_pair: Mapping[str, Any],
    paper_passages: Sequence[Tuple[int, str]],
) -> List[int]:
    """Match QASPER paragraph-level evidence to global passage IDs."""
    evidence = set(collect_textual_evidence(qa_pair))
    if not evidence:
        return []

    matched = [passage_id for passage_id, text in paper_passages if text in evidence]
    return sorted(set(matched))


def prepare_evaluation_data(
    processed_dataset: Mapping[str, Any],
    preprocessed_dataset: Mapping[str, Any],
    test_split: str,
    max_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepare evidence-grounded retrieval evaluation data for one split.

    Queries without textual paragraph evidence (for example unanswerable or
    figure/table-only annotations) are excluded because they do not have a
    valid passage-level relevance label for this retriever.
    """
    if test_split not in processed_dataset:
        raise ValueError(f"Test split {test_split!r} does not exist")
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be positive when provided")

    passage_texts, passages_by_paper = build_passage_lookup(
        preprocessed_dataset, test_split
    )

    queries: List[str] = []
    ground_truth: List[List[int]] = []
    skipped_no_text_evidence = 0
    skipped_unmatched_evidence = 0

    for document in processed_dataset[test_split]:
        paper_id = str(document.get("paper_id", ""))
        paper_passages = passages_by_paper.get(paper_id, [])

        for qa_pair in document.get("qa_pairs", []):
            evidence = collect_textual_evidence(qa_pair)
            if not evidence:
                skipped_no_text_evidence += 1
                continue

            relevant_passage_ids = match_relevant_passage_ids(
                qa_pair, paper_passages
            )
            if not relevant_passage_ids:
                skipped_unmatched_evidence += 1
                continue

            queries.append(str(qa_pair.get("question", "")))
            ground_truth.append(relevant_passage_ids)

            if max_queries is not None and len(queries) >= max_queries:
                return {
                    "queries": queries,
                    "ground_truth": ground_truth,
                    "passage_texts": passage_texts,
                    "skipped_no_text_evidence": skipped_no_text_evidence,
                    "skipped_unmatched_evidence": skipped_unmatched_evidence,
                }

    return {
        "queries": queries,
        "ground_truth": ground_truth,
        "passage_texts": passage_texts,
        "skipped_no_text_evidence": skipped_no_text_evidence,
        "skipped_unmatched_evidence": skipped_unmatched_evidence,
    }
