"""Evaluation failure aggregation contracts."""

from typing import Mapping


class MethodEvaluationError(RuntimeError):
    """Raised after all requested retrieval methods have been attempted."""

    def __init__(self, failures: Mapping[str, BaseException]) -> None:
        self.failures = dict(failures)
        detail = "; ".join(
            f"{method}: {type(error).__name__}: {error}"
            for method, error in self.failures.items()
        )
        super().__init__(
            f"{len(self.failures)} requested retrieval method(s) failed: {detail}"
        )


def raise_for_method_failures(failures: Mapping[str, BaseException]) -> None:
    """Fail the pipeline after collecting all requested method failures."""
    if failures:
        raise MethodEvaluationError(failures)
