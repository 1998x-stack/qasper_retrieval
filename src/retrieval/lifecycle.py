"""Small dependency-ownership helpers for retriever composition."""

from typing import Callable, Optional, Tuple, TypeVar


T = TypeVar("T")


def resolve_resource(
    provided: Optional[T],
    factory: Callable[[], T],
) -> Tuple[T, bool]:
    """Return a resource and whether the caller owns its lifecycle."""
    if provided is not None:
        return provided, False
    return factory(), True
