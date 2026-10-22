"""Passage ordering strategies used by the baseline experiment."""

from collections.abc import Sequence
from typing import TypeVar, cast

T = TypeVar("T")


def original_order(passages: Sequence[T]) -> list[T]:
    """Return passages in their retrieval order."""
    return list(passages)


def odd_even_reorder(passages: Sequence[T]) -> list[T]:
    """Arrange passages as odd-ranked passages, then even-ranked passages reversed.

    Input passages are assumed to be sorted by decreasing retrieval score. The
    two-pointer implementation places odd ranks from the left and even ranks
    from the right in O(k) time.
    """
    sentinel = object()
    result: list[T | object] = [sentinel] * len(passages)
    left, right = 0, len(passages) - 1

    for rank, passage in enumerate(passages, start=1):
        if rank % 2:
            result[left] = passage
            left += 1
        else:
            result[right] = passage
            right -= 1

    return [cast(T, passage) for passage in result]
