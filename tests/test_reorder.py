"""Tests for passage ordering."""

import pytest

from src.reorder import odd_even_reorder, original_order


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, []),
        (1, [1]),
        (2, [1, 2]),
        (5, [1, 3, 5, 4, 2]),
        (6, [1, 3, 5, 6, 4, 2]),
        (8, [1, 3, 5, 7, 8, 6, 4, 2]),
    ],
)
def test_odd_even_reorder_examples(count: int, expected: list[int]) -> None:
    assert odd_even_reorder(list(range(1, count + 1))) == expected


@pytest.mark.parametrize("count", range(11))
def test_odd_even_reorder_is_a_permutation(count: int) -> None:
    passages = [f"passage-{index}" for index in range(count)]
    reordered = odd_even_reorder(passages)
    assert len(reordered) == len(passages)
    assert sorted(reordered) == sorted(passages)


def test_original_order_returns_a_copy() -> None:
    passages = ["first", "second"]
    ordered = original_order(passages)
    assert ordered == passages
    assert ordered is not passages
