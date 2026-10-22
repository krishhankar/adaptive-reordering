"""Tests for evaluation helpers."""

from src.evaluate import is_correct, normalize_answer, retrieval_metrics


def test_normalize_answer() -> None:
    assert normalize_answer("The quick, brown fox!") == "quick brown fox"


def test_is_correct_uses_normalized_substring_match() -> None:
    assert is_correct("The answer is New York.", ["new york"])
    assert not is_correct("Boston", ["New York"])


def test_retrieval_metrics() -> None:
    recall, precision = retrieval_metrics(
        ["The capital is Paris.", "Nothing relevant here", "Paris is in France."],
        ["paris"],
    )
    assert recall == 1.0
    assert precision == 2 / 3


def test_retrieval_metrics_empty_inputs() -> None:
    assert retrieval_metrics([], ["answer"]) == (0.0, 0.0)
    assert retrieval_metrics(["some passage"], []) == (0.0, 0.0)
