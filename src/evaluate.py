"""Answer normalization and retrieval/RAG metrics."""

import re
import unicodedata
from collections.abc import Iterable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation and articles, and collapse whitespace."""
    text = "".join(
        character
        for character in text.lower()
        if not unicodedata.category(character).startswith("P")
    )
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def is_correct(prediction: str, gold_answers: Iterable[str]) -> bool:
    """Return whether any normalized gold answer occurs in the prediction."""
    normalized_prediction = normalize_answer(prediction)
    return any(
        answer and answer in normalized_prediction
        for answer in (normalize_answer(gold) for gold in gold_answers)
    )


def retrieval_metrics(
    passages: Sequence[str],
    gold_answers: Iterable[str],
) -> tuple[float, float]:
    """Compute Recall@k and Precision@k for passages containing a gold answer."""
    normalized_gold = [
        answer for answer in (normalize_answer(gold) for gold in gold_answers) if answer
    ]
    if not passages:
        return 0.0, 0.0

    hits = sum(
        any(answer in normalize_answer(passage) for answer in normalized_gold)
        for passage in passages
    )
    return float(hits > 0), hits / len(passages)
