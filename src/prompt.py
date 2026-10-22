"""Pure prompt construction helpers."""

from collections.abc import Mapping, Sequence
from typing import Any

from src.reorder import odd_even_reorder, original_order

INSTRUCTION = (
    "Answer the question based on the given document. "
    "Only give me the answer and do not output any other words."
)


def _passage_text(passage: Any) -> str:
    """Extract text from a passage string or a common passage mapping."""
    if isinstance(passage, str):
        return passage
    if isinstance(passage, Mapping):
        for key in ("text", "contents", "passage"):
            value = passage.get(key)
            if isinstance(value, str):
                return value
    value = getattr(passage, "text", None)
    if isinstance(value, str):
        return value
    raise TypeError("Each passage must be text or a mapping containing text.")


def build_prompt(
    passages: Sequence[Any],
    question: str,
    reorder: bool = False,
) -> str:
    """Build the baseline prompt, optionally applying odd-even reordering."""
    ordered_passages = (
        odd_even_reorder(passages) if reorder else original_order(passages)
    )
    documents = "\n\n".join(
        f"Document {index}: {_passage_text(passage)}"
        for index, passage in enumerate(ordered_passages, start=1)
    )
    document_section = f"\n\n{documents}" if documents else ""
    return (
        f"{INSTRUCTION}\n"
        "The following are given documents."
        f"{document_section}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
