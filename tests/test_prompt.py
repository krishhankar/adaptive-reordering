"""Tests for pure prompt construction."""

import re

from src.prompt import INSTRUCTION, build_prompt


def _documents(prompt: str) -> list[str]:
    return re.findall(r"Document \d+: (.+)", prompt)


def test_prompt_contains_expected_instruction_and_question() -> None:
    prompt = build_prompt(["alpha", "beta"], "Where?")
    assert prompt.startswith(INSTRUCTION)
    assert "The following are given documents.\n\nDocument 1: alpha" in prompt
    assert "Question: Where?\nAnswer:" in prompt
    assert _documents(prompt) == ["alpha", "beta"]


def test_reorder_changes_only_passage_order() -> None:
    passages = ["d1", "d2", "d3", "d4", "d5"]
    original = build_prompt(passages, "question", reorder=False)
    reordered = build_prompt(passages, "question", reorder=True)
    assert sorted(_documents(original)) == sorted(_documents(reordered))
    assert _documents(original) == passages
    assert _documents(reordered) == ["d1", "d3", "d5", "d4", "d2"]
    assert original.replace(
        "\n\n".join(f"Document {i}: {p}" for i, p in enumerate(passages, 1)),
        "DOCUMENTS",
    ) == reordered.replace(
        "\n\n".join(f"Document {i}: {p}" for i, p in enumerate(_documents(reordered), 1)),
        "DOCUMENTS",
    )
