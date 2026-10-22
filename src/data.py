"""Dataset loading and normalization for NQ Open and PopQA."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _first(record: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _answers(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(answer) for answer in value if str(answer).strip()]
    return []


def _normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    question = _first(record, ("question", "query"))
    answers = _answers(
        _first(
            record,
            (
                "answers",
                "answer",
                "possible_answers",
                "gold_answers",
                "golden_answers",
                "short_answers",
            ),
            [],
        )
    )
    if question is None or not answers:
        raise ValueError(f"Dataset row {index} is missing a question or gold answer.")
    return {
        "id": str(_first(record, ("id", "qid", "question_id"), index)),
        "question": str(question),
        "answers": answers,
    }


def load_dataset(
    name: str,
    hf_name: str,
    split: str,
    limit: int | None = None,
    input_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load and normalize a dataset from HuggingFace or FlashRAG-style JSONL."""
    if input_file is not None:
        rows = []
        with Path(input_file).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in {input_file}, line {line_number}."
                        ) from error
    else:
        try:
            from datasets import get_dataset_split_names
            from datasets import load_dataset as hf_load_dataset
        except ImportError as error:
            raise RuntimeError(
                "Install the 'datasets' package or provide --dataset_file."
            ) from error
        available_splits = get_dataset_split_names(hf_name)
        selected_split = split
        if selected_split not in available_splits:
            candidates = (
                ("validation", "dev", "test")
                if name.lower() in {"nq", "nq_open"}
                else ("test", "validation", "dev", "train")
            )
            selected_split = next(
                (candidate for candidate in candidates if candidate in available_splits),
                "",
            )
            if not selected_split:
                raise ValueError(
                    f"Dataset {hf_name!r} has no supported evaluation split: "
                    f"{available_splits}"
                )
            logger.warning(
                "Configured split %s is unavailable for %s; using %s from %s.",
                split,
                hf_name,
                selected_split,
                available_splits,
            )
        logger.info(
            "Loading %s (%s) from HuggingFace split %s",
            name,
            hf_name,
            selected_split,
        )
        rows = hf_load_dataset(hf_name, split=selected_split)

    normalized = [_normalize_record(dict(row), index) for index, row in enumerate(rows)]
    return normalized[:limit] if limit is not None else normalized
