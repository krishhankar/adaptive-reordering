"""CPU-only integration test for cached retrieval and mock generation."""

import json
from pathlib import Path
import sys

import pytest
import yaml

from src.run_experiment import main


def test_mock_pipeline_runs_both_orders_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_file = tmp_path / "questions.jsonl"
    data_file.write_text(
        json.dumps({"id": "q1", "question": "Capital of France?", "answers": ["Paris"]})
        + "\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache" / "retrieval"
    cache_dir.mkdir(parents=True)
    corpus_file = tmp_path / "corpus.tsv"
    corpus_file.write_text(
        "id\ttitle\ttext\n"
        "p1\tFrance\tParis is the capital of France.\n"
        "p2\tEurope\tFrance is in Europe.\n",
        encoding="utf-8",
    )
    raw_dir = tmp_path / "results" / "raw"
    config = {
        "seed": 7,
        "paths": {
            "cache": str(tmp_path / "cache"),
            "retrieval_cache": str(cache_dir),
            "indexes": str(tmp_path / "indexes"),
            "results": str(tmp_path / "results"),
            "raw_results": str(raw_dir),
            "bm25_index": "unused",
            "e5_index": "unused",
        },
        "datasets": {"nq": {"hf_name": "unused", "split": "test"}},
        "retrievers": {"bm25": {"type": "bm25"}},
        "models": {
            "gemma": {
                "hf_name": "unused-model",
                "max_k": 2,
                "context_window": 2048,
                "ks": [1, 2],
            }
        },
        "generation": {
            "dtype": "bfloat16",
            "quantization_4bit": False,
            "max_new_tokens": 32,
            "batch_size": 2,
            "backend": "auto",
            "mock_prediction": "Paris",
        },
        "retrieval": {"max_passages": 200},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment",
            "--config",
            str(config_path),
            "--dataset",
            "nq",
            "--retriever",
            "bm25",
            "--model",
            "gemma",
            "--dataset_file",
            str(data_file),
            "--ks",
            "1",
            "2",
            "--mock_llm",
            "--corpus_subset",
            "--corpus_file",
            str(corpus_file),
        ],
    )
    main()
    sys.argv.append("--resume")
    main()

    raw_path = next(raw_dir.glob("nq_bm25_gemma_*.jsonl"))
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    questions = [record for record in records if record.get("type") != "metadata"]
    assert len(questions) == 4
    assert {record["order"] for record in questions} == {"original", "reordered"}
    assert {record["k"] for record in questions} == {1, 2}
    assert all(record["correct"] for record in questions)
    assert all(record["prompt_token_count"] > 0 for record in questions)

    cache_records = [
        json.loads(line)
        for line in (cache_dir / "nq_bm25.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(cache_records) == 1
    assert len(cache_records[0]["passages"]) == 2

    summary_path = tmp_path / "results" / "summary.csv"
    with summary_path.open(encoding="utf-8") as stream:
        summary = stream.read()
    assert "accuracy,recall,precision,n_questions" in summary
    assert summary.count("\n") == 5
