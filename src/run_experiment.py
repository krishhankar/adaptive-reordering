"""Run baseline RAG experiments from cached retrieval results."""

import argparse
import csv
import hashlib
import json
import logging
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.data import load_dataset
from src.evaluate import is_correct, retrieval_metrics
from src.generate import Generator
from src.prompt import build_prompt
from src.retrieve import (
    create_retriever,
    load_retrieval_cache,
    save_retrieval_cache,
    write_retrieval_cache,
)
from src.reorder import odd_even_reorder, original_order

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        logger.warning("Unable to determine Git commit.")
        return "unknown"


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    def package_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not-installed"

    return {
        "python": __import__("platform").python_version(),
        "datasets": package_version("datasets"),
        "pandas": package_version("pandas"),
        "numpy": package_version("numpy"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "vllm": package_version("vllm"),
        "pyserini": package_version("pyserini"),
        "sentence-transformers": package_version("sentence-transformers"),
        "faiss-cpu": package_version("faiss-cpu"),
    }


def _metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "metadata", **metadata}, ensure_ascii=False) + "\n")


def _write_summary(results_dir: Path, raw_dir: Path | None = None) -> None:
    raw_dir = raw_dir or results_dir / "raw"
    result_files = sorted(raw_dir.glob("*.jsonl"))
    latest_config: dict[tuple[str, str, str], tuple[str, str]] = {}
    for path in result_files:
        last_metadata: dict[str, Any] | None = None
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("type") == "metadata":
                    last_metadata = record
        if last_metadata is not None:
            key = (
                last_metadata["dataset"],
                last_metadata["model"],
                last_metadata["retriever"],
            )
            timestamp = str(last_metadata.get("created_at", ""))
            if key not in latest_config or timestamp >= latest_config[key][0]:
                latest_config[key] = (timestamp, last_metadata["config_id"])

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in result_files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("type") == "metadata":
                    continue
                base_key = (
                    record["dataset"],
                    record["model"],
                    record["retriever"],
                )
                if record["config_id"] != latest_config.get(base_key, ("", None))[1]:
                    continue
                grouped[
                    (
                        record["dataset"],
                        record["model"],
                        record["retriever"],
                        record["k"],
                        record["order"],
                    )
                ].append(record)
    summary = results_dir / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "model",
        "retriever",
        "k",
        "order",
        "accuracy",
        "recall",
        "precision",
        "n_questions",
    ]
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
            writer.writerow(
                dict(
                    zip(
                        columns,
                        [
                            *key[:5],
                            sum(row["correct"] for row in rows) / len(rows),
                            sum(row["recall"] for row in rows) / len(rows),
                            sum(row["precision"] for row in rows) / len(rows),
                            len(rows),
                        ],
                        strict=True,
                    )
                )
            )


def _load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid configuration file: {path}")
    return loaded


def run(args: argparse.Namespace) -> None:
    """Execute retrieval (if uncached) and paired generation/evaluation."""
    config = _load_yaml(args.config)
    seed = int(config["seed"])
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.warning("NumPy unavailable; its RNG was not seeded.")
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.warning("Torch unavailable; only the Python random seed was set.")

    dataset_config = config["datasets"][args.dataset]
    if args.corpus_subset and args.retriever != "bm25":
        raise ValueError("--corpus_subset is available only with the bm25 retriever.")
    model_config = config["models"][args.model]
    retriever_config = config["retrievers"][args.retriever]
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    dataset = load_dataset(
        args.dataset,
        dataset_config["hf_name"],
        dataset_config["split"],
        args.limit,
        args.dataset_file,
    )
    requested_ks = args.ks or model_config["ks"]
    ks = sorted({int(k) for k in requested_ks})
    if not ks or ks[0] < 0:
        raise ValueError("At least one non-negative k is required.")
    if any(k > int(model_config["max_k"]) for k in ks):
        raise ValueError(f"k exceeds model maximum {model_config['max_k']}.")
    orders = list(dict.fromkeys(args.orders or ["original", "reordered"]))
    if not set(orders).issubset({"original", "reordered"}):
        raise ValueError("Orders must be 'original' and/or 'reordered'.")
    generation_config = config["generation"]
    batch_size = int(args.batch_size or generation_config["batch_size"])
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if not dataset:
        raise ValueError("The selected dataset contains no questions.")

    cache_dir = Path(config["paths"]["retrieval_cache"])
    cache_path = cache_dir / f"{args.dataset}_{args.retriever}.jsonl"
    if cache_path.exists() and not args.refresh_retrieval:
        retrieval = load_retrieval_cache(cache_path)
        logger.info("Using cached retrieval from %s", cache_path)
    else:
        retrieval = {}
    missing_ids = [row["id"] for row in dataset if row["id"] not in retrieval]
    if missing_ids:
        retriever = create_retriever(
            args.retriever,
            retriever_config,
            config["paths"],
            args.corpus_subset,
            args.corpus_file,
            args.corpus_subset_size,
            seed,
        )
        fresh_cache = cache_path.with_suffix(".missing.jsonl")
        write_retrieval_cache(
            [row for row in dataset if row["id"] in missing_ids],
            retriever,
            fresh_cache,
            int(config["retrieval"]["max_passages"]),
        )
        retrieval.update(load_retrieval_cache(fresh_cache))
        fresh_cache.unlink()
        save_retrieval_cache(retrieval, cache_path)

    generator = Generator(
        str(model_config["hf_name"]),
        backend=args.backend or generation_config["backend"],
        dtype=args.dtype or generation_config["dtype"],
        quantization_4bit=args.quantization_4bit
        or generation_config["quantization_4bit"],
        max_new_tokens=int(generation_config["max_new_tokens"]),
        mock_llm=args.mock_llm,
        mock_prediction=str(generation_config["mock_prediction"]),
    )
    resolved_dtype = args.dtype or generation_config["dtype"]
    resolved_backend = generator.backend
    signature = {
        "dataset": args.dataset,
        "split": dataset_config["split"] if args.dataset_file is None else None,
        "dataset_file": str(Path(args.dataset_file).resolve())
        if args.dataset_file
        else None,
        "retriever": args.retriever,
        "model": model_config,
        "generation": generation_config,
        "backend": resolved_backend,
        "dtype": resolved_dtype,
        "quantization_4bit": args.quantization_4bit
        or generation_config["quantization_4bit"],
        "ks": ks,
        "orders": orders,
        "mock_llm": args.mock_llm,
        "seed": seed,
    }
    config_id = hashlib.sha256(
        json.dumps(signature, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    result_dir = Path(config["paths"]["results"])
    raw_dir = Path(config["paths"]["raw_results"])
    raw_path = raw_dir / (
        f"{args.dataset}_{args.retriever}_{args.model}_{config_id}.jsonl"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "retriever": args.retriever,
        "config_id": config_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "seed": seed,
        "mock_llm": args.mock_llm,
    }
    if not args.resume:
        raw_path.write_text("", encoding="utf-8")
    _metadata(raw_path, metadata)
    done: set[tuple[str, int, str]] = set()
    if args.resume and raw_path.exists():
        with raw_path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if (
                    record.get("type") != "metadata"
                    and record.get("config_id") == config_id
                ):
                    done.add((str(record["id"]), int(record["k"]), record["order"]))

    metrics_path = cache_dir / f"{args.dataset}_{args.retriever}_metrics.jsonl"
    metrics: dict[tuple[str, int], tuple[float, float, str, str]] = {}
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                metrics[(str(row["id"]), int(row["k"]))] = (
                    float(row["recall"]),
                    float(row["precision"]),
                    json.dumps(row.get("gold_answers", []), sort_keys=True),
                    str(row.get("passage_hash", "")),
                )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as output:
        for k in ks:
            for order in orders:
                batch: list[tuple[dict[str, Any], list[Any], str, int, float, float]] = []
                for example in dataset:
                    if (example["id"], k, order) in done:
                        continue
                    top = retrieval[example["id"]][:k]
                    gold_signature = json.dumps(example["answers"], sort_keys=True)
                    passage_signature = hashlib.sha256(
                        json.dumps(
                            [(passage.id, passage.text) for passage in top],
                            ensure_ascii=False,
                        ).encode("utf-8")
                    ).hexdigest()
                    cached_metrics = metrics.get((example["id"], k))
                    if (
                        cached_metrics is None
                        or cached_metrics[2] != gold_signature
                        or cached_metrics[3] != passage_signature
                    ):
                        recall, precision = retrieval_metrics(
                            [passage.text for passage in top], example["answers"]
                        )
                        metrics[(example["id"], k)] = (
                            recall,
                            precision,
                            gold_signature,
                            passage_signature,
                        )
                    else:
                        recall, precision = cached_metrics[:2]
                    ordered = (
                        odd_even_reorder(top) if order == "reordered" else original_order(top)
                    )
                    prompt = generator.format_chat(
                        build_prompt(ordered, example["question"])
                    )
                    prompt_tokens = generator.token_count(prompt)
                    if prompt_tokens > int(model_config["context_window"]):
                        logger.warning(
                            "Skipping %s k=%s order=%s: prompt has %s tokens, "
                            "model context is %s.",
                            example["id"],
                            k,
                            order,
                            prompt_tokens,
                            model_config["context_window"],
                        )
                        continue
                    batch.append((example, ordered, prompt, prompt_tokens, recall, precision))
                    if len(batch) >= batch_size:
                        _generate_batch(
                            generator, batch, output, args, k, order, metadata
                        )
                        batch.clear()
                if batch:
                    _generate_batch(generator, batch, output, args, k, order, metadata)
                    batch.clear()
    with metrics_path.open("w", encoding="utf-8") as stream:
        for (identifier, k), (
            recall,
            precision,
            gold_signature,
            passage_signature,
        ) in sorted(metrics.items()):
            stream.write(
                json.dumps(
                    {
                        "id": identifier,
                        "k": k,
                        "recall": recall,
                        "precision": precision,
                        "gold_answers": json.loads(gold_signature),
                        "passage_hash": passage_signature,
                    }
                )
                + "\n"
            )
    _write_summary(result_dir, raw_dir)


def _generate_batch(
    generator: Generator,
    batch: list[tuple[dict[str, Any], list[Any], str, int, float, float]],
    output: Any,
    args: argparse.Namespace,
    k: int,
    order: str,
    metadata: dict[str, Any],
) -> None:
    prompts = [item[2] for item in batch]
    start = time.perf_counter()
    predictions = generator.generate(prompts)
    elapsed = max(time.perf_counter() - start, 1e-9)
    generated_tokens = sum(
        generator.generated_token_count(prediction) for prediction in predictions
    )
    logger.info(
        "Generated %s questions in %.2fs (%.2f tokens/sec, GPU memory: %s bytes)",
        len(batch),
        elapsed,
        generated_tokens / elapsed,
        generator.gpu_memory_bytes(),
    )
    for (example, _passages, _prompt, token_count, recall, precision), prediction in zip(
        batch, predictions, strict=True
    ):
        output.write(
            json.dumps(
                {
                    **metadata,
                    "id": example["id"],
                    "dataset": args.dataset,
                    "model": args.model,
                    "retriever": args.retriever,
                    "gold_answers": example["answers"],
                    "order": order,
                    "k": k,
                    "prompt_token_count": token_count,
                    "prediction": prediction,
                    "correct": is_correct(prediction, example["answers"]),
                    "recall": recall,
                    "precision": precision,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    output.flush()


def main() -> None:
    """Parse CLI options and run an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("nq", "popqa"))
    parser.add_argument("--retriever", required=True, choices=("bm25", "e5"))
    parser.add_argument("--model", required=True, choices=("gemma", "mistral"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset_file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ks", nargs="+", type=int)
    parser.add_argument("--orders", nargs="+", choices=("original", "reordered"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh_retrieval", action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--backend", choices=("auto", "vllm", "transformers"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--quantization_4bit", action="store_true")
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--corpus_subset", action="store_true")
    parser.add_argument("--corpus_file")
    parser.add_argument("--corpus_subset_size", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
