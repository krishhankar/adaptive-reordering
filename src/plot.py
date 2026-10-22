"""Plot baseline RAG metrics from results/summary.csv."""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def read_summary(path: Path) -> list[dict[str, Any]]:
    """Read the run summary and parse typed metric columns."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("k", "n_questions"):
            row[key] = int(row[key])
        for key in ("accuracy", "recall", "precision"):
            row[key] = float(row[key])
    return rows


def create_plots(summary_path: Path, output_dir: Path) -> None:
    """Create accuracy curves, NQ metric curves, and largest-k tables."""
    rows = read_summary(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    accuracies = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["model"], row["retriever"], row["order"])
        accuracies[key][row["k"]] = row["accuracy"]

    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    retrievers = sorted({row["retriever"] for row in rows})
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False)
    for panel, (dataset, model, retriever) in enumerate(
        (dataset, model, retriever)
        for dataset in datasets
        for model in models
        for retriever in retrievers
    ):
        axis = axes.flat[panel]
        for order, style, label in (
            ("original", "-", "Original"),
            ("reordered", "--", "Odd-even"),
        ):
            values = accuracies[(dataset, model, retriever, order)]
            if values:
                ks = sorted(values)
                axis.plot(ks, [values[k] for k in ks], linestyle=style, marker="o", label=label)
        axis.set_title(f"{dataset.upper()} / {model} / {retriever}")
        axis.set_xlabel("k")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    for panel in range(len(datasets) * len(models) * len(retrievers), 8):
        axes.flat[panel].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_k.png", dpi=160)
    plt.close(fig)

    nq_rows = [row for row in rows if row["dataset"] == "nq" and row["order"] == "original"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for panel, (model, retriever) in enumerate(
        (model, retriever) for model in models for retriever in retrievers
    ):
        axis = axes.flat[panel]
        selected = [
            row for row in nq_rows
            if row["model"] == model and row["retriever"] == retriever
        ]
        for metric in ("accuracy", "recall", "precision"):
            metric_rows = sorted(selected, key=lambda row: row["k"])
            if metric_rows:
                axis.plot(
                    [row["k"] for row in metric_rows],
                    [row[metric] for row in metric_rows],
                    marker="o",
                    label=metric.title(),
                )
        axis.set_title(f"NQ / {model} / {retriever}")
        axis.set_xlabel("k")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "nq_metrics_vs_k.png", dpi=160)
    plt.close(fig)

    highest: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        key = (row["dataset"], row["model"], row["retriever"])
        highest[key] = max(highest[key], row["k"])
    table = []
    for dataset, model, retriever in sorted(highest):
        max_k = highest[(dataset, model, retriever)]
        values = {
            row["order"]: row["accuracy"]
            for row in rows
            if row["dataset"] == dataset
            and row["model"] == model
            and row["retriever"] == retriever
            and row["k"] == max_k
        }
        original = values.get("original")
        reordered = values.get("reordered")
        table.append(
            {
                "dataset": dataset,
                "model": model,
                "retriever": retriever,
                "k": max_k,
                "original": original,
                "reordered": reordered,
                "change": reordered - original
                if reordered is not None and original is not None
                else None,
            }
        )
    with (output_dir / "accuracy_at_max_k.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]) if table else [])
        if table:
            writer.writeheader()
            writer.writerows(table)
    markdown = [
        "| Dataset | Model | Retriever | k | Original | Reordered | Change |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    markdown.extend(
        "| {dataset} | {model} | {retriever} | {k} | {original} | {reordered} | {change} |".format(
            **{
                **row,
                "original": _format_metric(row["original"]),
                "reordered": _format_metric(row["reordered"]),
                "change": _format_metric(row["change"]),
            }
        )
        for row in table
    )
    (output_dir / "accuracy_at_max_k.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    logger.info("Wrote plots and max-k tables to %s", output_dir)


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--output_dir", default="results/plots")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    create_plots(Path(args.summary), Path(args.output_dir))


if __name__ == "__main__":
    main()
