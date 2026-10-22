# passage-reordering-rag

A reproducible **inference-only baseline** for the long-context RAG passage
ordering experiment from Jin et al., “Long-Context LLMs Meet RAG” (ICLR 2025).
For each question, the system retrieves passages once and compares the same
generator prompt in two conditions:

1. **Original:** passages remain in descending retrieval-score order.
2. **Odd-even:** odd-ranked passages are followed by even-ranked passages in
   reverse order (for example, `d1, d3, d5, d4, d2`).

Everything other than passage order is held constant. This repository contains
only retrieval, inference, and evaluation: it does not train or fine-tune
models, and does not implement adaptive reordering or LoRA.

## Architecture

```text
                         ┌─ original order ─┐
Question -> Retriever -> top-k              ├-> same prompt -> LLM -> accuracy
                         └─ odd-even order ─┘
```

The retrieval cache stores up to 200 passages per question and is reused across
all configured values of `k` and both order conditions.

## Components and defaults

| Component | Baseline |
|---|---|
| Datasets | NQ Open held-out validation/dev split and PopQA, or JSONL files |
| Corpus | DPR Wikipedia December 2018 `psgs_w100` |
| Sparse retrieval | Pyserini prebuilt `wikipedia-dpr-100w` index |
| Dense retrieval | `intfloat/e5-large-v2` with a FAISS inner-product index |
| Generators | `google/gemma-2-9b-it` and `mistralai/Mistral-Nemo-Instruct-2407` |
| Decoding | Greedy, temperature 0, 32 new tokens |
| Evaluation | Normalized substring answer match; Recall@k and Precision@k |

The dataset source, split, k lists, model maximums, paths, and inference
settings are in [`configs/default.yaml`](configs/default.yaml). Change the E5
model there to `intfloat/e5-base-v2` if preferred. NQ and PopQA may also be
loaded from FlashRAG-style JSONL using `--dataset_file`; each JSONL record needs
a `question` and an answer field such as `answers`, `golden_answers`, or
`answer`.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Optional faster inference backend:
python -m pip install vllm
```

Pyserini requires a compatible Java runtime. For CUDA inference, install a
PyTorch build appropriate for the machine before installing the remaining
requirements if needed.

**Large downloads require deliberate approval before running the relevant
command.** Installing GPU-enabled PyTorch, obtaining the 21M-passage corpus or
prebuilt index, creating the full E5 index, loading a HuggingFace dataset, and
loading generator weights can transfer substantial data. No such downloads
are performed by the tests or mock demo below.

## CPU quick demo

The repository includes a tiny local sample corpus at
[`data/dev/sample_corpus.tsv`](data/dev/sample_corpus.tsv). The BM25
`--corpus_subset` development path builds a lightweight in-memory BM25 index
from a deterministic reservoir sample; it does not need Pyserini, the 21M
corpus, or model weights.

For 100 NQ Open questions with mock generation:

```bash
python -m src.run_experiment \
  --dataset nq \
  --retriever bm25 \
  --model gemma \
  --limit 100 \
  --ks 1 5 \
  --mock_llm \
  --corpus_subset \
  --corpus_file data/dev/sample_corpus.tsv
```

This fetches/caches the selected dataset split if it is not already available,
but the mock generator does not download any LLM. The tiny corpus is for
pipeline validation only, so its retrieval quality is not representative of
the paper. For a completely local test with no dataset download:

```bash
python -m src.run_experiment \
  --dataset nq \
  --retriever bm25 \
  --model gemma \
  --dataset_file path/to/nq_dev.jsonl \
  --limit 100 \
  --mock_llm \
  --corpus_subset \
  --corpus_file data/dev/sample_corpus.tsv
```

The JSONL file should contain up to 100 records with `id`, `question`, and gold
answers. The CPU end-to-end mock path is also exercised by `pytest`.

## Full experiments

### Corpus and retrieval

The corpus download script downloads the large DPR TSV:

```bash
bash scripts/download_corpus.sh
```

Run it only after approving that download. The standard BM25 backend uses
Pyserini's `wikipedia-dpr-100w` prebuilt index. It may download the index on
first use. To create the dense index, first approve the corpus and E5 model
downloads, then run:

```bash
python scripts/build_e5_index.py \
  --corpus data/corpus/psgs_w100.tsv \
  --output indexes/e5 \
  --model intfloat/e5-large-v2
```

The index builder streams the corpus, uses `passage: ` and `query: ` prefixes,
and writes a non-training FP16 scalar-quantized FAISS index. Building the
21M-passage dense index requires substantial disk space, compute, and memory.

Retrieval is run once per dataset/retriever and cached in
`cache/retrieval/<dataset>_<retriever>.jsonl`:

```bash
python -m src.retrieve --dataset nq --retriever bm25
python -m src.retrieve --dataset popqa --retriever bm25
python -m src.retrieve --dataset nq --retriever e5
python -m src.retrieve --dataset popqa --retriever e5
```

Use `--refresh` on `src.retrieve` or `--refresh_retrieval` on the experiment
runner only when you explicitly want to replace cached retrievals. Existing
question IDs are not re-retrieved during normal runs; missing IDs are added to
the cache. The cache keeps passage IDs, texts, and scores, so all k values use
the same retrieval result prefix.

### Generation and evaluation

Run one setting:

```bash
python -m src.run_experiment \
  --dataset nq \
  --retriever bm25 \
  --model gemma \
  --resume
```

The command loads configured k values, runs original and odd-even conditions,
and stores per-question records under `results/raw/`. `--limit N` enables a
quick subset; `--ks 1 5 10` and `--orders original reordered` override the
configured run settings. Use `--resume` to skip completed question/config/k/
order combinations. Prompts exceeding the model context window are warned
about and skipped rather than truncated. `--batch_size`, `--dtype`, and
`--quantization_4bit` adjust inference behavior.

For a full configured sweep, after approving any needed corpus, index, dataset,
and model downloads:

```bash
bash scripts/run_all.sh
```

The runner prefers vLLM when installed and otherwise uses Transformers.
`--mock_llm` replaces generation with the configured fixed prediction and is
intended for plumbing tests, not evaluation.

## Hardware notes

- Gemma 2 9B Instruct typically needs about **24 GB GPU memory** in bf16.
- Mistral Nemo Instruct 12B with 200 passages can create roughly **30K-token
  prompts** and needs substantial VRAM. The context guard warns and skips
  over-limit prompts.
- `--quantization_4bit` enables optional bitsandbytes 4-bit Transformers
  loading for smaller GPUs. It may change real-model outputs and should be
  treated as a separate experiment configuration.
- E5 indexing all 21M passages is also compute-, memory-, and disk-intensive.

## Outputs

- `cache/retrieval/`: reusable top-passage retrieval results and per-question
  retrieval metric cache.
- `results/raw/`: JSONL metadata and per-question outputs with question ID,
  gold answers, order, k, prompt token count, prediction, correctness, Recall,
  and Precision. A stable configuration ID keeps resume runs distinct.
- `results/summary.csv`: aggregate `dataset, model, retriever, k, order,
  accuracy, recall, precision, n_questions`. The summary uses the most recently
  run configuration for each dataset/model/retriever combination.
- `results/plots/accuracy_vs_k.png`: 2x4 accuracy panels, original solid and
  odd-even dashed.
- `results/plots/nq_metrics_vs_k.png`: NQ accuracy, Recall@k, and Precision@k
  by model/retriever for original order.
- `results/plots/accuracy_at_max_k.{md,csv}`: largest-k original/reordered
  accuracies and their difference.

Generate plots and largest-k tables from an existing summary:

```bash
python -m src.plot
```

Correctness is true when any normalized gold answer is a substring of the
normalized prediction. Normalization lowercases, removes Unicode punctuation
and articles (`a`, `an`, `the`), and collapses whitespace. Recall@k is one if
any top-k passage contains a normalized gold answer; Precision@k is the
fraction of top-k passages that do.

## Tests

```bash
pytest
```

The suite covers odd-even examples (including k=0), permutation behavior,
evaluation, prompt order invariance, and a CPU-only local BM25/mock-generation
end-to-end run.

## Limitations

- One greedy inference run per configuration; no repeated-sampling variance
  estimates.
- Accuracy uses normalized substring matching rather than semantic judging.
- Only two datasets (NQ Open and PopQA) and two generators (Gemma 2 9B and
  Mistral Nemo) are configured.
- The local `--corpus_subset` BM25 path is a development convenience and is
  not equivalent to the full DPR Wikipedia index.
