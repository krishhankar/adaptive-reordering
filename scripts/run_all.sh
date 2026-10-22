#!/usr/bin/env bash
set -euo pipefail

for dataset in nq popqa; do
  for retriever in bm25 e5; do
    for model in gemma mistral; do
      python -m src.run_experiment \
        --dataset "$dataset" \
        --retriever "$retriever" \
        --model "$model" \
        --resume
    done
  done
done

python -m src.plot
