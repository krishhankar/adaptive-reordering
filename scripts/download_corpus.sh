#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/corpus/psgs_w100.tsv}"
mkdir -p "$(dirname "$destination")"
archive="${destination}.gz"
curl --fail --location --output "$archive" \
  https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip --decompress --force "$archive"
