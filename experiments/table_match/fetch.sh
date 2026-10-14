#!/usr/bin/env bash
# Download the entity-matching benchmarks of the Leipzig University database group (raw tables plus the complete human mapping).
# Abt-Buy: product catalogues of two retailers, about 135 KB. DBLP-Scholar: two bibliographic databases, about 4 MB.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
for name in Abt-Buy DBLP-Scholar; do
  if [ ! -d "data/$name" ]; then
    curl -sS --fail -L -o "data/$name.zip" "https://dbs.uni-leipzig.de/files/datasets/$name.zip"
    unzip -q -o "data/$name.zip" -d "data/$name"
  fi
  echo "$name: $(find "data/$name" -name '*.csv' | wc -l | tr -d ' ') CSV files"
done
