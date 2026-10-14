#!/usr/bin/env bash
# Download the JOB benchmark: the IMDb data snapshot (about 1.26 GB, hosted by CWI) and its 113 queries (GitHub, a few dozen KB).
# Safe to re-run: finished downloads are skipped or resumed.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/imdb

if [ ! -f data/imdb/.extracted ]; then
  curl -L -C - --fail -o data/imdb.tgz https://event.cwi.nl/da/job/imdb.tgz
  tar -xzf data/imdb.tgz -C data/imdb
  touch data/imdb/.extracted
fi

if [ ! -d data/job-queries ]; then
  git clone --depth 1 https://github.com/gregrahn/join-order-benchmark.git data/job-queries
fi

echo "CSV files: $(ls data/imdb/*.csv | wc -l | tr -d ' '), queries: $(ls data/job-queries/[0-9]*.sql | wc -l | tr -d ' ')"
