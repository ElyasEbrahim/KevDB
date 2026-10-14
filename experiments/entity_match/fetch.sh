#!/usr/bin/env bash
# Download the validation and test splits of the Magellan entity-matching benchmarks (the version curated in the Ditto repository, Apache-2.0), about 6 MB in total.
# The zero-shot experiment needs no training split, so it is not downloaded.
set -euo pipefail
cd "$(dirname "$0")"
base=https://raw.githubusercontent.com/megagonlabs/ditto/master/data/er_magellan

fetch() {  # fetch <local name> <path in the repository>
  mkdir -p "data/$1"
  for split in valid test; do
    [ -s "data/$1/$split.txt" ] || curl -sS --fail -o "data/$1/$split.txt" "$base/$2/$split.txt"
  done
  echo "$1: $(wc -l < "data/$1/valid.txt" | tr -d ' ') validation pairs, $(wc -l < "data/$1/test.txt" | tr -d ' ') test pairs"
}

fetch abt-buy        Textual/Abt-Buy
fetch amazon-google  Structured/Amazon-Google
fetch walmart-amazon Structured/Walmart-Amazon
fetch dblp-acm       Structured/DBLP-ACM
