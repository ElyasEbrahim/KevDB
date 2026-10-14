#!/bin/sh
# The two extra downloads of phase 10, about 28 MB together, each checked against its SHA-256. Safe to re-run: existing files with the right checksum are not downloaded again.
#
# 1. The training splits of the four Magellan record-matching datasets (the version curated in the Ditto repository, Apache-2.0; pinned to the commit below), into experiments/entity_match/data/<dataset>/train.txt.
#    The validation and test splits were already downloaded by experiments/entity_match/fetch.sh and are left alone.
# 2. The training part of Kev's own base training set decision-v7 (12,576 records), replayed during fine-tuning against forgetting. Kev's repository ships only the manifest and checksums;
#    the file itself is in its Hub mirror jaredpalmer/kev-suites (pinned to the version Kev's code names), into open_models/kev/evals/v7/decision-v7/train.jsonl.
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# main-branch commit of megagonlabs/ditto on 2026-10-13; data/er_magellan has not changed since 2020-09-20
DITTO=https://raw.githubusercontent.com/megagonlabs/ditto/52985564a93fb11308439516d3e17a033d43ec8f/data/er_magellan
SUITES=https://huggingface.co/datasets/jaredpalmer/kev-suites/resolve/cc4bac803e73112689ec327ffa481c519cbc7a05

fetch() {  # target file  URL  SHA-256
    if [ -s "$1" ] && [ "$(shasum -a 256 "$1" | cut -d' ' -f1)" = "$3" ]; then
        echo "already present: $1"; return 0
    fi
    mkdir -p "$(dirname "$1")"
    curl -sS --fail -L -m 600 -o "$1.part" "$2"
    if [ "$(shasum -a 256 "$1.part" | cut -d' ' -f1)" != "$3" ]; then
        echo "$1 has the wrong SHA-256 and was not kept" >&2; rm -f "$1.part"; return 1
    fi
    mv "$1.part" "$1"
    echo "downloaded: $1 ($(wc -l < "$1" | tr -d ' ') lines)"
}

D="$ROOT/experiments/entity_match/data"
fetch "$D/abt-buy/train.txt"        "$DITTO/Textual/Abt-Buy/train.txt"           c1370966c08c246e9cf844b7cd2cf580b2559e9831f1eb3cca560fd53c1bcbb2
fetch "$D/amazon-google/train.txt"  "$DITTO/Structured/Amazon-Google/train.txt"  84ef5f7747296990a3775d2c8774949e720df643106b1b447682265f41929b26
fetch "$D/walmart-amazon/train.txt" "$DITTO/Structured/Walmart-Amazon/train.txt" 092fccd11c1cc1ffe54a702af276641218ce9506e6d60fb5a29a3c67d87892c3
fetch "$D/dblp-acm/train.txt"       "$DITTO/Structured/DBLP-ACM/train.txt"       4f757d2d5dae4f7d7d118b796c683a232b4e11bf000e2012b64f8af95a981a4e

[ -d "$ROOT/open_models/kev/evals/v7/decision-v7" ] || { echo "open_models/kev is missing: run ./scripts/open_models.sh setup first" >&2; exit 2; }
fetch "$ROOT/open_models/kev/evals/v7/decision-v7/train.jsonl" "$SUITES/v7/decision-v7/train.jsonl" 7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad
