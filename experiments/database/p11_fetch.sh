#!/bin/sh
# The extra downloads of phase 11, about 8.4 MB together, each checked against its SHA-256 (recorded when first fetched on 2026-10-20, after a
# check against the git blob hash GitHub reports at the pinned commit). Safe to re-run: files already present with the right checksum are kept.
#
# The training and validation splits of four more Magellan datasets in the version curated in the Ditto repository (Apache-2.0; the same pinned
# commit as phase 10), into experiments/entity_match/data/<dataset>/: Beer, Fodors-Zagats, iTunes-Amazon and DBLP-GoogleScholar.
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DITTO=https://raw.githubusercontent.com/megagonlabs/ditto/52985564a93fb11308439516d3e17a033d43ec8f/data/er_magellan

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
fetch "$D/beer/train.txt" "$DITTO/Structured/Beer/train.txt" 88efa35036f8f8c2c60bd10abfb79a2207517faaac7c1c0a12eaca29e868a2bc
fetch "$D/beer/valid.txt" "$DITTO/Structured/Beer/valid.txt" 1c8974c31b49cc73925cca57226df2152ae5af3f715a9a26bef542a636f26cdf
fetch "$D/fodors-zagats/train.txt" "$DITTO/Structured/Fodors-Zagats/train.txt" 5bc0d4095183b325811460407086ced9ec65e634d6b4e713a87b0e823d32b28a
fetch "$D/fodors-zagats/valid.txt" "$DITTO/Structured/Fodors-Zagats/valid.txt" c9022749f6c396b75c8cd0e28e3e17710282cfa3b066c6a72172a0c3e3a3453b
fetch "$D/itunes-amazon/train.txt" "$DITTO/Structured/iTunes-Amazon/train.txt" 2a6fa5bcbf2b91af2418631db5235ac5a3ba53594509902b43a8b2abc87cea64
fetch "$D/itunes-amazon/valid.txt" "$DITTO/Structured/iTunes-Amazon/valid.txt" 5c17220ad1f6e721c3ecb5ebb3afa73022865c0618573beeb060a735a796fc42
fetch "$D/dblp-googlescholar/train.txt" "$DITTO/Structured/DBLP-GoogleScholar/train.txt" 7f3530e4dad75ee2d2f60271036e8390e18fe449ae38c9350409c9dde93cd838
fetch "$D/dblp-googlescholar/valid.txt" "$DITTO/Structured/DBLP-GoogleScholar/valid.txt" 167193bb11b635e10d8d792f8aaba4f6c5dee82c8b78164cc58c51cc59b95a65
