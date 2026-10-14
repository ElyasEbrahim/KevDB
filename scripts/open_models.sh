#!/bin/sh
# Local runtime for the open, Jev-compatible models (used from phase 6 on, see experiments/database/p6_collect.py).
#
# Everything goes into open_models/ at the repository root (not committed): the uv package manager, the Python 3.13 it downloads,
# one virtual environment per model, the models' code and weights, and all caches. Nothing is installed system-wide and no shell
# configuration is changed. To remove everything: rm -rf open_models
#
#   ./scripts/open_models.sh setup            install the environments and download the weights (about 49 GB). Safe to re-run; finished files are not downloaded again.
#                                              If Kev's weights are interrupted, unfinished files restart from zero (huggingface_hub
#                                              does not resume across processes); Clef-flash is downloaded with curl and resumes where it stopped
#   ./scripts/open_models.sh setup laya       also install Laya (about 2 GB). It was behind Jev on all three datasets in phase 6 and was dropped,
#                                              so it is not installed by default; only needed to reproduce its rows
#   ./scripts/open_models.sh setup kev-0.8b   also download the Kev-0.8B weights (1.84 GB), used from phase 7 on. Shares Kev-4B's environment, so run the plain setup first
#   ./scripts/open_models.sh serve kev-4b     start a server; Ctrl-C stops it. kev-4b listens on port 8009, kev-9b on 8010, laya on 8011, clef-flash on 8012,
#                                              laya-multilingual on 8013, kev-0.8b on 8014. All listen on localhost only and run inside the system sandbox with no network access except loopback
#   ./scripts/open_models.sh size             disk usage of each part
#
# Versions: Kev's code is pinned to the release tag kev-1.0, its weights to the Hub tag v1.0, its dependencies to its own uv.lock;
# Laya is pinned to 0.4.1 with dependencies from scripts/requirements-laya.txt, and both its English and multilingual weights, which live in one repository, to the commit below;
# Clef-flash's weights and its bundled loading code are pinned to the commit below, with dependencies from scripts/requirements-clef.txt.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/open_models"
UV="$DIR/tools/bin/uv"
KEV_TAG=kev-1.0
WEIGHTS_TAG=v1.0
LAYA_VERSION=0.4.1
# main-branch commit of convaiinnovations/laya on 2026-10-09
export LAYA_REVISION=7b928d828b7b0e022f929d9bd2e44165aa270148
# main-branch commit of Cloudflare/clef-flash on 2026-10-10
CLEF_COMMIT=8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3
CLEF="$DIR/clef-flash/$CLEF_COMMIT"

# Interpreters, caches and weights all point into open_models/, never into ~/.cache, ~/.local or ~/Library/Caches
export HF_HOME="$DIR/huggingface" HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
# Plain HTTP downloads: Hugging Face's xet protocol measured about 1 MB/s on this connection, a single plain HTTP connection almost 4 MB/s
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR="$DIR/uv/cache" UV_PYTHON_INSTALL_DIR="$DIR/uv/python" UV_PYTHON_BIN_DIR="$DIR/uv/bin"
export UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1
export XDG_CACHE_HOME="$DIR/cache" XDG_DATA_HOME="$DIR/data" PIP_CACHE_DIR="$DIR/cache/pip" TORCH_HOME="$DIR/cache/torch"
export LAYA_INDUCTOR_CACHE_DIR="$DIR/cache/laya/inductor" TORCHINDUCTOR_CACHE_DIR="$DIR/cache/laya/torchinductor"

port() {
    case "$1" in
        kev-4b) echo 8009 ;;
        kev-9b) echo 8010 ;;
        laya) echo 8011 ;;
        clef-flash) echo 8012 ;;
        laya-multilingual) echo 8013 ;;
        kev-0.8b) echo 8014 ;;
        *) echo "unknown model: $1 (kev-4b, kev-9b, kev-0.8b, laya, laya-multilingual, clef-flash)" >&2; exit 2 ;;
    esac
}

# Download every file of a Hub repository at one commit into a plain folder. Can be interrupted at any time and resumes on the next run; large files are checked one by one against their SHA-256.
# Each failure starts a fresh curl: curl's own --retry would truncate what this attempt already downloaded.
fetch_one() {  # repository  commit  target folder  file name  bytes  SHA-256 (or - if none)
    [ -f "$3/$4" ] && return 0
    mkdir -p "$(dirname "$3/$4")"
    n=0
    until [ "$(stat -f %z "$3/$4.part" 2>/dev/null || echo -1)" = "$5" ] ||
          curl -fsS -L -C - --connect-timeout 30 --speed-limit 1000 --speed-time 60 -o "$3/$4.part" "https://huggingface.co/$1/resolve/$2/$4" 2>/dev/null; do
        n=$((n + 1)); [ "$n" -ge 300 ] && { echo "$4 still incomplete after 300 reconnects" >&2; return 1; }
        sleep 3
    done
    if [ "$6" != - ] && [ "$(shasum -a 256 "$3/$4.part" | cut -d' ' -f1)" != "$6" ]; then
        echo "$4 has the wrong SHA-256; deleted, the next run downloads it again" >&2; rm -f "$3/$4.part"; return 1
    fi
    mv "$3/$4.part" "$3/$4"
}

fetch() {  # repository  commit  target folder
    mkdir -p "$3"
    curl -fsS -m 60 "https://huggingface.co/api/models/$1/revision/$2?blobs=true" | "$DIR/tools/bin/python" -c '
import json, sys
for s in json.load(sys.stdin)["siblings"]:
    print(s["rfilename"], s.get("size") or 0, (s.get("lfs") or {}).get("sha256", "-"))' > "$3/.files"
    while read -r name size sha; do fetch_one "$1" "$2" "$3" "$name" "$size" "$sha" & done < "$3/.files"
    wait
    while read -r name size sha; do [ -f "$3/$name" ] || { echo "$1 is incomplete: $name is missing" >&2; return 1; }; done < "$3/.files"
}

# Download Kev's weights into open_models/huggingface: the adapter and decision head come from jaredpalmer/kev-*, the base Qwen model is the name and revision recorded in head.pt
kev_weights() {  # model names...
    (cd "$DIR/kev" && .venv/bin/python - "$WEIGHTS_TAG" "$@" <<'EOF'
import sys
from kev.checkpoint import read_meta, resolve_run
for name in sys.argv[2:]:
    run = resolve_run(f"jaredpalmer/{name}@{sys.argv[1]}")
    meta = read_meta(run)
    base = resolve_run(f"{meta.base}@{meta.base_revision or ''}")
    print(f"{name}: adapter {run.rsplit('/', 1)[-1][:12]}, base {meta.base}@{(meta.base_revision or base.rsplit('/', 1)[-1])[:12]}, temperature {meta.temperature}")
EOF
    )
}

setup() {
    mkdir -p "$DIR"
    if [ "${1:-}" = kev-0.8b ]; then
        [ -x "$DIR/kev/.venv/bin/python" ] || { echo "run setup without arguments first to install Kev's environment" >&2; exit 2; }
        echo "== Kev-0.8B (weights ${WEIGHTS_TAG})"
        kev_weights kev-0.8b
        return
    fi
    if [ ! -x "$UV" ]; then
        python3 -m venv "$DIR/tools"
        "$DIR/tools/bin/pip" install --quiet --no-cache-dir --disable-pip-version-check uv
    fi

    echo "== Kev (${KEV_TAG})"
    [ -d "$DIR/kev" ] || git clone --quiet --depth 1 --branch "$KEV_TAG" https://github.com/jaredpalmer/kev.git "$DIR/kev"
    (cd "$DIR/kev" && "$UV" sync --quiet --frozen --no-dev --extra serve)
    kev_weights kev-4b kev-9b

    if [ "${1:-}" = laya ]; then
        echo "== Laya (${LAYA_VERSION})"
        [ -x "$DIR/laya/bin/python" ] || "$UV" venv --quiet --python 3.13 "$DIR/laya"
        "$UV" pip install --quiet --python "$DIR/laya/bin/python" "laya[serve]==$LAYA_VERSION" --constraint "$ROOT/scripts/requirements-laya.txt"
        "$DIR/laya/bin/python" -c 'import laya; laya.load("convaiinnovations/laya"); print("laya: weights ready")'
        "$DIR/laya/bin/python" -c 'import laya; laya.load("convaiinnovations/laya", subfolder="multilingual"); print("laya-multilingual: weights ready")'
    fi

    echo "== Clef-flash ($(echo "$CLEF_COMMIT" | cut -c1-12))"
    [ -x "$DIR/clef/bin/python" ] || "$UV" venv --quiet --python 3.13 "$DIR/clef"
    "$UV" pip install --quiet --python "$DIR/clef/bin/python" --requirement "$ROOT/scripts/requirements-clef.txt"
    fetch Cloudflare/clef-flash "$CLEF_COMMIT" "$CLEF"
    echo "clef-flash: weights ready"
    size
}

# The models' code and dependencies are third-party and have not been reviewed line by line. So the servers run inside the macOS sandbox,
# which blocks every outgoing network connection except loopback: whatever they read on this machine cannot leave it.
# OPEN_MODELS_NO_SANDBOX=1 turns the sandbox off (for example if a future macOS no longer ships sandbox-exec).
SANDBOX='(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'

sandboxed() {
    if [ "${OPEN_MODELS_NO_SANDBOX:-}" = 1 ]; then exec "$@"; fi
    exec sandbox-exec -p "$SANDBOX" "$@"
}

serve() {
    PORT=$(port "$1")
    export HF_HUB_OFFLINE=1  # the weights are already local: serving never goes online, so it cannot silently switch to a newer version
    case "$1" in
        kev-*) cd "$DIR/kev" && sandboxed .venv/bin/python -m kev.serve --run "jaredpalmer/$1@$WEIGHTS_TAG" --host 127.0.0.1 --port "$PORT" ;;
        laya*) [ -x "$DIR/laya/bin/laya-serve" ] || { echo "Laya is not installed by default (dropped after phase 6). To use it, run first: open_models.sh setup laya" >&2; exit 2; } ;;
    esac
    case "$1" in
        laya) export LAYA_HOST=127.0.0.1 LAYA_PORT="$PORT" LAYA_MODELS=laya LAYA_PRELOAD=1
              sandboxed "$DIR/laya/bin/laya-serve" ;;
        # Same software with the multilingual weights; a separate server, so neither version can answer the other's requests
        laya-multilingual) export LAYA_HOST=127.0.0.1 LAYA_PORT="$PORT" LAYA_MODELS=multilingual LAYA_DEFAULT_MODEL=multilingual LAYA_PRELOAD=1
              sandboxed "$DIR/laya/bin/laya-serve" ;;
        # Clef-flash ships no server; scripts/clef_server.py wraps its bundled loading code in the same interface
        clef-flash) export TRANSFORMERS_OFFLINE=1
              sandboxed "$DIR/clef/bin/python" "$ROOT/scripts/clef_server.py" "$CLEF" --port "$PORT" ;;
    esac
}

size() {
    du -sh "$DIR"/tools "$DIR"/uv "$DIR"/kev "$DIR"/laya "$DIR"/clef "$DIR"/huggingface "$DIR"/clef-flash "$DIR" 2>/dev/null
}

case "${1:-}" in
    setup) setup "${2:-}" ;;
    serve) serve "${2:?usage: open_models.sh serve kev-4b|kev-9b|kev-0.8b|laya|laya-multilingual|clef-flash}" ;;
    fetch-clef) mkdir -p "$DIR" && fetch Cloudflare/clef-flash "$CLEF_COMMIT" "$CLEF" && du -sh "$CLEF" ;;
    size) size ;;
    *) sed -n '2,15p' "$0"; exit 2 ;;
esac
