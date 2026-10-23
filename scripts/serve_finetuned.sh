#!/bin/sh
# Serve a Kev checkpoint you fine-tuned yourself (a run directory written by kev.train) for KevDB, the way scripts/open_models.sh serves the
# released ones: Kev's own server and environment in open_models/kev, offline, inside the macOS sandbox with no network access except loopback.
#
#   ./scripts/serve_finetuned.sh open_models/p10/kev-4b/all          listens on 127.0.0.1:8016; Ctrl-C stops it
#   ./scripts/serve_finetuned.sh open_models/p10/kev-4b/all 8017     another port
#
# Then in Python: kevdb.finetuned.finetuned_backend("open_models/p10/kev-4b/all") (with port=... for another port). A fine-tuned run is
# served at temperature 1.0; fit its temperature on your own labels with kevdb.finetuned.calibrate (see docs/open-models.md).
# OPEN_MODELS_NO_SANDBOX=1 turns the sandbox off, as in scripts/open_models.sh.
set -eu

RUN=${1:?usage: serve_finetuned.sh <run directory> [port]}
PORT=${2:-8016}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/open_models"
[ -d "$RUN" ] || { echo "no such run directory: $RUN" >&2; exit 2; }
RUN=$(cd "$RUN" && pwd -P)  # absolute, symbolic links resolved: the server reports this path and kevdb.finetuned compares it
for f in head.pt adapter_model.safetensors training_config.json; do
    [ -s "$RUN/$f" ] || { echo "$RUN/$f is missing: kev.serve would silently serve another run" >&2; exit 2; }
done
[ -x "$DIR/kev/.venv/bin/python" ] || { echo "Kev is not installed: run ./scripts/open_models.sh setup first" >&2; exit 2; }
case "$PORT" in 8009|8010|8011|8012|8013|8014|8015) echo "port $PORT belongs to a released model or to the experiments; pick another" >&2; exit 2 ;; esac

export HF_HOME="$DIR/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$DIR/uv/cache" UV_PYTHON_INSTALL_DIR="$DIR/uv/python" UV_PYTHON_BIN_DIR="$DIR/uv/bin" UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1
export XDG_CACHE_HOME="$DIR/cache" XDG_DATA_HOME="$DIR/data" PIP_CACHE_DIR="$DIR/cache/pip" TORCH_HOME="$DIR/cache/torch"
SANDBOX='(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'

cd "$DIR/kev"
if [ "${OPEN_MODELS_NO_SANDBOX:-}" = 1 ]; then
    exec .venv/bin/python -m kev.serve --run "$RUN" --host 127.0.0.1 --port "$PORT"
fi
exec sandbox-exec -p "$SANDBOX" .venv/bin/python -m kev.serve --run "$RUN" --host 127.0.0.1 --port "$PORT"
