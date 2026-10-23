#!/bin/sh
# One phase 12 fine-tuning run: Kev-4B on n labels of one dataset, by the recipe Kev's author documents for one's own data. Criteria: experiments/database/protocol/p12.txt.
#
#   ./experiments/database/p12_train.sh kev-4b abt-buy.loo.300     <dataset>.<start>.<n>: start loo = phase 10's loo-<dataset> checkpoint, rel = the released one
#   P12_REHEARSAL=1 ./experiments/database/p12_train.sh kev-4b abt-buy.loo.20     rehearsal: made-up rows, results under open_models/p12_rehearsal
#
# The recipe (open_models/kev/README.md, "Fine-Tune on Your Own Data", By Hand): --init_from a checkpoint, 2 epochs, learning rate 2e-5,
# --batch 1 --accum 8, gradient checkpointing, nothing replayed. On this Mac as in phases 10 and 11: full-precision weights on Apple's GPU,
# sandboxed without network, through p10_kev_train.py (Kev's MPS shape padding applied in training, phase 10's rule-8 fix).
# Checkpoints go to open_models/p12/<model>/<run>/ (not committed), the log to a .log of the same name.
set -eu

MODEL=${1:?usage: p12_train.sh kev-4b <dataset>.<loo|rel>.<n>}
ARM=${2:?usage: p12_train.sh kev-4b <dataset>.<loo|rel>.<n>}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DIR="$ROOT/open_models"
WEIGHTS_TAG=v1.0

case "$MODEL" in
    kev-4b) BASE=Qwen/Qwen3.5-4B-Base; BASE_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b ;;
    *) echo "phase 12 trains kev-4b only, not $MODEL" >&2; exit 2 ;;
esac
DATASET=${ARM%%.*}
REST=${ARM#*.}
START=${REST%%.*}
N=${REST#*.}
case "$DATASET" in abt-buy|amazon-google|walmart-amazon|dblp-acm) ;; *) echo "unknown dataset in $ARM" >&2; exit 2 ;; esac
case "$N" in ''|*[!0-9]*) echo "the budget in $ARM is not a number" >&2; exit 2 ;; esac
case "$START" in
    loo) INIT="$DIR/p10/$MODEL/loo-$DATASET"  # rehearsals start from the same checkpoints: phase 10's rehearsal ones are deleted
         [ -s "$INIT/head.pt" ] && [ -s "$INIT/adapter_model.safetensors" ] || { echo "the starting checkpoint $INIT is missing" >&2; exit 2; } ;;
    rel) INIT="jaredpalmer/$MODEL@$WEIGHTS_TAG" ;;
    *) echo "unknown starting point in $ARM (loo, rel)" >&2; exit 2 ;;
esac
# Effective batch size 8 (one record per forward pass, an update every 8), as the recipe says; gradient checkpointing on (Kev-4B uses about 26 GB with it)
BATCH=1; ACCUM=8; CHECKPOINTING=1; WEIGHTS_DTYPE=fp32; EPOCHS=2; LR=2e-5

if [ "${P12_REHEARSAL:-}" = 1 ]; then
    DATA="$ROOT/out/database/p12_rehearsal/train/$DATASET.$N.jsonl"; RUNS="$DIR/p12_rehearsal"
else
    DATA="$ROOT/out/database/p12/train/$DATASET.$N.jsonl"; RUNS="$DIR/p12"
fi
OUT="$RUNS/$MODEL/$ARM"
LOG="$OUT.log"

[ -s "$DATA" ] || { echo "no training data $DATA: run python experiments/database/p12_data.py build first" >&2; exit 2; }
[ -e "$OUT" ] && { echo "$OUT already exists: the training script never overwrites a checkpoint; move it away to retrain" >&2; exit 2; }
mkdir -p "$RUNS/$MODEL"

# As in scripts/open_models.sh: interpreters, caches and weights all point into open_models/; no network, so nothing can silently switch to a newer version
export HF_HOME="$DIR/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$DIR/cache" TORCH_HOME="$DIR/cache/torch" TOKENIZERS_PARALLELISM=false
SANDBOX='(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'

cd "$DIR/kev"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  start $MODEL $ARM: $(wc -l < "$DATA" | tr -d ' ') training records, $EPOCHS epochs, from $INIT" | tee "$LOG"
# Should the trainer drop any record as too long, this run does not count (see below)
sandbox-exec -p "$SANDBOX" .venv/bin/python "$ROOT/experiments/database/p10_kev_train.py" kev.train \
    --data "$DATA" \
    --base "$BASE" --base_revision "$BASE_REVISION" --init_from "$INIT" \
    --epochs "$EPOCHS" --lr "$LR" --batch "$BATCH" --accum "$ACCUM" --checkpointing "$CHECKPOINTING" --weights_dtype "$WEIGHTS_DTYPE" \
    --device mps --seed 1 --out "$OUT" >> "$LOG" 2>&1
if grep -q "^dropped " "$LOG"; then
    echo "the training script dropped over-long records (see $LOG); this run does not count" | tee -a "$LOG" >&2; exit 1
fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  end $MODEL $ARM" | tee -a "$LOG"
