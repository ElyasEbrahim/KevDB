#!/bin/sh
# One phase 14 fine-tuning run: one of phase 10's Kev-0.8B arms trained again with phase 10's own seed 1, now through the padding launcher of
# phase 10's rule-8 fix (which Kev-0.8B's phase 10 runs predate). Criteria: experiments/database/protocol/p14.txt.
#
#   ./experiments/database/p14_train.sh kev-0.8b loo-walmart-amazon.seed1     <phase 10 arm>.seed1
#   P14_REHEARSAL=1 ./experiments/database/p14_train.sh kev-0.8b loo-walmart-amazon.seed1     rehearsal: phase 10's made-up rows, results under open_models/p14_rehearsal
#
# Everything is phase 13's p13_train.sh (that is, phase 10's p10_train.sh with the launcher), with seed 1 allowed and the checkpoints in
# open_models/p14/<model>/<arm>/ (not committed), the log in a .log of the same name.
set -eu

MODEL=${1:?usage: p14_train.sh kev-0.8b <arm>.seed1}
ARM=${2:?usage: p14_train.sh kev-0.8b <arm>.seed1}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DIR="$ROOT/open_models"
WEIGHTS_TAG=v1.0
SUITE=evals/v7/decision-v7

case "$MODEL" in
    kev-0.8b) BASE=Qwen/Qwen3.5-0.8B-Base; BASE_REVISION=dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 ;;
    kev-4b)   BASE=Qwen/Qwen3.5-4B-Base;   BASE_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b ;;
    *) echo "unknown model: $MODEL (kev-0.8b, kev-4b)" >&2; exit 2 ;;
esac
PHASE10_ARM=${ARM%.seed*}
SEED=${ARM##*.seed}
case "$PHASE10_ARM" in loo-abt-buy|loo-amazon-google|loo-walmart-amazon|loo-dblp-acm) ;; *) echo "not a phase 10 leave-one-out arm: $ARM" >&2; exit 2 ;; esac
case "$SEED" in ''|*[!0-9]*) echo "no seed in $ARM (<arm>.seed<N>)" >&2; exit 2 ;; esac
[ "$MODEL" = kev-0.8b ] && [ "$SEED" = 1 ] || { echo "phase 14 trains Kev-0.8B with seed 1 only, not $MODEL $ARM" >&2; exit 2; }
BATCH=1; ACCUM=8; CHECKPOINTING=1; WEIGHTS_DTYPE=fp32

if [ "${P14_REHEARSAL:-}" = 1 ]; then
    DATA="$ROOT/out/database/p10_rehearsal/train/$PHASE10_ARM.jsonl"; RUNS="$DIR/p14_rehearsal"; REPLAY=16
else
    DATA="$ROOT/out/database/p10/train/$PHASE10_ARM.jsonl"; RUNS="$DIR/p14"; REPLAY=2000
fi
OUT="$RUNS/$MODEL/$ARM"
LOG="$OUT.log"

[ -s "$DATA" ] || { echo "no training data $DATA: run python experiments/database/p10_data.py build $PHASE10_ARM first" >&2; exit 2; }
[ -s "$DIR/kev/$SUITE/train.jsonl" ] || { echo "the replay data $SUITE/train.jsonl is missing: run ./experiments/database/p10_fetch.sh first" >&2; exit 2; }
[ -e "$OUT" ] && { echo "$OUT already exists: the training script never overwrites a checkpoint; move it away to retrain" >&2; exit 2; }
mkdir -p "$RUNS/$MODEL"

# As in scripts/open_models.sh: interpreters, caches and weights all point into open_models/; no network, so nothing can silently switch to a newer version
export HF_HOME="$DIR/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$DIR/cache" TORCH_HOME="$DIR/cache/torch" TOKENIZERS_PARALLELISM=false
SANDBOX='(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'

cd "$DIR/kev"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  start $MODEL $ARM: $(wc -l < "$DATA" | tr -d ' ') training records, replaying $REPLAY, seed $SEED" | tee "$LOG"
# Through p10_kev_train.py: the one difference to Kev-0.8B's phase 10 run of the same arm and seed (see p14.txt)
sandbox-exec -p "$SANDBOX" .venv/bin/python "$ROOT/experiments/database/p10_kev_train.py" kev.train \
    --data "$DATA" --suite "$SUITE" --replay "$REPLAY" \
    --base "$BASE" --base_revision "$BASE_REVISION" --init_from "jaredpalmer/$MODEL@$WEIGHTS_TAG" \
    --epochs 1 --lr 2e-5 --batch "$BATCH" --accum "$ACCUM" --checkpointing "$CHECKPOINTING" --weights_dtype "$WEIGHTS_DTYPE" --p_none_pair 0.25 \
    --device mps --seed "$SEED" --out "$OUT" >> "$LOG" 2>&1
if grep -q "^dropped " "$LOG"; then
    echo "the training script dropped over-long records (see $LOG); this run does not count" | tee -a "$LOG" >&2; exit 1
fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  end $MODEL $ARM" | tee -a "$LOG"
