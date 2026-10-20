#!/bin/sh
# One phase 10 fine-tuning run: continue training a released Kev checkpoint on one arm's training data. Criteria: experiments/database/protocol/p10.txt.
#
#   ./experiments/database/p10_train.sh kev-0.8b loo-abt-buy     the model is kev-0.8b or kev-4b; the arm is loo-<held-out dataset> or all
#   P10_REHEARSAL=1 ./experiments/database/p10_train.sh kev-0.8b loo-abt-buy     rehearsal: made-up training rows, results under open_models/p10_rehearsal
#
# Build the training data first with python experiments/database/p10_data.py build. Checkpoints go to open_models/p10/<model>/<arm>/ (not committed), the log to a .log of the same name.
# Settings follow the last fine-tuning stage of the released checkpoint: continue from --init_from, learning rate 2e-5, one epoch, effective batch size 8, replaying decision-v7 training records against forgetting.
# The training code and its dependencies are third-party, so like the servers it runs inside the system sandbox: no network, and the weights and data are already local.
# Run one training at a time and no model server alongside it: two jobs on one GPU slow each other down a lot.
set -eu

MODEL=${1:?usage: p10_train.sh kev-0.8b|kev-4b <arm>}
ARM=${2:?usage: p10_train.sh kev-0.8b|kev-4b <arm>}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DIR="$ROOT/open_models"
WEIGHTS_TAG=v1.0
SUITE=evals/v7/decision-v7

case "$MODEL" in
    kev-0.8b) BASE=Qwen/Qwen3.5-0.8B-Base; BASE_REVISION=dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 ;;
    kev-4b)   BASE=Qwen/Qwen3.5-4B-Base;   BASE_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b ;;
    *) echo "unknown model: $MODEL (kev-0.8b, kev-4b)" >&2; exit 2 ;;
esac
# Effective batch size 8 (one record per forward pass, an update every 8), the same for both models; more records per forward pass are slower on this machine (measured in the rehearsal).
# Gradient checkpointing is on for both models; it affects only speed and memory, not results: without it Kev-4B grows past 80 GB on this 64 GB machine,
# and Kev-0.8B, normally about 10 GB, spikes to 24 GB on long records; with it Kev-4B uses about 26 GB and Kev-0.8B under 9 GB.
# Weights in full precision, as when the released checkpoint was trained. Rehearsals may vary these through environment variables; official runs ignore them.
BATCH=1; ACCUM=8; CHECKPOINTING=1; WEIGHTS_DTYPE=fp32; TAG=
if [ "${P10_REHEARSAL:-}" = 1 ]; then
    BATCH=${P10_BATCH:-$BATCH}; ACCUM=${P10_ACCUM:-$ACCUM}; CHECKPOINTING=${P10_CHECKPOINTING:-$CHECKPOINTING}; WEIGHTS_DTYPE=${P10_WEIGHTS_DTYPE:-$WEIGHTS_DTYPE}; TAG=${P10_TAG:+.$P10_TAG}
fi
[ $((BATCH * ACCUM)) = 8 ] || { echo "BATCH × ACCUM must be 8" >&2; exit 2; }

if [ "${P10_REHEARSAL:-}" = 1 ]; then
    DATA="$ROOT/out/database/p10_rehearsal/train/$ARM.jsonl"; RUNS="$DIR/p10_rehearsal"; REPLAY=${P10_REPLAY:-16}
else
    DATA="$ROOT/out/database/p10/train/$ARM.jsonl"; RUNS="$DIR/p10"; REPLAY=2000
fi
OUT="$RUNS/$MODEL/$ARM$TAG"
LOG="$OUT.log"

[ -s "$DATA" ] || { echo "no training data $DATA: run python experiments/database/p10_data.py build $ARM first" >&2; exit 2; }
[ -s "$DIR/kev/$SUITE/train.jsonl" ] || { echo "the replay data $SUITE/train.jsonl is missing: run ./experiments/database/p10_fetch.sh first" >&2; exit 2; }
[ -e "$OUT" ] && { echo "$OUT already exists: the training script never overwrites a checkpoint; move it away to retrain" >&2; exit 2; }
mkdir -p "$RUNS/$MODEL"

# As in scripts/open_models.sh: interpreters, caches and weights all point into open_models/; no network, so nothing can silently switch to a newer version
export HF_HOME="$DIR/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$DIR/cache" TORCH_HOME="$DIR/cache/torch" TOKENIZERS_PARALLELISM=false
SANDBOX='(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'

cd "$DIR/kev"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  start $MODEL $ARM: $(wc -l < "$DATA" | tr -d ' ') training records, replaying $REPLAY" | tee "$LOG"
# The longest state across the four training splits is 311 tokens, within the training script's default of 384; should it drop any record as too long, this run does not count (see below)
# Through p10_kev_train.py since 2026-10-17 (rule 8, see p10.txt): kev.train unchanged, with Kev's own MPS shape bucketing applied in training too,
# which keeps the process's memory flat over a long run; without it Kev-4B passed 41 GB after 2,160 records
sandbox-exec -p "$SANDBOX" .venv/bin/python "$ROOT/experiments/database/p10_kev_train.py" kev.train \
    --data "$DATA" --suite "$SUITE" --replay "$REPLAY" \
    --base "$BASE" --base_revision "$BASE_REVISION" --init_from "jaredpalmer/$MODEL@$WEIGHTS_TAG" \
    --epochs 1 --lr 2e-5 --batch "$BATCH" --accum "$ACCUM" --checkpointing "$CHECKPOINTING" --weights_dtype "$WEIGHTS_DTYPE" --p_none_pair 0.25 \
    --device mps --seed 1 --out "$OUT" >> "$LOG" 2>&1
if grep -q "^dropped " "$LOG"; then
    echo "the training script dropped over-long records (see $LOG); this run does not count" | tee -a "$LOG" >&2; exit 1
fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  end $MODEL $ARM" | tee -a "$LOG"
