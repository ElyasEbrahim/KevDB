#!/bin/sh
# Phase 10 memory probe (2026-10-17, after Phase 2 was stopped by its memory guard): what makes the training process's memory grow over a long run?
# Runs kev.train with exactly the settings of p10_train.sh on a given data file, sandboxed, and reads the process's footprint every 5 s together with
# the trainer's last step line. Made-up rows only. Output: open_models/p10_probe/<model>/<name> (checkpoint, not used for anything) and
# out/database/p10_probe/<model>.<name>.tsv (seconds, step, footprint bytes). Extra environment variables after the data file are passed to the trainer.
#
#   ./experiments/database/p10_memprobe.sh kev-0.8b same out/database/p10_probe/same.jsonl 0
#   ./experiments/database/p10_memprobe.sh kev-0.8b varied out/database/p10_probe/varied.jsonl 0 SOME_VAR=1
#   PROBE_BUCKET=1 ./experiments/database/p10_memprobe.sh kev-0.8b varied-bucketed out/database/p10_probe/varied.jsonl 640
set -eu

MODEL=${1:?model}; NAME=${2:?name}; DATA=${3:?data file}; REPLAY=${4:-0}
shift 4 2>/dev/null || shift $#
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DIR="$ROOT/open_models"
case "$MODEL" in
    kev-0.8b) BASE=Qwen/Qwen3.5-0.8B-Base; BASE_REVISION=dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 ;;
    kev-4b)   BASE=Qwen/Qwen3.5-4B-Base;   BASE_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b ;;
    *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac
OUT="$DIR/p10_probe/$MODEL/$NAME"
TSV="$ROOT/out/database/p10_probe/$MODEL.$NAME.tsv"
LOG="$OUT.log"
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
mkdir -p "$DIR/p10_probe/$MODEL" "$ROOT/out/database/p10_probe"
case "$DATA" in /*) ;; *) DATA="$ROOT/$DATA" ;; esac

export HF_HOME="$DIR/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$DIR/cache" TORCH_HOME="$DIR/cache/torch" TOKENIZERS_PARALLELISM=false
SANDBOX='(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'

cd "$DIR/kev"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  probe $MODEL $NAME: $(wc -l < "$DATA" | tr -d ' ') records, replay $REPLAY${*:+, env $*}" | tee "$LOG"
# PROBE_BUCKET=1: through experiments/database/p10_kev_train.py (Kev's MPS shape bucketing in training too) instead of python -m kev.train
if [ "${PROBE_BUCKET:-}" = 1 ]; then set -- "$@" PROBE_TRAINER=bucketed; TRAINER="$ROOT/experiments/database/p10_kev_train.py kev.train"; else TRAINER="-m kev.train"; fi
# shellcheck disable=SC2086
env "$@" sandbox-exec -p "$SANDBOX" .venv/bin/python $TRAINER \
    --data "$DATA" --suite evals/v7/decision-v7 --replay "$REPLAY" \
    --base "$BASE" --base_revision "$BASE_REVISION" --init_from "jaredpalmer/$MODEL@v1.0" \
    --epochs 1 --lr 2e-5 --batch 1 --accum 8 --checkpointing 1 --weights_dtype fp32 --p_none_pair 0.25 \
    --device mps --seed 1 --out "$OUT" >> "$LOG" 2>&1 &
job=$!
printf 'seconds\tstep\tfootprint\n' > "$TSV"
started=$(date +%s)
while kill -0 "$job" 2>/dev/null; do
    bytes=$(footprint -f bytes "$job" 2>/dev/null | awk '/phys_footprint:/ {print $2; exit}')
    step=$(grep "^ep0 step" "$LOG" | tail -1 | awk '{print $3}')
    printf '%s\t%s\t%s\n' "$(($(date +%s) - started))" "${step:--}" "${bytes:-0}" >> "$TSV"
    sleep 5
done
wait "$job"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  probe end $MODEL $NAME (code $?)" | tee -a "$LOG"
