#!/bin/sh
# Phase 14 in one command: is Kev-0.8B's drop on movie reviews in phase 10 a property of seed 1 or of training without padding? Each of phase
# 13's three Kev-0.8B arms is trained again with seed 1 through the padding launcher (experiments/database/p14_train.sh) and evaluated,
# then the analysis. Criteria: experiments/database/protocol/p14.txt. Built from phase 13's p13_run.sh: the same pre-flight checks, memory
# watchdog, rules 2, 3, 4 and 7 and resume behaviour, with phase 14's scripts.
#
#   ./experiments/database/p14_run.sh                          every run in order, then inventory and analysis
#   ./experiments/database/p14_run.sh --rehearsal              rehearsal: phase 10's made-up rows, checkpoints in open_models/p14_rehearsal, results in out/database/p14_rehearsal
#
# Start it detached from any session, keep the lid open and the power connected. The log is out/database/p14/run.log.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=.venv/bin/python
MODELS="kev-0.8b"
BASE=out/database/p14
RUNS=open_models/p14
PORT=8015
NEED_GB=14
# Memory watchdog: a training process above this footprint is stopped and handled under rule 2 (Kev-4B needs about 26 GB, Kev-0.8B under 9 GB)
LIMIT=$((44 * 1024 * 1024 * 1024))
while [ $# -gt 0 ]; do
    case "$1" in
        --rehearsal) export P14_REHEARSAL=1; BASE=out/database/p14_rehearsal; RUNS=open_models/p14_rehearsal ;;
        *) echo "unknown argument: $1" >&2; sed -n '2,10p' "$0"; exit 2 ;;
    esac
    shift
done
for model in $MODELS; do
    case "$model" in kev-0.8b) ;; *) echo "phase 14 trains kev-0.8b only, not $model" >&2; exit 2 ;; esac
done
mkdir -p "$BASE"
LOG="$BASE/run.log"

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" | tee -a "$LOG"; }
deviation() { say "deviation: $*"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" >> "$OUT/deviations.txt"; }  # OUT: the model's folder, set below

# ---------- Pre-flight checks (rule 1): nothing starts unless all pass ----------
fail() { say "pre-flight check: $*"; exit 1; }
for tool in footprint sandbox-exec lsof curl caffeinate vm_stat pmset shasum pgrep; do
    command -v "$tool" > /dev/null || fail "$tool is not installed"
done
lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1 && fail "port $PORT is in use; stop the process using it and try again"
others=$(pgrep -f 'kev\.(serve|train)' | tr '\n' ' ')
[ -n "$others" ] && fail "another Kev training run or server is using the GPU (process ${others}); stop it first"
pmset -g batt | head -1 | grep -q "AC Power" || fail "not on mains power"
free_kb=$(df -k "$ROOT" | awk 'NR == 2 {print $4}')
[ "$free_kb" -gt $((10 * 1024 * 1024)) ] || fail "less than 10 GB of free disk"
available=$(vm_stat | awk '/^Pages (free|inactive|speculative|purgeable):/ {gsub("\\.", "", $NF); n += $NF} END {print int(n * 16384 / 1073741824)}')
[ "$available" -ge "$NEED_GB" ] || fail "only ${available} GB of memory available, Kev-0.8B needs ${NEED_GB}; close other programs"
"$PY" -m pytest -q > "$BASE/tests.log" 2>&1 || fail "offline tests failed (see $BASE/tests.log)"
caffeinate -i -s -w $$ &
say "starting phase 14${P14_REHEARSAL:+ (rehearsal)}; models: ${MODELS}; ${available} GB of memory available"

# ---------- Step 1: phase 10's inputs and training files, checked against phase 10's frozen checksums ----------
"$PY" experiments/database/p14_eval.py verify >> "$LOG" 2>&1 || fail "phase 10's inputs or training files differ from the frozen ones (see $LOG)"

# ---------- Step 2: each run is trained, then evaluated ----------
# Move a checkpoint and its log out of the way (never deleted): the training script refuses to overwrite one
set_aside() {  # arm  label
    mkdir -p "$RUNS/$MODEL/failed"
    [ -e "$RUNS/$MODEL/$1" ] && mv "$RUNS/$MODEL/$1" "$RUNS/$MODEL/failed/$1.$2"
    [ -e "$RUNS/$MODEL/$1.log" ] && mv "$RUNS/$MODEL/$1.log" "$RUNS/$MODEL/failed/$1.$2.log"
}

# One training run under the memory watchdog. Sets OUTCOME and PEAK; returns 0 if the checkpoint counts
train_once() {  # arm  attempt
    limit=$LIMIT
    if [ -n "${P14_REHEARSAL:-}" ] && [ "${P14_DRILL_WATCHDOG_ARM:-}" = "$1" ] && [ "$2" = 1 ]; then
        limit=$((1024 * 1024 * 1024))  # rehearsal drill: the watchdog fires on this run's first attempt
    fi
    started=$(date +%s)
    ./experiments/database/p14_train.sh "$MODEL" "$1" >> "$LOG" 2>&1 &
    job=$!
    PEAK=0
    stopped=""
    while kill -0 "$job" 2>/dev/null; do
        pid=$(pgrep -P "$job" -f 'kev\.train' | head -1)
        if [ -n "$pid" ]; then
            bytes=$(footprint -f bytes "$pid" 2>/dev/null | awk '/phys_footprint:/ {print $2; exit}')
            bytes=${bytes:-0}
            [ "$bytes" -gt "$PEAK" ] && PEAK=$bytes
            if [ "$bytes" -gt "$limit" ] && [ -z "$stopped" ]; then
                stopped=1
                say "${MODEL} ${1}: training process at $((bytes / 1048576)) MB, over the watchdog limit of $((limit / 1048576)) MB; stopping it"
                kill -TERM "$pid" 2>/dev/null; sleep 5; kill -KILL "$pid" 2>/dev/null
            fi
        fi
        sleep 5
    done
    wait "$job"
    code=$?
    seconds=$(($(date +%s) - started))
    reason=$("$PY" experiments/database/p14_eval.py trained "$MODEL" "$1")
    if [ -z "$stopped" ] && [ "$code" -eq 0 ] && [ "$reason" = complete ]; then
        OUTCOME=complete
    elif [ -n "$stopped" ]; then
        OUTCOME="stopped by the memory watchdog"
    elif grep -Eq "^dropped |non-finite" "$RUNS/$MODEL/$1.log" 2>/dev/null; then
        OUTCOME="dropped records or non-finite loss"
    else
        OUTCOME="exited with code ${code}: ${reason}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$OUTCOME" "$seconds" "$PEAK" "$(date -u -r "$started" +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/training.tsv"
    [ "$OUTCOME" = complete ]
}

train_arm() {  # arm; returns 1 if the run is missing
    arm=$1
    if [ -e "$OUT/$arm/MISSING" ]; then
        say "${arm}: missing for good ($(cat "$OUT/$arm/MISSING")), not trained again"
        return 1
    fi
    if "$PY" experiments/database/p14_eval.py trained "$MODEL" "$arm" > /dev/null 2>&1; then
        say "${arm}: already trained"
        return 0
    fi
    if [ -e "$RUNS/$MODEL/$arm" ] || [ -e "$RUNS/$MODEL/$arm.log" ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "-" "interrupted (earlier invocation)" "-" "0" "-" >> "$OUT/training.tsv"
        set_aside "$arm" "interrupted.$(date +%Y%m%dT%H%M%S)"
        deviation "rule 7  ${arm}: an unfinished training run of an earlier invocation was set aside to open_models/.../failed/; training starts again from the beginning"
    fi
    attempt=1
    while [ "$attempt" -le 2 ]; do
        say "${arm}: training, attempt ${attempt}"
        if train_once "$arm" "$attempt"; then
            say "${arm}: trained in ${seconds} s, peak footprint $((PEAK / 1048576)) MB"
            return 0
        fi
        set_aside "$arm" "attempt$attempt"
        case "$OUTCOME" in "dropped"*) rule=3 ;; *) rule=2 ;; esac
        if [ "$attempt" -eq 1 ]; then
            deviation "rule ${rule}  ${arm}: training attempt 1 ${OUTCOME} after ${seconds} s; set aside, retrained from the beginning once"
        else
            deviation "rule ${rule}  ${arm}: training attempt 2 ${OUTCOME} after ${seconds} s; the run is missing"
        fi
        attempt=$((attempt + 1))
    done
    mkdir -p "$OUT/$arm"
    echo "training failed twice (rule 2): $OUTCOME" > "$OUT/$arm/MISSING"
    return 1
}

for MODEL in $MODELS; do
    OUT="$BASE/$MODEL"
    mkdir -p "$OUT" "$RUNS/$MODEL"
    [ -f "$OUT/training.tsv" ] || printf 'arm\tattempt\toutcome\tseconds\tpeak_bytes\tstarted\n' > "$OUT/training.tsv"
    ARMS=$("$PY" experiments/database/p14_eval.py arms "$MODEL") || fail "could not list ${MODEL}'s runs"
    say "${MODEL}: runs ${ARMS}"
    for arm in $ARMS; do
        train_arm "$arm" || continue
        say "${MODEL} ${arm}: evaluating"
        "$PY" experiments/database/p14_eval.py arm "$MODEL" "$arm" >> "$LOG" 2>&1
        code=$?
        case "$code" in
            0) say "${MODEL} ${arm}: evaluated" ;;
            3) say "${MODEL} ${arm}: evaluation incomplete after two attempts (rule 4); what exists is kept" ;;
            *) say "${MODEL} ${arm}: evaluation ended with code ${code}, see $LOG" ;;
        esac
    done
done

# ---------- Wrap-up: inventory, then the analysis once every run is settled ----------
settled=1
for MODEL in kev-0.8b; do
    "$PY" experiments/database/p14_eval.py check "$MODEL" 2>&1 | tee -a "$LOG" > "$BASE/inventory.$MODEL.txt"
    "$PY" experiments/database/p14_eval.py check "$MODEL" > /dev/null 2>&1 || settled=""
done
if [ -n "$settled" ]; then
    if "$PY" experiments/database/p14_report.py > "$BASE/report.txt" 2>> "$LOG"; then
        say "analysis done: $BASE/report.txt"
    else
        say "the analysis script failed (see $LOG); every collected result is kept: fix it and run experiments/database/p14_report.py again"
    fi
else
    say "runs still pending (see $BASE/inventory.*.txt); the analysis runs once every run is settled"
fi
say "finished this invocation"
