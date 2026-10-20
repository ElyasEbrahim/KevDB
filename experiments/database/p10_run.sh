#!/bin/sh
# Phase 10 for one model in one command: does fine-tuning Kev on public matching pairs bring it to Jev's zero-shot level on matching data it was not trained on?
# Criteria: experiments/database/protocol/p10.txt. The steps and the rules for every mishap are fixed in protocol/p10_plan.md (sections 6 and 8) and carried out here,
# so nobody needs to decide anything midway. No paid interface is called; training and serving run sandboxed without network access.
#
#   ./experiments/database/p10_run.sh --model kev-0.8b                  every arm in order (loo-abt-buy, loo-amazon-google, loo-walmart-amazon, loo-dblp-acm, all, released), then inventory and analysis
#   ./experiments/database/p10_run.sh --model kev-4b --arms loo-abt-buy,loo-amazon-google
#                                                                      only these arms (one phase of the schedule); the analysis waits until every arm of the model is settled
#   ./experiments/database/p10_run.sh --model kev-0.8b --rehearsal      rehearsal: made-up rows, checkpoints in open_models/p10_rehearsal, results in out/database/p10_rehearsal
#
# Each arm is trained (p10_train.sh) and evaluated at once (p10_eval.py arm: its checkpoint's server is started, asked, stopped), so the arms finished
# before an interruption are complete. One training run or one server uses the GPU at a time. Finished steps are never repeated: after an
# interruption run the same command again. Start it detached from any session (it outlives the session that starts it), keep the lid open and the
# power connected; the script prevents sleep itself. The log is out/database/p10/<model>/run.log, deviations go to deviations.txt next to it.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=.venv/bin/python
MODEL=""
ARMS_ALL="loo-abt-buy loo-amazon-google loo-walmart-amazon loo-dblp-acm all released"
ARMS=""
BASE=out/database/p10
RUNS=open_models/p10
PORT=8015
# Memory watchdog: a training process above this footprint is stopped and handled under rule 2 (Kev-4B needs about 26 GB, Kev-0.8B under 9 GB)
LIMIT=$((44 * 1024 * 1024 * 1024))
while [ $# -gt 0 ]; do
    case "$1" in
        --model) shift; MODEL=${1:-} ;;
        --arms) shift; ARMS=$(echo "${1:-}" | tr ',' ' ') ;;
        --rehearsal) export P10_REHEARSAL=1; BASE=out/database/p10_rehearsal; RUNS=open_models/p10_rehearsal ;;
        *) echo "unknown argument: $1" >&2; sed -n '2,12p' "$0"; exit 2 ;;
    esac
    shift
done
case "$MODEL" in
    kev-0.8b) NEED_GB=14 ;;
    kev-4b) NEED_GB=34 ;;
    *) echo "--model kev-0.8b or --model kev-4b" >&2; exit 2 ;;
esac
ARMS=${ARMS:-$ARMS_ALL}
for arm in $ARMS; do
    case " $ARMS_ALL " in *" $arm "*) ;; *) echo "unknown arm: $arm" >&2; exit 2 ;; esac
done
OUT="$BASE/$MODEL"
mkdir -p "$OUT" "$RUNS/$MODEL"
LOG="$OUT/run.log"

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" | tee -a "$LOG"; }
deviation() { say "deviation: $*"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" >> "$OUT/deviations.txt"; }

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
[ "$available" -ge "$NEED_GB" ] || fail "only ${available} GB of memory available, ${MODEL} needs ${NEED_GB}; close other programs"
"$PY" -m pytest -q > "$OUT/tests.log" 2>&1 || fail "offline tests failed (see $OUT/tests.log)"
caffeinate -i -s -w $$ &
say "starting ${MODEL}${P10_REHEARSAL:+ (rehearsal)}; arms: ${ARMS}; ${available} GB of memory available"

# ---------- Step 1: data, checked against the checksums frozen in p10_eval.py ----------
if ! ./experiments/database/p10_fetch.sh >> "$LOG" 2>&1 || ! "$PY" experiments/database/p10_data.py build >> "$LOG" 2>&1 || ! "$PY" experiments/database/p10_data.py pools >> "$LOG" 2>&1; then
    fail "the training data could not be built (see $LOG)"
fi
"$PY" experiments/database/p10_eval.py verify >> "$LOG" 2>&1 || fail "the inputs or the built training files differ from the frozen ones (see $LOG)"

# ---------- Step 2: each arm is trained, then evaluated ----------
[ -f "$OUT/training.tsv" ] || printf 'arm\tattempt\toutcome\tseconds\tpeak_bytes\tstarted\n' > "$OUT/training.tsv"

# Move a checkpoint and its log out of the way (never deleted): the training script refuses to overwrite one
set_aside() {  # arm  label
    mkdir -p "$RUNS/$MODEL/failed"
    [ -e "$RUNS/$MODEL/$1" ] && mv "$RUNS/$MODEL/$1" "$RUNS/$MODEL/failed/$1.$2"
    [ -e "$RUNS/$MODEL/$1.log" ] && mv "$RUNS/$MODEL/$1.log" "$RUNS/$MODEL/failed/$1.$2.log"
}

# One training run under the memory watchdog. Sets OUTCOME and PEAK; returns 0 if the checkpoint counts
train_once() {  # arm  attempt
    limit=$LIMIT
    if [ -n "${P10_REHEARSAL:-}" ] && [ "${P10_DRILL_WATCHDOG_ARM:-}" = "$1" ] && [ "$2" = 1 ]; then
        limit=$((1024 * 1024 * 1024))  # rehearsal drill: the watchdog fires on this arm's first attempt
    fi
    started=$(date +%s)
    ./experiments/database/p10_train.sh "$MODEL" "$1" >> "$LOG" 2>&1 &
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
    reason=$("$PY" experiments/database/p10_eval.py trained "$MODEL" "$1")
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

train_arm() {  # arm; returns 1 if the arm is missing
    arm=$1
    if [ -e "$OUT/$arm/MISSING" ]; then
        say "${arm}: missing for good ($(cat "$OUT/$arm/MISSING")), not trained again"
        return 1
    fi
    if "$PY" experiments/database/p10_eval.py trained "$MODEL" "$arm" > /dev/null 2>&1; then
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
            deviation "rule ${rule}  ${arm}: training attempt 2 ${OUTCOME} after ${seconds} s; the arm is missing"
        fi
        attempt=$((attempt + 1))
    done
    mkdir -p "$OUT/$arm"
    echo "training failed twice (rule 2): $OUTCOME" > "$OUT/$arm/MISSING"
    return 1
}

for arm in $ARMS; do
    if [ "$arm" != released ]; then
        train_arm "$arm" || continue
    fi
    say "${arm}: evaluating"
    "$PY" experiments/database/p10_eval.py arm "$MODEL" "$arm" >> "$LOG" 2>&1
    code=$?
    case "$code" in
        0) say "${arm}: evaluated" ;;
        3) say "${arm}: evaluation incomplete after two attempts (rule 4); what exists is kept" ;;
        *) say "${arm}: evaluation ended with code ${code}, see $LOG" ;;
    esac
done

# ---------- Wrap-up: inventory, then the analysis once every arm is settled ----------
"$PY" experiments/database/p10_eval.py check "$MODEL" 2>&1 | tee -a "$LOG" > "$OUT/inventory.txt"
if "$PY" experiments/database/p10_eval.py check "$MODEL" > /dev/null 2>&1; then
    if "$PY" experiments/database/p10_report.py "$MODEL" > "$OUT/report.txt" 2>> "$LOG"; then
        say "analysis done: $OUT/report.txt"
    else
        say "the analysis script failed (see $LOG); every collected result is kept: fix it and run experiments/database/p10_report.py $MODEL again"
    fi
else
    say "arms still pending (see $OUT/inventory.txt); the analysis runs once every arm is settled"
fi
say "finished this invocation"
