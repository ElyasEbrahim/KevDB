#!/bin/sh
# The whole of phase 6 in one command, from start to report. What each step does and how every mishap is handled is fixed here according to experiments/database/protocol/p6_plan.md,
# so nobody needs to make decisions midway.
#
#   ./experiments/database/p6_run.sh                 Jev control group → the three local models (Kev-4B, Kev-9B, Clef-flash) in turn → inventory → merge caches → analysis
#                                        the official run of 2026-10-11 also ran laya and laya-multilingual; both were behind Jev on all three datasets and were dropped from the default afterwards
#   ./experiments/database/p6_run.sh --rehearsal     rehearsal: the same steps on a few dozen made-up rows, results under out/database/p6_rehearsal, no analysis
#   ./experiments/database/p6_run.sh laya kev-4b     run only some local models (control group and wrap-up as usual). Answered requests are cached and not asked again
#
# The log is out/database/p6/run.log. Keep the lid open while it runs; the script itself prevents sleep.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=.venv/bin/python
OUT=out/database/p6
if [ "${1:-}" = --rehearsal ]; then
    export P6_REHEARSAL=1
    OUT=out/database/p6_rehearsal
    shift
fi
LOCAL="${*:-kev-4b kev-9b clef-flash}"
mkdir -p "$OUT"
LOG="$OUT/run.log"

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" | tee -a "$LOG"; }

port() {
    case "$1" in kev-4b) echo 8009 ;; kev-9b) echo 8010 ;; laya) echo 8011 ;; clef-flash) echo 8012 ;; laya-multilingual) echo 8013 ;; *) echo "unknown model: $1" >&2; exit 2 ;; esac
}

# Start a local model's server and wait until it answers. Returns 0 on success, with the server's process id in SERVER
start() {
    ./scripts/open_models.sh serve "$1" > "$OUT/serve.$1.log" 2>&1 &
    SERVER=$!
    i=0
    while [ "$i" -lt 300 ]; do
        curl -s -o /dev/null -m 2 "http://127.0.0.1:$(port "$1")/v1/models" && return 0
        kill -0 "$SERVER" 2>/dev/null || return 1
        sleep 2
        i=$((i + 1))
    done
    return 1
}

stop() {
    kill "$SERVER" 2>/dev/null
    wait "$SERVER" 2>/dev/null
}

# Run one collection command while watching the server: if the server exits midway the collector does not stop by itself (it retries every request until it times out), so it is stopped here
guarded() {
    "$@" >> "$LOG" 2>&1 &
    job=$!
    while kill -0 "$job" 2>/dev/null; do
        if ! kill -0 "$SERVER" 2>/dev/null; then
            kill "$job" 2>/dev/null
            wait "$job" 2>/dev/null
            return 9
        fi
        sleep 5
    done
    wait "$job"
}

# ---------- Pre-flight checks: nothing starts unless they pass, and once they pass the run no longer stops for environment problems ----------
for m in $LOCAL; do
    if lsof -nP -iTCP:"$(port "$m")" -sTCP:LISTEN > /dev/null 2>&1; then
        say "pre-flight check: port $(port "$m") is in use; stop the process using it and try again"
        exit 1
    fi
done
if ! "$PY" -m pytest -q > "$OUT/tests.log" 2>&1; then
    say "pre-flight check: offline tests failed (see $OUT/tests.log)"
    exit 1
fi
caffeinate -i -s -w $$ &
say "starting. Local models: $LOCAL${P6_REHEARSAL:+ (rehearsal)}"

# ---------- 1. Jev control group: runs first and also tests the collection process itself ----------
if "$PY" experiments/database/p6_collect.py run jev.fresh >> "$LOG" 2>&1; then
    "$PY" experiments/database/p6_collect.py control >> "$LOG" 2>&1
    code=$?
    tail -1 "$LOG"
    if [ "$code" -eq 3 ]; then
        say "Jev control group failed: today's Jev does not match the cached Jev, so the collection process or Jev itself has changed. Stopping here as planned; no local model has been asked"
        exit 3
    fi
    say "Jev control group: done"
else
    say "Jev control group did not run (key, quota or network, see the log). Continuing as planned: the cached Jev stays the reference, and the control group is run later"
fi

# ---------- 2. Local models one at a time (running them together would compete for the GPU and spoil the throughput numbers) ----------
for m in $LOCAL; do
    attempt=1
    while [ "$attempt" -le 2 ]; do
        say "${m}: starting the server (attempt $attempt)"
        if ! start "$m"; then
            say "${m}: the server did not start (see $OUT/serve.$m.log)"
            stop
            attempt=$((attempt + 1))
            continue
        fi
        [ -f "$OUT/probe.$m.json" ] || guarded "$PY" experiments/database/p6_collect.py probe "$m"
        guarded "$PY" experiments/database/p6_collect.py run "$m"
        code=$?
        stop
        if [ "$code" -eq 0 ]; then
            say "${m}: done"
            break
        fi
        say "${m}: collection interrupted (code $code). Answered requests are cached; the retry only asks what is missing"
        attempt=$((attempt + 1))
    done
    [ "$attempt" -le 2 ] || say "${m}: did not finish in two attempts. Keeping what was collected as planned and moving on to the next model"
done

# ---------- 3. Wrap-up: inventory → merge caches → analysis ----------
"$PY" experiments/database/p6_collect.py check $LOCAL jev.fresh 2>&1 | tee -a "$LOG"
if [ -z "${P6_REHEARSAL:-}" ]; then
    "$PY" experiments/database/p6_collect.py merge >> "$LOG" 2>&1
    "$PY" experiments/database/p6_report.py > "$OUT/report.txt" 2>> "$LOG" && say "analysis done: $OUT/report.txt" || say "the analysis script failed (see the log). Every collected judgement is kept; fix the script and re-run experiments/database/p6_report.py"
fi
say "all done"
