#!/bin/sh
# The whole of phase 7 in one command, from start to report: does a local decision model inside DuckDB offer anything a cloud API cannot?
# What each step does and how every mishap is handled is fixed here according to experiments/database/protocol/p7_plan.md, so nobody needs to make decisions midway. No paid API is called at any point.
#
#   ./experiments/database/p7_run.sh                  four steps in order: sql (SemBench's 10 queries, server at factory settings) → budget (the 10 per-review queries, cache opened to the memory budget)
#                                         → cache (prefix cache: one server start per dataset, arm and pass) → calibration (calibration parameters in a table) → inventory → analysis
#   ./experiments/database/p7_run.sh --rehearsal      rehearsal: the same steps with every row made up, results under out/database/p7_rehearsal
#   ./experiments/database/p7_run.sh cache            run only some steps (inventory and analysis as usual). Parts that already have results are not redone
#   ./experiments/database/p7_run.sh --model kev-0.8b use another local model (default kev-4b); results are kept in one directory per model
#   P7_NO_REPORT=1 ./experiments/database/p7_run.sh   collect and take inventory only, no analysis (for running several models in a row: analyse once everything is collected)
#
# The log is out/database/p7/<model>/run.log. Keep the lid open while it runs; the script itself prevents sleep.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=.venv/bin/python
MODEL=kev-4b
BASE=out/database/p7
STEPS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --rehearsal) export P7_REHEARSAL=1; BASE=out/database/p7_rehearsal ;;
        --model) shift; MODEL=$1 ;;
        sql|budget|cache|calibration) STEPS="$STEPS $1" ;;
        *) echo "unknown argument: $1" >&2; sed -n '2,12p' "$0"; exit 2 ;;
    esac
    shift
done
STEPS="${STEPS:- sql budget cache calibration}"
export P7_MODEL="$MODEL"
OUT="$BASE/$MODEL"
mkdir -p "$OUT/sql"
LOG="$OUT/run.log"
# The 10 queries whose requests each contain a single review (joins in their rewritten form); the budget step runs only these
PER_REVIEW="Q1 Q2 Q3 Q4 Q5r Q6r Q7r Q8 Q9 Q10"
# Cache setting of the budget step: opened to this model's memory budget (CAPACITY in experiments/database/p7_data.py)
BUDGET=$("$PY" experiments/database/p7_cache.py env budget) || { echo "cannot read the budget setting" >&2; exit 2; }

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" | tee -a "$LOG"; }

case "$MODEL" in kev-4b) PORT=8009 ;; kev-9b) PORT=8010 ;; kev-0.8b) PORT=8014 ;; *) echo "unknown model: $MODEL" >&2; exit 2 ;; esac

# Start the server and wait until it answers. $1 holds this run's environment variables for the server (empty = factory settings). Returns 0 on success, with the process id in SERVER.
# Clear any cache settings inherited from outside first: the server's settings are decided here alone
start() {
    # shellcheck disable=SC2086
    env -u KEV_PREFIX_CACHE -u KEV_PREFIX_MAX_TOKENS -u KEV_PREFIX_MIN_TOKENS -u KEV_TEMPERATURE -u KEV_DTYPE -u KEV_BACKEND -u KEV_TRUNCATE_STATES -u KEV_DATE_FACTS $1 \
        ./scripts/open_models.sh serve "$MODEL" > "$OUT/serve.log" 2>&1 &
    SERVER=$!
    i=0
    while [ "$i" -lt 300 ]; do
        curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/v1/models" && return 0
        kill -0 "$SERVER" 2>/dev/null || return 1
        sleep 2
        i=$((i + 1))
    done
    return 1
}

stop() {
    kill "$SERVER" 2>/dev/null
    wait "$SERVER" 2>/dev/null
    # Wait until the port is really free before starting the next one
    i=0
    while lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1 && [ "$i" -lt 30 ]; do sleep 1; i=$((i + 1)); done
}

# Run one collection command while watching the server: if the server exits midway the collector does not stop by itself, so it is stopped here
guarded() {
    "$@" >> "$LOG" 2>&1 &
    job=$!
    while kill -0 "$job" 2>/dev/null; do
        if ! kill -0 "$SERVER" 2>/dev/null; then
            kill "$job" 2>/dev/null
            wait "$job" 2>/dev/null
            return 9
        fi
        sleep 2
    done
    wait "$job"
}

card() { curl -s -m 30 "http://127.0.0.1:$PORT/v1/models" | "$PY" -c 'import json, sys; json.dump(json.load(sys.stdin)["models"][0], open(sys.argv[1], "w"), indent=1)' "$1"; }

# Start the server → run one command → stop the server. On failure (server does not start, exits midway, command fails) the whole thing is retried once; returns 1 if both attempts fail.
# $1 label, $2 the server's environment variables, $3 file prefix for the server's counters (empty: not stored), the rest is the command
session() {
    label=$1; settings=$2; cards=$3; shift 3
    attempt=1
    while [ "$attempt" -le 2 ]; do
        say "${label}: starting the server (attempt $attempt)${settings:+, settings $settings}"
        if start "$settings"; then
            [ -n "$cards" ] && card "$cards.before.json"
            guarded "$@"
            code=$?
            [ -n "$cards" ] && [ "$code" -eq 0 ] && card "$cards.after.json"
            stop
            if [ "$code" -eq 0 ]; then
                say "${label}: done"
                return 0
            fi
            say "${label}: interrupted (code ${code}), see $LOG"
        else
            say "${label}: the server did not start, see $OUT/serve.log"
            stop
        fi
        attempt=$((attempt + 1))
    done
    say "${label}: did not finish in two attempts. Keeping what exists as planned and moving on to the next step"
    return 1
}

# ---------- Pre-flight checks: nothing starts unless they pass, and once they pass the run no longer stops for environment problems ----------
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
    say "pre-flight check: port $PORT is in use; stop the process using it and try again"
    exit 1
fi
for tool in footprint sandbox-exec lsof vm_stat; do
    command -v "$tool" > /dev/null || { say "pre-flight check: $tool is not installed"; exit 1; }
done
if ! "$PY" -m pytest -q > "$OUT/tests.log" 2>&1; then
    say "pre-flight check: offline tests failed (see $OUT/tests.log)"
    exit 1
fi
caffeinate -i -s -w $$ &
say "starting. Model ${MODEL}, steps: $STEPS${P7_REHEARSAL:+ (rehearsal)}"

TABLES=""
if [ -n "${P7_REHEARSAL:-}" ]; then
    TABLES="--tables $("$PY" experiments/database/p7_data.py tables)"
fi

for step in $STEPS; do
    case "$step" in
        sql)
            # shellcheck disable=SC2086
            session "sql" "" "$OUT/sql/open.default.server" "$PY" experiments/database/p1_movie.py run open --open "$MODEL" --out "$OUT/sql" --tag default --concurrency 16 --cache "$OUT/cache.sqlite" $TABLES
            ;;
        budget)
            # shellcheck disable=SC2086
            session "budget" "$BUDGET" "$OUT/sql/open.budget.server" "$PY" experiments/database/p1_movie.py run open --open "$MODEL" --out "$OUT/sql" --tag budget --concurrency 16 --cache "$OUT/cache.sqlite" --only $PER_REVIEW $TABLES
            ;;
        cache)
            "$PY" experiments/database/p7_cache.py plan > "$OUT/cache.plan"
            while read -r name arm rep; do
                [ -f "$OUT/cache/$name.$arm.r$rep.json" ] && continue
                session "cache $name $arm pass $rep" "$("$PY" experiments/database/p7_cache.py env "$name" "$arm")" "" "$PY" experiments/database/p7_cache.py run "$name" "$arm" "$rep"
            done < "$OUT/cache.plan"
            ;;
        calibration)
            session "calibration" "" "" "$PY" experiments/database/p7_calibration.py run
            if [ -n "${P7_REHEARSAL:-}" ]; then
                # Rehearsals have no phase 6 cache to replay, so the replay step really asks the made-up rows and needs a server too
                session "calibration replay" "" "" "$PY" experiments/database/p7_calibration.py replay
            else
                "$PY" experiments/database/p7_calibration.py replay >> "$LOG" 2>&1 && say "calibration replay: done (no model asked)" || say "calibration replay: failed, see $LOG"
            fi
            ;;
    esac
done

# ---------- Wrap-up: inventory → analysis ----------
"$PY" experiments/database/p7_cache.py check 2>&1 | tee -a "$LOG"
"$PY" experiments/database/p7_calibration.py check 2>&1 | tee -a "$LOG"
if [ -z "${P7_NO_REPORT:-}" ]; then
    "$PY" experiments/database/p7_report.py > "$OUT/report.txt" 2>> "$LOG" && say "analysis done: $OUT/report.txt" || say "the analysis script failed (see the log). Every collected result is kept; fix the script and re-run experiments/database/p7_report.py"
fi
say "all done"
