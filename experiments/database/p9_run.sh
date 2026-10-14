#!/bin/sh
# The whole of phase 9 in one command, from start to report: does phase 8's packed form carry over to joins other than record matching (SemBench's sentiment pairs)?
# Criteria: experiments/database/protocol/p9.txt. Jev runs first (official API, paid, capped by BUDGET_USD in experiments/database/p9_join.py), then the local Kev-4B runs its arm, then inventory and analysis.
#
#   ./experiments/database/p9_run.sh                  everything
#   ./experiments/database/p9_run.sh --rehearsal      rehearsal: the same steps with every row made up, results under out/database/p9_rehearsal
#
# Finished steps are not redone; after an interruption just run it again. The log is out/database/p9/run.log.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=.venv/bin/python
BASE=out/database/p9
if [ "${1:-}" = "--rehearsal" ]; then
    export P9_REHEARSAL=1
    BASE=out/database/p9_rehearsal
fi
MODEL=kev-4b
PORT=8009
mkdir -p "$BASE"
LOG="$BASE/run.log"
rm -f "$BASE/DONE"

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" | tee -a "$LOG"; }

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
    say "pre-flight check: port ${PORT} is in use; stop the process using it and try again"
    exit 1
fi
if ! "$PY" -m pytest -q > "$BASE/tests.log" 2>&1; then
    say "pre-flight check: offline tests failed (see ${BASE}/tests.log)"
    exit 1
fi
caffeinate -i -s -w $$ &
say "starting${P9_REHEARSAL:+ (rehearsal)}"

say "jev: collecting"
"$PY" experiments/database/p9_join.py run jev >> "$LOG" 2>&1
say "jev: finished (code $?)"

# The local model: start the server, wait until it answers, stop it when done. The server uses its factory settings, so clear any settings inherited from outside first
say "${MODEL}: starting the server"
env -u KEV_PREFIX_CACHE -u KEV_PREFIX_MAX_TOKENS -u KEV_PREFIX_MIN_TOKENS -u KEV_TEMPERATURE -u KEV_DTYPE -u KEV_BACKEND -u KEV_TRUNCATE_STATES -u KEV_DATE_FACTS \
    ./scripts/open_models.sh serve "$MODEL" > "$BASE/serve.log" 2>&1 &
SERVER=$!
i=0
up=1
while [ "$i" -lt 300 ]; do
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/v1/models"; then up=0; break; fi
    kill -0 "$SERVER" 2>/dev/null || break
    sleep 2
    i=$((i + 1))
done
if [ "$up" -eq 0 ]; then
    "$PY" experiments/database/p9_join.py run "$MODEL" >> "$LOG" 2>&1 &
    JOB=$!
    # If the server exits midway the collector does not stop by itself, so it is watched
    while kill -0 "$JOB" 2>/dev/null; do
        if ! kill -0 "$SERVER" 2>/dev/null; then kill "$JOB" 2>/dev/null; fi
        sleep 2
    done
    wait "$JOB"
    say "${MODEL}: finished (code $?)"
else
    say "${MODEL}: the server did not start, see ${BASE}/serve.log"
fi
kill "$SERVER" 2>/dev/null
wait "$SERVER" 2>/dev/null

"$PY" experiments/database/p9_join.py check >> "$LOG" 2>&1
say "inventory: code $? (0 = complete)"
"$PY" experiments/database/p9_join.py report > "$BASE/report.txt" 2>> "$LOG"
say "analysis: code $?, see ${BASE}/report.txt"
touch "$BASE/DONE"
