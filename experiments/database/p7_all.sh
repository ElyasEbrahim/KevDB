#!/bin/sh
# The official phase 7 run in one command: the two local models one after the other (each the full four steps of experiments/database/p7_run.sh), then Jev's long-document reference,
# and the analysis only after everything is collected. Nobody needs to make decisions midway; for mishaps see section 6 of experiments/database/protocol/p7_plan.md.
#
#   ./experiments/database/p7_all.sh                 Kev-4B → Kev-0.8B → Jev's long-document reference (paid, about $0.03) → one report per model
#   ./experiments/database/p7_all.sh --rehearsal     rehearsal: the same steps with every row made up (the Jev step costs about $0.003)
#
# The capacity probe must have been run first, with its result written into CAPACITY in experiments/database/p7_data.py (experiments/database/p7_capacity.py).
# The main log is out/database/p7/all.log, each model's log out/database/p7/<model>/run.log. Parts that already have results are not redone.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=.venv/bin/python
MODELS="kev-4b kev-0.8b"
FLAG=""
BASE=out/database/p7
if [ "${1:-}" = --rehearsal ]; then
    FLAG=--rehearsal
    export P7_REHEARSAL=1
    BASE=out/database/p7_rehearsal
fi
mkdir -p "$BASE"
LOG="$BASE/all.log"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" | tee -a "$LOG"; }

caffeinate -i -s -w $$ &
say "starting${P7_REHEARSAL:+ (rehearsal)}. Models: $MODELS"

for m in $MODELS; do
    say "${m}: starting"
    # shellcheck disable=SC2086
    P7_NO_REPORT=1 ./experiments/database/p7_run.sh $FLAG --model "$m" >> "$LOG" 2>&1
    code=$?
    if [ "$code" -ne 0 ]; then
        # Only a failed pre-flight check returns non-zero: then this model has not sent a single request. As planned, fix the problem first rather than carrying on with it
        say "${m}: pre-flight check failed (code ${code}), stopping. See $BASE/$m/run.log"
        exit "$code"
    fi
    say "${m}: collection finished"
done

say "Jev long-document reference: starting"
"$PY" experiments/database/p7_jev_long.py run >> "$LOG" 2>&1 && say "Jev long-document reference: done" || say "Jev long-document reference: did not run (key, quota or network, see the log). Continuing as planned; the report leaves this column empty to fill in later"
"$PY" experiments/database/p7_jev_long.py check 2>&1 | tee -a "$LOG"

for m in $MODELS; do
    P7_MODEL="$m" "$PY" experiments/database/p7_report.py > "$BASE/$m/report.txt" 2>> "$LOG" && say "${m}: analysis done, $BASE/$m/report.txt" || say "${m}: the analysis script failed (see the log). Every collected result is kept; fix it and re-run P7_MODEL=$m experiments/database/p7_report.py"
done
say "all done"
