#!/bin/sh
# Phase 16, one command: every cell of experiments/database/protocol/p16.txt, then the analysis. Replays cached answers; calls no model; free.
#   ./experiments/database/p16_run.sh               formal run, into out/database/p16
#   ./experiments/database/p16_run.sh --rehearsal   the same on simulated labels, into out/database/p16_rehearsal
# Finished cells are skipped, so running it again resumes an interrupted run.
set -eu
cd "$(dirname "$0")/../.."
if [ "${1:-}" = "--rehearsal" ]; then
  export P16_REHEARSAL=1
  OUT=out/database/p16_rehearsal
else
  unset P16_REHEARSAL || true
  OUT=out/database/p16
fi
mkdir -p "$OUT"
LOG="$OUT/run.log"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(stamp) start: $OUT" >> "$LOG"
.venv/bin/python experiments/database/p16_run.py --workers "${P16_WORKERS:-14}" >> "$LOG" 2>&1
echo "$(stamp) cells done, analysis" >> "$LOG"
.venv/bin/python experiments/database/p16_report.py > "$OUT/report.console.txt" 2>> "$LOG"
echo "$(stamp) finished this invocation" >> "$LOG"
