#!/bin/bash
# Wait until no other job is on the GPU, then run the voice-cloning test and evaluation.
# Requires 3 idle checks in a row, so a job pausing between steps is not mistaken for finished.
cd "$(dirname "$0")/.." || exit 1
LOG=~/Documents/Arch/ssh_session.log
say() { echo "[$(date '+%F %T')] [persian_voice] $*" | tee -a "$LOG"; }

say "waiting for a free GPU (used < 1000 MiB, 3 checks 60 s apart)"
idle=0
while [ "$idle" -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -lt 1000 ]; then idle=$((idle + 1)); else idle=0; fi
    [ "$idle" -lt 3 ] && sleep 60
done

say "GPU free: running smoke test"
if ! .venv/bin/python -m scripts.smoke_tts parsvoice-xtts > results/smoke_parsvoice.log 2>&1; then
    tail -5 results/smoke_parsvoice.log | tee -a "$LOG"
    say "SMOKE_FAILED"
    exit 1
fi
tail -1 results/smoke_parsvoice.log | tee -a "$LOG"

say "running cloning evaluation"
if .venv/bin/python -m scripts.evaluate --no-stt --tts > results/eval_tts.log 2>&1; then
    say "EVAL_DONE"
else
    tail -5 results/eval_tts.log | tee -a "$LOG"
    say "EVAL_FAILED"
fi
