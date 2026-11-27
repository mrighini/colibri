#!/usr/bin/env bash
# Verify graphing falls back when the shared peer context is requested.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1 Q36_SYCL_SHARED_GRAPH=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=1 COLI_CUDA=1 USAGE_SAVE=0
unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS COLI_DENSE_IDOT COLI_SYCL_XMX PPL
binary="$PWD/reports/b60/qwen-graph-safe-sandbox/qwen36-graph-safe"
run=qwen-graph-safe-peer-fallback-20261127
reports/b60/profile-run.sh "$run" 0,1 /usr/bin/timeout --signal=TERM --kill-after=10s 180s \
    "$binary" 256 4 "$PWD/reports/b60/qwen-graph-safe-sandbox/reference-16.json"
grep -q 'experimental B60 peer context enabled for 2 devices' "reports/b60/$run/run.log"
grep -q 'Matching tokens: 16/16' "reports/b60/$run/run.log"
if grep -q '\[SYCL graph experiment\] captures=' "reports/b60/$run/run.log"; then
    printf 'Graph unexpectedly used with peer context.\n' >&2; exit 1
fi
