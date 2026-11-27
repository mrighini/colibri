#!/usr/bin/env bash
# Check the single-B60 fallback on hp while holding the GPU lock.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=device ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 COLI_CUDA=1 USAGE_SAVE=0
unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS COLI_DENSE_IDOT COLI_SYCL_XMX PPL
binary="$PWD/reports/b60/qwen-graph-dn-sandbox/qwen36-graph-dn"
reference="$PWD/reports/b60/qwen-graph-dn-sandbox/reference-16.json"
for mode in control requested; do
    unset Q36_SYCL_SHARED_GRAPH Q36_SYCL_DN_GRAPH
    if [[ "$mode" == requested ]]; then export Q36_SYCL_SHARED_GRAPH=1 Q36_SYCL_DN_GRAPH=1; fi
    run="qwen-graph-dn-single-$mode-20261127"
    printf 'start %s\n' "$mode"
    reports/b60/profile-run.sh "$run" 0 /usr/bin/timeout --signal=TERM --kill-after=10s 180s "$binary" 256 4 "$reference"
    grep -q 'Matching tokens: 16/16' "reports/b60/$run/run.log"
    if grep -q '\[SYCL graph experiment\] captures=' "reports/b60/$run/run.log"; then
        printf 'Unexpected graph capture in single-device fallback.\n' >&2
        exit 1
    fi
    printf 'finish %s\n' "$mode"
done
