#!/usr/bin/env bash
# Reverse the long-run order on hp while holding the GPU lock.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1 N_NEW=512 NOSTREAM=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 COLI_CUDA=1 USAGE_SAVE=0
unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS COLI_DENSE_IDOT COLI_SYCL_XMX PPL
binary="$PWD/reports/b60/qwen-graph-dn-sandbox/qwen36-graph-dn"
for mode in shared control; do
    unset Q36_SYCL_SHARED_GRAPH Q36_SYCL_DN_GRAPH
    if [[ "$mode" == shared ]]; then export Q36_SYCL_SHARED_GRAPH=1; fi
    export Q36_GEN_IDS="$PWD/reports/b60/qwen-graph-dn-sandbox/greedy-$mode-512-r2.json"
    run="qwen-graph-dn-$mode-gen512-r2-20261127"
    printf 'start %s\n' "$mode"
    reports/b60/profile-run.sh "$run" 0,1 /usr/bin/timeout --signal=TERM --kill-after=10s 240s \
        "$binary" 256 4 "$PWD/reports/b60/qwen-prompt.txt"
    test -s "$Q36_GEN_IDS"
    if [[ "$mode" == shared ]]; then
        grep -Eq '\[SYCL graph experiment\] captures=[1-9][0-9]* replays=[1-9][0-9]* invalidations=[0-9]+ errors=0' "reports/b60/$run/run.log"
    fi
    printf 'finish %s\n' "$mode"
done
