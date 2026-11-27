#!/usr/bin/env bash
# Run sequentially on hp while holding its GPU lock. Output paths must not exist.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1 PPL=1 ROUTE_DEBUG=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 COLI_CUDA=1 USAGE_SAVE=0
unset COLI_PLACE COLI_CUDA_CHAIN COLI_DENSE_IDOT DUMP_LAYERS
binary="$PWD/reports/b60/qwen36-nll-diagnostic"
quality="$PWD/reports/b60/qwen-quality-first.json"
for mode in split chain; do
    unset COLI_PLATFORM_DEFAULTS COLI_CUDA_EXPERT_ARENA
    if [[ "$mode" == split ]]; then
        export COLI_PLATFORM_DEFAULTS=0 COLI_CUDA_EXPERT_ARENA=0
    fi
    export Q36_NLL_TRACE="$PWD/reports/b60/qwen-routing-$mode.csv"
    printf 'start %s\n' "$mode"
    reports/b60/profile-run.sh "qwen-prefill-route-$mode" 0,1 "$binary" 256 4 "$quality"
    printf 'finish %s\n' "$mode"
done
