#!/usr/bin/env bash
# Repeat one- and two-card generation after the public-base build has run.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 COLI_CUDA=1 COLI_V4_SAVE_USAGE=0 USAGE_SAVE=0
unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS COLI_DENSE_IDOT COLI_SYCL_XMX PPL N_NEW
binary="$PWD/reports/b60/upstream-pr/c/qwen36"
for run in 1gpu-generation-r2 2gpu-generation-r2; do
    case "$run" in 1gpu-*) devices=0 ;; 2gpu-*) devices=0,1 ;; esac
    dir="qwen-upstream-port-$run-20261127"
    printf 'start %s\n' "$run"
    reports/b60/profile-run.sh "$dir" "$devices" "$binary" 256 4 \
        "$PWD/reports/b60/qwen-chain_greedy-128.json"
    grep -E 'warmstart|TTFT:|Speed:|Matching tokens:' "reports/b60/$dir/run.log" || true
    printf 'finish %s\n' "$run"
done
