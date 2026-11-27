#!/usr/bin/env bash
# Run sequentially on hp while holding its GPU lock. Output paths must not exist.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 COLI_CUDA=1 USAGE_SAVE=0 N_NEW=128 NOSTREAM=1
unset COLI_PLACE COLI_CUDA_CHAIN PPL Q36_NLL_TRACE
binary="$PWD/reports/b60/qwen36-nll-diagnostic"
for mode in split_f32 chain_no_xmx; do
    unset COLI_PLATFORM_DEFAULTS COLI_CUDA_EXPERT_ARENA COLI_DENSE_IDOT COLI_SYCL_XMX
    case "$mode" in
        split_f32) export COLI_PLATFORM_DEFAULTS=0 COLI_CUDA_EXPERT_ARENA=0 COLI_DENSE_IDOT=0 ;;
        chain_no_xmx) export COLI_SYCL_XMX=0 ;;
    esac
    export Q36_GEN_IDS="$PWD/reports/b60/qwen-$mode-greedy-128.json"
    printf 'start %s\n' "$mode"
    reports/b60/profile-run.sh "qwen-greedy-$mode" 0,1 "$binary" 256 4 "$PWD/reports/b60/qwen-prompt.txt"
    printf 'finish %s\n' "$mode"
done
