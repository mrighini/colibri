#!/usr/bin/env bash
# Run sequentially on hp while holding its GPU lock. Output paths must not exist.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1 PPL=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 COLI_CUDA=1 USAGE_SAVE=0
unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS COLI_DENSE_IDOT
unset COLI_CUDA_ADDNORM COLI_CUDA_CHAIN_ROUTER_GATE COLI_CUDA_ATTN_DEC COLI_CUDA_DNREC_DEC COLI_CUDA_DNREC_ROWS
binary="$PWD/reports/b60/qwen36-nll-diagnostic"
quality="$PWD/reports/b60/qwen-quality-first.json"

for mode in cpu_f32 split chain; do
    unset COLI_PLATFORM_DEFAULTS COLI_CUDA_EXPERT_ARENA COLI_DENSE_IDOT
    export COLI_CUDA=1
    devices=0,1
    case "$mode" in
        cpu_f32) export COLI_CUDA=0 COLI_DENSE_IDOT=0; devices=0 ;;
        split) export COLI_PLATFORM_DEFAULTS=0 COLI_CUDA_EXPERT_ARENA=0 ;;
    esac
    export DUMP_LAYERS="$PWD/reports/b60/qwen-prefill-$mode.f32"
    export Q36_NLL_TRACE="$PWD/reports/b60/qwen-prefill-$mode.csv"
    if [[ -e "$DUMP_LAYERS" || -e "$Q36_NLL_TRACE" ]]; then
        printf 'Diagnostic output already exists for %s\n' "$mode" >&2
        exit 1
    fi
    printf 'start %s\n' "$mode"
    reports/b60/profile-run.sh "qwen-prefill-$mode" "$devices" "$binary" 256 4 "$quality"
    printf 'finish %s\n' "$mode"
done
