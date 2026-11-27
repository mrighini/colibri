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
binary="$PWD/reports/b60/qwen36-nll-diagnostic"
quality="$PWD/reports/b60/qwen-quality-320.json"

for mode in no_addnorm no_router_gate no_attn_dec no_dnrec_dec no_dnrec_rows; do
    unset COLI_CUDA_ADDNORM COLI_CUDA_CHAIN_ROUTER_GATE COLI_CUDA_ATTN_DEC COLI_CUDA_DNREC_DEC COLI_CUDA_DNREC_ROWS
    case "$mode" in
        no_addnorm) export COLI_CUDA_ADDNORM=0 ;;
        no_router_gate) export COLI_CUDA_CHAIN_ROUTER_GATE=0 ;;
        no_attn_dec) export COLI_CUDA_ATTN_DEC=0 ;;
        no_dnrec_dec) export COLI_CUDA_DNREC_DEC=0 ;;
        no_dnrec_rows) export COLI_CUDA_DNREC_ROWS=0 ;;
    esac
    export Q36_NLL_TRACE="$PWD/reports/b60/qwen-nll-$mode-320.csv"
    printf 'start %s\n' "$mode"
    reports/b60/profile-run.sh "qwen-chain-ablate-$mode" 0,1 \
        "$binary" 256 4 "$quality"
    printf 'finish %s\n' "$mode"
done
