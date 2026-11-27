#!/usr/bin/env bash
# Run sequentially on hp while holding its GPU lock. Output paths must not exist.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 USAGE_SAVE=0
binary="$PWD/reports/b60/qwen36-nll-diagnostic"
quality="$PWD/reports/b60/qwen-quality-320.json"

for mode in cpu_f32 split chain cpu_greedy chain_greedy; do
    unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS
    unset COLI_DENSE_IDOT Q36_NLL_TRACE Q36_GEN_IDS PPL N_NEW NOSTREAM
    export COLI_CUDA=1
    devices=0,1
    case "$mode" in
        cpu_f32|cpu_greedy)
            export COLI_CUDA=0 COLI_DENSE_IDOT=0
            devices=0 ;;
        split)
            export COLI_PLATFORM_DEFAULTS=0 COLI_CUDA_EXPERT_ARENA=0 ;;
    esac
    case "$mode" in
        *_greedy)
            export N_NEW=128 NOSTREAM=1
            export Q36_GEN_IDS="$PWD/reports/b60/qwen-$mode-128.json"
            input="$PWD/reports/b60/qwen-prompt.txt" ;;
        *)
            export PPL=1
            export Q36_NLL_TRACE="$PWD/reports/b60/qwen-nll-$mode-320.csv"
            input="$quality" ;;
    esac
    printf 'start %s\n' "$mode"
    reports/b60/profile-run.sh "qwen-diagnostic-$mode" "$devices" \
        "$binary" 256 4 "$input"
    printf 'finish %s\n' "$mode"
done
