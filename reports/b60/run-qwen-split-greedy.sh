#!/usr/bin/env bash
# Run on hp while holding its GPU lock. Output paths must not exist.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 USAGE_SAVE=0 COLI_CUDA=1
export COLI_PLATFORM_DEFAULTS=0 COLI_CUDA_EXPERT_ARENA=0
export N_NEW=128 NOSTREAM=1
export Q36_GEN_IDS="$PWD/reports/b60/qwen-split_greedy-128.json"
unset COLI_PLACE COLI_CUDA_CHAIN COLI_DENSE_IDOT PPL Q36_NLL_TRACE
reports/b60/profile-run.sh qwen-diagnostic-split_greedy 0,1 \
    "$PWD/reports/b60/qwen36-nll-diagnostic" 256 4 "$PWD/reports/b60/qwen-prompt.txt"
