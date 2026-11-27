#!/usr/bin/env bash
# Run on hp while holding its GPU lock. Output paths must not exist.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1 PPL=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 COLI_CUDA=1 COLI_CUDA_DNREC_SPLIT=0 USAGE_SAVE=0
export Q36_NLL_TRACE="$PWD/reports/b60/qwen-nll-no_dnrec_split-320.csv"
unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS COLI_DENSE_IDOT
unset COLI_CUDA_DNREC_ROWS COLI_CUDA_DNREC_DEC COLI_CUDA_DNREC_MIN
reports/b60/profile-run.sh qwen-chain-ablate-no_dnrec_split 0,1 \
    "$PWD/reports/b60/qwen36-nll-diagnostic" 256 4 "$PWD/reports/b60/qwen-quality-320.json"
