#!/usr/bin/env bash
# Profile the no-graph Qwen control on hp while holding the GPU lock.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 COLI_CUDA=1 USAGE_SAVE=0
unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS COLI_DENSE_IDOT COLI_SYCL_XMX PPL
printf 'start generation\n'
reports/b60/profile-run.sh qwen-graph-baseline-generation-20261127 0,1 \
    "$PWD/c/qwen36" 256 4 "$PWD/reports/b60/qwen-chain_greedy-128.json"
grep -q 'Matching tokens: 128/128' reports/b60/qwen-graph-baseline-generation-20261127/run.log
printf 'start scoring\n'
export PPL=1
reports/b60/profile-run.sh qwen-graph-baseline-score-20261127 0,1 \
    "$PWD/c/qwen36" 256 4 "$PWD/reports/b60/qwen-quality-320.json"
printf 'baseline complete\n'
