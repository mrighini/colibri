#!/usr/bin/env bash
# Validate the public-base B60 port while holding the GPU lock on hp.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_LOOKUP=0 COLI_TIMERS=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_CUDA=1 COLI_V4_SAVE_USAGE=0 USAGE_SAVE=0
unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS COLI_DENSE_IDOT COLI_SYCL_XMX PPL N_NEW
base="$PWD/reports/b60/upstream-pr/c"
for run in peer-separate peer-shared 1gpu-generation 2gpu-generation 2gpu-score; do
    export COLI_SYCL_PEER_CONTEXT=0
    unset PPL
    case "$run" in
        peer-separate) devices=0,1; command=("$base/sycl/tests/test_sycl_peer_copy" --device 0,1) ;;
        peer-shared) devices=0,1; export COLI_SYCL_PEER_CONTEXT=1
                     command=("$base/sycl/tests/test_sycl_peer_copy" --device 0,1) ;;
        1gpu-generation) devices=0; command=("$base/qwen36" 256 4 "$PWD/reports/b60/qwen-chain_greedy-128.json") ;;
        2gpu-generation) devices=0,1; command=("$base/qwen36" 256 4 "$PWD/reports/b60/qwen-chain_greedy-128.json") ;;
        2gpu-score) devices=0,1; export PPL=1
                    command=("$base/qwen36" 256 4 "$PWD/reports/b60/qwen-quality-320.json") ;;
    esac
    dir="qwen-upstream-port-$run-20261127"
    printf 'start %s\n' "$run"
    reports/b60/profile-run.sh "$dir" "$devices" "${command[@]}"
    grep -E 'PASS|warmstart|Speed:|Matching tokens:|TF-NLL:' "reports/b60/$dir/run.log" || true
    printf 'finish %s\n' "$run"
done
