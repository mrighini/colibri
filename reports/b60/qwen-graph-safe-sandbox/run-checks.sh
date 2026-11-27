#!/usr/bin/env bash
# Profile the hardened graph guard on hp while holding the GPU lock.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=host ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export Q36_MAXT=2048 Q36_KV=f32 COLI_TIMERS=1
export SNAP=/home/mrighini/models/qwen36-35b-a3b
export TOK=/home/mrighini/models/qwen36-35b-a3b/tokenizer.json
export COLI_SYCL_PEER_CONTEXT=0 COLI_CUDA=1 USAGE_SAVE=0
unset COLI_PLACE COLI_CUDA_CHAIN COLI_CUDA_EXPERT_ARENA COLI_PLATFORM_DEFAULTS COLI_DENSE_IDOT COLI_SYCL_XMX
binary="$PWD/reports/b60/qwen-graph-safe-sandbox/qwen36-graph-safe"
sha256sum "$binary" > reports/b60/qwen-graph-safe-sandbox/binary.sha256
for run in control-gen graph-gen control-score graph-score default-lookup; do
    unset Q36_SYCL_SHARED_GRAPH Q36_NLL_TRACE PPL COLI_LOOKUP
    export COLI_LOOKUP=0
    case "$run" in graph-*|default-lookup) export Q36_SYCL_SHARED_GRAPH=1 ;; esac
    case "$run" in
        *-score) export PPL=1 Q36_NLL_TRACE="$PWD/reports/b60/qwen-graph-safe-sandbox/nll-$run.csv"
                 ref="$PWD/reports/b60/qwen-quality-320.json" ;;
        default-lookup) unset COLI_LOOKUP; ref="$PWD/reports/b60/qwen-graph-safe-sandbox/reference-16.json" ;;
        *) ref="$PWD/reports/b60/qwen-chain_greedy-128.json" ;;
    esac
    dir="qwen-graph-safe-$run-20261127"
    printf 'start %s\n' "$run"
    reports/b60/profile-run.sh "$dir" 0,1 /usr/bin/timeout --signal=TERM --kill-after=10s 180s "$binary" 256 4 "$ref"
    case "$run" in
        *-score) grep -q 'TF-NLL:' "reports/b60/$dir/run.log" ;;
        default-lookup) grep -q 'Matching tokens: 16/16' "reports/b60/$dir/run.log"
                        if grep -q '\[SYCL graph experiment\] captures=' "reports/b60/$dir/run.log"; then
                            printf 'Graph unexpectedly used with default lookup.\n' >&2; exit 1
                        fi ;;
        *) grep -q 'Matching tokens: 128/128' "reports/b60/$dir/run.log" ;;
    esac
    case "$run" in graph-*)
        grep -Eq '\[SYCL graph experiment\] captures=[1-9][0-9]* replays=[1-9][0-9]* invalidations=[0-9]+ errors=0' "reports/b60/$dir/run.log" ;;
    esac
    printf 'finish %s\n' "$run"
done
