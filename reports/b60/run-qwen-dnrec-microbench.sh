#!/usr/bin/env bash
# Synthetic recurrence benchmark on hp while holding its GPU lock.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export COLI_B60_TRACE=device ONEAPI_DEVICE_SELECTOR='level_zero:*'
export OMP_NUM_THREADS=18 OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE
export COLI_CUDA_CHAIN_BENCH=dnrec COLI_CUDA_BENCH_N=30
export COLI_CUDA_BENCH_DNREC='32 16 128 128 30 17'
export COLI_SYCL_PEER_CONTEXT=0 USAGE_SAVE=0
binary="$PWD/reports/b60/qwen-chain-ops-test"
for mode in split rows scalar; do
    unset COLI_CUDA_DNREC_SPLIT COLI_CUDA_DNREC_ROWS COLI_CUDA_DNREC_DEC COLI_CUDA_DNREC_MIN
    case "$mode" in
        rows) export COLI_CUDA_DNREC_SPLIT=0 ;;
        scalar) export COLI_CUDA_DNREC_ROWS=0 ;;
    esac
    printf 'start %s\n' "$mode"
    reports/b60/profile-run.sh "qwen-dnrec-b60-$mode" 0 "$binary"
    logfile="$PWD/reports/b60/qwen-dnrec-b60-$mode/run.log"
    # The existing benchmark reports correctness in its text, not its exit code.
    if ! grep -q 'bench dnrec VH 32 KH 16 KD 128 VD 128, 17 rows' "$logfile" || grep -q '\[FAIL\|frame failed' "$logfile"; then
        printf 'The %s recurrence comparison did not pass.\n' "$mode" >&2
        exit 1
    fi
    printf 'finish %s\n' "$mode"
done
