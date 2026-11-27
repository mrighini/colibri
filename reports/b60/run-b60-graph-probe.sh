#!/usr/bin/env bash
# Profile a synthetic command-graph probe on hp while holding the GPU lock.
set -euo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60
export ONEAPI_DEVICE_SELECTOR='level_zero:*' USAGE_SAVE=0
sha256sum reports/b60/b60-graph-probe reports/b60/b60-graph-probe.cpp > reports/b60/b60-graph-probe.sha256
for run in gpu0-r1 gpu0-r2 gpu1-r1 gpu0-host; do
    case "$run" in
        gpu0-host) ordinal=0; export COLI_B60_TRACE=host ;;
        gpu0-*) ordinal=0; export COLI_B60_TRACE=device ;;
        gpu1-*) ordinal=1; export COLI_B60_TRACE=host ;;
    esac
    printf 'start %s\n' "$run"
    reports/b60/profile-run.sh "b60-graph-$run-20261127" "$ordinal" \
        /usr/bin/timeout --signal=TERM --kill-after=10s 90s "$PWD/reports/b60/b60-graph-probe" "$ordinal"
    printf 'finish %s\n' "$run"
done
