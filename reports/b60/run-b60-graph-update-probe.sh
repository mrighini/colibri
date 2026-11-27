#!/usr/bin/env bash
# Profile graph updates on hp while holding its GPU lock.
set -uo pipefail
cd /home/mrighini/colibri-sycl/worktrees/b60 || exit 1
export ONEAPI_DEVICE_SELECTOR='level_zero:*' COLI_B60_TRACE=host USAGE_SAVE=0
sha256sum reports/b60/b60-graph-update-probe reports/b60/b60-graph-update-probe.cpp > reports/b60/b60-graph-update-probe.sha256
failed=0
for ordinal in 0 1; do
    printf 'start gpu%s\n' "$ordinal"
    if reports/b60/profile-run.sh "b60-graph-update-gpu${ordinal}-20261127" "$ordinal" \
        /usr/bin/timeout --signal=TERM --kill-after=10s 90s "$PWD/reports/b60/b60-graph-update-probe" "$ordinal"; then
        printf 'finish gpu%s status=0\n' "$ordinal"
    else
        status=$?
        printf 'finish gpu%s status=%s\n' "$ordinal" "$status"
        failed=1
    fi
done
exit "$failed"
