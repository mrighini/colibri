#!/usr/bin/env bash
# Profile one command on hp. The output directory must not already exist.
set -u
if [ "$#" -lt 3 ]; then
    printf 'usage: %s RUN_NAME DEVICE_LIST COMMAND [ARG...]\n' "$0" >&2
    exit 2
fi
run_name=$1
devices=$2
shift 2
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 2 ;; esac
root=/home/mrighini/colibri-sycl/worktrees/b60
out="$root/reports/b60/$run_name"
mkdir "$out" || exit 2
cd "$root" || exit 2
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:*}"
export PCM_KEEP_NMI_WATCHDOG=1 PCM_NO_MSR=1
# Model-local expert history is user data. Keep it read-only unless the caller
# deliberately opts back in; repeated comparisons then see the same history.
export COLI_V4_SAVE_USAGE="${COLI_V4_SAVE_USAGE:-0}" USAGE_SAVE="${USAGE_SAVE:-0}"
export COLI_GPUS="$devices"
case "${COLI_B60_TRACE:-device}" in
    device) trace_opts=(-d -h -s) ;;
    host) trace_opts=(-h) ;;
    *) printf 'Unknown COLI_B60_TRACE mode\n' >&2; exit 2 ;;
esac

{
    date -u '+start_utc=%FT%TZ'
    uname -a
    uptime
    free -h
    printf 'devices=%s\n' "$devices"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
    env | sort | grep -E '^(COLI_.*|CTX|CUDA_EXPERT.*|DSV4_.*|DUMP(_LAYERS)?|HEAT_FILE|N_NEW|OMP_.*|ONEAPI_DEVICE.*|PCM_.*|PPL|Q36_.*|SNAP|TF|TOK|USAGE_SAVE|V4_.*)=' || :
    sha256sum "$1"
    xpu-smi ps
} > "$out/preflight.log" 2>&1
if fuser -v /dev/dri/card1 /dev/dri/card2 /dev/dri/renderD129 /dev/dri/renderD130 > "$out/gpu-users.log" 2>&1; then
    printf 'A GPU device is already in use; the workload was not started.\n' > "$out/status.log"
    exit 1
fi

/home/mrighini/tool/prefix/bin/pcm 1 -nc -ns -csv="$out/pcm.csv" > "$out/pcm.log" 2>&1 &
pcm_pid=$!
xpu-smi dump --device 0 --metrics MEMORY,UTILIZATION,CLOCK,PCI,TEMPERATURE,POWER --interval 1 -f "$out/xpu0.csv" > "$out/xpu0.log" 2>&1 &
xpu0_pid=$!
xpu-smi dump --device 1 --metrics MEMORY,UTILIZATION,CLOCK,PCI,TEMPERATURE,POWER --interval 1 -f "$out/xpu1.csv" > "$out/xpu1.log" 2>&1 &
xpu1_pid=$!
cleanup() {
    kill -INT "$pcm_pid" 2>/dev/null || :
    kill -TERM "$xpu0_pid" "$xpu1_pid" 2>/dev/null || :
    wait "$pcm_pid" 2>/dev/null || :
    wait "$xpu0_pid" 2>/dev/null || :
    wait "$xpu1_pid" 2>/dev/null || :
}
trap cleanup EXIT
sleep 2
for pid in "$pcm_pid" "$xpu0_pid" "$xpu1_pid"; do
    if ! kill -0 "$pid" 2>/dev/null; then
        printf 'A required monitoring process exited before the workload started.\n' > "$out/status.log"
        exit 1
    fi
done

timeout --signal=TERM --kill-after=30s 1200 \
    /opt/intel/oneapi/vtune/2026.7/bin64/vtune -collect hotspots \
    -knob sampling-mode=sw -knob enable-characterization-insights=false \
    -return-app-exitcode -result-dir "$out/vtune" -- \
    /home/mrighini/tool/prefix/bin/unitrace "${trace_opts[@]}" -o "$out/unitrace" "$@" > "$out/run.log" 2>&1
status=$?
printf 'exit_status=%s\n' "$status" > "$out/status.log"
date -u '+end_utc=%FT%TZ' >> "$out/status.log"
exit "$status"
