# Qwen on one and two B60s with matched tracing

The earlier table compared a one-card run with device tracing to a two-card
run with host API tracing. That table cannot establish single- versus
two-card generation speed. These follow-up runs use the same production
binary (SHA-256 `1812bf06eee5c2860b0f0f6d442f1ba28839766a3060597b33d6a0c82c9d5c90`),
the same 23-token prompt and 128-token greedy reference, float32 KV, XMX on,
18 CPU threads, lookup off, and graphs off. All experts were resident. The
one- and two-card host-traced runs were ordered 1, 2, 2, 1; the two one-card
device-traced runs followed. The expert data was already in the OS page cache.

Every run used Intel GPU Tracer, VTune software CPU sampling, PCM and xpu-smi.
Two-device tracing used only host API timing because device timestamp tracing
had previously stalled. The throughput numbers include profiler overhead.

| Configuration | Warmstart | Generation | Reference |
| --- | --- | --- | --- |
| 1 B60, host API tracing | 12.0 / 12.0 s | 77.99 / 78.04 tok/s | 128 / 128 twice |
| 2 B60s, host API tracing | 12.0 / 12.0 s | 62.26 / 61.63 tok/s | 128 / 128 twice |
| 1 B60, host and device tracing | 12.0 / 12.0 s | 31.18 / 31.10 tok/s | 128 / 128 twice |

For this workload, the two-card average is about 21% below the one-card
average with matched host tracing. Enabling device tracing and its summary
reduces the measured one-card rate by about 60% relative to host tracing. Neither is an
uninstrumented performance measurement or a model-wide quality qualification.

The one-card run kept all 10,240 experts and the complete dense trunk on one
card, with all 5,120 layer steps' routed experts inside the device
chain. It reported 130 chain frames for the 128 generated tokens. The
two-card run placed 5,120 experts on each card and the complete trunk on card
0; its existing expert path is host-mediated and is not the P2P inference
pipeline. It reported 10,369 chain frames and 4.80 / 4.86 ms per decode
token for host-routed expert work. These counters are not an additive
breakdown and do not prove which operation causes the difference. Isolating
the extra dispatch, synchronization and transfers requires further profiling.

The scripts are `run-qwen-matched-trace.sh` and
`run-qwen-matched-device-trace.sh`; each run's output is under
`qwen-matched-{hosttrace,devicetrace}-*-20261127/`, with full VTune results
retained on `hp`. `qwen-matched-results-20261127.csv` extracts the engine's
measurements and correctness counts from the run logs. The host-traced runs
completed successfully, but the
post-run console summary in the original script invoked `rg`, which was not
installed on `hp`; this affected only that summary, not profiling or the
workloads. The retained script uses `grep` for future runs. All six run status
files report exit status 0, and the GPUs were idle after the runs.

Next: profile a matched single-card host-routed ablation and the two-card
communication path; test asynchronous P2P activation and result transfers,
fewer chain frames and graph replay with the same correctness references.
