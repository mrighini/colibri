# Arc Pro B60 SYCL command-graph probe

Qwen inference does **not** use command graphs in this checkout. These
experiments are standalone B60 probes, not Qwen benchmarks or a production
graph implementation. The tests ran on `mrighini@hp`, whose two B60s were
tested one at a time, after checking the GPU device nodes and acquiring the
shared lock. No result here is from PVC.

`b60-graph-probe.cpp` enqueues 64 identical 1,024-element integer increment
kernels per frame, 40 times, on an in-order queue, waiting after each frame.
It compares ordinary submissions with a captured graph, verifies all output
elements, and reports capture/finalization time separately. The binary was
built with oneAPI 2026.2 `icpx -O2 -std=c++17 -fsycl`; it is a synthetic
dispatch test, not a measurement of Qwen kernels, mixed workloads, multi-GPU
execution, or token throughput.

| Device and tracing | Ordinary frame | Graph frame | Graph setup | Correctness |
| --- | ---: | ---: | ---: | --- |
| B60 `0000:2f:00.0`, host API | 268.987 us | 98.415 us | 1,557.919 us | pass |
| B60 `0000:34:00.0`, host API | 258.361 us | 98.508 us | 1,516.991 us | pass |
| `0000:2f:00.0`, device trace, run 1 | 1,550.022 us | 467.106 us | 2,831.979 us | pass |
| `0000:2f:00.0`, device trace, run 2 | 2,231.233 us | 470.731 us | 3,614.257 us | pass |

Both cards reported the full and limited command-graph aspects. The timing
includes tracer and other profiler overhead; compare only like trace modes.
The two device-traced ordinary-submission results varied substantially, and
the table is not evidence of a model-level speedup.

`b60-graph-update-probe.cpp` also tested changing 32 captured scalar argument
sets through whole-graph update and, separately, changing values in a stable
device parameter buffer before replay. Both produced all expected values on
each B60. With host API tracing, whole-graph record/update/replay averaged
17.865 and 17.920 us per iteration on devices 0 and 1; parameter copy/replay
averaged 15.618 and 16.025 us. These single-kernel tests do not establish the
cost or safety of updating a complete Qwen frame.

Every run used Intel GPU Tracer (`unitrace`), VTune software CPU sampling, PCM
and xpu-smi through `profile-run.sh`; each probe also had a 90-second inner
timeout. Two-GPU timestamp collection was not used. The source files, scripts,
preflight records, profiler summaries and run logs are under `reports/b60/`;
the full VTune data is on `hp` in each run's `vtune/` directory. Hashes of the
probe sources and their host executables are in `b60-graph-probe.sha256` and
`b60-graph-update-probe.sha256`.

The next model experiment needs separate graphs around Qwen's host routing,
correct token-position and KV/DeltaNet state updates, correct expert buffer
lifetimes, and a fallback for unsupported graph operations. Compare its
generated tokens and fixed-history scores with the existing no-graph reference
before measuring end-to-end throughput on one and two B60s.
