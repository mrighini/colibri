# Arc Pro B60 measurements

These are measurements on `mrighini@hp`, not PVC results. The source checkout
was based on `bb45a348ed48673be9d4db110d320edbfa8083bc` and was modified as
described below. The host has two Arc Pro B60 devices, PCI IDs `8086:e211`,
at `0000:2f:00.0` and `0000:34:00.0`; there is no three- or four-GPU result.
Each card reported 160 EUs, subgroup sizes 16 and 32, 128 KiB of local
memory, XMX support and 24,480 MiB of device memory. xpu-smi did not report a
usable PCIe link rate or throughput, so the peer measurements below do not
infer one.
The OS page cache was not flushed, so these are not cold-start measurements.
The models were local to the host. Other GPU workloads were stopped and each
run checked the device nodes before starting.

Every benchmark and GPU operation test in this report used `profile-run.sh` to start Intel GPU Tracer
(`unitrace`), VTune software CPU sampling, PCM and xpu-smi. VTune's GPU hardware
collection was unavailable to this user, and PCM memory/energy counters and
xpu-smi PCI throughput counters were unavailable. Tracer device timing worked
on one GPU; with two GPUs it stalled on timestamp queries, so two-GPU runs use
host API tracing and lack per-kernel GPU timestamps. These figures include
profiling overhead; compare only runs with the same trace mode.
Run logs, tracer summaries, PCM/xpu-smi samples, reference data and source
hashes are copied into this directory. The full VTune result directories remain
on `hp` under the corresponding `reports/b60/<run>/vtune/` paths.
The Git snapshot retains the scripts, reference data, tracer summaries and
telemetry CSV files. Ignored run logs, build artifacts, raw activation dumps
and the copied historical checkout remain in the worktree and on `hp` rather
than in Git. The DeepSeek checkout can be reconstructed from the recorded
revision and experiment patch.
The two archived earlier harness versions differ from the current harness only
in whether `DUMP`/`DUMP_LAYERS` and `ROUTE_*` are included in the environment
record; the profiler commands and run conditions are unchanged.

## Qwen3.6 35B A3B, native int4 group size 64

Prompt: `qwen-prompt.txt` (23 input tokens), 64 new tokens, 18 CPU threads,
all 10,240 experts resident. Rates are the engine's generation rates.

| Configuration | Trace | Expert warmstart | Generation | TTFT | Runs |
| --- | --- | --- | --- | --- | --- |
| 1 B60, original placement and allocations | device | 22.2 s / 22.2 s | 28.67 / 30.03 tok/s | 0.203 / 0.162 s | `qwen-baseline-1gpu-r1/r2` |
| 1 B60, expert arena | device | 12.2 s / 12.2 s | 30.02 / 29.97 tok/s | 0.139 / 0.138 s | `qwen-arena-1gpu-r1/r2` |
| 2 B60s, original automatic placement | host API | 17.0 s / 17.1 s | 12.62 / 12.32 tok/s | 1.151 / 1.261 s | `qwen-baseline-2gpu-r2/r3` |
| 2 B60s, manual trunk co-location, no arena | host API | 16.8 s / 17.0 s | 35.27 / 37.43 tok/s | 0.669 / 0.562 s | `qwen-colocate-2gpu-r1/r2` |
| 2 B60s, B60 defaults: co-location and arena | host API | 12.1 s / 12.1 s | 55.98 / 56.11 tok/s | 0.194 / 0.189 s | `qwen-candidate-2gpu-r1/r2` |

The old two-card placement split the dense trunk and disabled the device chain.
The arena also cut single-card Level Zero device-allocation calls from 61,917
to 478 in these traced runs. The identical generated-text hash was verified on
the short baseline and candidate runs. These are results for one prompt and do
not establish peak throughput across workloads.

The longer fixed reference `qwen-quality-320.json` has a human-written,
320-token continuation. It uses float32 KV storage; the CPU comparison disables
its int8 activation approximation (`COLI_DENSE_IDOT=0`). NLL is measured in
nats per target token; no acceptance threshold was established for this text.

| Configuration | NLL | Scoring rate |
| --- | ---: | ---: |
| CPU, float activation | 2.260724 | 10.10 tok/s |
| Original 2-B60 GPU placement | 2.257804 | 14.93 tok/s |
| B60 defaults, XMX on | 2.269182 | 58.53 tok/s |
| B60 defaults, XMX off | 2.267411 | 55.35 tok/s |
| B60 defaults, XMX M=16 | 2.269182 | 58.57 tok/s |

The difference is a diagnostic, not an accuracy qualification. The 320-token
sample is too small to set a model-wide quality bound, and disabling XMX does
not eliminate it. The candidate reported use of its XMX prompt GEMM.
The M=16 result shows no meaningful improvement in this run. Forcing the split
DeltaNet recurrence on one B60 reduced the short-prompt rate from 30.25 to
28.37 tok/s, so neither experiment changed the production default.
The tested model had every expert resident, so storage-prefetching missing
experts was not a relevant explanation for this result. The tested Qwen SYCL
path did not exercise a command graph; graph capture and hardware prefetching
need separate workload-specific measurements and correctness checks.

### Separating the Qwen numerical paths

Further runs used the same 320 target tokens, float32 KV storage,
18 CPU threads and host API tracing. The first four GPU runs used the same
production binary from before the registry-only rebuild; the float-activation
split run used the diagnostic copy, and the CPU run used the original binary.
The arena alone leaves the chain off with the original split placement;
co-location enables the chain.
`COLI_DENSE_IDOT=0` selects float activations for the CPU's int8-weight
dense path; the separate int4-weight path can still quantize activations.

| Change from original two-card placement | NLL | Scoring rate | Expert warmstart |
| --- | ---: | ---: | ---: |
| Original split, arena off | 2.257804 | 14.63 tok/s | 16.8 s |
| Arena on, split retained | 2.257804 | 15.23 tok/s | 13.9 s |
| Chain enabled by co-location, arena off | 2.269182 | 52.94 tok/s | 16.9 s |
| Chain and arena enabled | 2.269182 | 58.38 tok/s | 11.9 s |
| Original split, CPU float-activation dense math | 2.267411 | 14.83 tok/s | 17.0 s |
| CPU, int8-activation dense math | 2.279692 | 11.30 tok/s | N/A |

The arena does not change the displayed aggregate score in either placement;
the chain and original execution paths do. With CPU float activations the
split-to-chain average NLL difference decreases from 0.011378 to 0.001771;
7 of 320 top-1 predictions still differ. The CPU-only path also changes score
when its dense activation approximation is enabled: its float-activation
score was 2.260724. These results do not identify a single faulty operation or
establish which implementation is the most accurate.

`qwen36-nll-diagnostic.c` is a copy of the production C entry point with
optional per-token CSV and generated-token recording; it is not a change to
the production engine. Its three fixed-continuation runs reproduce the
aggregate scores above. Compared on the same 320 teacher-forced target tokens,
the split GPU and CPU float-activation paths disagree on top-1 at 9 positions,
the chain and split GPU paths at 8, and the chain and CPU paths at 11. A top-1
disagreement is not an accuracy error by itself. The first-token NLL already
differs between the chain and split paths, before any decode step in this
sample, so differences are not confined to decode. Per-token traces and
`compare-qwen-diagnostics.py` are included here.

On the separate 23-token prompt, the original split and two-card chain greedy
outputs match for all 128 tokens. Both match the CPU float-activation output
for 81 tokens before their first divergence. The CPU generation was 7.42
tok/s, the original split 14.18 tok/s and the chain 61.34 tok/s, with host tracing.
After divergence the continuations no longer have the same history, so their
later positions are not a per-token accuracy comparison. This CPU run is a
self-reference using the checkpoint's quantized weights; it is not an
independent full-precision model oracle.

To narrow the chain difference, each of the following uses the diagnostic
binary, the same fixed continuation and host tracing. Top-1 changes compare
the same target histories with the default chain, not generated continuations.

| Chain configuration | NLL | Scoring rate | Top-1 changes vs. chain |
| --- | ---: | ---: | ---: |
| Default chain | 2.269182 | 60.17 tok/s | 0 / 320 |
| Fused add/norm off | 2.269182 | 59.92 tok/s | 0 / 320 |
| Combined router/gate projection off | 2.269182 | 59.69 tok/s | 0 / 320 |
| Decode-attention specialization off | 2.269182 | 54.75 tok/s | 0 / 320 |
| Decode DeltaNet specialization off | 2.269182 | 62.03 tok/s | 0 / 320 |
| Split prompt DeltaNet kernel off | 2.262932 | 59.99 tok/s | 6 / 320 |
| DeltaNet row kernels off for prompt and decode | 2.262936 | 61.81 tok/s | 6 / 320 |
| XMX off | 2.267411 | 56.33 tok/s | 7 / 320 |
| XMX and split prompt DeltaNet kernel off | 2.267411 | 56.32 tok/s | 7 / 320 |

Fused add/norm and router/gate ablations were identical to the default in
every recorded target NLL. Decode-attention and decode-recurrence ablations
changed mean absolute per-token NLL by about 1.36e-6 and 1.09e-6 respectively.
Disabling the row kernels also changes the first target's NLL, before decode,
so the prompt recurrence implementation contributes to the numerical
difference. Disabling only the split prompt kernel gives nearly the same
result as disabling all row kernels (zero top-1 differences, mean absolute
NLL difference 7.84e-5), narrowing the contribution to that prompt kernel.
The one sample does not show that either recurrence is a better
oracle: the row-kernel-off result still differs from the CPU in 11 top-1
predictions. Scoring rates exclude warmstart and are single profiled runs;
they are not enough to choose a performance default.

With XMX off, the chain and original split path with CPU float-activation
dense math select the same top-1 for all 320 fixed histories. Their mean
absolute per-token NLL difference is 1.58e-6, with a maximum of 1.82e-5.
When XMX is off, turning the split DeltaNet prompt kernel off changes mean
absolute per-token NLL by only 1.11e-6 and no top-1 predictions, showing an
interaction between the two changes in this sample. On the diagnostic run,
the XMX-off chain scored at 56.33 tok/s versus 60.17 tok/s for XMX on and
14.83 tok/s for the float-activation split path. This provides a selectable
comparison mode (`COLI_SYCL_XMX=0`), not an external accuracy guarantee.
On the 128-token greedy prompt, the float-activation split and XMX-off chain
match all token IDs; their generation rates were 13.94 and 52.18 tok/s and
their TTFTs 1.249 and 0.541 seconds. The XMX-on chain produced the same 128
tokens at 61.34 tok/s and 0.187 seconds TTFT. These are individual profiled
runs, not an assurance for other prompts.

The environment registry now recognizes `COLI_SYCL_XMX` and the opt-in
`COLI_SYCL_PEER_CONTEXT`, and accepts `OMP_PROC_BIND` for Qwen. Previously,
strict environment checking rejected the first two controls and Qwen's own
OpenMP tuner could cause it to reject `OMP_PROC_BIND`. The registry consistency
check and its unit test pass; an actual Qwen tokenizer startup in strict mode
now succeeds with both SYCL controls and the OpenMP tuner enabled. The rebuilt
engine was then profiled on two B60s with XMX off: it reproduced all 128
recorded tokens at 52.10 tok/s and 0.544 seconds TTFT. The old and rebuilt
binaries are retained on `hp` as `qwen36-before-registry` and
`qwen36-registry-updated`; the rebuilt one is installed as `c/qwen36` in this
worktree.

The existing chain operation test was built for B60 and run with all four
profilers on GPU 0, using a synthetic 17-row recurrence at the Qwen dimensions
(32 value heads, 16 key heads, 128 key and value dimensions, 30 states).
The split, nonsplit row and scalar variants each matched its CPU reference
within the test's 1e-4 maximum absolute output and state tolerance. Their best
of five pipelined call times, with device tracing, were 105.64, 110.10 and
109.81 microseconds respectively. These are synthetic microbenchmarks, not
end-to-end latency or a quality qualification.

For the initial 17-token prompt, `DUMP_LAYERS` records the last prompt row
at three points per layer. It disables the fused add/norm while dumping; the
separate fusion ablation above produced identical per-token NLL, and each
dump run reproduced the corresponding uninstrumented first-token NLL exactly.
Relative L2 error of chain versus split, using the split tensor's norm as the
denominator, was 0.000137 for layer 0's DeltaNet branch, 0.000084 after its
residual, and 0.007997 after its MoE residual. The latter includes both the
shared and routed experts, so it does not by itself identify which part causes
the increase. Raw float32 dumps remain local and `compare-qwen-layers.py` is
included.
`ROUTE_DEBUG` on that prompt reports the same eight experts at layer 0 but a
different ordering and weights; 13 of 40 layers select a different expert set.
With the original split path's CPU int8 activation approximation disabled
(`COLI_DENSE_IDOT=0`), all 40 expert sets match the chain for the prompt's
last row, although three layers order their selected experts differently. At
layer 0, the largest printed pre-normalization expert-weight difference from
the chain falls from 0.0038348 to 0.0000297. This setting affects other CPU
dense math too, so the result is evidence about the source of the difference,
not a proof of router arithmetic alone.

A separate staged-int4 experiment combined each expert's weight and scale
uploads into one copy. Device tests on both B60s matched the existing upload
byte-for-byte; the second card needed host-only tracing because device
timestamp queries stalled. On one card, traced memory-copy calls fell from
21,159 to 10,919, but warmstarts remained 12.1/12.0 s for the control and
12.2/12.0 s for the experiment. On two cards they were 12.1/12.1 s and
12.0/12.0 s, respectively; generation rates and output matched. The production
change was removed as an unmeasurable gain. Its reproducible patch is
`qwen-s4-onecopy-experiment.patch`.

On this host `OMP_NUM_THREADS=18 OMP_PROC_BIND=close` without `OMP_PLACES`
put the OpenMP team on nine physical cores. `OMP_PLACES=cores` used all 18.
In alternating two-card runs, the default placement warmed experts in 12.0 /
12.0 s and generated 56.13 / 55.84 tok/s; the core placement took 11.7 /
11.9 s and 55.32 / 55.88 tok/s. The output matched; this sample does not show
a reliable overall improvement, so no affinity default was changed.

The production changes make the arena and co-location defaults specific to
`8086:e211` and the measured model geometry. Co-location is limited to two
cards with complete trunk offers and enough budget for all experts on their
existing homes. Explicit settings and general fallback paths remain available;
see `c/sycl/README.md`. The oneAPI 2026.2 DPCT compatibility change was built
with that toolchain; an older compiler was not tested here.

## DeepSeek V4 Flash REAP, historical single-card port

The current checkout has no integrated DeepSeek SYCL backend. The experiment
uses the historical engine snapshot at `dsv4-sandbox/`, copied from sibling
worktree `q38-best` commit `1d93555528c1e1a7b0ddb39406e83645aec5c113`.
The patch against that snapshot is `deepseek-resident-experiment.patch`; it is
not integrated into the current engine. Its SYCL port exposes only one GPU.

The B60 allocation, tiny-matrix and historical full-operation suites passed;
the full suite reported worst absolute error `2.98023e-08`. The experimental
resident reference also matched the existing reference bit-for-bit in small
and 2048-by-4096 matrix tests, including grouped offsets and 1/5-token batches.

The following uses the local 79 GiB REAP checkpoint, 11 prompt tokens, 8
generated tokens, 18 CPU threads, device tracing, CPU-order GPU dense
arithmetic and CPU routed experts. Thus this is a hybrid correctness-oriented
configuration, not a GPU-expert or multi-GPU result.

The historical engine's default behavior wrote the model's `.coli_usage`
expert-history file during the initial runs. Its original contents were not
captured, so the updated file has been left in place. The history changed
between the following runs, which limits their comparability. Later runs set
`COLI_V4_SAVE_USAGE=0` and `USAGE_SAVE=0` and check the history hash.

| Configuration | Total generation | TTFT | After-first phase | Output / check |
| --- | ---: | ---: | ---: | --- |
| CPU reference | 27.420 s | 20.046 s | 7.372 s | CPU self-oracle |
| Historical B60 hybrid | 52.471 / 51.751 s | 32.393 / 31.753 s | 20.076 / 19.997 s | 10/10 teacher-forced positions, 8/8 greedy tokens |
| Experimental resident-weight hybrid | 40.709 / 40.718 s | 29.636 / 29.659 s | 11.071 / 11.057 s | 10/10 teacher-forced positions, 8/8 greedy tokens |

With history writes disabled and the same history hash, the matched repeat was
32.032 s / 24.783 s TTFT for CPU, 52.036 s / 32.067 s TTFT for the original
hybrid and 40.117 s / 29.136 s TTFT for the resident-weight hybrid. All three
produced the same eight-token text.

With the same read-only history and `OMP_PLACES=cores`, CPU repeats took
25.953 / 26.541 s and the resident-weight hybrid took 36.225 / 35.975 s.
Their output matched the earlier runs. This affinity result is for this host
and short prompt; the wider score comparison below used the original affinity.

The measured device host-to-device copy time fell from 12.314 s to 1.399 s
when the CPU-order kernel read the existing device mirror. The GPU path remains
slower than the CPU on this short run. The oracle was produced by the CPU
implementation, and this short match is not a full checkpoint-quality gate.
The new path remains opt-in (`DSV4_SYCL_DENSE_RESIDENT_REF=1`).

An additional read-only sequential score used a fixed human-written 64-token
continuation after an 11-token prompt. All 74 scored positions had identical
top-1 predictions and identical per-token NLL between CPU and the experimental
B60 path; the continuation NLL was `3.0072612801903484` on both. These are
score and top-1 comparisons, not a full-logit comparison or a broad model
quality qualification. The model history hash was unchanged over both runs.

## Peer access

The B60 topology is reported as PHB; there is no reported MDF fabric or peer
atomic support. Both peer access directions were available. The two-card
`p2p_probe.cpp` copies were verified byte-for-byte. `p2p_read_probe.cpp` also
verified kernels reading the other card's allocation: for 64 MiB, the
instrumented median was 1.676 and 1.664 GB/s for remote reads versus 180.017
and 184.488 GB/s for local reads. These are application-level elapsed rates,
not a measured PCIe wire rate; hardware PCI traffic counters were unavailable.

Production SYCL uses separate contexts by default. The opt-in
`COLI_SYCL_PEER_CONTEXT=1` creates a shared context for two to four mutually
accessible B60s on one Level Zero platform and enables the synchronous
`coli_cuda_pipe_peer_copy` API. It checks allocation ownership, orders source
queues and waits for the transfer. On the two-card host the separate-context
refusal and shared-context copies passed three open/shutdown cycles, both
directions, interior allocation offsets and invalid-owner tests. Under host
tracing the API's median 8 KiB latency was 19.353 / 18.980 microseconds and
the 1 MiB latency was 683.126 / 681.519 microseconds (five copies, each
direction). These timings include source-queue waits and validation.

Qwen's routed-expert path and the historical DeepSeek port do not yet use
this primitive for P2P inference. The experimental context is off by default;
the next implementation should place expert computation beside its weights,
exchange small activations and partial results, add explicit asynchronous
dependencies and lifetime tests, and retain a safe fallback.
With the same Qwen binary and short prompt, separate contexts measured
56.05 / 56.30 tok/s and the shared context 53.57 / 53.87 tok/s; their generated
text matched. The shared-context 320-token NLL was 2.269182, the same printed
value as the separate-context candidate. It is not a Qwen speedup yet.

## Remaining work

Run an independently defined, longer quality gate on both models; integrate
and qualify a DeepSeek backend against the current engine; profile a correct
GPU-expert path; implement and test activation/result P2P; and obtain suitable
hardware for three- and four-card scaling. Long contexts, serving concurrency,
and broad model coverage have not been qualified by these measurements.
