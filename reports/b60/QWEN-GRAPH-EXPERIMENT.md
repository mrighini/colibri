# Qwen graph replay on two Arc Pro B60s

The experiment is isolated under `qwen-graph-sandbox/`; the production engine
and its defaults have not been changed. The model is the local native-int4
Qwen3.6 35B A3B on `hp`, not a PVC result. Both B60s were used, with the
trunk on device 0, host-routed experts, 18 CPU threads, float32 KV cache,
XMX enabled, and the same 23-token prompt and 128-token greedy reference as
the earlier B60 report. Each run used Intel GPU Tracer, VTune software CPU
sampling, PCM and xpu-smi. Two-card tracing is host API only, so these timings
include profiling overhead and do not contain per-kernel GPU timestamps.
The source baseline is commit `97b608f66ddc528df28c437faf34bd9b9d8b522b`;
`qwen-graph-experiment-source.sha256` records the source and script hashes.

The first sandbox records the stable, single-row shared-expert frame for each
layer, then replays it on the same in-order queue while routed experts run via
the existing host path. It is explicitly limited to this model shape and two
B60s, and is disabled for serving, explicitly enabled lookup or MTP,
peer-context mode, fault injection, and unsupported tensor formats. The tested
runs explicitly set `COLI_LOOKUP=0`; the sandbox guard does not fully exclude
default or other speculative modes and must be strengthened before use.
It invalidates cached graphs
when chain buffers are freed or graph inputs change. The graph code is an
experimental prototype; it has not received the lifetime, concurrency, error
recovery and broad-model testing required for a production default.

| Configuration | Generation rate | TTFT | Greedy reference |
| --- | --- | --- | --- |
| Graph off | 61.49 / 61.44 tok/s | 0.186 / 0.187 s | 128 / 128 on both runs |
| Shared-expert graph on | 62.64 / 62.01 tok/s | 0.184 / 0.192 s | 128 / 128 on both runs |

These two samples suggest a small gain but do not establish a reliable
general speedup. On the separate 320-token fixed continuation, both variants
reported 2.269182 nats/token and every recorded target NLL and top-1 token
was identical. The graph run captured 40 graphs, replayed them 12,720 times,
and reported no errors or invalidations. On the first 128-token run, unitrace
recorded 145,516 kernel-append calls without the graph and 120,316 with it;
the graph run also issued 5,080 immediate command-list appends.

The fresh production control scored 59.70 tok/s and generated 128 tokens at
61.69 tok/s. The prototype source, build commands, reference and profiler
logs are in the workspace and on `hp`. The reproducible source difference is
`qwen-graph-shared-experiment.patch`; it is not applied to the production
tree. Generated sandbox source copies and raw profiler output remain local;
the patches, scripts, per-token comparisons and generated-token IDs are
retained in Git. Hashes of the generated copies are comments in the source
manifest. Results on a single B60, three or four GPUs, other prompts, longer
contexts, and peer-to-peer inference are not established by these runs.

## Shared-expert plus DeltaNet-prefix graphs

A second isolated build in `qwen-graph-dn-sandbox/` also captures the
position-independent part of each single-row DeltaNet step. The final
projection remains outside the graph because its split-K scratch allocation
can move. This variant is subject to the same strict B60, model, execution-mode
and buffer-lifetime restrictions as the first. Its source difference is
`qwen-graph-dn-experiment.patch`; it has not been applied to production.

The following alternating runs use the **same second binary**, so the control
also includes the experimental dispatch checks but has graphing disabled:

| Configuration | 128-token generation | TTFT | Greedy reference |
| --- | --- | --- | --- |
| Graph off | 61.05 / 60.76 tok/s | 0.185 / 0.190 s | 128 / 128 twice |
| Shared-expert graph | 62.31 / 62.03 tok/s | 0.187 / 0.192 s | 128 / 128 twice |
| Shared and DeltaNet graphs | 60.74 / 60.92 tok/s | 0.189 / 0.188 s | 128 / 128 twice |

With both graph segments, the 320-token score was 2.269182 nats/token,
identical to the control in every recorded per-token field. It captured 70
graphs and replayed them 22,260 times, including 9,540 DeltaNet replays, with
zero errors or cache invalidations. In the first 128-token run, tracer counted
145,516 kernel appends in the control, 120,316 with shared graphs, and 105,196
with both. More captured kernels did not improve this short workload: the
combined graph variant was slower than shared-only. The chain's operation and
matmul counters exclude operations inside graph replays and must not be
compared as total executed GPU work.

For a separate 512-token greedy continuation, all three variants generated
exactly the same token IDs. The graph-off, shared-only, and combined rates
were 64.54, 67.07, and 65.42 tok/s respectively in their first profiled runs.
A reverse-order repeat measured 65.25 tok/s graph-off and 66.98 tok/s
shared-only, again with all 512 tokens matching. These two runs of one prompt
are not an external accuracy reference or a model-wide performance claim.

With one B60, requesting the experimental graph paths correctly fell back to
normal dispatch: no graphs were captured and all 16 reference tokens matched.
The graph-off and requested runs were 25.00 and 24.83 tok/s respectively with
device tracing. This does not test graph acceleration for a one-card MoE path.

## Tightened shared-graph guard

`qwen-graph-safe-sandbox/` is a third experimental build using the first
sandbox's graph backend and a stricter engine guard. It requires explicit
`COLI_LOOKUP=0`, no speculative snapshots, no DFlash, no active MTP, no
serving or embedded-engine build, and no shared peer context. The revised
source difference is `qwen-graph-safe-experiment.patch`; it is preserved as an
experimental patch and is not applied to the production backend.

In fresh same-binary runs the control and guarded graph generated 128 reference
tokens at 61.70 and 62.63 tok/s, respectively. Their fixed-history 320-token
scores were 2.269182 nats/token in both cases, and every recorded per-token
field was identical; the graph run captured 40 graphs and replayed them 12,720
times without errors. A run with graphing requested but the default lookup
setting matched the 16-token reference and captured no graphs, as intended.
The same was true with the experimental two-device shared peer context enabled:
the context opened successfully, the 16 reference tokens matched, and no
graphs were captured.
These checks do not cover serving, multiple concurrent requests, cache
reallocation, device-loss recovery, three or four B60s, or broad model
quality. They are prerequisites for making graphing a production default.
