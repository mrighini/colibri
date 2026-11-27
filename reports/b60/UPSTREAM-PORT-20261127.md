# B60 validation of the upstream SYCL draft

These tests used the B60 port commit `b27ed949074c5bff77d5c8b20b09ff9356ada6b0`
on top of public SYCL backend commit `e31af166a46c7051e5bca27ded6950ce20492b4c`.
They ran on `hp` with two Arc Pro B60s, oneAPI 2026.2 and local Qwen3.6 35B
A3B native int4. The source hashes are in `upstream-pr-source.sha256`, relative
to a checkout of that port commit. The Qwen binary's SHA-256 was
`a87419d1725291ed501746ef1afd60be64a5e08e23c42117e30e54692ffea9d4`;
the peer test binary's was
`4fed8156cd570991232b7117e09a927f644c36da541203b7238a2bbd6f69ba0b`.

The build used `make -B -j4 qwen36 sycl/tests/test_sycl_peer_copy SYCL=1
SYCL_TARGETS=spir64 ARCH=x86-64-v4`. The host did not have `ocloc`, so this
build uses JIT device compilation. Every run used Intel GPU Tracer, VTune
software CPU sampling, PCM and xpu-smi through `profile-run.sh`; all used
host API tracing and include profiler overhead. VTune GPU hardware metrics
and PCM hardware counters were unavailable. The model data was already in the
OS page cache; this was not a cold-storage benchmark.

All Qwen runs used 18 CPU threads, float32 KV, XMX on, lookup off, graphs off,
and all 10,240 experts resident. The generation reference contains a 23-token
prompt and 128 greedy output tokens; the score reference contains 320 fixed
continuation tokens. Initial and repeat runs used the same binary.

| Run | Expert warmstart | TTFT | Generation / score | Reference |
| --- | --- | --- | --- | --- |
| 1 B60, first run | 15.2 s | 25.847 s | 4.68 tok/s, 128 tokens | 128 / 128 |
| 1 B60, repeat | 11.9 s | 0.130 s | 78.05 tok/s, 128 tokens | 128 / 128 |
| 2 B60s, first run | 12.0 s | 0.197 s | 60.81 tok/s, 128 tokens | 128 / 128 |
| 2 B60s, repeat | 12.0 s | 0.194 s | 60.53 tok/s, 128 tokens | 128 / 128 |
| 2 B60s, fixed-history score | 12.0 s | — | 2.269182 nats/token, 58.72 tok/s | 320 target tokens |

The first Qwen run followed a JIT build and incurred a much longer TTFT; a
first-use device compilation cost is a likely explanation, not independently
isolated here. The later 1-B60 result is only one warm repeat. All matching
counts are on one prompt; the aggregate fixed-history score reproduces the
earlier development snapshot but is not a model-wide accuracy guarantee.

The peer-copy test passed three open/shutdown cycles in each of the separate
and shared-context modes. It checked refusal across separate contexts,
both directions, source-write ordering, interior pointers, bounds and invalid
owners. Under host tracing its five-copy median was 19.559 / 19.051 us at
8 KiB and 684.687 / 684.991 us at 1 MiB in the two directions. The driver may
choose its transfer mechanism, and Qwen inference does not yet use this API.

`run-qwen-upstream-port.sh` and `run-qwen-upstream-warm-repeat.sh` record the
commands. Each run's profiler summaries and telemetry are under
`qwen-upstream-port-*-20261127/`; raw run logs and full VTune data remain in
the worktree and on `hp`. All seven run status files reported exit status 0.
The GPUs were idle when the lock was released. Three- and four-card runs,
long contexts, serving and broad quality qualification remain untested.
