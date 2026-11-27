#!/usr/bin/env python3
"""Compare completed teacher-forced score files; this does not compare logits."""
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    cpu = json.load(f)
with open(sys.argv[2], encoding="utf-8") as f:
    gpu = json.load(f)
if not cpu.get("complete") or not gpu.get("complete"):
    raise SystemExit("incomplete score file")
a, b = cpu["rows"], gpu["rows"]
if len(a) != len(b) or any((x["position"], x["input"], x["target"]) !=
                           (y["position"], y["input"], y["target"]) for x, y in zip(a, b)):
    raise SystemExit("score files use different positions or tokens")
top = sum(x["top1"] == y["top1"] for x, y in zip(a, b))
same = sum(x["nll"] == y["nll"] for x, y in zip(a, b))
maximum = max((abs(x["nll"] - y["nll"]) for x, y in zip(a, b)), default=0.0)
print(f"positions={len(a)} top1_matches={top} identical_nll={same} max_abs_nll={maximum:.17g}")
print(f"cpu_continuation_nll={cpu['nll_continuation']:.17g} gpu_continuation_nll={gpu['nll_continuation']:.17g}")
