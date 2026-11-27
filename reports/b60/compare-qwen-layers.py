#!/usr/bin/env python3
"""Compare Qwen's last-prompt-token DUMP_LAYERS float32 tensors."""
import argparse
import math
import struct
from pathlib import Path


def load(path, layers, width):
    data = Path(path).read_bytes()
    n = layers * 3 * width
    if len(data) != n * 4:
        raise ValueError(f"{path}: expected {n * 4} bytes, got {len(data)}")
    values = struct.unpack(f"<{n}f", data)
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"{path}: non-finite value")
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--width", type=int, default=2048)
    args = ap.parse_args()
    a = load(args.reference, args.layers, args.width)
    b = load(args.candidate, args.layers, args.width)
    print("layer,sublayer,relative_l2,max_abs")
    for layer in range(args.layers):
        for slot, name in enumerate(("branch", "post_branch", "post_moe")):
            begin = (layer * 3 + slot) * args.width
            aa = a[begin:begin + args.width]
            bb = b[begin:begin + args.width]
            scale = sum(x * x for x in aa)
            delta = sum((x - y) ** 2 for x, y in zip(aa, bb))
            rel = math.sqrt(delta / scale) if scale else math.sqrt(delta)
            maximum = max(abs(x - y) for x, y in zip(aa, bb))
            print(f"{layer},{name},{rel:.9g},{maximum:.9g}")


if __name__ == "__main__":
    main()
