#!/usr/bin/env python3
"""Compare ROUTE_DEBUG output for the last row of one Qwen prompt."""
import argparse
import re
from pathlib import Path


def read(path):
    routes = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.search(r"\[route\] layer (\d+): (.*?) \| next", line)
        if not m:
            continue
        layer = int(m.group(1))
        if layer in routes:
            raise ValueError(f"duplicate route for layer {layer} in {path}")
        entries = re.findall(r"(\d+)\(([-+0-9.eE]+)\)", m.group(2))
        if not entries:
            raise ValueError(f"missing route entries for layer {layer}")
        routes[layer] = [(int(e), float(w)) for e, w in entries]
    if not routes:
        raise ValueError(f"no routes in {path}")
    return routes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    args = ap.parse_args()
    a, b = read(args.reference), read(args.candidate)
    if a.keys() != b.keys():
        raise ValueError("the layer sets differ")
    print("layer,ordered_ids_equal,sets_equal,max_weight_delta,reference_ids,candidate_ids")
    for layer in sorted(a):
        x, y = a[layer], b[layer]
        ids_a, ids_b = [v[0] for v in x], [v[0] for v in y]
        weights_a, weights_b = dict(x), dict(y)
        common = weights_a.keys() & weights_b.keys()
        dw = max((abs(weights_a[k] - weights_b[k]) for k in common), default=0.0)
        print(f"{layer},{int(ids_a == ids_b)},{int(set(ids_a) == set(ids_b))},{dw:.9g},"
              f"{' '.join(map(str, ids_a))},{' '.join(map(str, ids_b))}")


if __name__ == "__main__":
    main()
