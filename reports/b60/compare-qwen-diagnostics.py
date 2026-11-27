#!/usr/bin/env python3
"""Compare per-token Qwen score traces or two recorded greedy continuations."""
import argparse
import csv
import json
import math
from pathlib import Path


def read_scores(path):
    with Path(path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    return [
        (int(r["position"]), int(r["target"]), int(r["top1"]), float(r["target_nll"]))
        for r in rows
    ]


def scores(ref_path, candidate_path):
    ref, candidate = read_scores(ref_path), read_scores(candidate_path)
    if len(ref) != len(candidate) or not ref:
        raise ValueError("traces must have the same nonzero row count")
    if [(r[0], r[1]) for r in ref] != [(r[0], r[1]) for r in candidate]:
        raise ValueError("positions and target token IDs must match")
    differences = [b[3] - a[3] for a, b in zip(ref, candidate)]
    if not all(math.isfinite(v) for v in differences):
        raise ValueError("non-finite NLL difference")
    top_mismatches = [a[0] for a, b in zip(ref, candidate) if a[2] != b[2]]
    largest = max(range(len(ref)), key=lambda i: abs(differences[i]))
    return {
        "rows": len(ref),
        "reference_mean_nll": sum(r[3] for r in ref) / len(ref),
        "candidate_mean_nll": sum(r[3] for r in candidate) / len(candidate),
        "mean_delta_nll": sum(differences) / len(differences),
        "mean_absolute_delta_nll": sum(abs(x) for x in differences) / len(differences),
        "max_absolute_delta_nll": abs(differences[largest]),
        "max_delta_position": ref[largest][0],
        "top1_mismatches": len(top_mismatches),
        "first_top1_mismatch": top_mismatches[0] if top_mismatches else None,
    }


def greedy(ref_path, candidate_path):
    ref = json.loads(Path(ref_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    if ref["prompt_ids"] != candidate["prompt_ids"]:
        raise ValueError("prompt token IDs must match")
    prompt_len = len(ref["prompt_ids"])
    if ref["full_ids"][:prompt_len] != ref["prompt_ids"] or candidate["full_ids"][:prompt_len] != candidate["prompt_ids"]:
        raise ValueError("prompt must be a prefix of each continuation")
    a, b = ref["full_ids"][prompt_len:], candidate["full_ids"][prompt_len:]
    if len(a) != len(b) or not a:
        raise ValueError("continuations must have the same nonzero length")
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    return {"tokens": len(a), "matching_prefix": first,
            "same_position_tokens": sum(x == y for x, y in zip(a, b)),
            "first_reference_id": a[first] if first < len(a) else None,
            "first_candidate_id": b[first] if first < len(a) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("scores", "greedy"))
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    print(json.dumps((scores if args.kind == "scores" else greedy)(args.reference, args.candidate), indent=2))


if __name__ == "__main__":
    main()
