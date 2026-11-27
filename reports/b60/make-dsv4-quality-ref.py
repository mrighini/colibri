#!/usr/bin/env python3
"""Build a fixed human continuation after a recorded DeepSeek chat prefix."""
import json
import sys
from tokenizers import Tokenizer

continuation = """GPU memory stores the data an accelerator needs while it is working. Model weights, temporary activations, and the record of earlier tokens all compete for that space. When the model fits, the device can read its weights locally and avoid repeated transfers from system memory. If it does not fit, the program must choose which data to keep and which to move.

Memory capacity and memory bandwidth are different. Capacity determines how much data can stay on the device; bandwidth affects how quickly the processor can read or write that data. A larger memory does not automatically make a computation faster if the program is waiting for synchronization or for a connection between devices.

For an interactive application, the delay before the first token matters to the user. Later tokens should arrive steadily. Engineers measure both phases, check that results remain accurate, and repeat the test with different prompt lengths and realistic workloads."""
tokenizer = Tokenizer.from_file(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as f:
    prompt = json.load(f)["prompt_ids"]
continuation_ids = tokenizer.encode(continuation, add_special_tokens=False).ids
if len(continuation_ids) < 64:
    raise SystemExit("continuation must contain at least 64 tokens")
data = {"prompt_ids": prompt, "full_ids": prompt + continuation_ids[:64],
        "continuation": continuation, "target_tokens": 64,
        "tokenizer_version": __import__("tokenizers").__version__}
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"prompt_tokens={len(prompt)} target_tokens=64")
