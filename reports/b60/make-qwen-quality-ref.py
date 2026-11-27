#!/usr/bin/env python3
"""Build the fixed, human-written 320-token Qwen continuation used by B60 tests."""
import json
import sys
from tokenizers import Tokenizer

prompt = "Explain how GPUs and CPUs cooperate when generating text with a large language model.\n"
continuation = """The CPU prepares the request and arranges the data that the model will need. It can read weights from storage, manage the conversation, choose which operations to run, and send work to the accelerator. The GPU performs many arithmetic operations at once. In a language model, those operations transform the current token and the earlier context into scores for possible next tokens.

The two processors must exchange information, but transfers take time. A system can reduce that cost by keeping frequently used weights and intermediate results in device memory. It should also avoid waiting for a small copy whenever useful work can proceed independently. When memory is limited, the program may keep some weights in host memory and move only the pieces needed for the current step.

There are several ways to judge performance. Time to the first token includes processing the prompt and preparing the initial result. The rate of later tokens measures how quickly a response continues once generation is underway. These measurements can change with prompt length, model size, batch size, and the amount of memory available to each device.

Using more than one GPU can provide additional memory and computing power, but it also introduces coordination costs. The devices must exchange activations or partial results, and a slow connection can offset the benefit of the extra hardware. A useful design places related operations together and measures the real transfer path before choosing how to divide the model.

Correctness matters as much as speed. Changing the order of floating point operations, reducing precision, or using a different kernel can alter the scores. Engineers compare outputs with a reference on several inputs and check that any differences remain within an agreed tolerance. They also repeat measurements under controlled conditions, because startup, cached files, other processes, and profiling tools can all affect the observed numbers.

In practice, the best configuration depends on the workload. A single interactive request favors low latency, while a server with many requests may favor batching and higher total throughput. Profiling helps identify the limiting resource so that the next change addresses a measured cost."""

tokenizer = Tokenizer.from_file(sys.argv[1])
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
all_ids = tokenizer.encode(prompt + continuation, add_special_tokens=False).ids
if all_ids[:len(prompt_ids)] != prompt_ids or len(all_ids) < len(prompt_ids) + 320:
    raise SystemExit("reference tokenization did not produce the expected prefix and 320 targets")
result = {"prompt_ids": prompt_ids, "full_ids": all_ids[:len(prompt_ids) + 320],
          "prompt": prompt, "continuation": continuation,
          "target_tokens": 320, "tokenizer_version": __import__("tokenizers").__version__}
with open(sys.argv[2], "w", encoding="utf-8") as out:
    json.dump(result, out, indent=2, ensure_ascii=False)
    out.write("\n")
print(f"prompt_tokens={len(prompt_ids)} target_tokens=320")
