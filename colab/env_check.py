#!/usr/bin/env python3
"""Run with the TRAINING environment's interpreter: print package versions and the GPU, and say which experiments this GPU supports.

Exit code 1 when there is no GPU. Used by shaer_colab.check_environment().
"""
import sys

import torch
import transformers
import peft
import bitsandbytes
import sentencepiece  # noqa: F401  (import check: the slow Llama tokenizer needs it)
import google.protobuf

print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| bitsandbytes", bitsandbytes.__version__, "| protobuf", google.protobuf.__version__)
if not torch.cuda.is_available():
    print("no GPU: choose Runtime > Change runtime type > A100 GPU")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
cc = (p.major, p.minor)
gb = p.total_memory / 2**30
bf16 = torch.cuda.is_bf16_supported()
print(f"GPU: {p.name} | {gb:.0f} GB | compute capability {cc[0]}.{cc[1]} | bf16 supported: {bf16}")
ok_gen = bf16 and cc < (12, 0) and gb >= 22
ok_train = bf16 and cc < (12, 0) and gb >= 38
print("E0 (scoring)    : fine on any runtime")
print("E1/E2 (generate):", "OK" if ok_gen else "NOT OK on this GPU (needs bf16, older than Blackwell, >= 22 GB)")
print("E3 (training)   :", "OK" if ok_train else "NOT RECOMMENDED on this GPU (needs bf16, older than Blackwell, a 40 GB class card)")
