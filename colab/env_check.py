#!/usr/bin/env python3
"""Run with the TRAINING environment's interpreter: print Python, package versions and the GPU, and say which experiments this GPU
supports. Every package pinned in requirements-colab.txt is compared with what is installed; a difference is flagged.

Exit code 1 when there is no GPU. Used by shaer_colab.check_environment().
"""
import pathlib
import platform
import re
import sys
from importlib import metadata

import torch
import transformers  # noqa: F401  (import checks: these must load, not only be installed)
import peft  # noqa: F401
import bitsandbytes  # noqa: F401
import sentencepiece  # noqa: F401  (the slow Llama tokenizer needs it)
import google.protobuf  # noqa: F401

print(f"training environment: Python {platform.python_version()} ({sys.executable})")
print(f"  torch {torch.__version__} | built for CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")

# the pins of requirements-colab.txt, plus torch (installed separately by the notebook)
req = pathlib.Path(__file__).with_name("requirements-colab.txt")
pins = {"torch": "2.3.1"}
for line in req.read_text().splitlines():
    m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*(?:==\s*([^\s#;]+))?", line.split("#", 1)[0])
    if m and m.group(1):
        pins[m.group(1)] = m.group(2)
mismatches = []
rows = []
for name, wanted in pins.items():
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        have = "MISSING"
    ok = have != "MISSING" and (wanted is None or have.split("+")[0] == wanted)
    rows.append(f"{name}=={have}" + ("" if ok else f" (expected {wanted or 'installed'})"))
    if not ok:
        mismatches.append(name)
for i in range(0, len(rows), 5):
    print("  " + " | ".join(rows[i:i + 5]))
print("  ✔ all packages match requirements-colab.txt" if not mismatches
      else f"  ⚠ differs from requirements-colab.txt: {', '.join(mismatches)} (re-run the installation cell)")

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
