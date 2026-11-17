#!/usr/bin/env python3
"""Run with the TRAINING environment's interpreter. Checks, before any long step, that

  1. the Hugging Face token can read the base model (BASE_MODEL; the paper's Yehia is gated), and
  2. the slow sentencepiece tokenizer that train_sft.py and generate_shaer_test.py load really loads
     (it needs protobuf with transformers 4.43).

Environment: BASE_MODEL, HF_TOKEN (may be empty for a public model). Exit code 2 on failure.
"""
import os
import sys

from huggingface_hub import hf_hub_download

model = os.environ["BASE_MODEL"]
token = os.environ.get("HF_TOKEN") or None
try:
    hf_hub_download(model, "config.json", token=token)
except Exception as exc:  # gated without approval, wrong token, offline ...
    print(f"cannot access {model}: {type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
    sys.exit(2)
print("access to", model, ": OK")

from transformers import AutoTokenizer  # noqa: E402

try:
    tok = AutoTokenizer.from_pretrained(model, use_fast=False, token=token)
except Exception as exc:
    print(f"the slow tokenizer of {model} does not load: {type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
    sys.exit(2)
print("slow tokenizer loads OK | vocab", len(tok), "|", type(tok).__name__)
