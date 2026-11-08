#!/usr/bin/env python3
"""Merge the released Shaer PEFT adapter into the bf16 Yehia-7B base (CPU), for experiment E1.

The official adapter (Shaer-AI/Shaer-adapters, adapters/fresh_sft/train/best) was trained on a
bitsandbytes NF4 base, which cannot be loaded on Apple Silicon. We therefore load the base in
bf16 on the CPU, apply the adapter with PEFT, merge it (W + scale * B A) and save a plain HF
checkpoint, which is then converted to MLX 4-bit with:
    mlx_lm.convert --hf-path models/shaer-official-merged-bf16 -q --q-bits 4 --q-group-size 64 \
                   --mlx-path models/shaer-official-mlx-4bit
Needs about 16 GB of free memory and a Hugging Face token with access to the gated base model.
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ADAPTER_REPO_ID, ADAPTER_SUBFOLDER, BASE_MODEL_ID


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL_ID)
    ap.add_argument("--adapter", default=ADAPTER_REPO_ID)
    ap.add_argument("--subfolder", default=ADAPTER_SUBFOLDER)
    ap.add_argument("--out", default="models/shaer-official-merged-bf16")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                device_map={"": "cpu"})
    model = PeftModel.from_pretrained(base, args.adapter, subfolder=args.subfolder)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(args.out)
    print(f"merged model written to {args.out}")


if __name__ == "__main__":
    main()
