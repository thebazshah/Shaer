#!/usr/bin/env python3
"""Download the official Shaer splits, tokenise, pack and write MLX-ready training data.

Mirrors sft/train_sft.py: same prompt/full-text fields, add_special_tokens=False,
keep_end truncation to 1024 tokens, completion-only mask, packing within sampler groups,
and WeightedRandomSampler weights (alpha = 0.6). The weighted sampler is applied per epoch
in train_mlx_qlora.py using the weights saved here.

Outputs (in --out):
  train_packed.npz      tokens [N,1024] int32, mask [N,1024] uint8, lengths [N], weights [N]
  eval.jsonl            tokenised eval rows (unpacked), used for eval loss
  eval_subset.jsonl     fixed random subset of eval rows (periodic evaluation)
  test_prompts.jsonl    test rows needed for generation and scoring
  test.jsonl            tokenised test rows (final test loss)
  stats.json            counts for the report
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from common import DATASET_ID, MAX_SEQ_LENGTH, SEED, length_bucket, pack_by_group, sampler_weights, tokenize_example, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF id or local dir with the Yehia tokenizer (e.g. the MLX-converted model dir)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-len", type=int, default=MAX_SEQ_LENGTH)
    ap.add_argument("--eval-subset", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="debug: only use the first N train/eval/test rows")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset(DATASET_ID)
    print({k: len(v) for k, v in ds.items()})

    def rows(split):
        d = ds[split]
        if args.limit:
            d = d.select(range(min(args.limit, len(d))))
        return d

    # ---- train: tokenise -> pack -> weights ----
    examples = []
    n_trunc = 0
    for r in rows("train"):
        ids, mask, truncated = tokenize_example(tok, r["sft_prompt"], r["sft_full_text"], args.max_len)
        n_trunc += int(truncated)
        examples.append({"input_ids": ids, "mask": mask, "sampler_group": r["sampler_group"]})
    packed = pack_by_group(examples, args.max_len)
    weights = sampler_weights([p["sampler_group"] for p in packed])
    N, L = len(packed), args.max_len
    tokens = np.zeros((N, L), dtype=np.int32)
    masks = np.zeros((N, L), dtype=np.uint8)
    lengths = np.zeros((N,), dtype=np.int32)
    for i, p in enumerate(packed):
        n = len(p["input_ids"])
        tokens[i, :n] = p["input_ids"]
        masks[i, :n] = p["mask"]
        lengths[i] = n
    np.savez_compressed(out / "train_packed.npz", tokens=tokens, mask=masks, lengths=lengths,
                        weights=np.asarray(weights, dtype=np.float64),
                        groups=np.asarray([p["sampler_group"] for p in packed]))

    # ---- eval (unpacked, as in the official run) ----
    eval_rows = []
    for r in rows("eval"):
        ids, mask, _ = tokenize_example(tok, r["sft_prompt"], r["sft_full_text"], args.max_len)
        eval_rows.append({"input_ids": ids, "mask": mask})
    write_jsonl(out / "eval.jsonl", eval_rows)
    rng = random.Random(SEED)
    subset = rng.sample(eval_rows, min(args.eval_subset, len(eval_rows)))
    write_jsonl(out / "eval_subset.jsonl", subset)

    # ---- test: what generation and scoring need ----
    test_rows = []
    for i, r in enumerate(rows("test")):
        test_rows.append({
            "id": f"shaer_test_{i + 1:06d}",
            "sft_prompt": r["sft_prompt"],
            "base_meter": r["base_meter"],
            "form": r["form"],
            "requested_bayts": int(r["requested_bayts"]),
            "requested_num_lines": int(r["sft_num_lines"]),
            "length_bucket": length_bucket(r["requested_bayts"]),
            "description": r["description"],
            "enhanced_description": r["enhanced_description"],
            "reference_completion": r["sft_completion"],
        })
    write_jsonl(out / "test_prompts.jsonl", test_rows)
    write_jsonl(out / "test.jsonl", ({"input_ids": ids, "mask": mask} for ids, mask, _ in
                                     (tokenize_example(tok, r["sft_prompt"], r["sft_full_text"], args.max_len)
                                      for r in rows("test"))))

    stats = {
        "train_rows": len(examples), "packed_sequences": N, "max_len": L,
        "train_tokens": int(lengths.sum()), "supervised_tokens": int(masks.sum()),
        "steps_per_epoch_at_eff_batch_16": int(np.ceil(N / 16)),
        "eval_rows": len(eval_rows), "eval_subset": len(subset), "test_rows": len(test_rows),
        "truncated_train_rows": n_trunc,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
