#!/usr/bin/env python3
"""Export the authors' released Shaer test generations (Shaer-AI/shaer-sft-test) to JSONL for experiment E0.

E0 re-scores the released poems with the repository's evaluator on your own machine. The released columns `meter`
and `count_adherence` are kept under `released_meter` / `released_count_adherence` so the re-scored values can be
compared row by row (the scorer overwrites `meter` and `count_adherence`).

Usage: python evaluation/export_released_generations.py --out runs/e0/generations.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-id", default="Shaer-AI/shaer-sft-test")
    ap.add_argument("--limit-rows", type=int, default=0)
    args = ap.parse_args()
    ds = load_dataset(args.dataset_id, split="test", token=os.getenv("HF_TOKEN") or None)
    if args.limit_rows:
        ds = ds.select(range(args.limit_rows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            row = dict(row)
            row["released_meter"] = row.pop("meter", None)
            row["released_count_adherence"] = row.pop("count_adherence", None)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {len(ds)} rows -> {out}")


if __name__ == "__main__":
    main()
