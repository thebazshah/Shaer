#!/usr/bin/env python3
"""Export the authors' released Shaer test generations (Shaer-AI/shaer-sft-test, split 'test') to JSONL
for experiment E0, which re-scores them with the official evaluator on our machine.

The released rows already carry the authors' own 'meter' and 'count_adherence' values; they are kept
under 'released_meter' / 'released_count_adherence' so that the re-scored values can be compared
row by row.

Usage:
  python export_released_generations.py --out runs/e0/generations.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from common import RELEASED_GENERATIONS_ID


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/e0/generations.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ds = load_dataset(RELEASED_GENERATIONS_ID, split="test")
    if args.limit:
        ds = ds.select(range(args.limit))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in ds:
            row = dict(r)
            row["released_meter"] = row.pop("meter", None)
            row["released_count_adherence"] = row.pop("count_adherence", None)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {len(ds)} rows -> {out}")


if __name__ == "__main__":
    main()
