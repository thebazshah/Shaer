#!/usr/bin/env python3
"""Write the Shaer data splits as small JSONL files for the verifier tools: {"id", "verses", "base_meter", "form"} per poem.

The train split builds the pronunciation lexicon; eval tunes the verifier's thresholds; test is used only for the final
report. Source: the released dataset Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits.

Usage: python prepare_splits.py --out-dir data_splits [--splits train eval test]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DATASET = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()
    from datasets import load_dataset
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        target = out / f"{split}.jsonl"
        ds = load_dataset(args.dataset, split=split, token=os.getenv("HF_TOKEN") or None)
        tmp = target.with_name(target.name + ".partial")
        with tmp.open("w", encoding="utf-8") as f:
            for r in ds:
                f.write(json.dumps({"id": r["id"], "verses": list(r["poem verses"]), "base_meter": r["base_meter"],
                                    "form": r["form"]}, ensure_ascii=False) + "\n")
        os.replace(tmp, target)
        print(f"{split}: {len(ds):,} poems -> {target}")


if __name__ == "__main__":
    main()
