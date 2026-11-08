#!/usr/bin/env python3
"""Memorisation / copying analysis, following the Shaer paper (Section 5.2):
exact normalised match, longest shared contiguous word span and word 5-gram Jaccard of each
generated poem against (a) all training poems and (b) its paired held-out source poem.

Usage:
  python memorisation.py --generations runs/e3/generations.jsonl --out runs/e3/memorisation.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from common import DATASET_ID, read_jsonl

DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")


def norm_words(text: str) -> list[str]:
    text = DIACRITICS.sub("", text or "")
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def ngrams(words, n=5):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def longest_common_span(a: list[str], b: list[str]) -> int:
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds = load_dataset(DATASET_ID)
    train = [norm_words(r["sft_completion"]) for r in ds["train"]]
    test_ref = {f"shaer_test_{i + 1:06d}": norm_words(r["sft_completion"]) for i, r in enumerate(ds["test"])}
    train_text = {" ".join(w) for w in train}
    index = defaultdict(set)
    train_grams = []
    for k, w in enumerate(train):
        g = ngrams(w)
        train_grams.append(g)
        for x in g:
            index[x].add(k)

    rows, exact, max_span, max_jac, max_span_src, max_jac_src = [], 0, 0, 0.0, 0, 0.0
    for r in read_jsonl(Path(args.generations)):
        w = norm_words(r["generated_text"])
        g = ngrams(w)
        is_exact = " ".join(w) in train_text
        cands = set().union(*(index[x] for x in g)) if g else set()
        jac = max((len(g & train_grams[k]) / len(g | train_grams[k]) for k in cands), default=0.0)
        span = max((longest_common_span(w, train[k]) for k in cands), default=0)
        src = test_ref.get(r["id"], [])
        gs = ngrams(src)
        jac_src = len(g & gs) / len(g | gs) if (g or gs) else 0.0
        span_src = longest_common_span(w, src)
        exact += int(is_exact)
        max_span, max_jac = max(max_span, span), max(max_jac, jac)
        max_span_src, max_jac_src = max(max_span_src, span_src), max(max_jac_src, jac_src)
        rows.append({"id": r["id"], "exact_train": is_exact, "span_train": span, "jaccard5_train": jac,
                     "span_source": span_src, "jaccard5_source": jac_src})
    summary = {"poems": len(rows), "exact_copies": exact, "max_shared_span": max_span, "max_jaccard5": max_jac,
               "max_shared_span_source": max_span_src, "max_jaccard5_source": max_jac_src,
               "poems_span_ge_13_train": sum(x["span_train"] >= 13 for x in rows)}
    Path(args.out).write_text(json.dumps({**summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
