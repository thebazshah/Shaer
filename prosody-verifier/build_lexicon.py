#!/usr/bin/env python3
"""Build the pronunciation prior (prosody/lexicon.py) from the TRAIN split.

Every hemistich of every distinct train poem is scanned in its labelled meter; the readings of the best scansion are counted
per bare word form. Round 2 repeats the scan with round 1's prior so the choices settle. Poems are deduplicated first (the
released train split repeats poems for weighted sampling).

Input: JSONL rows with `verses` (list of hemistichs) and `base_meter` (the released train split, or anything shaped alike).
Output: a gzipped JSON lexicon (a few MB).

Usage: python build_lexicon.py --train train.jsonl --out lexicon.json.gz [--rounds 2] [--workers 4] [--limit 0]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prosody import arudi  # noqa: E402
from prosody.lexicon import Lexicon, reading_key, word_key  # noqa: E402
from prosody.meters import NAME_TO_KEY  # noqa: E402
from prosody.scan import scan  # noqa: E402

_LEX = None


def _init(lex_path):
    global _LEX
    _LEX = Lexicon.load(lex_path) if lex_path else None


def _count(poems: list[tuple[str, list[str]]]) -> tuple[dict, int, int]:
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    scanned = matched = 0
    for meter, verses in poems:
        for line in verses:
            scanned += 1
            ms = scan(line, lexicon=_LEX, meters=(meter,))
            if not ms:
                continue
            matched += 1
            words = arudi.tokenize(line)
            for w, r in zip(words, ms[0].choices):
                if r is not None:
                    counts[word_key(w)][reading_key(r)] += 1.0
    return {k: dict(v) for k, v in counts.items()}, scanned, matched


def load_poems(path: str, limit: int = 0) -> list[tuple[str, list[str]]]:
    seen, poems = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            verses = tuple(v.strip() for v in r["verses"] if str(v).strip())
            if not verses or verses in seen or r["base_meter"] not in NAME_TO_KEY:
                continue
            seen.add(verses)
            poems.append((NAME_TO_KEY[r["base_meter"]], list(verses)))
            if limit and len(poems) >= limit:
                break
    return poems


def build_round(poems, lex_path, workers: int, chunk: int = 200):
    chunks = [poems[i:i + chunk] for i in range(0, len(poems), chunk)]
    total: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    scanned = matched = 0
    t0 = time.time()
    with mp.Pool(workers, initializer=_init, initargs=(lex_path,)) as pool:
        for i, (counts, s, m) in enumerate(pool.imap_unordered(_count, chunks), 1):
            scanned, matched = scanned + s, matched + m
            for k, v in counts.items():
                for rk, c in v.items():
                    total[k][rk] += c
            if i % 20 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {scanned:,} hemistichs, {100 * matched / max(scanned, 1):.1f}% scan in "
                      f"their meter, {time.time() - t0:.0f}s", flush=True)
    return {k: dict(v) for k, v in total.items()}, scanned, matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--limit", type=int, default=0, help="number of distinct poems (0 = all)")
    ap.add_argument("--min-total", type=float, default=2.0)
    args = ap.parse_args()
    poems = load_poems(args.train, args.limit)
    print(f"{len(poems):,} distinct train poems", flush=True)
    lex_path = None
    out = Path(args.out)
    for rnd in range(1, args.rounds + 1):
        print(f"round {rnd}/{args.rounds}", flush=True)
        counts, scanned, matched = build_round(poems, lex_path, args.workers)
        counts = {k: v for k, v in counts.items() if sum(v.values()) >= args.min_total}
        tmp = out.with_name(out.name + f".round{rnd}")
        lex = Lexicon(counts, min_total=args.min_total)
        lex.train_chars()                         # letter model for words the lexicon has not seen
        lex.save(tmp, meta={"round": rnd, "poems": len(poems), "hemistichs": scanned, "matched": matched,
                            "source": str(args.train)})
        lex_path = str(tmp)
        print(f"  round {rnd}: {len(counts):,} word forms, {100 * matched / max(scanned, 1):.1f}% of hemistichs scanned in "
              f"their meter", flush=True)
    Path(lex_path).replace(out)
    for p in out.parent.glob(out.name + ".round*"):
        p.unlink()
    print(f"lexicon -> {out}")


if __name__ == "__main__":
    main()
