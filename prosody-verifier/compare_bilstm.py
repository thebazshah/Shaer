#!/usr/bin/env python3
"""Bayt-by-bayt agreement between the prosody verifier and the BiLSTM meter classifier used by the Shaer evaluation.

Input: rows scored by evaluation/score_meter_count.py (or meter_score_batched.py), which carry the classifier's per-bayt
prediction (`meter_details_json`). For every complete bayt:
  BiLSTM on target   the classifier's predicted meter equals the requested base meter
  verifier ok        both hemistichs scan in the requested meter (any form, and the requested form) with an allowed pair
The report gives the 2x2 table, Cohen's kappa, per-meter rates, and writes a sample of disagreements for manual review.

Usage: python compare_bilstm.py --scored scored.jsonl --lexicon lexicon.json.gz --out compare.json [--limit 500]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prosody.lexicon import Lexicon  # noqa: E402
from prosody.meters import METERS, shaer_target  # noqa: E402
from prosody.verify import Verifier  # noqa: E402

V: Verifier | None = None


def _init(lex_path, ignore_diacritics=False):
    global V
    V = Verifier(Lexicon.load(lex_path) if lex_path else None, diagnose=False, ignore_diacritics=ignore_diacritics)


def _work(rows):
    out = []
    for r in rows:
        try:
            meter, form = shaer_target(r["base_meter"], r.get("form") or "تام")
        except KeyError:
            continue
        lines = [x.strip() for x in str(r.get("generated_text") or "").split("\n") if x.strip()]
        try:
            details = json.loads(r.get("meter_details_json") or "{}").get("per_bayt_details", [])
        except json.JSONDecodeError:
            details = []
        target_ar = METERS[meter].name
        pv = V.poem(lines[: len(lines) // 2 * 2], meter, form)
        # the same bayts in any form of the meter (the Shaer metric only knows base meters)
        forms_ok = {}
        for fk in METERS[meter].forms:
            if fk != form:
                forms_ok[fk] = [b.ok for b in V.poem(lines[: len(lines) // 2 * 2], meter, fk).bayts]
        for b in pv.bayts:
            d = next((x for x in details if x.get("bayt_index") == b.index), None)
            if d is None or d.get("skipped") or not d.get("pred"):
                continue
            any_form = b.ok or any(v[b.index] for v in forms_ok.values() if b.index < len(v))
            out.append({"id": r.get("id"), "meter": meter, "form": form, "bayt": b.index, "bilstm": d["pred"] == target_ar,
                        "bilstm_pred": d["pred"], "p_target": d.get("p_target"), "verifier": b.ok, "verifier_any_form": any_form,
                        "reason": b.reason, "sadr": b.sadr.text, "ajuz": b.ajuz.text,
                        "sadr_feet": b.sadr.match and b.sadr.match.feet_text, "ajuz_feet": b.ajuz.match and b.ajuz.match.feet_text})
    return out


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--lexicon", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--ignore-diacritics", action="store_true", help="scan bare letters (written vowels do not count)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.scored, encoding="utf-8") if l.strip()]
    if "sample_idx" in rows[0]:
        rows = [r for r in rows if int(r.get("sample_idx", 0) or 0) == 0]
    random.Random(3).shuffle(rows)
    rows = rows[: args.limit] if args.limit else rows
    chunks = [rows[i:i + 20] for i in range(0, len(rows), 20)]
    recs, t0 = [], time.time()
    with mp.Pool(args.workers, initializer=_init, initargs=(args.lexicon or None, args.ignore_diacritics)) as pool:
        for k, part in enumerate(pool.imap_unordered(_work, chunks), 1):
            recs.extend(part)
            if k % 10 == 0 or k == len(chunks):
                print(f"  {k}/{len(chunks)} chunks, {len(recs):,} bayts, {time.time() - t0:.0f}s", flush=True)
    res = {"source": args.scored, "poems": len(rows), "bayts": len(recs), "ignore_diacritics": args.ignore_diacritics}
    for key in ("verifier", "verifier_any_form"):
        t = Counter((r["bilstm"], r[key]) for r in recs)
        res[key] = {"both_ok": t[(True, True)], "bilstm_only": t[(True, False)], "verifier_only": t[(False, True)],
                    "both_not": t[(False, False)],
                    "agreement_pct": round(100 * (t[(True, True)] + t[(False, False)]) / len(recs), 2),
                    "kappa": round(kappa([r["bilstm"] for r in recs], [r[key] for r in recs]), 3),
                    "bilstm_ok_pct": round(100 * sum(r["bilstm"] for r in recs) / len(recs), 2),
                    "verifier_ok_pct": round(100 * sum(r[key] for r in recs) / len(recs), 2)}
    per = defaultdict(list)
    for r in recs:
        per[r["meter"]].append(r)
    res["by_meter"] = {m: {"bayts": len(v), "bilstm_ok_pct": round(100 * sum(r["bilstm"] for r in v) / len(v), 1),
                           "verifier_ok_pct": round(100 * sum(r["verifier"] for r in v) / len(v), 1),
                           "verifier_any_form_pct": round(100 * sum(r["verifier_any_form"] for r in v) / len(v), 1)}
                       for m, v in sorted(per.items(), key=lambda kv: -len(kv[1]))}
    rng = random.Random(5)
    dis = [r for r in recs if r["bilstm"] != r["verifier_any_form"]]
    res["disagreement_sample"] = rng.sample(dis, min(args.samples, len(dis)))
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("disagreement_sample", "by_meter")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
