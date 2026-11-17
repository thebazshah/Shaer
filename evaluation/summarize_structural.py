#!/usr/bin/env python3
"""Summarise rows scored by evaluation/score_meter_count.py into the paper's structural metrics.

Definitions (each checked against the released generations Shaer-AI/shaer-sft-test: re-scoring them on a different
machine reproduces the paper's Tables 20-21 for Shaer, overall and per length bucket):

  meter conformity          mean of the per-poem `meter` score                       -> 0.9064
  base-meter accuracy       `meter_predicted` == `meter_target_used`, i.e. the prediction for the FIRST bayt -> 95.17
  exact count accuracy      parsed hemistich count == requested hemistich count     -> 83.40
  count adherence           mean of max(0, 1 - |generated - requested| / requested) -> 0.9792

Note: the paper's text describes base-meter accuracy as a majority vote over the bayt-level predictions, but the
published 95.17 % (and every per-bucket figure) is reproduced only by the first-bayt prediction; the majority vote
gives 98.76 % on the same poems. Both are reported here (`base_meter_accuracy_pct` is the one that matches the
paper's tables; `base_meter_accuracy_majority_pct` follows the text). Likewise `count_adherence == 1` counts bayts and
is slightly more lenient (84.43 %) than the paper's hemistich-count match (83.40 %).

Rows may come from the released dataset or from generate_shaer_test.py. If the input has several samples per prompt (a
`sample_idx` column), the script reports the first sample, the mean over samples and the best-of-k sample per prompt
(highest `meter`), so a single-sample run and a selected-best run can be compared.

Usage:
  python evaluation/summarize_structural.py --scored runs/e1/scored.jsonl --out runs/e1/summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BUCKETS = ["1-3", "4-6", "7-10", "11-20"]


def bucket(requested_bayts: int) -> str:
    n = int(requested_bayts)
    return "1-3" if n <= 3 else "4-6" if n <= 6 else "7-10" if n <= 10 else "11-20"


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _target(row: dict) -> str:
    return row.get("meter_target_used") or row.get("base_meter") or ""


def meter_hit_first(row: dict) -> bool:
    """Paper-table definition: the first bayt's predicted meter equals the target meter."""
    return bool(_target(row)) and row.get("meter_predicted") == _target(row)


def meter_hit_majority(row: dict) -> bool:
    """Definition described in the paper text: plurality of the bayt-level predictions equals the target meter."""
    try:
        details = json.loads(row.get("meter_details_json") or "{}").get("per_bayt_details", [])
    except json.JSONDecodeError:
        return False
    preds = [d.get("pred") for d in details if not d.get("skipped") and d.get("pred")]
    return bool(preds and _target(row) and Counter(preds).most_common(1)[0][0] == _target(row))


def exact_count_hit(row: dict) -> bool:
    return int(row.get("parsed_num_lines", -1)) == int(row.get("requested_num_lines", -2))


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    count = [float(r.get("count_adherence") or 0.0) for r in rows]
    return {
        "n": n,
        "meter_conformity": sum(float(r.get("meter") or 0.0) for r in rows) / n,
        "base_meter_accuracy_pct": 100.0 * sum(meter_hit_first(r) for r in rows) / n,
        "base_meter_accuracy_majority_pct": 100.0 * sum(meter_hit_majority(r) for r in rows) / n,
        "exact_count_accuracy_pct": 100.0 * sum(exact_count_hit(r) for r in rows) / n,
        "count_adherence": sum(count) / n,
    }


def by_bucket(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("length_bucket") or bucket(r["requested_bayts"])].append(r)
    return {b: summarise(groups[b]) for b in BUCKETS if groups.get(b)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="JSONL written by evaluation/score_meter_count.py")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = read_jsonl(Path(args.scored))
    result: dict = {"input": args.scored}
    samples = {int(r.get("sample_idx", 0)) for r in rows}
    if len(samples) <= 1:
        result["overall"] = summarise(rows)
        result["by_length_bucket"] = by_bucket(rows)
    else:
        first = [r for r in rows if int(r.get("sample_idx", 0)) == min(samples)]
        per_id = defaultdict(list)
        for r in rows:
            per_id[r["id"]].append(r)
        best = [max(v, key=lambda r: float(r.get("meter") or 0.0)) for v in per_id.values()]
        result["samples_per_prompt"] = len(samples)
        result["first_sample"] = summarise(first)
        result["mean_over_samples"] = {k: sum(summarise([r for r in rows if int(r.get("sample_idx", 0)) == s]).get(k, 0.0)
                                              for s in sorted(samples)) / len(samples)
                                       for k in ("meter_conformity", "base_meter_accuracy_pct",
                                                 "base_meter_accuracy_majority_pct", "exact_count_accuracy_pct",
                                                 "count_adherence")}
        result["best_of_k_by_meter"] = summarise(best)
        result["by_length_bucket_first_sample"] = by_bucket(first)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
