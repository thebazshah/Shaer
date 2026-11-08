#!/usr/bin/env python3
"""Compute the paper's structural metrics from rows scored by the official evaluator
(evaluation/score_meter_count.py) and write them as LaTeX macros for the report.

Metrics (as defined in the Shaer paper, Section 5.2):
  * base-meter accuracy  : poem-level majority vote of the bayt-level meter predictions == target meter
  * meter conformity     : mean over poems of the per-poem geometric-mean probability ('meter' field)
  * exact count accuracy : share of poems whose bayt count equals the request (count_adherence == 1)
  * count adherence      : mean of max(0, 1 - |Hg - Hr| / Hr)

Usage:
  python collect_results.py --runs-dir runs --out ../report/results/values.tex
Expects runs/<exp>/scored.jsonl for exp in e0, e1, e2, e3 (missing ones stay as placeholders) and
optionally runs/e3_train/train_log.jsonl, runs/e3/generation_summary.json, runs/e4/judge_summary.json,
runs/e3/memorisation.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import LENGTH_BUCKETS, length_bucket, read_jsonl

EXPS = {"e0": "Ezero", "e1": "Eone", "e2": "Etwo", "e3": "Ethree"}


def meter_hit(row: dict) -> bool:
    try:
        details = json.loads(row.get("meter_details_json") or "{}").get("per_bayt_details", [])
    except json.JSONDecodeError:
        return False
    preds = [d.get("pred") for d in details if not d.get("skipped") and d.get("pred")]
    target = row.get("meter_target_used") or row.get("base_meter")
    if not preds or not target:
        return False
    return Counter(preds).most_common(1)[0][0] == target


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "meter_acc": 100.0 * sum(meter_hit(r) for r in rows) / n,
        "meter_conf": sum(float(r.get("meter", 0.0) or 0.0) for r in rows) / n,
        "count_acc": 100.0 * sum(float(r.get("count_adherence", 0.0) or 0.0) >= 0.999999 for r in rows) / n,
        "count_adh": sum(float(r.get("count_adherence", 0.0) or 0.0) for r in rows) / n,
    }


def fmt_pct(x):
    return f"{x:.2f}"


def fmt3(x):
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--template", default=None, help="values.tex with placeholders (default: --out itself)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = Path(args.runs_dir)
    template = Path(args.template or args.out).read_text(encoding="utf-8")
    values: dict[str, str] = {}
    summary: dict[str, dict] = {}

    for exp, prefix in EXPS.items():
        path = runs / exp / "scored.jsonl"
        if not path.exists():
            continue
        rows = list(read_jsonl(path))
        s = summarise(rows)
        summary[exp] = s
        values[f"{prefix}MeterAcc"] = fmt_pct(s["meter_acc"])
        values[f"{prefix}MeterConf"] = fmt3(s["meter_conf"])
        values[f"{prefix}CountAcc"] = fmt_pct(s["count_acc"])
        values[f"{prefix}CountAdh"] = fmt3(s["count_adh"])
        if exp == "e3":
            by_b = defaultdict(list)
            for r in rows:
                by_b[r.get("length_bucket") or length_bucket(r["requested_bayts"])].append(r)
            for b, key in zip(LENGTH_BUCKETS, "ABCD"):
                sb = summarise(by_b.get(b, []))
                if sb:
                    values[f"EthreeCountBucket{key}"] = fmt_pct(sb["count_acc"])
                    values[f"EthreeMeterBucket{key}"] = fmt_pct(sb["meter_acc"])
            summary["e3_by_bucket"] = {b: summarise(by_b.get(b, [])) for b in LENGTH_BUCKETS}

    log = runs / "e3_train" / "train_log.jsonl"
    if log.exists():
        recs = list(read_jsonl(log))
        tr = [r for r in recs if "train_loss" in r]
        ev = [r for r in recs if "eval_loss" in r and r.get("step", 0) > 0]
        if tr:
            values["EthreeSteps"] = f"{tr[-1]['step']:,}"
            values["EthreeTrainLoss"] = f"{sum(r['train_loss'] for r in tr) / len(tr):.3f}"
            tps = [r["tokens_per_sec"] for r in tr]
            values["EthreeTokPerSec"] = f"{sum(tps) / len(tps):.0f}"
            values["EthreePeakMem"] = f"{max(r['peak_mem_gb'] for r in tr):.1f}\\,GB"
            hours = (recs[-1]["time"] - recs[0]["time"]) / 3600
            values["EthreeWallClock"] = f"{hours:.1f}\\,h"
        if ev:
            best = min(ev, key=lambda r: r["eval_loss"])
            values["EthreeBestEvalLoss"] = f"{best['eval_loss']:.3f}"
            values["EthreeBestStep"] = f"{best['step']:,}"
    test_loss = runs / "e3_train" / "test_loss.json"
    if test_loss.exists():
        values["EthreeTestLoss"] = f"{json.loads(test_loss.read_text())['test_loss']:.3f}"
    gen = runs / "e3" / "generation_summary.json"
    if gen.exists():
        values["EthreeGenTime"] = f"{json.loads(gen.read_text())['seconds'] / 3600:.1f}\\,h"
    judge = runs / "e4" / "judge_summary.json"
    if judge.exists():
        j = json.loads(judge.read_text())
        for key, macro in [("description_adherence", "EfourDA"), ("meaning", "EfourMeaning"), ("fluency", "EfourFluency"),
                           ("coherence", "EfourCoherence"), ("poeticness", "EfourPoeticness")]:
            if key in j:
                values[macro] = f"{j[key]:.2f}"
    mem = runs / "e3" / "memorisation.json"
    if mem.exists():
        m = json.loads(mem.read_text())
        values["EthreeExactCopies"] = str(m["exact_copies"])
        values["EthreeMaxSpan"] = str(m["max_shared_span"])
        values["EthreeJaccardMax"] = f"{m['max_jaccard5']:.3f}"

    out_lines = []
    for line in template.splitlines():
        if line.startswith("\\newcommand{\\"):
            name = line[len("\\newcommand{\\"):].split("}", 1)[0]
            if name in values:
                line = f"\\newcommand{{\\{name}}}{{{values[name]}}}"
        out_lines.append(line)
    Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    (runs / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"filled {len(values)} values -> {args.out}")


if __name__ == "__main__":
    main()
