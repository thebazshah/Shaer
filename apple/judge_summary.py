#!/usr/bin/env python3
"""Average the per-metric LLM-judge scores written by the official evaluation/judge_llm.py (experiment E4).

judge_llm.py scores one metric per call and writes one JSONL file per metric, e.g.
runs/e4/judge_meaning.jsonl with rows {"id", "metric", "score", "judge_error", ...}.
This script averages the 1-5 scores per metric (rows with a judge error or a score outside 1-5 are
dropped and counted) and writes runs/e4/judge_summary.json, which collect_results.py reads.

Usage:
  python judge_summary.py --judge-dir runs/e4 --out runs/e4/judge_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl

METRICS = ["description_adherence", "meaning", "fluency", "coherence", "poeticness"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dir", default="runs/e4")
    ap.add_argument("--out", default="runs/e4/judge_summary.json")
    args = ap.parse_args()
    summary: dict = {"counts": {}}
    for metric in METRICS:
        path = Path(args.judge_dir) / f"judge_{metric}.jsonl"
        if not path.exists():
            continue
        rows = list(read_jsonl(path))
        ok = [r["score"] for r in rows if not r.get("judge_error") and 1 <= int(r.get("score") or 0) <= 5]
        summary["counts"][metric] = {"rows": len(rows), "valid": len(ok), "dropped": len(rows) - len(ok)}
        if ok:
            summary[metric] = sum(ok) / len(ok)
            summary["judge_model"] = rows[0].get("judge_model", "")
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
