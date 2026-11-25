#!/usr/bin/env python3
"""Same output as evaluation/score_meter_count.py, with the BiLSTM forward pass batched (fast on a CPU runtime).

The repository's scorer calls the meter classifier one bayt at a time, which is slow on a CPU (about 10 poems per minute
here). The classifier pads every bayt to the same fixed length, so running many bayts in one batch is the same computation.
This wrapper precomputes those predictions in batches and then runs the repository's own `score_generation_row` (unchanged)
on every row, with the classifier's `predict` answered from the precomputed table.

As a safeguard it re-runs the original one-at-a-time `predict` on the first `--check` bayts and stops if any prediction
differs (or a probability differs by more than 1e-4). On the 3,481 released poems the batched path reproduced the paper's
table exactly (meter conformity 0.9064, base-meter accuracy 95.17 %, exact count 83.40 %, count adherence 0.9792).

Needs METER_MODEL_ID in the environment, like the repository's scorer.

Usage: python meter_score_batched.py --input-jsonl gens.jsonl --output-jsonl scored.jsonl [--batch-size 256] [--check 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation"))
sys.path.insert(0, str(REPO / "grpo"))


def resume_prefix(out: Path, rows: list[dict]) -> list[str]:
    """The complete lines of an earlier, interrupted output that are the first rows of `rows` (the generation file only
    grows, so what was scored is its beginning). A line cut by an interrupted write ends it; an output that does not match
    the input is ignored (everything is scored again)."""
    kept: list[str] = []
    if not Path(out).exists():
        return kept
    for line in open(out, encoding="utf-8"):
        try:
            json.loads(line)
        except json.JSONDecodeError:
            break
        kept.append(line if line.endswith("\n") else line + "\n")
    if len(kept) > len(rows):
        return []
    for k, r in zip(kept, rows):
        d = json.loads(k)
        if d.get("id") != r.get("id") or d.get("sample_idx") != r.get("sample_idx"):
            return []
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--check", type=int, default=200, help="bayts re-scored one at a time to confirm identical results")
    args = ap.parse_args()

    import numpy as np
    import torch
    import metrics                                    # evaluation/metrics.py (unchanged)
    from rewards import meter as M                    # grpo/rewards/meter.py (unchanged)
    from rewards.common import extract_text, pair_lines_to_bayts, poem_structure

    rows = [json.loads(line) for line in open(args.input_jsonl, encoding="utf-8") if line.strip()]
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = resume_prefix(out, rows)
    print(f"{len(kept):,} of {len(rows):,} poems already scored", flush=True)
    todo_rows = rows[len(kept):]
    if not todo_rows:
        print(f"scored_rows={len(rows)} output={out}")
        return 0
    rows_all, rows = rows, todo_rows
    pred = M.get_meter_predictor()
    original_predict = pred.predict
    texts = set()
    for r in rows:
        for b in pair_lines_to_bayts(poem_structure(extract_text(r.get("generated_text") or r.get("completion") or ""))["lines"]):
            c = pred.clean_to_stoi(b)
            if len(c.replace(" ", "")) >= 20:          # the scorer skips shorter bayts; nothing to precompute for them
                texts.add(c)
    texts = sorted(texts)
    print(f"{len(rows):,} poems, {len(texts):,} distinct bayts to classify on {pred.device}", flush=True)
    cache = {}
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i:i + args.batch_size]
            x = torch.cat([pred._encode(t) for t in chunk]).to(pred.device)
            probs = torch.softmax(pred.model(x), dim=-1).cpu().numpy()
            cache.update(zip(chunk, probs))
            print(f"classified {min(i + args.batch_size, len(texts)):,}/{len(texts):,} bayts", flush=True)

    def batched_predict(text: str, target_meter: str) -> dict:
        p = cache[text]
        pid, tid = int(np.argmax(p)), pred.label2id.get(target_meter)
        return {"pred": pred.classes[pid], "p_target": float(p[tid]) if tid is not None else 0.0, "p_pred": float(p[pid])}

    target = next(iter(pred.label2id))
    for t in texts[: args.check]:
        a, b = original_predict(t, target), batched_predict(t, target)
        if a["pred"] != b["pred"] or abs(a["p_target"] - b["p_target"]) > 1e-4 or abs(a["p_pred"] - b["p_pred"]) > 1e-4:
            raise SystemExit(f"batched and one-at-a-time predictions differ for {t!r}: {a} vs {b}")
    print(f"check passed: {min(args.check, len(texts))} bayts give identical predictions one at a time", flush=True)

    pred.predict = batched_predict
    with out.open("w", encoding="utf-8") as f:         # the complete rows kept above, then the new ones, each flushed
        f.writelines(kept)
        for i, r in enumerate(rows, 1):
            s = dict(r)
            s.update(metrics.score_generation_row(r))
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            if i % 100 == 0 or i == len(rows):
                f.flush()
            if i % 500 == 0 or i == len(rows):
                print(f"scored {len(kept) + i:,}/{len(rows_all):,} poems", flush=True)
    print(f"scored_rows={len(rows_all)} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
