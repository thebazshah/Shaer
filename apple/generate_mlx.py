#!/usr/bin/env python3
"""Generate the 3,481 test poems with MLX (replaces the HF-generate / vLLM path of the official repo).

Decoding follows the paper (Appendix I, Table 15): temperature 0.55, top-p 0.90,
repetition penalty 1.08, at most 768 new tokens, stop at EOS. The prompt is the official
`sft_prompt` tokenised without extra special tokens (it already starts with '<s>'), exactly
as during training. Output rows carry the fields read by the official scorer
(evaluation/score_meter_count.py): generated_text, base_meter, form, requested_bayts,
requested_num_lines.

Usage:
  python generate_mlx.py --model models/shaer-official-mlx-4bit --prompts data/test_prompts.jsonl --out runs/e1/generations.jsonl
  python generate_mlx.py --model models/yehia-7b-mlx-4bit --adapter runs/e3_train/best --prompts ... --out runs/e3/generations.jsonl
Re-running resumes: ids already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from common import DECODING, SEED, read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="MLX adapter dir (adapters.safetensors + adapter_config.json)")
    ap.add_argument("--prompts", default="data/test_prompts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--temperature", type=float, default=DECODING["temperature"])
    ap.add_argument("--top-p", type=float, default=DECODING["top_p"])
    ap.add_argument("--repetition-penalty", type=float, default=DECODING["repetition_penalty"])
    ap.add_argument("--max-tokens", type=int, default=DECODING["max_tokens"])
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
    processors = make_logits_processors(repetition_penalty=args.repetition_penalty)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    rows = list(read_jsonl(Path(args.prompts)))
    if args.limit:
        rows = rows[: args.limit]

    t_start, n_new, n_tok = time.time(), 0, 0
    with out.open("a", encoding="utf-8") as f:
        for k, row in enumerate(rows):
            if row["id"] in done:
                continue
            mx.random.seed(args.seed + k)  # one fixed seed per prompt -> resumable and repeatable
            prompt_ids = tokenizer.encode(row["sft_prompt"], add_special_tokens=False)
            t0 = time.time()
            text = generate(model, tokenizer, prompt=prompt_ids, max_tokens=args.max_tokens,
                            sampler=sampler, logits_processors=processors, verbose=False)
            gen_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            rec = {k2: row[k2] for k2 in ("id", "base_meter", "form", "requested_bayts", "requested_num_lines",
                                           "length_bucket", "description", "enhanced_description")}
            rec.update({"generated_text": text.strip(), "gen_tokens": gen_tokens, "gen_seconds": time.time() - t0,
                        "model": str(args.model), "adapter": args.adapter,
                        "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                                     "repetition_penalty": args.repetition_penalty, "max_tokens": args.max_tokens}})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            n_tok += gen_tokens
            if n_new % 25 == 0:
                el = time.time() - t_start
                print(f"{len(done) + n_new}/{len(rows)} poems | {n_tok / el:.1f} tok/s | {el / 3600:.2f} h elapsed", flush=True)
    el = time.time() - t_start
    # Totals over every row in --out (not just this invocation), so resumed runs are counted once.
    all_rows = list(read_jsonl(out))
    secs = sum(r.get("gen_seconds", 0.0) for r in all_rows)
    toks = sum(r.get("gen_tokens", 0) for r in all_rows)
    summary = {"poems": len(all_rows), "seconds": secs, "tokens": toks, "tok_per_sec": toks / max(secs, 1e-9),
               "this_invocation": {"new_poems": n_new, "seconds": el, "tokens": n_tok},
               "model": str(args.model), "adapter": args.adapter, "seed": args.seed}
    (out.parent / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
