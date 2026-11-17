#!/usr/bin/env python3
"""Generate poems for the 3,481 held-out Shaer test prompts with Hugging Face `generate` (E1 / E2 / E3).

The repository ships the released generations (Shaer-AI/shaer-sft-test) but no script that produces them, so this one
fills the gap. Prompts are the official `sft_prompt` strings of the test split of
Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits; row i gets the id
`shaer_test_{i+1:06d}`, which matches the released dataset row for row.

Modes
  E1  released adapter   : --adapter-repo Shaer-AI/Shaer-adapters --adapter-subfolder adapters/fresh_sft/train/best   (default)
  E3  re-trained adapter : --adapter-path sft/outputs/train/<run>/checkpoint-<best>      (a local PEFT checkpoint)
  E2  untuned base model : --no-adapter

Decoding follows paper Table 15: temperature 0.55, top-p 0.90, repetition penalty 1.08, 768 new tokens, stop at EOS.
Top-k is "not set" in the paper and therefore follows the inference framework's default. Hugging Face defaults to
top-k 50 whereas vLLM disables it, so this script disables it (--top-k 0) unless told otherwise.

The base model is loaded in bf16 without quantisation by default (an A100 80GB has room; this is also how the repo's
vLLM path serves the adapter). Use --quantize nf4 to run on the same NF4 base the adapter was trained on.
The prompt is tokenised with add_special_tokens=False like the training data: the prompt text already starts with "<s>".

Output: one JSON line per (prompt, sample) with the fields read by evaluation/score_meter_count.py. Re-running resumes.
Score with:  python evaluation/score_meter_count.py --input-jsonl OUT --output-jsonl OUT_scored
and summarise with evaluation/summarize_structural.py.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

DATASET_ID = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
BASE_MODEL_ID = "Navid-AI/Yehia-7B-preview"      # the paper's base model (gated)
YEHIA_BASE_MODEL_ID = BASE_MODEL_ID
ADAPTER_REPO_ID = "Shaer-AI/Shaer-adapters"
ADAPTER_SUBFOLDER = "adapters/fresh_sft/train/best"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSONL (resumable)")
    ap.add_argument("--base-model", default=BASE_MODEL_ID)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-adapter", action="store_true", help="E2: generate with the untuned base model")
    g.add_argument("--adapter-path", default="", help="local PEFT adapter / checkpoint directory (E3)")
    ap.add_argument("--adapter-repo", default=ADAPTER_REPO_ID)
    ap.add_argument("--adapter-subfolder", default=ADAPTER_SUBFOLDER)
    ap.add_argument("--allow-adapter-mismatch", action="store_true",
                    help="exploratory only: apply the released adapter (trained on Yehia-7B-preview) to a different base model")
    ap.add_argument("--merge", action="store_true", help="merge the adapter into the weights (faster, tiny rounding change)")
    ap.add_argument("--quantize", choices=["none", "nf4"], default="none")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="auto", help="auto (GPU if present) | cuda | cpu")
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=1, help="samples per prompt (rows get sample_idx)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--top-p", type=float, default=0.90)
    ap.add_argument("--top-k", type=int, default=0, help="0 disables top-k (vLLM-like); HF's own default is 50")
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument("--add-special-tokens", action="store_true", help="let the tokenizer prepend BOS (training did not)")
    ap.add_argument("--use-fast-tokenizer", action="store_true", help="training used use_fast=False (ignored by transformers>=5)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def check_adapter_matches_base(args: argparse.Namespace) -> None:
    """The released Shaer adapter was trained on Yehia-7B-preview's weights. On any other base model (e.g. its parent
    ALLaM-7B-Instruct-preview) it is not the paper's E1, so refuse unless explicitly asked for an exploratory run."""
    uses_released_adapter = not args.no_adapter and not args.adapter_path and args.adapter_repo == ADAPTER_REPO_ID
    if uses_released_adapter and args.base_model != YEHIA_BASE_MODEL_ID and not args.allow_adapter_mismatch:
        raise SystemExit(
            f"refusing to apply the released adapter ({ADAPTER_REPO_ID}) to base model '{args.base_model}': it was trained on "
            f"'{YEHIA_BASE_MODEL_ID}'. Use --adapter-path with your own adapter (E3), --no-adapter (E2), "
            "or --allow-adapter-mismatch for an exploratory run that is not the paper's E1.")


def read_done(path: Path) -> set[tuple[str, int]]:
    """Ids already generated. A truncated last line (interrupted write, e.g. a Colab disconnect) is dropped from the file."""
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    kept, dropped = [], 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r["id"], int(r.get("sample_idx", 0))))
                kept.append(line if line.endswith("\n") else line + "\n")
            except (json.JSONDecodeError, KeyError):
                dropped += 1
    if dropped:
        print(f"dropping {dropped} unreadable line(s) from {path}", flush=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        tmp.replace(path)
    return done


def main() -> None:
    args = parse_args()
    check_adapter_matches_base(args)
    import torch
    import transformers
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.getenv("HF_TOKEN") or None
    use_cuda = torch.cuda.is_available() and args.device in ("auto", "cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = getattr(torch, args.dtype)

    ds = load_dataset(args.dataset_id, split=args.split, token=token)
    rows = [{"idx": i, "id": f"shaer_test_{i + 1:06d}", "row": ds[i]} for i in range(len(ds))]
    if args.limit_rows:
        rows = rows[: args.limit_rows]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=args.use_fast_tokenizer, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: dict = {"torch_dtype": dtype, "token": token}
    if args.quantize == "nf4":
        if not use_cuda:
            raise SystemExit("--quantize nf4 needs a CUDA GPU")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    load_kwargs["device_map"] = {"": 0} if use_cuda else {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    adapter_desc = "none"
    if not args.no_adapter:
        from peft import PeftModel

        if args.adapter_path:
            adapter_dir, adapter_desc = args.adapter_path, args.adapter_path
        else:
            root = snapshot_download(args.adapter_repo, allow_patterns=[f"{args.adapter_subfolder}/*"], token=token)
            adapter_dir = str(Path(root) / args.adapter_subfolder)
            adapter_desc = f"{args.adapter_repo}:{args.adapter_subfolder}"
        model = PeftModel.from_pretrained(model, adapter_dir)
        if args.merge:
            if args.quantize != "none":
                raise SystemExit("--merge is only supported on an unquantised base")
            model = model.merge_and_unload()
    model.eval()

    # longest prompts first -> the first batch reveals the memory peak; batches are fixed so a resume reuses the same seeds
    lengths = [len(tokenizer(r["row"]["sft_prompt"], add_special_tokens=args.add_special_tokens)["input_ids"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -lengths[i])
    batches = [order[k:k + args.batch_size] for k in range(0, len(order), args.batch_size)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    meta = {
        "args": vars(args), "adapter": adapter_desc,
        "paper_flow": args.base_model == YEHIA_BASE_MODEL_ID,   # False => substitute base model, not the paper's setup "n_prompts": len(rows), "max_prompt_tokens": max(lengths),
        "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    eos = tokenizer.eos_token_id
    t_start, n_new, tok_total = time.time(), 0, 0
    with out.open("a", encoding="utf-8") as f:
        for sample_idx in range(args.num_samples):
            for b_no, batch in enumerate(batches):
                todo = [i for i in batch if (rows[i]["id"], sample_idx) not in done]
                if not todo:
                    continue
                torch.manual_seed(args.seed * 1_000_003 + sample_idx * 10_007 + b_no)
                prompts = [rows[i]["row"]["sft_prompt"] for i in todo]
                enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=args.add_special_tokens)
                enc = {k: v.to(device) for k, v in enc.items()}
                width = int(enc["input_ids"].shape[1])
                t0 = time.time()
                with torch.inference_mode():
                    gen = model.generate(
                        **enc, do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
                dt = time.time() - t0
                for j, i in enumerate(todo):
                    new = gen[j][width:]
                    n_tok = int(new.shape[0])
                    finished = False
                    hit = (new == eos).nonzero()
                    if len(hit):
                        n_tok, finished = int(hit[0].item()) + 1, True
                    r = rows[i]["row"]
                    rec = {
                        "id": rows[i]["id"], "sample_idx": sample_idx, "base_meter": r["base_meter"], "form": r["form"],
                        "requested_bayts": int(r["requested_bayts"]), "requested_num_lines": int(r["sft_num_lines"]),
                        "length_bucket": r.get("length_bucket"), "description": r.get("description"),
                        "enhanced_description": r.get("enhanced_description"), "reference_completion": r["sft_completion"],
                        "generated_text": tokenizer.decode(new, skip_special_tokens=True).strip(),
                        "gen_tokens": n_tok, "finished_with_eos": finished,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    tok_total += n_tok
                f.flush()
                n_new += len(todo)
                el = time.time() - t_start
                print(f"sample {sample_idx} batch {b_no + 1}/{len(batches)} | {n_new} new rows | batch {dt:.0f}s | "
                      f"{tok_total / el:.0f} tok/s | {el / 60:.1f} min elapsed", flush=True)
    summary = {"new_rows": n_new, "seconds": time.time() - t_start, "tokens": tok_total,
               "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta_path = out.with_suffix(".meta.json")
    previous = json.loads(meta_path.read_text()).get("runs", []) if meta_path.exists() else []
    meta["runs"] = previous + [summary]  # one entry per invocation, so resumed runs keep their history
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
