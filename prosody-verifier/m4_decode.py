#!/usr/bin/env python3
"""M4 generation on a GPU: whole poems (I0/I1) or the hemistich search (I2), with the released Shaer adapter.

  python m4_decode.py sample --split eval --subset 400 --num-samples 16 --out OUT.jsonl     # I1 (best-of-N pool), I0 = N 1
  python m4_decode.py search --split eval --subset 400 --k 4 --rounds 1 --out OUT.jsonl     # I2

The model, adapter, prompts and decoding are those of evaluation/generate_shaer_test.py (paper Table 15: temperature 0.55,
top-p 0.90, top-k off, repetition penalty 1.08, bf16, prompt tokenised without special tokens); its constants and
checks are imported, not copied. `sample` is that script's decoding, run on a subset and with the samples of one prompt
in the same batch (so all rows of a batch have similar lengths).

`search` writes one hemistich per step (m4_search.SearchRunner). Each step:
  1. the context (prompt + accepted hemistichs, as token ids: never re-tokenised text) of every poem in the batch is
     encoded once, without computing logits;
  2. its key/value cache is repeated k times, and k candidates per poem are sampled from it until a line break or the
     end of text (or --line-max-tokens);
  3. the candidates are checked in sample order by the verifier (worker processes, early exit at the first that passes);
  4. poems with no passing candidate get another round (up to --rounds), then the candidate adding the fewest defects.
The repetition penalty sees the whole context, exactly as in one-shot generation. A candidate that ends the poem early
(end of text before the last hemistich) is kept as a line and the end of text is replaced by a line break.

Output: one JSON line per (prompt, sample) in the schema of generate_shaer_test.py (read by
evaluation/score_meter_count.py), plus "m4" accounting (generated tokens, encoded context tokens, model and checker
seconds). Each finished poem is written at once; re-running resumes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from m4_search import CheckerPool, PoemState, SearchRunner, SHIPPED_LEXICON, Candidate, state_record  # noqa: E402
from prosody.meters import shaer_target  # noqa: E402


def _load_generate_shaer_test():
    """evaluation/generate_shaer_test.py, imported by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("generate_shaer_test", REPO / "evaluation" / "generate_shaer_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GST = _load_generate_shaer_test()


# ---------------------------------------------------------------------------------------------- arguments and rows
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sample", "search", "selfcheck"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="eval", choices=["train", "eval", "test"])
    ap.add_argument("--subset", type=int, default=0, help="a seeded random subset of this many prompts (0: all)")
    ap.add_argument("--subset-seed", type=int, default=2026)
    ap.add_argument("--prompts-jsonl", default="", help="read the prompt rows from this file instead of the dataset (tests)")
    ap.add_argument("--num-samples", type=int, default=1, help="sample mode: poems per prompt")
    ap.add_argument("--k", type=int, default=4, help="search mode: candidates per hemistich and round")
    ap.add_argument("--rounds", type=int, default=1, help="search mode: rounds of k candidates before falling back")
    ap.add_argument("--line-max-tokens", type=int, default=64, help="search mode: tokens per candidate hemistich at most")
    ap.add_argument("--no-ita", action="store_true", help="search mode: do not treat a repeated rhyme word as a defect")
    ap.add_argument("--check", choices=["letters", "binding"], default="letters",
                    help="search mode: verify the letters alone (default) or with the written vowels binding")
    ap.add_argument("--workers", type=int, default=0, help="search mode: checker processes (0: about the CPU count)")
    ap.add_argument("--lexicon", default=str(SHIPPED_LEXICON))
    ap.add_argument("--batch-size", default="auto", help="prompts per batch, or auto (from free GPU memory)")
    ap.add_argument("--max-batch", type=int, default=64)
    # the model and decoding: as generate_shaer_test.py
    ap.add_argument("--base-model", default=GST.BASE_MODEL_ID)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-adapter", action="store_true")
    g.add_argument("--adapter-path", default="")
    ap.add_argument("--adapter-repo", default=GST.ADAPTER_REPO_ID)
    ap.add_argument("--adapter-subfolder", default=GST.ADAPTER_SUBFOLDER)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dataset-id", default=GST.DATASET_ID, help="a Hugging Face dataset or a folder built by m5_data.py")
    ap.add_argument("--max-new-tokens", type=int, default=768, help="sample mode")
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--top-p", type=float, default=0.90)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def select_indices(n_rows: int, subset: int, seed: int) -> list[int]:
    """The prompts of a run: all, or a seeded random subset (the same for every condition with the same seed)."""
    if not subset or subset >= n_rows:
        return list(range(n_rows))
    return sorted(random.Random(seed).sample(range(n_rows), subset))


def prompt_rows(args) -> list[dict]:
    """Rows with the fields generate_shaer_test.py writes, plus the prompt."""
    if args.prompts_jsonl:
        raw = [json.loads(x) for x in open(args.prompts_jsonl, encoding="utf-8") if x.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset(args.dataset_id, split=args.split, token=os.getenv("HF_TOKEN") or None)
        raw = None
    n = len(raw) if raw is not None else len(ds)
    rows = []
    for i in select_indices(n, args.subset, args.subset_seed):
        r = raw[i] if raw is not None else ds[i]
        rows.append({
            "idx": i, "id": r.get("id") if raw is not None and r.get("id") else f"shaer_{args.split}_{i + 1:06d}",
            "sft_prompt": r["sft_prompt"], "base_meter": r["base_meter"], "form": r["form"],
            "requested_bayts": int(r["requested_bayts"]), "requested_num_lines": int(r["sft_num_lines"]),
            "length_bucket": r.get("length_bucket"), "description": r.get("description"),
            "enhanced_description": r.get("enhanced_description"), "reference_completion": r.get("sft_completion"),
            **({"requested_rawi": r["requested_rawi"]} if r.get("requested_rawi") is not None else {}),   # M5 prompts
        })
    return rows


# ---------------------------------------------------------------------------------------------- the model
def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.allow_adapter_mismatch = False         # the released adapter runs only on its own base model, Yehia-7B-preview
    GST.check_adapter_matches_base(args)
    token = os.getenv("HF_TOKEN") or None
    use_cuda = torch.cuda.is_available() and args.device in ("auto", "cuda")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    print(f"loading {args.base_model} in {args.dtype} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=getattr(torch, args.dtype), token=token,
                                                 device_map={"": 0} if use_cuda else {"": "cpu"})
    desc = "none"
    if not args.no_adapter:
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        if args.adapter_path:
            adir, desc = args.adapter_path, args.adapter_path
        else:
            root = snapshot_download(args.adapter_repo, allow_patterns=[f"{args.adapter_subfolder}/*"], token=token)
            adir, desc = str(Path(root) / args.adapter_subfolder), f"{args.adapter_repo}:{args.adapter_subfolder}"
        model = PeftModel.from_pretrained(model, adir)
    model.eval()
    return model, tok, desc, torch.device("cuda" if use_cuda else "cpu")


def inner_model(model):
    """The transformer without the language-model head (LoRA layers stay in place: PEFT injects them into the modules)."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model


def kv_bytes_per_token(model) -> int:
    c = model.config
    head_dim = c.hidden_size // c.num_attention_heads
    return 2 * c.num_hidden_layers * getattr(c, "num_key_value_heads", c.num_attention_heads) * head_dim * 2


def free_memory_tokens(model, device) -> int:
    """How many key/value tokens fit in the GPU memory left (conservative); a large number on CPU."""
    import torch
    if device.type != "cuda":
        return 10 ** 7
    free, _ = torch.cuda.mem_get_info()
    return int(max(0, 0.80 * free - 2.0e9) // kv_bytes_per_token(model))


# ---------------------------------------------------------------------------------------------- the line generator
class LineStop:
    """Per-row stop when the last token is a line break or the end of text (transformers >= 4.39 stops rows one by one)."""

    def __init__(self, stop_mask):
        self.mask = stop_mask

    def __call__(self, input_ids, scores, **kwargs):
        return self.mask[input_ids[:, -1]]


class HFGenerator:
    """propose(states, k): k candidate hemistichs per partial poem, sampled from one shared encoding of its context."""

    def __init__(self, model, tok, device, args, logits_processor=None):
        import torch
        self.torch, self.model, self.tok, self.device, self.args = torch, model, tok, device, args
        self.eos = tok.eos_token_id
        self.pad = tok.pad_token_id
        vocab = len(tok)
        nl = [i for i in range(vocab) if "\n" in tok.decode([i])]
        if not nl:
            raise RuntimeError("the tokenizer has no token containing a line break")
        enc = tok("أ\nب", add_special_tokens=False).input_ids
        self.newline = next((i for i in enc if i in set(nl)), nl[0])
        mask = torch.zeros(max(vocab, model.config.vocab_size), dtype=torch.bool)
        mask[nl] = True
        mask[self.eos] = True
        self.stop = LineStop(mask.to(device))
        self.stop_ids = set(nl) | {self.eos}
        self.logits_processor = logits_processor
        self.calls = 0
        self._last = None       # the encoding of the current step's contexts, reused by its later rounds

    def _decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def encode_contexts(self, contexts: list[list[int]]):
        """Left-padded ids and mask, and the key/value cache of every position but the last (no logits computed)."""
        torch = self.torch
        width = max(len(c) for c in contexts)
        ids = torch.full((len(contexts), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(contexts), width), dtype=torch.long)
        for i, c in enumerate(contexts):
            ids[i, width - len(c):] = torch.tensor(c, dtype=torch.long)
            mask[i, width - len(c):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        from transformers import DynamicCache
        pos = mask.cumsum(-1) - 1
        pos.masked_fill_(mask == 0, 1)
        with torch.inference_mode():
            out = inner_model(self.model)(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], position_ids=pos[:, :-1],
                                          past_key_values=DynamicCache(), use_cache=True, return_dict=True)
        return ids, mask, out.past_key_values

    @staticmethod
    def expand_cache(cache, index):
        from transformers import DynamicCache
        new = DynamicCache()
        for layer in range(len(cache)):
            new.update(cache.key_cache[layer].index_select(0, index), cache.value_cache[layer].index_select(0, index), layer)
        return new

    def sample_lines(self, contexts: list[list[int]], k: int, seed: int):
        """Generated ids (after the context) for k samples of each context, rows ordered context by context, and the
        number of context tokens encoded for each (0 when a later round of the same step reuses the encoding)."""
        torch = self.torch
        keys = [tuple(c) for c in contexts]
        last = self._last
        if last is not None and all(key in last["pos"] for key in keys):
            ids, mask, cache = last["ids"], last["mask"], last["cache"]
            rows, encoded = [last["pos"][key] for key in keys], [0] * len(keys)
        else:
            self._last = None
            ids, mask, cache = self.encode_contexts(contexts)
            self._last = {"pos": {key: i for i, key in enumerate(keys)}, "ids": ids, "mask": mask, "cache": cache}
            rows, encoded = list(range(len(keys))), [len(c) for c in contexts]
        index = torch.tensor(rows, device=self.device).repeat_interleave(k)
        cache_k = self.expand_cache(cache, index)
        ids_k, mask_k = ids.index_select(0, index), mask.index_select(0, index)
        from transformers import StoppingCriteriaList
        torch.manual_seed(seed)
        a = self.args
        with torch.inference_mode():
            sampling = dict(do_sample=True, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k) if a.temperature > 0 \
                else dict(do_sample=False)              # temperature 0: greedy (used by the equivalence tests)
            out = self.model.generate(
                input_ids=ids_k, attention_mask=mask_k, past_key_values=cache_k, **sampling,
                repetition_penalty=a.repetition_penalty, max_new_tokens=a.line_max_tokens,
                stopping_criteria=StoppingCriteriaList([self.stop]), pad_token_id=self.pad, eos_token_id=self.eos,
                use_cache=True, logits_processor=self.logits_processor)
        return out[:, ids.shape[1]:].tolist(), encoded

    def candidate(self, state: PoemState, new: list[int]) -> Candidate:
        stop_at = next((j for j, t in enumerate(new) if t in self.stop_ids), None)
        gen = new if stop_at is None else new[: stop_at + 1]
        eos = stop_at is not None and gen[-1] == self.eos
        completion = state.context[state.prompt_len:]
        before = self._decode(completion)
        full = self._decode(completion + [t for t in gen if t != self.eos])
        line = full.split("\n")[before.count("\n")] if full.startswith(before) else full[len(before):].split("\n")[0]
        ids = [t for t in gen if t != self.eos]
        if not ids or ids[-1] not in self.stop_ids:      # no line break yet (end of text, or the token limit)
            ids = ids + [self.newline]
        return Candidate(line.strip(), ids, len(gen), eos)

    def propose(self, states: list[PoemState], k: int):
        torch = self.torch
        self.calls += 1
        t0 = time.perf_counter()
        try:
            rows, encoded = self.sample_lines([s.context for s in states], k, self.args.seed * 1_000_003 + self.calls)
        except torch.cuda.OutOfMemoryError:
            self._last = None
            if len(states) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(states) // 2
            print(f"out of memory with {len(states)} poems x {k}: retrying as {h} + {len(states) - h}", flush=True)
            a, ia = self.propose(states[:h], k)
            b, ib = self.propose(states[h:], k)
            return a + b, {key: ia[key] + ib[key] for key in ia}
        dt = time.perf_counter() - t0
        props = [[self.candidate(s, rows[i * k + j]) for j in range(k)] for i, s in enumerate(states)]
        return props, {"seconds": [dt / len(states)] * len(states), "prefill_tokens": encoded}


# ---------------------------------------------------------------------------------------------- runs
def read_done(path: Path) -> set:
    return GST.read_done(path)


def plan_batches(rows: list[dict], prompt_lens: list[int], rows_per_prompt: int, line_tokens: int, budget_tokens: int,
                 cap: int, fixed: int = 0) -> list[list[int]]:
    """Batches of similar requests (sorted by requested length, then prompt length): at most `fixed` prompts, or (auto)
    as many as fit, rows_per_prompt rows each, a cache as long as the longest poem of the batch, and at most `cap`."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["requested_num_lines"], prompt_lens[i]))
    batches, cur, cur_width = [], [], 0
    for i in order:
        need = prompt_lens[i] + (rows[i]["requested_num_lines"] + 1) * line_tokens
        width = max(cur_width, need)
        if fixed:
            full = len(cur) >= fixed
        else:
            full = len(cur) >= cap or (len(cur) + 1) * rows_per_prompt * width > budget_tokens
        if cur and full:
            batches.append(cur)
            cur, width = [], need
        cur.append(i)
        cur_width = width
    if cur:
        batches.append(cur)
    return batches


def write_meta(out: Path, args, extra: dict) -> None:
    import torch
    import transformers
    meta_path = out.with_suffix(".meta.json")
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = {"args": vars(args), "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", **extra,
            "runs": old.get("runs", [])}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def add_run(out: Path, summary: dict) -> None:
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["runs"] = meta.get("runs", []) + [summary]
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def _bar(done: int, total: int, t_start: float, new: int, extra: str = "") -> str:
    el = time.time() - t_start
    eta = (total - done) / (new / el) / 60 if new else float("nan")
    filled = int(round(20 * done / max(1, total)))
    return (f"[{'█' * filled}{'░' * (20 - filled)}] {100 * done / max(1, total):5.1f}% {done}/{total} poems | "
            f"{el / 60:.1f} min elapsed | ETA ~{eta:.0f} min{extra}")


def run_sample(args, model, tok, device, rows: list[dict], out: Path) -> None:
    import torch
    lens = [len(tok(r["sft_prompt"], add_special_tokens=False).input_ids) for r in rows]
    n = args.num_samples
    # rows per batch as in generate_shaer_test.py (KV cache of prompt + new tokens, and the fp32 logits of the prompt pass)
    per_batch = (GST.auto_batch_size(model, max(lens), args.max_new_tokens, cap=args.max_batch)
                 if str(args.batch_size).lower() == "auto" else int(args.batch_size))
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["requested_num_lines"], lens[i]))
    pairs = [(i, s) for i in order for s in range(n)]           # one prompt's samples side by side: similar lengths
    batches = [pairs[c:c + per_batch] for c in range(0, len(pairs), per_batch)]
    done = read_done(out)
    total = len(rows) * n
    print(f"sample: {len(rows)} prompts x {n} = {total} poems ({len(done)} done) in {len(batches)} batches of {per_batch}",
          flush=True)
    write_meta(out, args, {"mode": "sample", "n_prompts": len(rows)})
    eos = tok.eos_token_id
    t_start, new_rows, tok_total = time.time(), 0, 0
    with out.open("a", encoding="utf-8") as f:
        for b_no, batch in enumerate(batches):
            todo = [(i, s) for i, s in batch if (rows[i]["id"], s) not in done]
            if not todo:
                continue
            for chunk in (todo,):
                torch.manual_seed(args.seed * 1_000_003 + b_no * 10_007)
                enc = tok([rows[i]["sft_prompt"] for i, _ in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
                enc = {key: v.to(device) for key, v in enc.items()}
                width = int(enc["input_ids"].shape[1])
                t0 = time.time()
                with torch.inference_mode():
                    gen = GST.generate_with_oom_retry(model, enc, dict(
                        do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens,
                        pad_token_id=tok.pad_token_id, eos_token_id=eos, use_cache=True), tok.pad_token_id)
                dt = time.time() - t0
                for j, (i, s) in enumerate(chunk):
                    new = gen[j][width:]
                    n_tok, finished = int(new.shape[0]), False
                    hit = (new == eos).nonzero()
                    if len(hit):
                        n_tok, finished = int(hit[0].item()) + 1, True
                    rec = {k: v for k, v in rows[i].items() if k not in ("idx", "sft_prompt")}
                    rec.update({"sample_idx": s, "generated_text": tok.decode(new, skip_special_tokens=True).strip(),
                                "gen_tokens": n_tok, "finished_with_eos": finished,
                                "m4": {"config": {"method": "sample"}, "prefill_tokens": lens[i],
                                       "gen_seconds": round(dt / len(chunk), 4)}})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    tok_total += n_tok
                f.flush()
                new_rows += len(chunk)
                peak = f" | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device.type == "cuda" else ""
                print(_bar(len(done) + new_rows, total, t_start, new_rows,
                           f" | batch {b_no + 1}/{len(batches)} ({len(chunk)} rows, {dt:.0f}s) | "
                           f"{tok_total / (time.time() - t_start):.0f} tok/s{peak}"), flush=True)
    add_run(out, {"new_rows": new_rows, "seconds": time.time() - t_start, "tokens": tok_total,
                  "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})


def run_search(args, model, tok, device, rows: list[dict], out: Path, pool: CheckerPool, generator=None) -> None:
    import torch
    gen = generator or HFGenerator(model, tok, device, args)
    runner = SearchRunner(gen, pool, args.k, args.rounds)
    prompt_ids = [tok(r["sft_prompt"], add_special_tokens=False).input_ids for r in rows]
    budget = free_memory_tokens(model, device)
    fixed = 0 if str(args.batch_size).lower() == "auto" else int(args.batch_size)
    batches = plan_batches(rows, [len(p) for p in prompt_ids], args.k + 1, 24, budget, args.max_batch, fixed)
    done = read_done(out)
    config = {"method": "search", "k": args.k, "rounds": args.rounds, "ita_defect": not args.no_ita,
              "line_max_tokens": args.line_max_tokens, "check": args.check}
    print(f"search k={args.k} rounds={args.rounds}: {len(rows)} prompts ({len(done)} done) in {len(batches)} batches",
          flush=True)
    write_meta(out, args, {"mode": "search", "n_prompts": len(rows), "config": config, "newline_id": gen.newline})
    t_start, new_rows, stats = time.time(), 0, {"hemistichs": 0, "first": 0, "forced": 0, "tokens": 0}
    with out.open("a", encoding="utf-8") as f:
        for b_no, batch in enumerate(batches):
            todo = [i for i in batch if (rows[i]["id"], 0) not in done]
            if not todo:
                continue
            states = []
            for i in todo:
                meter, form = shaer_target(rows[i]["base_meter"], rows[i].get("form") or "تام")
                states.append(PoemState(rows[i]["id"], meter, form, rows[i]["requested_num_lines"], list(prompt_ids[i]),
                                        len(prompt_ids[i])))
            last = [time.time()]
            written = set()

            def progress(sts, b_no=b_no, todo=todo):
                for s, i in zip(sts, todo):
                    if s.done and s.key not in written:
                        rec = state_record(s, rows[i], config)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written.add(s.key)
                        stats["hemistichs"] += len(s.log)
                        stats["first"] += sum(1 for c, r, p in s.log if p and c == 1)
                        stats["forced"] += len(s.forced)
                        stats["tokens"] += s.gen_tokens
                f.flush()
                if time.time() - last[0] > 20:
                    step = max(len(s.lines) for s in sts)
                    print(f"  batch {b_no + 1}/{len(batches)}: hemistich {step}/{max(s.requested_lines for s in sts)}",
                          flush=True)
                    last[0] = time.time()

            runner.run(states, progress)
            new_rows += len(states)
            h = max(1, stats["hemistichs"])
            peak = f" | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device.type == "cuda" else ""
            print(_bar(len(done) + new_rows, len(rows), t_start, new_rows,
                       f" | batch {b_no + 1}/{len(batches)} ({len(states)} poems) | first candidate passes "
                       f"{100 * stats['first'] / h:.0f}% | forced {100 * stats['forced'] / h:.1f}% of hemistichs | "
                       f"{stats['tokens'] / (time.time() - t_start):.0f} tok/s{peak}"), flush=True)
    add_run(out, {"new_rows": new_rows, "seconds": time.time() - t_start, **stats,
                  "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})


def run_selfcheck(args, model, tok, device, rows: list[dict], out: Path) -> dict:
    """On the real model: (1) the next-token logits from the shared cache against a full forward pass, on contexts that
    end inside a poem; (2) greedy decoding one hemistich at a time with a checker that accepts everything, against
    one-shot greedy decoding of the same prompts. Writes a JSON summary to --out."""
    import torch
    from transformers import StoppingCriteriaList

    class AcceptAll:
        def run(self, jobs):
            from m4_search import Check
            return [(0, [Check(True, 0, 0.0)]) for _ in jobs]

    args.temperature = 0.0
    gen = HFGenerator(model, tok, device, args)
    ctxs = []
    for r in rows:
        p = tok(r["sft_prompt"], add_special_tokens=False).input_ids
        ref = tok(" " + "\n".join((r.get("reference_completion") or "").split("\n")[:2]), add_special_tokens=False).input_ids
        ctxs.append(p + ref[: max(1, len(ref) // 2)])
    diffs, agree = [], 0
    for c0 in range(0, len(ctxs), 4):
        chunk = ctxs[c0:c0 + 4]
        ids, mask, cache = gen.encode_contexts(chunk)
        pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
        with torch.inference_mode():
            full = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits[:, -1].float()
            step = model(input_ids=ids[:, -1:], attention_mask=mask, position_ids=pos[:, -1:], past_key_values=cache,
                         use_cache=True).logits[:, -1].float()
        diffs += (full - step).abs().max(dim=-1).values.tolist()
        agree += int((full.argmax(-1) == step.argmax(-1)).sum())
    n_lines = 4
    states = [PoemState(r["id"], *shaer_target(r["base_meter"], r.get("form") or "تام"), n_lines,
                        tok(r["sft_prompt"], add_special_tokens=False).input_ids,
                        len(tok(r["sft_prompt"], add_special_tokens=False).input_ids)) for r in rows]
    SearchRunner(gen, AcceptAll(), k=1, rounds=1).run(states)
    same, compared = 0, 0
    for s, r in zip(states, rows):
        enc = tok(r["sft_prompt"], return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            one = model.generate(**enc, do_sample=False, repetition_penalty=args.repetition_penalty,
                                 max_new_tokens=n_lines * args.line_max_tokens, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)[0, enc["input_ids"].shape[1]:].tolist()
        lines = [x.strip() for x in tok.decode(one, skip_special_tokens=True).split("\n")]
        k = min(len(lines), n_lines) - (0 if tok.eos_token_id in one else 1)   # a line cut by the token limit is not compared
        if k <= 0:
            continue
        compared += 1
        same += s.lines[:k] == lines[:k]
    res = {"contexts": len(ctxs), "logit_max_abs_diff": round(max(diffs), 4), "logit_mean_abs_diff": round(sum(diffs) / len(diffs), 4),
           "argmax_agreement_pct": round(100 * agree / len(ctxs), 2), "greedy_poems_compared": compared,
           "greedy_identical_pct": round(100 * same / max(1, compared), 2), "dtype": args.dtype}
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = prompt_rows(args)
    pool = None
    if args.mode == "search":      # the checker processes start before CUDA is initialised
        workers = args.workers or max(1, min(12, (os.cpu_count() or 2) - 1))
        print(f"starting {workers} checker processes ...", flush=True)
        pool = CheckerPool(args.lexicon, workers=workers, ita=not args.no_ita, binding=args.check == "binding")
    model, tok, desc, device = load_model(args)
    print(f"adapter: {desc}; device: {device}; {len(rows)} prompts", flush=True)
    try:
        if args.mode == "sample":
            run_sample(args, model, tok, device, rows, out)
        elif args.mode == "selfcheck":
            run_selfcheck(args, model, tok, device, rows, out)
        else:
            run_search(args, model, tok, device, rows, out, pool)
    finally:
        if pool:
            pool.close()


if __name__ == "__main__":
    main()
