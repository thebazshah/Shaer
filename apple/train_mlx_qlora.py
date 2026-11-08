#!/usr/bin/env python3
"""QLoRA-style fine-tuning of Shaer on Apple Silicon with MLX.

Replaces the CUDA-only parts of sft/train_sft.py (bitsandbytes NF4 + PEFT + HF Trainer +
paged_adamw_8bit) while keeping the official recipe:
  * frozen 4-bit base (MLX affine, group 64; produced by `mlx_lm.convert -q`),
  * LoRA r=64 on q,k,v,o,gate,up,down in every block, dropout 0.05,
    scale = alpha/sqrt(r) = 128/8 = 16 (RS-LoRA, as use_rslora=True in the official config),
  * AdamW, lr 8e-5, linear warmup 3% + cosine decay, weight decay 0, grad-norm clip 1.0,
  * batch 1 x 16 gradient-accumulation steps, packed 1024-token sequences,
  * completion-only loss (prompt tokens masked), weighted sampler re-drawn every epoch,
  * eval every 200 steps (fixed eval subset), checkpoints every 200 steps, best = lowest eval loss.

Usage:
  python train_mlx_qlora.py --model models/yehia-7b-mlx-4bit --data data --out runs/e3_train
  (re-run the same command to resume from the latest checkpoint)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from common import SEED, read_jsonl

LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MLX 4-bit base model dir")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/e3_train")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=128.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--rslora", type=int, default=1)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--grad-checkpoint", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="debug/benchmark: stop after N optimiser steps")
    ap.add_argument("--final-eval", type=int, default=1, help="evaluate the best adapter on the full eval and test splits")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def masked_loss(model, tokens, mask):
    """Mean cross-entropy over poem tokens only (labels == -100 on the prompt in the official code)."""
    logits = model(tokens[:, :-1]).astype(mx.float32)
    targets = tokens[:, 1:]
    m = mask[:, 1:].astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, targets, reduction="none")
    ntok = m.sum()
    return (ce * m).sum() / mx.maximum(ntok, 1.0), ntok


def epoch_order(weights: np.ndarray, epoch: int, seed: int) -> np.ndarray:
    """WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), one draw per epoch."""
    rng = np.random.default_rng(seed + epoch)
    p = weights / weights.sum()
    return rng.choice(len(weights), size=len(weights), replace=True, p=p)


def lr_schedule(total_steps: int, lr: float, warmup_ratio: float):
    warmup = max(1, int(math.ceil(total_steps * warmup_ratio)))
    return optim.join_schedules(
        [optim.linear_schedule(0.0, lr, warmup), optim.cosine_decay(lr, max(1, total_steps - warmup), 0.0)],
        [warmup],
    ), warmup


def evaluate(model, rows) -> float:
    total, ntok = 0.0, 0.0
    for r in rows:
        toks = mx.array([r["input_ids"]])
        m = mx.array([r["mask"]])
        loss, n = masked_loss(model, toks, m)
        mx.eval(loss, n)
        total += loss.item() * n.item()
        ntok += n.item()
    return total / max(ntok, 1.0)


def save_adapter(model, path: Path, adapter_config: dict, state: dict):
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (path / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    (path / "trainer_state.json").write_text(json.dumps(state, indent=2))


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)

    data = np.load(Path(args.data) / "train_packed.npz")
    tokens_np, mask_np, weights, lengths_np = data["tokens"], data["mask"], data["weights"], data["lengths"]
    eval_rows = list(read_jsonl(Path(args.data) / "eval_subset.jsonl"))
    n_seq = tokens_np.shape[0]
    steps_per_epoch = math.ceil(n_seq / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    model, tokenizer = load(args.model)
    model.freeze()
    scale = args.alpha / math.sqrt(args.rank) if args.rslora else args.alpha / args.rank
    lora_cfg = {"rank": args.rank, "scale": scale, "dropout": args.dropout, "keys": LORA_KEYS}
    linear_to_lora_layers(model, len(model.layers), lora_cfg)
    if args.grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint
        grad_checkpoint(model.layers[0])
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable parameters: {n_train / 1e6:.1f}M | packed sequences {n_seq} | steps {total_steps}")
    adapter_config = {"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": lora_cfg,
                      "base_model": str(args.model), "source": "assignment-02 MLX port of Shaer sft/train_sft.py"}

    schedule, warmup = lr_schedule(total_steps, args.lr, args.warmup_ratio)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)

    # ---- resume ----
    state = {"step": 0, "best_eval_loss": None, "best_step": None, "checkpoints": []}
    latest = out / "latest"
    if (latest / "trainer_state.json").exists():
        state = json.loads((latest / "trainer_state.json").read_text())
        model.load_weights(str(latest / "adapters.safetensors"), strict=False)
        if (latest / "optimizer.safetensors").exists():
            optimizer.init(model.trainable_parameters())
            opt_state = mx.load(str(latest / "optimizer.safetensors"))
            from mlx.utils import tree_unflatten
            optimizer.state = tree_unflatten(list(opt_state.items()))
        print(f"resumed from step {state['step']}")

    loss_and_grad = nn.value_and_grad(model, masked_loss)
    log_f = (out / "train_log.jsonl").open("a", encoding="utf-8")

    def log(rec):
        rec["time"] = time.time()
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    if state["step"] == 0 and not (out / "eval_step0.done").exists():
        e0 = evaluate(model, eval_rows)
        log({"step": 0, "eval_loss": e0})
        (out / "eval_step0.done").write_text(str(e0))
        print(f"step 0 eval loss {e0:.4f}")

    step = state["step"]
    t0, tok_count, losses = time.time(), 0, []
    model.train()
    while step < total_steps:
        epoch = step // steps_per_epoch
        order = epoch_order(weights, epoch, args.seed)
        start = (step % steps_per_epoch) * args.grad_accum
        for s in range(start, len(order), args.grad_accum):
            if step >= total_steps:
                break
            idx = order[s:s + args.grad_accum]
            acc_grads, acc_loss = None, 0.0
            for i in idx:
                L = int(lengths_np[i])
                toks = mx.array(tokens_np[i:i + 1, :L])
                m = mx.array(mask_np[i:i + 1, :L])
                (loss, _), grads = loss_and_grad(model, toks, m)
                # transformers 4.43 (pinned by the official repo) averages the per-sequence mean losses
                # over the accumulation steps, so every packed sequence gets equal weight.
                acc_grads = grads if acc_grads is None else tree_map(mx.add, acc_grads, grads)
                acc_loss = acc_loss + loss
                mx.eval(acc_grads, acc_loss)
                tok_count += L
            k = len(idx)
            acc_grads = tree_map(lambda g: g / k, acc_grads)
            acc_grads, gnorm = optim.clip_grad_norm(acc_grads, args.max_grad_norm)
            optimizer.update(model, acc_grads)
            mx.eval(model.trainable_parameters(), optimizer.state)
            step += 1
            losses.append((acc_loss / k).item())

            if step % args.log_every == 0:
                dt = time.time() - t0
                peak = mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory() / 1e9
                rec = {"step": step, "epoch": step / steps_per_epoch, "train_loss": float(np.mean(losses)),
                       "lr": float(schedule(step)), "grad_norm": float(gnorm.item()),
                       "tokens_per_sec": tok_count / dt, "peak_mem_gb": peak}
                log(rec)
                print(json.dumps(rec))
                t0, tok_count, losses = time.time(), 0, []

            if step % args.eval_every == 0 or step == total_steps:
                model.eval()
                ev = evaluate(model, eval_rows)
                model.train()
                log({"step": step, "eval_loss": ev})
                print(f"step {step} eval loss {ev:.4f}")
                if state["best_eval_loss"] is None or ev < state["best_eval_loss"]:
                    state["best_eval_loss"], state["best_step"] = ev, step
                    save_adapter(model, out / "best", adapter_config, {**state, "step": step})

            if step % args.save_every == 0 or step == total_steps:
                state["step"] = step
                ck = out / f"checkpoint-{step}"
                save_adapter(model, ck, adapter_config, state)
                # keep the newest --keep-last checkpoints found on disk (robust to resumes and moved run dirs)
                cks = sorted((d for d in out.glob("checkpoint-*") if d.is_dir()), key=lambda d: int(d.name.split("-")[1]))
                for old in cks[:-args.keep_last] if args.keep_last > 0 else []:
                    for f in old.glob("*"):
                        f.unlink()
                    old.rmdir()
                state["checkpoints"] = [d.name for d in cks[-args.keep_last:]] if args.keep_last > 0 else [d.name for d in cks]
                save_adapter(model, ck, adapter_config, state)
                save_adapter(model, latest, adapter_config, state)
                mx.save_safetensors(str(latest / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    log({"event": "done", "step": step, "best_eval_loss": state["best_eval_loss"], "best_step": state["best_step"]})
    if args.final_eval and (out / "best" / "adapters.safetensors").exists():
        model.load_weights(str(out / "best" / "adapters.safetensors"), strict=False)
        model.eval()
        full_eval = evaluate(model, read_jsonl(Path(args.data) / "eval.jsonl"))
        test_loss = evaluate(model, read_jsonl(Path(args.data) / "test.jsonl"))
        res = {"best_step": state["best_step"], "full_eval_loss": full_eval, "test_loss": test_loss}
        (out / "test_loss.json").write_text(json.dumps(res, indent=2))
        log({"event": "final_eval", **res})
        print(res)
    print("done", state)


if __name__ == "__main__":
    main()
