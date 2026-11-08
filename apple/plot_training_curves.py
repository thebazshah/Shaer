#!/usr/bin/env python3
"""Plot E3 training curves from runs/e3_train/train_log.jsonl.

Shows the training loss (logged every 10 steps), the eval loss on the 500-prompt subset
(every 200 steps), the official run's eval loss on the full eval split (from the official
repository's training_details.md, for comparison) and the learning rate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import read_jsonl

# Official eval-loss trajectory (Shaer repository, training_details.md, section 14).
OFFICIAL_EVAL = [(0, 3.0803), (200, 2.3408), (400, 2.2992), (600, 2.2703), (800, 2.2623), (1000, 2.2710),
                 (2800, 2.2295), (3000, 2.2074), (3200, 2.2103), (3400, 2.2326), (3600, 2.2288),
                 (3800, 2.2309), (4000, 2.2301), (4200, 2.2318)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/e3_train/train_log.jsonl")
    ap.add_argument("--out", default="runs/training_curves.pdf")
    args = ap.parse_args()
    recs = list(read_jsonl(Path(args.log)))
    tr = [r for r in recs if "train_loss" in r]
    ev = [r for r in recs if "eval_loss" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot([r["step"] for r in tr], [r["train_loss"] for r in tr], lw=0.8, alpha=0.8, label="train loss (ours, M4)")
    ax1.plot([r["step"] for r in ev], [r["eval_loss"] for r in ev], "o-", ms=3, label="eval loss (ours, 500-prompt subset)")
    ax1.plot(*zip(*OFFICIAL_EVAL), "s--", ms=3, color="gray", label="eval loss (official, full eval split)")
    ax1.set_xlabel("optimiser step")
    ax1.set_ylabel("cross-entropy (poem tokens)")
    ax1.set_ylim(bottom=min([r["train_loss"] for r in tr] + [2.0]) - 0.2)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot([r["step"] for r in tr], [r["lr"] for r in tr], color="tab:green")
    ax2.set_xlabel("optimiser step")
    ax2.set_ylabel("learning rate")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
