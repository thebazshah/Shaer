"""Shared constants and helpers for the Shaer reproduction on Apple Silicon.

Part of the Apple port in the fork github.com/thebazshah/Shaer (folder apple/).
Everything here mirrors the official repository (github.com/AhmaddAbbass/Shaer, commit bba73cb):
dataset ids, the sampler-weight formula (sft/split_utils.py) and the packing routine
(sft/train_sft.py::pack_preprocessed_dataset).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

DATASET_ID = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
RELEASED_GENERATIONS_ID = "Shaer-AI/shaer-sft-test"
BASE_MODEL_ID = "Navid-AI/Yehia-7B-preview"
ADAPTER_REPO_ID = "Shaer-AI/Shaer-adapters"
ADAPTER_SUBFOLDER = "adapters/fresh_sft/train/best"
METER_MODEL_ID = "Shaer-AI/4BiLSTM-meter-classification-pytorch"

MAX_SEQ_LENGTH = 1024
WEIGHT_ALPHA = 0.6
SEED = 42

# Decoding settings from the paper (Appendix I, Table 15) for Shaer and Yehia-7B.
DECODING = {"temperature": 0.55, "top_p": 0.90, "repetition_penalty": 1.08, "max_tokens": 768}

LENGTH_BUCKETS = ["1-3", "4-6", "7-10", "11-20"]


def tokenize_example(tokenizer, prompt: str, full_text: str, max_len: int = MAX_SEQ_LENGTH):
    """Return (input_ids, mask, truncated) exactly as train_sft.py builds (input_ids, labels).

    mask[i] == 1 where labels[i] != -100 (poem tokens), 0 on prompt tokens.
    Tokenisation uses add_special_tokens=False because the official prompt text already
    starts with '<s>'. Truncation keeps the end of the sequence ('keep_end').
    """
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    truncated = max_len > 0 and len(full_ids) > max_len
    if truncated:
        trim = len(full_ids) - max_len
        full_ids = full_ids[-max_len:]
        prompt_len = max(0, len(prompt_ids) - trim)
    else:
        prompt_len = min(len(prompt_ids), len(full_ids))
    mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
    return full_ids, mask, truncated


def pack_by_group(examples: list[dict], max_len: int = MAX_SEQ_LENGTH) -> list[dict]:
    """Port of pack_preprocessed_dataset(): concatenate examples of the same sampler_group
    (groups in sorted order, rows in dataset order) into sequences of exactly max_len
    tokens; an example may be split across two packed sequences; the last partial
    sequence of each group is kept; sequences without supervised tokens are dropped."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_group[ex["sampler_group"]].append(ex)
    packed: list[dict] = []
    for group in sorted(by_group):
        cur_ids: list[int] = []
        cur_mask: list[int] = []

        def flush():
            nonlocal cur_ids, cur_mask
            if cur_ids and sum(cur_mask) > 0:
                packed.append({"input_ids": cur_ids, "mask": cur_mask, "sampler_group": group})
            cur_ids, cur_mask = [], []

        for ex in by_group[group]:
            ids, mask = ex["input_ids"], ex["mask"]
            cursor = 0
            while cursor < len(ids):
                take = min(max_len - len(cur_ids), len(ids) - cursor)
                cur_ids = cur_ids + ids[cursor:cursor + take]
                cur_mask = cur_mask + mask[cursor:cursor + take]
                cursor += take
                if len(cur_ids) == max_len:
                    flush()
        flush()
    return packed


def sampler_weights(groups: Iterable[str], alpha: float = WEIGHT_ALPHA) -> list[float]:
    """Port of split_utils.build_train_weights(): w(g) = (1/count(g))**alpha, normalised to mean 1."""
    groups = list(groups)
    counts = Counter(groups)
    raw = {g: (1.0 / c) ** alpha for g, c in counts.items()}
    mean_w = sum(raw.values()) / len(raw)
    return [raw[g] / mean_w for g in groups]


def length_bucket(requested_bayts: int) -> str:
    n = int(requested_bayts)
    if n <= 3:
        return "1-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    return "11-20"


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
