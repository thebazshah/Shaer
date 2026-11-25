#!/usr/bin/env python3
"""M5 data (idea 2): relabel the Shaer SFT corpus with the verifier, and build the training sets of 2a, 2b and 2c.

  relabel   every poem of the train / eval / test splits through the verifier (letters mode): its rawī (rhyme letter),
            rhyme verdict, meter verdict at margin 1.5 and with no margin, strict validity. Resumable, parallel.
  build     a training set in a folder the unchanged sft/train_sft.py loads with SFT_DATASET_ID=<folder> (Parquet splits
            train / eval / test and a README.md that names them), in a new folder named after a hash of its inputs, so a
            changed set never reuses an old data cache:
              2a  every prompt gets the line "حرف الروي: <letter>" (the rawī of its own poem) before "الموضوع:"; the
                  training text is the new prompt followed by the unchanged completion
              2b  2a with the train split limited to poems the verifier accepts (meter in the labelled form with no margin,
                  and rhyme); eval and test stay those of 2a, so eval losses compare
              2c  poems the model wrote with the hemistich search for train prompts that passed the quality gate
                  (`accept`), mixed with original train rows of the base set; eval and test are the base set's
  accept    the quality gate of 2c over a search run on train prompts: complete without a forced hemistich, exactly the
            requested length, strictly valid for the evaluation verifier (letters, margin 1.5), the rawī that was asked
            for, no hemistich repeated in the poem, none copied from the training poems, at least 3 words per hemistich.

The eval and test prompts of 2a carry the rawī of the human poem of the same prompt (column requested_rawi), so a model
trained on 2a is asked for that rhyme at test time and "rawī as requested" can be measured.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prosody import arudi  # noqa: E402
from prosody.lexicon import Lexicon  # noqa: E402
from prosody.meters import shaer_target  # noqa: E402
from prosody.verify import Verifier  # noqa: E402
from pv_text import split_poem_lines  # noqa: E402

SHIPPED_LEXICON = HERE / "data" / "lexicon_train.json.gz"
DATASET_ID = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"   # as generate_shaer_test.py
SPLITS = ("train", "eval", "test")
RAWI_LABEL = "حرف الروي"
TOPIC_LINE = "\nالموضوع:"
README = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: eval
    path: eval.parquet
  - split: test
    path: test.parquet
---
Built by prosody-verifier/m5_data.py; see manifest.json.
"""


# ---------------------------------------------------------------------------------------------- prompts
def add_rawi(prompt: str, rawi: str) -> str:
    """The prompt with the line 'حرف الروي: <rawi>' inserted before the topic line (the rest is unchanged)."""
    i = prompt.find(TOPIC_LINE)
    if i < 0:
        raise ValueError("prompt without a topic line")
    return prompt[: i + 1] + f"{RAWI_LABEL}: {rawi}\n" + prompt[i + 1:]


def with_prompt(row: dict, prompt: str, completion: str | None = None) -> dict:
    """A copy of a training row with a new prompt (and optionally a new completion), the training text rebuilt the way the
    corpus builds it: the prompt, then the completion exactly as it follows the prompt in the original text."""
    old_prompt, old_full = row["sft_prompt"], row["sft_full_text"]
    if not old_full.startswith(old_prompt):
        raise ValueError("sft_full_text does not start with sft_prompt")
    tail = old_full[len(old_prompt):]
    if completion is not None:
        old_completion = row["sft_completion"]
        if old_completion not in tail:
            raise ValueError("sft_completion is not in sft_full_text")
        tail = tail.replace(old_completion, completion, 1)
    out = dict(row)
    out["sft_prompt"], out["sft_full_text"] = prompt, prompt + tail
    if completion is not None:
        out["sft_completion"] = completion
    return out


# ---------------------------------------------------------------------------------------------- loading
def load_source(source: str) -> dict[str, list[dict]]:
    """The splits as lists of rows: a Hugging Face dataset id, a folder built by `build`, or a folder of Parquet files
    named <split>*.parquet (tests)."""
    p = Path(source)
    if p.is_dir() and not (p / "README.md").exists():
        import pyarrow.parquet as pq
        out = {}
        for split in SPLITS:
            files = sorted(glob.glob(str(p / f"{split}*.parquet")))
            out[split] = [r for f in files for r in pq.read_table(f).to_pylist()]
        return out
    from datasets import load_dataset
    ds = load_dataset(source, token=os.getenv("HF_TOKEN") or None)
    return {split: ds[split].to_list() for split in SPLITS}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


# ---------------------------------------------------------------------------------------------- relabel
_V: dict = {}


def _init(lex: str) -> None:
    lexicon = Lexicon.load(lex)
    _V["v"] = Verifier(lexicon, ignore_diacritics=True, diagnose=False)
    _V["inf"] = Verifier(lexicon, ignore_diacritics=True, diagnose=False, margin=math.inf).share_cache(_V["v"])


def label_row(split: str, index: int, row: dict) -> dict:
    meter, form = shaer_target(row["base_meter"], row.get("form") or "تام")
    lines = split_poem_lines(row.get("sft_completion") or "")
    req = int(row.get("sft_num_lines") or 0) or None
    pv = _V["v"].poem(lines, meter, form, req)
    meter_inf = pv.meter_ok or _V["inf"].poem(lines, meter, form, req).meter_ok
    return {"split": split, "index": index, "rawi": pv.rhyme.get("rawi") or "", "rhyme_ok": pv.rhyme_ok,
            "meter_ok": pv.meter_ok, "meter_ok_no_margin": meter_inf, "strict_valid": pv.ok, "lines": len(lines)}


def _label_chunk(job):
    split, items = job
    return [label_row(split, i, r) for i, r in items]


def relabel(source: str, out_dir: Path, lexicon: str, workers: int, chunk: int = 200, limit: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_source(source)
    meta = {"source": source, "lexicon_sha": _sha_file(Path(lexicon)),
            "verifier_sha": hashlib.sha256(b"".join(p.read_bytes() for p in sorted((HERE / "prosody").glob("*.py")))).hexdigest()[:16]}
    meta_p = out_dir / "labels.meta.json"
    if meta_p.exists() and json.loads(meta_p.read_text()) != meta:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for split in SPLITS:
            f = out_dir / f"{split}.jsonl"
            if f.exists():
                f.rename(out_dir / f"{split}.stale-{stamp}.jsonl")
        print(f"labels made by another verifier, lexicon or source were set aside (*.stale-{stamp}.jsonl)", flush=True)
    meta_p.write_text(json.dumps(meta, indent=1))
    _init(lexicon)
    for split in SPLITS:
        rows = data[split][: limit or None]
        path = out_dir / f"{split}.jsonl"
        kept = read_jsonl(path)
        if path.exists():                 # drop a line cut by an interrupted write: appending would glue a record onto it
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
            tmp.replace(path)
        done = {r["index"] for r in kept}
        todo = [(i, r) for i, r in enumerate(rows) if i not in done]
        print(f"{split}: {len(rows):,} poems, {len(done):,} already labelled, {len(todo):,} to label", flush=True)
        jobs = [(split, todo[k:k + chunk]) for k in range(0, len(todo), chunk)]
        t0, n = time.time(), 0
        with open(path, "a", encoding="utf-8") as f:
            it = Pool(workers, initializer=_init, initargs=(lexicon,)).imap_unordered(_label_chunk, jobs) if workers > 1 \
                else map(_label_chunk, jobs)
            for res in it:
                f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in res))
                f.flush()
                n += len(res)
                rate = n / max(1e-9, time.time() - t0)
                print(f"  {split}: {len(done) + n:,}/{len(rows):,} poems, {rate:.1f}/s, about "
                      f"{(len(todo) - n) / max(rate, 1e-9) / 60:.0f} min left", flush=True)


def load_labels(labels_dir: Path) -> dict[str, dict[int, dict]]:
    return {split: {r["index"]: r for r in read_jsonl(Path(labels_dir) / f"{split}.jsonl")} for split in SPLITS}


# ---------------------------------------------------------------------------------------------- the 2c gate
def _bare(line: str) -> str:
    return " ".join(arudi._fold(arudi.strip_marks(w)) for w in arudi.tokenize(line))


def train_line_keys(rows: list[dict]) -> set:
    keys = set()
    for r in rows:
        for line in split_poem_lines(r.get("sft_completion") or ""):
            k = _bare(line)
            if len(k) >= 8:
                keys.add(k)
    return keys


def gate(gen: dict, requested_rawi: str, verifier: Verifier, train_keys: set) -> str:
    """'' when a self-generated poem may join the 2c training set, else the first reason it may not."""
    m4 = gen.get("m4") or {}
    lines = split_poem_lines(gen.get("generated_text") or "")
    req = int(gen.get("requested_num_lines") or 0)
    if m4.get("forced"):
        return "forced hemistich"
    if len(lines) != req or m4.get("empty_lines"):
        return "length"
    meter, form = shaer_target(gen["base_meter"], gen.get("form") or "تام")
    pv = verifier.poem(lines, meter, form, req)
    if not pv.ok:
        return "not strictly valid"
    if requested_rawi and pv.rhyme.get("rawi") != requested_rawi:
        return "another rawī"
    keys = [_bare(x) for x in lines]
    if len(set(keys)) < len(keys):
        return "repeated hemistich"
    if any(k in train_keys for k in keys if len(k) >= 8):
        return "hemistich copied from the training poems"
    if any(len(arudi.tokenize(x)) < 3 for x in lines):
        return "hemistich under 3 words"
    return ""


def accept(gens_path: Path, source: str, base_dir: Path, lexicon: str, out: Path) -> dict:
    """The 2c gate over a search run on train prompts (ids shaer_train_<index + 1>). Writes accepted rows and counts."""
    base = load_source(str(base_dir))
    train_keys = train_line_keys(load_source(source)["train"])
    v = Verifier(Lexicon.load(lexicon), ignore_diacritics=True, diagnose=False)
    reasons, kept = {}, []
    for g in read_jsonl(gens_path):
        if int(g.get("sample_idx", 0) or 0) != 0:
            continue
        index = int(str(g["id"]).rsplit("_", 1)[1]) - 1
        row = base["train"][index]
        why = gate(g, row.get("requested_rawi") or "", v, train_keys)
        reasons[why or "accepted"] = reasons.get(why or "accepted", 0) + 1
        if not why:
            kept.append({"index": index, "generated_text": "\n".join(split_poem_lines(g["generated_text"])),
                         "requested_rawi": row.get("requested_rawi") or ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    summary = {"generations": str(gens_path), "poems": sum(reasons.values()), "accepted": len(kept), "reasons": reasons}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


# ---------------------------------------------------------------------------------------------- build
def _rawi_rows(rows: list[dict], labels: dict[int, dict]) -> tuple[list[dict], int]:
    out, missing = [], 0
    for i, r in enumerate(rows):
        rawi = (labels.get(i) or {}).get("rawi") or ""
        if rawi:
            nr = with_prompt(r, add_rawi(r["sft_prompt"], rawi))
        else:
            nr, missing = dict(r), missing + 1
        nr["requested_rawi"] = rawi
        out.append(nr)
    return out, missing


def _write(out_root: Path, variant: str, inputs: dict, splits: dict[str, list[dict]], manifest: dict) -> Path:
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    d = Path(out_root) / f"{variant}-{key}"
    if (d / "_COMPLETE").exists():
        print(f"already built: {d}", flush=True)
        return d
    import pyarrow as pa
    import pyarrow.parquet as pq
    tmp = d.with_name(d.name + ".building")
    tmp.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for rows in splits.values() for r in rows for k in r})
    for split, rows in splits.items():
        table = pa.Table.from_pylist([{c: r.get(c) for c in columns} for r in rows])
        pq.write_table(table, tmp / f"{split}.parquet")
    (tmp / "README.md").write_text(README, encoding="utf-8")
    manifest = {**manifest, "variant": variant, "inputs": inputs, "rows": {s: len(r) for s, r in splits.items()},
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (tmp / "_COMPLETE").write_text("")
    if d.exists():
        import shutil
        shutil.rmtree(d)
    tmp.rename(d)
    print(f"built {variant}: {d} {manifest['rows']}", flush=True)
    return d


def build(variant: str, source: str, labels_dir: Path, out_root: Path, base_dir: Path | None = None,
          accepted: Path | None = None, mix: float = 1.0, seed: int = 2026) -> Path:
    code = _sha_file(Path(__file__))
    labels_meta = json.loads((Path(labels_dir) / "labels.meta.json").read_text())
    label_shas = {s: _sha_file(Path(labels_dir) / f"{s}.jsonl") for s in SPLITS}
    inputs = {"variant": variant, "source": source, "labels": labels_meta, "label_files": label_shas, "code": code}
    if variant in ("2a", "2b"):
        data, labels = load_source(source), load_labels(labels_dir)
        splits, missing = {}, {}
        for s in SPLITS:
            if len(labels[s]) < len(data[s]):
                raise RuntimeError(f"{s}: {len(labels[s]):,} of {len(data[s]):,} poems labelled; run relabel first")
            splits[s], missing[s] = _rawi_rows(data[s], labels[s])
        manifest = {"no_rawi_rows": missing}
        if variant == "2b":
            before = len(splits["train"])
            splits["train"] = [r for i, r in enumerate(splits["train"])
                               if labels["train"][i]["meter_ok_no_margin"] and labels["train"][i]["rhyme_ok"]]
            manifest["train_kept"] = {"before": before, "after": len(splits["train"]),
                                      "rule": "meter in the labelled form with no margin, and the rhyme check passes"}
        return _write(out_root, variant, inputs, splits, manifest)
    if variant.startswith("2c"):
        if base_dir is None or accepted is None:
            raise ValueError("2c needs --base (the 2a or 2b folder) and --accepted (the gate's output)")
        base = load_source(str(base_dir))
        acc = read_jsonl(Path(accepted))
        inputs.update(base=Path(base_dir).name, accepted=_sha_file(Path(accepted)), mix=mix, seed=seed)
        new_rows = [with_prompt(base["train"][a["index"]], base["train"][a["index"]]["sft_prompt"], a["generated_text"])
                    for a in acc]
        for r in new_rows:
            r["m5_source"] = "self_generated"
        taken = {a["index"] for a in acc}
        pool = [i for i in range(len(base["train"])) if i not in taken]
        k = min(len(pool), int(round(mix * len(new_rows))))
        originals = [dict(base["train"][i], m5_source="original") for i in sorted(random.Random(seed).sample(pool, k))]
        splits = {"train": new_rows + originals, "eval": base["eval"], "test": base["test"]}
        manifest = {"self_generated": len(new_rows), "original": len(originals), "base": str(base_dir)}
        return _write(out_root, variant, inputs, splits, manifest)
    raise ValueError(f"unknown variant {variant!r} (2a, 2b, 2c...)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relabel")
    r.add_argument("--source", default=DATASET_ID)
    r.add_argument("--out", required=True)
    r.add_argument("--lexicon", default=str(SHIPPED_LEXICON))
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("--variant", required=True)
    b.add_argument("--source", default=DATASET_ID)
    b.add_argument("--labels", required=True)
    b.add_argument("--out-root", required=True)
    b.add_argument("--base", default="")
    b.add_argument("--accepted", default="")
    b.add_argument("--mix", type=float, default=1.0)
    b.add_argument("--record", default="", help="write the built folder's path to this file")
    a = sub.add_parser("accept")
    a.add_argument("--generations", required=True)
    a.add_argument("--source", default=DATASET_ID)
    a.add_argument("--base", required=True)
    a.add_argument("--lexicon", default=str(SHIPPED_LEXICON))
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "relabel":
        relabel(args.source, Path(args.out), args.lexicon, args.workers or max(1, (os.cpu_count() or 2) - 1), limit=args.limit)
    elif args.cmd == "build":
        d = build(args.variant, args.source, Path(args.labels), Path(args.out_root), Path(args.base) if args.base else None,
                  Path(args.accepted) if args.accepted else None, args.mix)
        if args.record:
            Path(args.record).write_text(str(d))
    else:
        accept(Path(args.generations), args.source, Path(args.base), args.lexicon, Path(args.out))


if __name__ == "__main__":
    main()
