#!/usr/bin/env python3
"""Export every generated poem of a split with its input prompt and its scores, one file per run plus one for all runs, for
analysis and for the research write-up (JSONL, and CSV that opens in Google Sheets / Excel).

For each row of <runs-dir>/<run>/generations.jsonl:
  * the prompt the model was given (the row's own `sft_prompt` when the run stored it, else looked up from the dataset by id:
    shaer_<split>_<n> is row n-1 of that split; a run named in --variant-data uses that M5 training-set folder instead),
  * the poem, the human poem of the same prompt, the requested meter / form / length,
  * the repository's scorer fields (scored.jsonl): meter conformity, parsed lines, count adherence,
  * the verifier's verdicts in both modes (m3/<run>/letters.jsonl and binding.jsonl): strictly valid, meter, rhyme, length,
    bayts on meter, issue kinds,
  * the generation accounting (tokens; for the search: candidates, rounds, forced hemistichs).

  python export_samples.py --runs-dir m4/eval/runs --m3-dir m4/eval/m3 --split eval --out-dir m4/eval/export
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COLUMNS = ["run", "id", "sample_idx", "base_meter", "form", "requested_bayts", "requested_lines", "lines", "prompt",
           "generated_text", "reference_completion", "valid_letters", "valid_binding", "meter_ok_letters", "rhyme_ok_letters",
           "count_ok_letters", "meter_ok_binding", "bayts_on_meter_letters", "bayts", "issues_letters", "bilstm_meter_score",
           "count_adherence", "gen_tokens", "candidates", "rounds", "forced_hemistichs"]


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:         # a line cut by an interrupted write
                        pass
    return out


def key_of(row: dict) -> str:
    return f"{row.get('id')}#{int(row.get('sample_idx', 0) or 0)}"


def prompt_lookup(source: str, split: str, cache: dict) -> dict:
    """{id: prompt} of one split of a dataset (a Hugging Face id or a folder built by m5_data.py), loaded once."""
    if source not in cache:
        import m5_data
        rows = m5_data.load_source(source)[split]
        cache[source] = {f"shaer_{split}_{i + 1:06d}": r.get("sft_prompt") for i, r in enumerate(rows)}
    return cache[source]


def export_run(run_dir: Path, m3_dir: Path, run: str, prompts: dict | None) -> list[dict]:
    gens = read_jsonl(run_dir / "generations.jsonl")
    scored = {key_of(r): r for r in read_jsonl(run_dir / "scored.jsonl")}
    modes = {m: {r.get("key"): r for r in read_jsonl(m3_dir / run / f"{m}.jsonl")} for m in ("letters", "binding")}
    rows = []
    for g in gens:
        k = key_of(g)
        s, lt, bd = scored.get(k, {}), modes["letters"].get(k, {}), modes["binding"].get(k, {})
        m4 = g.get("m4") or {}
        tgt = (lt.get("target") or {}).get("bayt_ok") or []
        rows.append({
            "run": run, "id": g.get("id"), "sample_idx": int(g.get("sample_idx", 0) or 0),
            "base_meter": g.get("base_meter"), "form": g.get("form"), "requested_bayts": g.get("requested_bayts"),
            "requested_lines": g.get("requested_num_lines"), "lines": lt.get("lines", s.get("parsed_num_lines")),
            "prompt": g.get("sft_prompt") or (prompts or {}).get(g.get("id")),
            "generated_text": g.get("generated_text"), "reference_completion": g.get("reference_completion"),
            "valid_letters": lt.get("ok"), "valid_binding": bd.get("ok"), "meter_ok_letters": lt.get("meter_ok"),
            "rhyme_ok_letters": lt.get("rhyme_ok"), "count_ok_letters": lt.get("count_ok"), "meter_ok_binding": bd.get("meter_ok"),
            "bayts_on_meter_letters": sum(bool(x) for x in tgt) if tgt else None, "bayts": len(tgt) if tgt else None,
            "issues_letters": ",".join(lt.get("issues") or []) if lt else None,
            "bilstm_meter_score": s.get("meter"), "count_adherence": s.get("count_adherence"),
            "gen_tokens": g.get("gen_tokens"), "candidates": m4.get("candidates"), "rounds": m4.get("rounds"),
            "forced_hemistichs": len(m4.get("forced") or []) if m4 else None,
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:         # utf-8-sig: Excel / Sheets read the Arabic correctly
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--m3-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-id", default="Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits")
    ap.add_argument("--variant-data", action="append", default=[], help="RUN=FOLDER: the training-set folder whose prompts RUN used")
    a = ap.parse_args()
    runs_dir, out = Path(a.runs_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    special = dict(x.split("=", 1) for x in a.variant_data)
    cache: dict = {}
    allrows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if (p / "generations.jsonl").exists()):
        gens = read_jsonl(run_dir / "generations.jsonl")
        need = any(not g.get("sft_prompt") for g in gens)
        prompts = None
        if need:
            try:
                prompts = prompt_lookup(special.get(run_dir.name, a.dataset_id), a.split, cache)
            except Exception as exc:                                   # offline: export without prompts, say so
                print(f"  ! {run_dir.name}: prompts could not be looked up ({exc.__class__.__name__}: {str(exc)[:120]})")
        rows = export_run(run_dir, Path(a.m3_dir), run_dir.name, prompts)
        missing = sum(1 for r in rows if not r["prompt"])
        write_csv(out / f"{run_dir.name}.csv", rows)
        with open(out / f"{run_dir.name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        allrows += rows
        print(f"  {run_dir.name}: {len(rows):,} poems" + (f", {missing:,} without a prompt" if missing else ""), flush=True)
    write_csv(out / "all_runs.csv", allrows)
    with open(out / "all_runs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in allrows)
    print(f"exported {len(allrows):,} poems of {len({r['run'] for r in allrows})} runs -> {out}")


if __name__ == "__main__":
    main()
