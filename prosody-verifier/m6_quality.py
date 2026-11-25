#!/usr/bin/env python3
"""M6: the quality study of the final systems: LLM judge, copying and diversity, and a small blinded human study.

  collect        gather one poem per test prompt for every final system into m6/systems/<name>.jsonl (generations
                 schema): the M4 conditions (selected poems), the M5 runs, and the human poems of the same prompts
  human-sheet    a blinded rating sheet (CSV, opens in Google Sheets or Excel): for a seeded sample of prompts, the poem of
                 every chosen system, shuffled, under item codes only; the key (item -> system) goes to a separate file
  human-score    read a filled sheet back, unblind it with the key, and summarise the ratings per system
  summarize      per system, on the prompts every system shares: the paper's five LLM-judge metrics (judged by
                 evaluation/run_llm_judge.py into m6/judge/<name>/judged.jsonl), strict validity (the verifier),
                 copying and diversity, and the human ratings; paired differences to a baseline with bootstrap intervals

The judge sees only the poem and its description, never which system wrote it. The human sheet hides the systems too;
raters should not be told which systems are in it.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import time
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import m3_rescore as m3  # noqa: E402
from m4_report import load_train_lines, quality  # noqa: E402
from prosody.lexicon import Lexicon  # noqa: E402
from prosody.meters import shaer_target  # noqa: E402
from prosody.verify import Verifier  # noqa: E402
from pv_text import split_poem_lines  # noqa: E402

JUDGE_METRICS = ("description_adherence", "meaning", "fluency", "coherence", "poeticness")
HUMAN_COLUMNS = ("overall", "meaning", "fluency", "meter_by_ear")
HUMAN_SCALE = "1 (poor) to 5 (excellent); leave blank if unsure"


def read_jsonl(path: Path) -> list[dict]:
    return m3._read_jsonl(Path(path)) if Path(path).exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------------------------------------- collect
def collect(out_dir: Path, m4_dir: Path | None, m5_dir: Path | None, m4_conditions: list[str] | None = None) -> dict:
    """Copy the final systems' poems into out_dir/systems. M4: the selected files its summary wrote, as m4_<condition>
    (all, or those whose name contains one of m4_conditions). M5: sample 0 of every run, under the run's name
    (e1_released, e3_retrained, m5_2a, ...). Human: the reference poems of the same prompts."""
    sysdir = Path(out_dir) / "systems"
    sysdir.mkdir(parents=True, exist_ok=True)
    found, refs = {}, {}
    if m4_dir and (Path(m4_dir) / "selected").exists():
        for p in sorted((Path(m4_dir) / "selected").glob("*.jsonl")):
            if m4_conditions and not any(c in p.stem for c in m4_conditions):
                continue
            name = f"m4_{p.stem}"
            shutil.copyfile(p, sysdir / f"{name}.jsonl")
            found[name] = str(p)
    if m5_dir and (Path(m5_dir) / "runs").exists():
        for run in sorted((Path(m5_dir) / "runs").iterdir()):
            g = run / "generations.jsonl"
            if g.exists():
                rows = [r for r in read_jsonl(g) if int(r.get("sample_idx", 0) or 0) == 0]
                write_jsonl(sysdir / f"{run.name}.jsonl", rows)       # e1_released, e3_retrained, m5_2a, grpo_v2, ...
                found[run.name] = str(g)
    for name in list(found):
        for r in read_jsonl(sysdir / f"{name}.jsonl"):
            if r.get("reference_completion") and r["id"] not in refs:
                refs[r["id"]] = r
    if refs:
        human = [dict(r, generated_text=r["reference_completion"], sample_idx=0, system="human") for r in refs.values()]
        write_jsonl(sysdir / "human.jsonl", sorted(human, key=lambda r: r["id"]))
        found["human"] = "reference_completion of the same prompts"
    (Path(out_dir) / "systems.json").write_text(json.dumps(found, ensure_ascii=False, indent=1))
    print(json.dumps(found, ensure_ascii=False, indent=1))
    return found


def systems(out_dir: Path) -> dict[str, list[dict]]:
    return {p.stem: read_jsonl(p) for p in sorted((Path(out_dir) / "systems").glob("*.jsonl"))}


def common_ids(sys_rows: dict[str, list[dict]], names: list[str] | None = None) -> list[str]:
    names = names or list(sys_rows)
    sets = [{r["id"] for r in sys_rows[n] if (r.get("generated_text") or "").strip()} for n in names]
    return sorted(set.intersection(*sets)) if sets else []


def judge_input(out_dir: Path, name: str, ids: list[str]) -> Path:
    """The judge's input for one system, limited to the shared prompts (sorted by id, so --limit-rows takes the same
    prompts for every system)."""
    rows = {r["id"]: r for r in read_jsonl(Path(out_dir) / "systems" / f"{name}.jsonl")}
    path = Path(out_dir) / "judge_input" / f"{name}.jsonl"
    write_jsonl(path, [dict(rows[i], sample_idx=0) for i in ids if i in rows])
    return path


# ---------------------------------------------------------------------------------------------- the human study
def human_sheet(out_dir: Path, names: list[str], prompts: int = 40, seed: int = 2026) -> tuple[Path, Path]:
    sys_rows = systems(out_dir)
    missing = [n for n in names if n not in sys_rows]
    if missing:
        raise ValueError(f"no such systems: {missing} (have: {sorted(sys_rows)})")
    ids = common_ids(sys_rows, names)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(ids, min(prompts, len(ids))))
    by = {n: {r["id"]: r for r in sys_rows[n]} for n in names}
    items, key = [], {}
    for pid in chosen:
        order = list(names)
        rng.shuffle(order)
        for n in order:
            code = f"P{chosen.index(pid) + 1:03d}-{len(items) + 1:04d}"
            r = by[n][pid]
            key[code] = {"system": n, "id": pid}
            items.append({"item": code, "prompt": chosen.index(pid) + 1, "meter": r.get("base_meter"), "form": r.get("form"),
                          "requested_bayts": r.get("requested_bayts"),
                          "description": r.get("enhanced_description") or r.get("description") or "",
                          "poem": "\n".join(split_poem_lines(r.get("generated_text") or "")),
                          **{c: "" for c in HUMAN_COLUMNS}, "comment": ""})
    hdir = Path(out_dir) / "human"
    hdir.mkdir(parents=True, exist_ok=True)
    sheet, key_path = hdir / "rating_sheet.csv", hdir / "rating_key.json"
    new_key = json.dumps({"seed": seed, "systems": names, "prompts": chosen, "scale": HUMAN_SCALE, "items": key},
                         ensure_ascii=False, indent=1)
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(items[0]))
    w.writeheader()
    w.writerows(items)
    new_sheet = buf.getvalue()
    if sheet.exists() and key_path.exists():
        same = key_path.read_text(encoding="utf-8") == new_key and sheet.read_text(encoding="utf-8-sig") == new_sheet
        if same:                                       # run again (Run all, a new session): nothing changes, raters keep matching
            print(f"sheet already made and unchanged: {sheet} ({len(items)} poems); key: {key_path}")
            return sheet, key_path
        if any((hdir / "filled").glob("*.csv")):
            raise ValueError("a rating sheet was already made and filled sheets exist, but these systems, prompts or poems "
                             "differ from it: a new sheet would no longer match the raters' item codes. Move the old "
                             "m6/human folder away (or restore the systems) if you really want a new study.")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for f in (sheet, key_path):
            f.rename(f.with_name(f"{f.stem}.old-{stamp}{f.suffix}"))
        print(f"the sheet differed from the earlier one (not yet filled): the earlier one is kept as *.old-{stamp}", flush=True)
    sheet.write_text(new_sheet, encoding="utf-8-sig", newline="")
    key_path.write_text(new_key, encoding="utf-8")
    (hdir / "instructions.md").write_text(
        "# Rating instructions\n\nEach row is one poem written for the description beside it. Rate every poem on "
        f"{', '.join(HUMAN_COLUMNS)}: {HUMAN_SCALE}. 'meter_by_ear': does it sound metrical in the requested meter? "
        "Rate each poem on its own; poems of the same prompt appear in a random order and come from different "
        "writers. Do not look anything up and do not try to guess who wrote a poem.\n", encoding="utf-8")
    print(f"sheet: {sheet} ({len(items)} poems, {len(chosen)} prompts x {len(names)} systems); key (keep it from raters): {key_path}")
    return sheet, key_path


def human_score(sheets: list[Path], key_path: Path, out: Path) -> dict:
    """Ratings from one or more filled copies of the sheet (one per rater): a poem's score is the mean of its raters', a
    system's score the mean over its prompts."""
    key = json.loads(Path(key_path).read_text())
    ratings: dict = {}
    rows_read = 0
    for sheet in sheets:
        with open(sheet, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows_read += 1
                k = key["items"].get(row["item"])
                if not k:
                    continue
                for c in HUMAN_COLUMNS:
                    v = (row.get(c) or "").strip()
                    if v:
                        try:
                            x = float(v)
                        except ValueError:
                            continue
                        if 1 <= x <= 5:
                            ratings.setdefault(k["system"], {}).setdefault(c, {}).setdefault(k["id"], []).append(x)
    res = {"systems": {}, "sheets": [str(x) for x in sheets], "rows_read": rows_read}
    for n, cols in ratings.items():
        res["systems"][n] = {c: {"mean": round(sum(sum(v) / len(v) for v in d.values()) / len(d), 3), "prompts": len(d)}
                             for c, d in cols.items()}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


# ---------------------------------------------------------------------------------------------- summarize
def _boot_mean(values: dict[str, float], ids: list[str], reps: int, seed: int, other: dict | None = None) -> tuple:
    import numpy as np
    ids = [i for i in ids if i in values and (other is None or i in other)]
    if not ids:
        return (float("nan"),) * 3
    a = np.array([values[i] for i in ids])
    if other is not None:
        a = a - np.array([other[i] for i in ids])
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(reps, len(ids)))
    lo, hi = np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
    return (round(float(a.mean()), 3), round(float(lo), 3), round(float(hi), 3))


def summarize(out_dir: Path, baseline: str = "", lexicon: str = "", train: Path | None = None, reps: int = 1000,
              seed: int = 7) -> dict:
    out_dir = Path(out_dir)
    sys_rows = systems(out_dir)
    res = {"systems": {}, "notes": [], "config": {"reps": reps, "seed": seed}}
    if not sys_rows:
        return res
    judged = {}
    for n in sys_rows:
        rows = read_jsonl(out_dir / "judge" / n / "judged.jsonl")
        judged[n] = {str(r.get("judge_source_id") or r.get("id")): r for r in rows}
    judged_names = [n for n in sys_rows if judged[n]]
    jids = sorted(set.intersection(*(set(judged[n]) for n in judged_names))) if judged_names else []
    res["judged_prompts"] = len(jids)
    if judged_names and len(judged_names) < len(sys_rows):
        res["notes"].append(f"not judged yet: {sorted(set(sys_rows) - set(judged_names))}")
    baseline = baseline if baseline in sys_rows else next((n for n in sys_rows if n == "e3_retrained"),
                                                          next((n for n in sys_rows if "i0_single" in n), next(iter(sys_rows))))
    res["baseline"] = baseline
    human = json.loads((out_dir / "human" / "human_scores.json").read_text()) if (out_dir / "human" / "human_scores.json").exists() else {}
    # strict validity on the judged prompts (or on a shared sample of 300 when nothing is judged yet)
    ids = jids or common_ids(sys_rows)[:300]
    v = Verifier(Lexicon.load(lexicon or str(HERE / "data" / "lexicon_train.json.gz")), ignore_diacritics=True, diagnose=False)
    train_lines = load_train_lines(train)
    scores = {}
    for n, rows in sys_rows.items():
        by = {r["id"]: r for r in rows}
        valid = {}
        for i in ids:
            r = by.get(i)
            if not r:
                continue
            meter, form = shaer_target(r["base_meter"], r.get("form") or "تام")
            req = int(r.get("requested_num_lines") or 0) or None
            valid[i] = float(v.poem(split_poem_lines(r.get("generated_text") or ""), meter, form, req).ok)
        jm = {}
        for m in JUDGE_METRICS:
            vals = {i: float(judged[n][i][m]) for i in jids if i in judged[n] and isinstance(judged[n][i].get(m), (int, float))
                    and float(judged[n][i][m]) > 0}
            jm[m] = vals
        scores[n] = {"valid": valid, "judge": jm}
        res["systems"][n] = {
            "strictly_valid_pct": round(100 * sum(valid.values()) / max(1, len(valid)), 2), "poems_checked": len(valid),
            "judge": {m: _boot_mean(jm[m], jids, reps, seed) for m in JUDGE_METRICS} if jids else None,
            "quality": quality([by[i].get("generated_text") or "" for i in ids if i in by],
                               [by[i].get("reference_completion") for i in ids if i in by], train_lines),
            "human": (human.get("systems") or {}).get(n),
        }
    res["vs_baseline"] = {}
    for n in sys_rows:
        if n == baseline:
            continue
        b, a = scores[baseline], scores[n]
        res["vs_baseline"][n] = {
            "strictly_valid_pp": _boot_mean({i: 100 * x for i, x in a["valid"].items()}, ids, reps, seed,
                                            {i: 100 * x for i, x in b["valid"].items()}),
            "judge": ({m: _boot_mean(a["judge"][m], jids, reps, seed, b["judge"][m]) for m in JUDGE_METRICS} if jids else None)}
    return res


def render(res: dict) -> str:
    f = m3._fmt
    out = ["# M6 — quality of the final systems", ""]
    if not res.get("systems"):
        return "\n".join(out + ["No system collected yet (pv.m6_collect)."])
    out += [f"Baseline: **{res['baseline']}**. LLM judge (the paper's: 1-5 per metric) on {res.get('judged_prompts', 0):,} "
            "prompts shared by every judged system; strict validity on the same prompts; 95% bootstrap intervals.", ""]
    out += [f"* note: {n}" for n in res.get("notes", [])]
    out += ["## LLM judge and strict validity", "",
            "| system | strictly valid | " + " | ".join(JUDGE_METRICS) + " |", "|---|---:|" + "---:|" * len(JUDGE_METRICS)]
    for n, s in res["systems"].items():
        j = s["judge"] or {}
        out.append(f"| {n} | {s['strictly_valid_pct']} | " + " | ".join(
            (f"{j[m][0]:.2f} [{j[m][1]:.2f}, {j[m][2]:.2f}]" if j.get(m) and j[m][0] == j[m][0] else "–") for m in JUDGE_METRICS) + " |")
    if res.get("vs_baseline"):
        out += ["", f"## Paired differences to {res['baseline']}", "",
                "| system | strictly valid (pp) | " + " | ".join(JUDGE_METRICS) + " |", "|---|---:|" + "---:|" * len(JUDGE_METRICS)]
        for n, x in res["vs_baseline"].items():
            j = x["judge"] or {}
            out.append(f"| {n} | {f(x['strictly_valid_pp'])} | " + " | ".join(
                (f"{j[m][0]:+.2f} [{j[m][1]:+.2f}, {j[m][2]:+.2f}]" if j.get(m) and j[m][0] == j[m][0] else "–")
                for m in JUDGE_METRICS) + " |")
    qk = ("distinct_2", "distinct_2_within_poem", "repeated_hemistichs_pct", "distinct_rhyme_words_pct",
          "copied_from_reference_pct", "copied_from_train_pct")
    out += ["", "## Copying and diversity", "", "| system | " + " | ".join(qk) + " |", "|---|" + "---:|" * len(qk)]
    for n, s in res["systems"].items():
        out.append(f"| {n} | " + " | ".join(str(s["quality"].get(k, "–")) for k in qk) + " |")
    if any(s.get("human") for s in res["systems"].values()):
        out += ["", "## Blinded human ratings (1-5, mean over prompts)", "",
                "| system | " + " | ".join(HUMAN_COLUMNS) + " |", "|---|" + "---:|" * len(HUMAN_COLUMNS)]
        for n, s in res["systems"].items():
            h = s.get("human") or {}
            out.append(f"| {n} | " + " | ".join(f"{h[c]['mean']} ({h[c]['prompts']})" if c in h else "–" for c in HUMAN_COLUMNS) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--m4", default="")
    c.add_argument("--m5", default="")
    c.add_argument("--m4-conditions", default="", help="comma-separated parts of M4 condition file names to keep")
    j = sub.add_parser("judge-input")
    j.add_argument("--out", required=True)
    j.add_argument("--names", default="")
    h = sub.add_parser("human-sheet")
    h.add_argument("--out", required=True)
    h.add_argument("--names", required=True)
    h.add_argument("--prompts", type=int, default=40)
    h.add_argument("--seed", type=int, default=2026)
    hs = sub.add_parser("human-score")
    hs.add_argument("--out", required=True)
    hs.add_argument("--sheet", action="append", required=True, help="a filled sheet; repeat for several raters")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--baseline", default="")
    s.add_argument("--train", default="")
    s.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(Path(a.out), Path(a.m4) if a.m4 else None, Path(a.m5) if a.m5 else None,
                [x for x in a.m4_conditions.split(",") if x] or None)
    elif a.cmd == "judge-input":
        rows = systems(Path(a.out))
        names = [x for x in a.names.split(",") if x] or list(rows)
        ids = common_ids(rows, names)
        for n in names:
            print(judge_input(Path(a.out), n, ids))
    elif a.cmd == "human-sheet":
        human_sheet(Path(a.out), [x for x in a.names.split(",") if x], a.prompts, a.seed)
    elif a.cmd == "human-score":
        human_score([Path(x) for x in a.sheet], Path(a.out) / "human" / "rating_key.json",
                    Path(a.out) / "human" / "human_scores.json")
    else:
        res = summarize(Path(a.out), a.baseline, train=Path(a.train) if a.train else None, reps=a.reps)
        Path(a.out, "m6_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        text = render(res)
        Path(a.out, "m6_report.md").write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
