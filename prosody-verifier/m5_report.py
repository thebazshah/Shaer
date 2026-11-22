#!/usr/bin/env python3
"""M5 summary: single samples of each trained variant (2a, 2b, 2c rounds) against the baselines, on the same test prompts.

Input: an M5 folder with runs/<system>/{generations,scored}.jsonl and m3/<system>/{letters,binding}.jsonl (as for M4),
and the verifier's labels of the test split (m5/labels/test.jsonl: the rawī of each human poem). Systems:
  e1_released   the released Shaer adapter, the paper's prompt (the baseline's E1 poems, read from its folder)
  e3_retrained  the adapter re-trained by the baseline notebook, the paper's prompt (if E3 was run)
  grpo_v2       the authors' best GRPO adapter, the paper's prompt (optional)
  m5_<variant>  an adapter trained here; its prompts name the rawī of the human poem

  python m5_report.py summarize --dir M5DIR/test --labels M5DIR/labels/test.jsonl --baseline e3_retrained

Every comparison is paired on the prompts all systems share, one sample per prompt (no selection), with bootstrap
intervals and an exact McNemar test on strict validity. "rawī as requested": the poem's rawī (verifier) is the rawī of the
human poem of the prompt; for the baselines, which were not asked for one, it shows how often they hit it unasked.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))

import m3_rescore as m3  # noqa: E402
from m4_report import MARGIN, boot, load_run, mcnemar, per_poem, quality  # noqa: E402

KEYS = ("strict_valid", "valid_ignoring_length", "meter_ok", "rhyme_ok", "count_exact", "bayt_ok", "no_reading", "bilstm_bayt_ok")


def _rawi_hit(rec: dict, want: str) -> bool:
    return bool(want) and rec["rhyme"].get("rawi") == want


def summarize(m5_dir: Path, labels: Path, baseline: str = "", train: Path | None = None, reps: int = 1000,
              seed: int = 7) -> dict:
    import summarize_structural as ss
    from m4_report import load_train_lines
    m5_dir = Path(m5_dir)
    names = sorted(p.name for p in (m5_dir / "runs").iterdir()
                   if (p / "generations.jsonl").exists() and (m5_dir / "m3" / p.name / "letters.jsonl").exists()) \
        if (m5_dir / "runs").exists() else []
    res = {"systems": {}, "notes": [], "checks": {}, "config": {"margin": MARGIN, "reps": reps, "seed": seed}}
    if not names:
        return res
    runs = {n: load_run(m5_dir, n) for n in names}
    first = {n: {k[0]: k for k in r["recs"]["letters"] if k[1] == 0} for n, r in runs.items()}
    common = sorted(set.intersection(*(set(f) for f in first.values())))
    res["prompts"] = len(common)
    want = {}
    for r in m3._read_jsonl(Path(labels)) if labels and Path(labels).exists() else []:
        want[f"shaer_test_{int(r['index']) + 1:06d}"] = r.get("rawi") or ""
    if not want:
        res["notes"].append("no test labels: rawī as requested is not reported (run pv.m5_relabel)")
    baseline = baseline if baseline in names else ("e3_retrained" if "e3_retrained" in names else
                                                   "e1_released" if "e1_released" in names else names[0])
    res["baseline"] = baseline
    train_lines = load_train_lines(train)
    pp_all = {}
    for n in names:
        run, sel = runs[n], {i: first[n][i] for i in common}
        L = {i: run["recs"]["letters"][k] for i, k in sel.items()}
        B = {i: run["recs"]["binding"].get(k) for i, k in sel.items()}
        pp = {i: per_poem(r, MARGIN) for i, r in L.items()}
        for i in pp:
            pp[i]["rawi_as_requested"] = _rawi_hit(L[i], want.get(i, ""))
        pp_all[n] = pp
        pp_b = {i: per_poem(r, MARGIN) for i, r in B.items() if r is not None}
        pp_inf = {i: per_poem(r, math.inf) for i, r in L.items()}
        rows = [run["scored"].get(k) for k in sel.values()]
        official = None
        if rows and all(r is not None and r.get("meter") not in (None, "") for r in rows):
            official = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ss.summarise(rows).items()}
        gens = [run["gens"].get(k) or {} for k in sel.values()]
        asked = [i for i in common if want.get(i)]
        hit = lambda d: {i: {"strict_valid": d[i]["rawi_as_requested"]} for i in asked}
        out = res["systems"][n] = {
            "letters": {k: boot(pp, common, k, reps, seed) for k in KEYS},
            "rawi_as_requested": m3.bootstrap(hit(pp), asked, "strict_valid", reps, seed) if asked else None,
            "asked_for_rawi": sum(1 for g in gens if g.get("requested_rawi")),
            "letters_no_margin": {k: boot(pp_inf, common, k, reps, seed) for k in ("strict_valid", "meter_ok")},
            "binding": ({k: boot(pp_b, common, k, reps, seed) for k in ("strict_valid", "meter_ok", "bayt_ok")}
                        if len(pp_b) == len(common) else None),
            "official_structural": official,
            "quality": quality([g.get("generated_text") or "" for g in gens], [g.get("reference_completion") for g in gens],
                               train_lines),
            "gen_tokens": round(sum(int(g.get("gen_tokens") or 0) for g in gens) / max(1, len(gens)), 1),
        }
        by_b = defaultdict(list)
        for i in common:
            by_b[m3.bucket(L[i]["requested_bayts"])].append(i)
        out["by_length"] = {b: {"poems": len(v), "strict_valid": round(100 * sum(pp[i]["strict_valid"] for i in v) / len(v), 2),
                                "rhyme_ok": round(100 * sum(pp[i]["rhyme_ok"] for i in v) / len(v), 2)}
                            for b, v in sorted(by_b.items(), key=lambda kv: m3.BUCKETS.index(kv[0]) if kv[0] in m3.BUCKETS else 9)}
        res["checks"][n] = {"missing_binding_records": len(common) - len(pp_b),
                            "missing_scored_rows": sum(r is None for r in rows),
                            "m5_prompts_without_rawi": (sum(1 for g in gens if not g.get("requested_rawi"))
                                                        if n.startswith("m5_") else None)}
    res["vs_baseline"] = {}
    for n in names:
        if n == baseline:
            continue
        a = [pp_all[n][i]["strict_valid"] for i in common]
        b = [pp_all[baseline][i]["strict_valid"] for i in common]
        asked = [i for i in common if want.get(i)]
        hit = lambda d: {i: {"strict_valid": d[i]["rawi_as_requested"]} for i in asked}
        res["vs_baseline"][n] = {**{k: boot(pp_all[n], common, k, reps, seed, other=pp_all[baseline])
                                    for k in ("strict_valid", "valid_ignoring_length", "meter_ok", "rhyme_ok", "bilstm_bayt_ok")},
                                 "rawi_as_requested": (m3.bootstrap(hit(pp_all[n]), asked, "strict_valid", reps, seed,
                                                                    hit(pp_all[baseline])) if asked else None),
                                 "mcnemar_strict_valid": mcnemar(a, b)}
    return res


def render(res: dict) -> str:
    f = m3._fmt
    out = ["# M5 — self-training on verifier-approved poems: results", ""]
    if not res.get("systems"):
        return "\n".join(out + ["No system has verifier records yet."])
    out += [f"{res['prompts']:,} test prompts shared by every system; one sample per prompt; letters mode, margin "
            f"{res['config']['margin']}; 95% bootstrap intervals over prompts. Baseline: **{res['baseline']}**.", ""]
    out += [f"* note: {n}" for n in res.get("notes", [])]
    out += ["## Results", "",
            "| system | strictly valid | valid ignoring length | meter | rhyme | rawī as requested | exact length | bayts on meter | BiLSTM bayts | gen tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, s in res["systems"].items():
        L = s["letters"]
        out.append(f"| {n} | {f(L['strict_valid'])} | {f(L['valid_ignoring_length'])} | {f(L['meter_ok'])} | {f(L['rhyme_ok'])} | "
                   f"{f(s['rawi_as_requested'])} | {f(L['count_exact'])} | {f(L['bayt_ok'])} | {f(L['bilstm_bayt_ok'])} | {s['gen_tokens']} |")
    out += ["", "**Independent views**", "",
            "| system | Shaer meter conformity | base-meter accuracy | strictly valid, no margin | strictly valid, binding |",
            "|---|---:|---:|---:|---:|"]
    for n, s in res["systems"].items():
        o, b = s["official_structural"] or {}, s["binding"] or {}
        out.append(f"| {n} | {o.get('meter_conformity', '–')} | {o.get('base_meter_accuracy_pct', '–')} | "
                   f"{f(s['letters_no_margin']['strict_valid'])} | {f(b.get('strict_valid'))} |")
    if res.get("vs_baseline"):
        out += ["", f"## Paired differences to {res['baseline']} (percentage points)", "",
                "| system | strictly valid | valid ignoring length | meter | rhyme | rawī as requested | BiLSTM bayts | McNemar (system only / baseline only, p) |",
                "|---|---:|---:|---:|---:|---:|---:|---|"]
        for n, x in res["vs_baseline"].items():
            mc = x["mcnemar_strict_valid"]
            out.append(f"| {n} | {f(x['strict_valid'])} | {f(x['valid_ignoring_length'])} | {f(x['meter_ok'])} | {f(x['rhyme_ok'])} | "
                       f"{f(x['rawi_as_requested'])} | {f(x['bilstm_bayt_ok'])} | {mc['a_only']} / {mc['b_only']}, p = {mc['p']:.2g} |")
    buckets = [b for b in m3.BUCKETS if any(b in s["by_length"] for s in res["systems"].values())]
    out += ["", "## By requested length (strictly valid / rhyme)", "", "| system | " + " | ".join(f"{b} bayts" for b in buckets) + " |",
            "|---|" + "---:|" * len(buckets)]
    for n, s in res["systems"].items():
        out.append(f"| {n} | " + " | ".join(f"{s['by_length'][b]['strict_valid']} / {s['by_length'][b]['rhyme_ok']}"
                                             if b in s["by_length"] else "–" for b in buckets) + " |")
    qk = ("distinct_2", "distinct_2_within_poem", "repeated_hemistichs_pct", "distinct_rhyme_words_pct",
          "copied_from_reference_pct", "copied_from_train_pct")
    out += ["", "## Quality and copying", "", "| system | " + " | ".join(qk) + " |", "|---|" + "---:|" * len(qk)]
    for n, s in res["systems"].items():
        out.append(f"| {n} | " + " | ".join(str(s["quality"].get(k, "–")) for k in qk) + " |")
    out += ["", "## Integrity checks (all must be 0)", "",
            "| system | missing binding records | missing scored rows | M5 prompts without a rawī |", "|---|---:|---:|---:|"]
    for n, x in res["checks"].items():
        out.append(f"| {n} | {x['missing_binding_records']} | {x['missing_scored_rows']} | "
                   f"{'–' if x['m5_prompts_without_rawi'] is None else x['m5_prompts_without_rawi']} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--dir", required=True)
    s.add_argument("--labels", default="")
    s.add_argument("--baseline", default="")
    s.add_argument("--train", default="")
    s.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    res = summarize(Path(a.dir), Path(a.labels) if a.labels else None, a.baseline, Path(a.train) if a.train else None, a.reps)
    Path(a.dir, "m5_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    text = render(res)
    Path(a.dir, "m5_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
