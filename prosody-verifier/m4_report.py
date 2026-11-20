#!/usr/bin/env python3
"""M4 summary: best-of-N (I0/I1) against the hemistich search (I2) at equal compute.

Input: an M4 folder with
    runs/<run>/generations.jsonl     m4_decode.py output (sample: several samples per prompt; search: one)
    runs/<run>/scored.jsonl          the repository's scorer on it (BiLSTM meter, line count)
    m3/<run>/{letters,binding}.jsonl m3_rescore.py score records (the M3 verifier, both reading modes)

  python m4_report.py summarize --dir M4DIR [--train train.jsonl] [--reps 1000]

Conditions:
  I1 best-of-n   for every n = 1, 2, 4, ... up to the samples of a sample run: per prompt, among samples 0..n-1, the first
                 strictly valid one (letters mode, margin 1.5) or else the one with the fewest defects. n = 1 is I0, the
                 paper's single sample. The budget of best-of-n is the tokens of all n samples.
  I2 k=…, R=…    a search run.
Every comparison is paired on the prompts all conditions share. Strict validity is reported with the length requirement
and without it (the search writes exactly the requested number of hemistichs by construction, so its length gain is
shown apart). The verifier that selects is the one that scores, so the independent views are reported next to it: the
BiLSTM metric of the Shaer paper, binding mode (written vowels count) and no margin.

Matched compute: each I2 run is compared with the smallest n whose best-of-n used at least as many generated tokens per
prompt (I1 gets at least the same budget), with paired bootstrap intervals and an exact McNemar test.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "evaluation"))

import m3_rescore as m3  # noqa: E402
from prosody import arudi  # noqa: E402
from pv_text import split_poem_lines  # noqa: E402

MARGIN = m3.DEFAULT_MARGIN
REPORT_KEYS = ("strict_valid", "valid_ignoring_length", "meter_ok", "rhyme_ok", "count_exact", "bayt_ok", "no_reading",
               "bilstm_bayt_ok")


# ---------------------------------------------------------------------------------------------- loading
def _jsonl(path: Path) -> list[dict]:
    return m3._read_jsonl(Path(path)) if Path(path).exists() else []


def load_run(m4_dir: Path, name: str) -> dict:
    run = m4_dir / "runs" / name
    gens = {(r["id"], int(r.get("sample_idx", 0) or 0)): r for r in _jsonl(run / "generations.jsonl")}
    scored = {(r["id"], int(r.get("sample_idx", 0) or 0)): r for r in _jsonl(run / "scored.jsonl")}
    recs = {}
    for mode in m3.MODES:
        recs[mode] = {(r["id"], int(r["sample_idx"])): r for r in _jsonl(m4_dir / "m3" / name / f"{mode}.jsonl")
                      if "error" not in r}
    meta_p = run / "generations.meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return {"name": name, "gens": gens, "scored": scored, "recs": recs, "meta": meta}


def defects_of(rec: dict) -> int:
    """Defects counted from an M3 record, as in the search: failing bayts plus every issue kind present."""
    return sum(not x for x in rec["target"]["bayt_ok"]) + len(rec["rhyme"]["issues"]) + len(rec["issues"])


def rank(rec: dict) -> tuple:
    return (not rec["ok"], defects_of(rec), abs(rec["lines"] - (rec["requested_lines"] or rec["lines"])) + (rec["lines"] % 2))


# ---------------------------------------------------------------------------------------------- conditions
def conditions(m4_dir: Path) -> tuple[dict, list]:
    """{condition name: {id: (run, key)}} and a list of notes; I1 from every sample run, I2 from every search run."""
    out, notes, runs = {}, [], {}
    for p in sorted((m4_dir / "runs").iterdir()) if (m4_dir / "runs").exists() else []:
        if not (p / "generations.jsonl").exists():
            continue
        r = load_run(m4_dir, p.name)
        if not r["recs"]["letters"]:
            notes.append(f"{p.name}: no verifier records yet (run m3_rescore.py score on it)")
            continue
        runs[p.name] = r
        mode = r["meta"].get("mode") or ("search" if any((g.get("m4") or {}).get("config", {}).get("method") == "search"
                                                         for g in r["gens"].values()) else "sample")
        by_id = defaultdict(list)
        for key in r["recs"]["letters"]:
            by_id[key[0]].append(key)
        if mode == "sample":
            n_max = min(len(v) for v in by_id.values())
            n = 1
            while n <= n_max:
                sel = {}
                for i, keys in by_id.items():
                    pool = sorted(k for k in keys if k[1] < n)
                    sel[i] = min(pool, key=lambda k: (rank(r["recs"]["letters"][k]), k[1]))
                label = "I0 single sample" if n == 1 else f"I1 best-of-{n}"
                out[f"{label} [{p.name}]"] = {"run": p.name, "n": n, "sel": sel}
                n *= 2
        else:
            cfg = (r["meta"].get("config") or {})
            label = f"I2 k={cfg.get('k', '?')} R={cfg.get('rounds', '?')}" + (" binding" if cfg.get("check") == "binding" else "")
            out[f"{label} [{p.name}]"] = {"run": p.name, "n": 1, "sel": {i: sorted(v)[0] for i, v in by_id.items()}}
    return {"conditions": out, "runs": runs}, notes


def _budget(run: dict, pid: str, n: int, search: bool) -> dict:
    keys = [(pid, s) for s in range(n)]
    gs = [run["gens"].get(k) or {} for k in keys]
    m4 = [g.get("m4") or {} for g in gs]
    return {"gen_tokens": sum(int(g.get("gen_tokens") or 0) for g in gs),
            "prefill_tokens": sum(int(x.get("prefill_tokens") or 0) for x in m4),
            "gen_seconds": sum(float(x.get("gen_seconds") or 0) for x in m4),
            "check_seconds": sum(float(x.get("check_seconds") or 0) for x in m4)}


def per_poem(rec: dict, margin: float) -> dict:
    v = m3.poem_metrics(rec, margin)
    v["valid_ignoring_length"] = v["meter_ok"] and rec["rhyme_ok"] and rec["even"]
    return v


# ---------------------------------------------------------------------------------------------- statistics
POEM_KEYS = m3.POEM_KEYS + ("valid_ignoring_length",)


def boot(pp: dict, ids: list, key: str, reps: int, seed: int, other: dict | None = None) -> tuple:
    if key == "valid_ignoring_length":          # a poem-level key m3 does not know: reuse its machinery
        a = {i: {"strict_valid": pp[i][key]} for i in ids}
        b = {i: {"strict_valid": other[i][key]} for i in ids} if other is not None else None
        return m3.bootstrap(a, ids, "strict_valid", reps, seed, b)
    return m3.bootstrap(pp, ids, key, reps, seed, other)


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired pass/fail outcomes."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n10, "b_only": n01, "p": p}


# ---------------------------------------------------------------------------------------------- quality
def _bare_words(text: str) -> list[str]:
    return [arudi._fold(arudi.strip_marks(w)) for w in arudi.tokenize(text)]


def _line_key(line: str) -> str:
    return " ".join(_bare_words(line))


def distinct(seqs: list[list[str]], n: int) -> float:
    """Share of distinct n-grams (grpo/rewards/common.py::distinct_n, pooled over the given sequences)."""
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return 100.0 * len(set(grams)) / len(grams) if grams else float("nan")


def load_train_lines(path: Path | None) -> set:
    if not path or not Path(path).exists():
        return set()
    out = set()
    for r in _jsonl(path):
        for line in r.get("verses") or r.get("poem verses") or []:
            k = _line_key(line)
            if len(k) >= 8:
                out.add(k)
    return out


def quality(texts: list[str], refs: list[str], train: set) -> dict:
    poems = [split_poem_lines(t) for t in texts]
    words = [_bare_words(t) for t in texts]
    lines = [l for p in poems for l in p]
    keys = [_line_key(l) for l in lines]
    rep = [len(p) - len({_line_key(x) for x in p}) for p in poems]
    ref_lines = [{_line_key(x) for x in split_poem_lines(r or "")} for r in refs]
    copied_ref = sum(sum(_line_key(x) in rl for x in p) for p, rl in zip(poems, ref_lines))
    out = {
        "words_per_hemistich": round(sum(len(_bare_words(l)) for l in lines) / max(1, len(lines)), 2),
        "distinct_1": round(distinct(words, 1), 2), "distinct_2": round(distinct(words, 2), 2),
        "distinct_2_within_poem": round(sum(distinct([w], 2) for w in words if len(w) > 1) / max(1, sum(len(w) > 1 for w in words)), 2),
        "repeated_hemistichs_pct": round(100 * sum(rep) / max(1, len(lines)), 2),
        "copied_from_reference_pct": round(100 * copied_ref / max(1, len(lines)), 2),
        "distinct_rhyme_words_pct": None,
    }
    ends = [[_bare_words(x)[-1] for x in p[1::2] if _bare_words(x)] for p in poems]
    tot = sum(len(e) for e in ends)
    out["distinct_rhyme_words_pct"] = round(100 * sum(len(set(e)) for e in ends) / tot, 2) if tot else None
    if train:
        out["copied_from_train_pct"] = round(100 * sum(k in train for k in keys if len(k) >= 8) / max(1, len(keys)), 2)
    return out


# ---------------------------------------------------------------------------------------------- summarize
def summarize(m4_dir: Path, train: Path | None = None, reps: int = 1000, seed: int = 7) -> dict:
    import summarize_structural as ss
    m4_dir = Path(m4_dir)
    found, notes = conditions(m4_dir)
    conds, runs = found["conditions"], found["runs"]
    res = {"conditions": {}, "notes": notes, "checks": {}, "config": {"margin": MARGIN, "reps": reps, "seed": seed},
           "run_configs": {n: (r["meta"].get("config") or {"method": r["meta"].get("mode")}) for n, r in runs.items()}}
    if not conds:
        return res
    common = sorted(set.intersection(*(set(c["sel"]) for c in conds.values())))
    res["prompts"] = len(common)
    train_lines = load_train_lines(train)
    pp_all = {}
    for name, c in conds.items():
        run = runs[c["run"]]
        search = c["n"] == 1 and "I2" in name
        binding_check = search and (res["run_configs"].get(c["run"]) or {}).get("check") == "binding"
        sel = {i: c["sel"][i] for i in common}
        L = {i: run["recs"]["letters"][k] for i, k in sel.items()}
        B = {i: run["recs"]["binding"].get(k) for i, k in sel.items()}
        pp = {i: per_poem(r, MARGIN) for i, r in L.items()}
        pp_inf = {i: per_poem(r, math.inf) for i, r in L.items()}
        pp_b = {i: per_poem(r, MARGIN) for i, r in B.items() if r is not None}
        pp_all[name] = pp
        budgets = {i: _budget(run, i, c["n"], search) for i in common}
        mean = lambda key: round(sum(b[key] for b in budgets.values()) / len(budgets), 2)
        rows = [run["scored"].get(k) for k in sel.values()]
        official = None
        if rows and all(r is not None and r.get("meter") not in (None, "") for r in rows):
            off = ss.summarise(rows)
            official = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in off.items()}
        gens = [run["gens"].get(k) or {} for k in sel.values()]
        out = res["conditions"][name] = {
            "run": c["run"], "n": c["n"], "kind": "search" if search else "sample",
            "budget": {"gen_tokens": mean("gen_tokens"), "prefill_tokens": mean("prefill_tokens"),
                       "gen_seconds": mean("gen_seconds"), "check_seconds": mean("check_seconds")},
            "letters": {k: boot(pp, common, k, reps, seed) for k in REPORT_KEYS},
            "letters_no_margin": {k: boot(pp_inf, common, k, reps, seed) for k in ("strict_valid", "meter_ok", "bayt_ok")},
            "binding": ({k: boot(pp_b, common, k, reps, seed) for k in ("strict_valid", "meter_ok", "rhyme_ok", "bayt_ok")}
                        if len(pp_b) == len(common) else None),
            "official_structural": official,
            "quality": quality([g.get("generated_text") or "" for g in gens], [g.get("reference_completion") for g in gens],
                               train_lines),
            "by_length": {},
            "selected_sample_mean": round(sum(k[1] for k in sel.values()) / len(sel), 3),
        }
        by_b = defaultdict(list)
        for i in common:
            by_b[m3.bucket(L[i]["requested_bayts"])].append(i)
        out["by_length"] = {b: {"poems": len(v), "strict_valid": m3._round(m3._stat([pp[i]["strict_valid"] for i in v], "strict_valid")),
                                "valid_ignoring_length": m3._round(100 * sum(pp[i]["valid_ignoring_length"] for i in v) / len(v)),
                                "gen_tokens": round(sum(budgets[i]["gen_tokens"] for i in v) / len(v), 1)}
                            for b, v in sorted(by_b.items(), key=lambda kv: m3.BUCKETS.index(kv[0]) if kv[0] in m3.BUCKETS else 9)}
        if search:
            logs = [((run["gens"].get(k) or {}).get("m4") or {}) for k in sel.values()]
            hem = [h for m in logs for h in m.get("log", [])]
            pos = defaultdict(lambda: [0, 0])
            for m in logs:
                for j, h in enumerate(m.get("log", [])):
                    pos["sadr" if j % 2 == 0 else "ajuz"][0] += bool(h[2])
                    pos["sadr" if j % 2 == 0 else "ajuz"][1] += 1
            out["search"] = {
                "hemistichs": len(hem),
                "first_candidate_passes_pct": round(100 * sum(1 for h in hem if h[2] and h[0] == 1) / max(1, len(hem)), 2),
                "passed_pct": round(100 * sum(1 for h in hem if h[2]) / max(1, len(hem)), 2),
                "checks_per_hemistich": round(sum(h[0] for h in hem) / max(1, len(hem)), 2),
                "rounds_per_hemistich": round(sum(h[1] for h in hem) / max(1, len(hem)), 3),
                "passed_by_position_pct": {p: round(100 * a / b, 2) for p, (a, b) in pos.items()},
                "early_end_of_text_per_poem": round(sum(m.get("early_eos", 0) for m in logs) / max(1, len(logs)), 3),
                "poems_with_a_forced_hemistich_pct": round(100 * sum(bool(m.get("forced")) for m in logs) / max(1, len(logs)), 2),
                "poems_with_an_empty_line": sum(bool(m.get("empty_lines")) for m in logs),
            }
        # integrity: every selected poem was scored by both scorers; I2 poems have exactly the requested length
        res["checks"][name] = {
            "missing_binding_records": len(common) - len(pp_b),
            "missing_scored_rows": sum(r is None for r in rows),
            # a search poem is shorter only when every candidate of a hemistich was empty (counted apart below)
            "length_mismatches": (sum(1 for i, k in sel.items() if not pp[i]["count_exact"]
                                      and not ((run["gens"].get(k) or {}).get("m4") or {}).get("empty_lines")) if search else None),
            # the search verifier only differs from the scoring one by refusing tadwīr and counting īṭāʾ, both stricter, and
            # the final poem is the last partial poem it checked: a poem without a forced hemistich must be strictly valid
            "unforced_but_invalid": (sum(1 for i, k in sel.items() if not ((run["gens"].get(k) or {}).get("m4") or {}).get("forced")
                                         and not (pp_b if binding_check else pp).get(i, {}).get("strict_valid"))
                                     if search else None),
            "text_mismatch_with_records": sum(
                len(split_poem_lines((run["gens"].get(k) or {}).get("generated_text") or "")) != L[i]["lines"]
                for i, k in sel.items()),
        }
    # paired comparisons: every condition against I0 of its sample run, and each search run at matched compute
    i0 = next((n for n in conds if n.startswith("I0")), None)
    samples = sorted((n for n in conds if not n.startswith("I2")), key=lambda n: res["conditions"][n]["budget"]["gen_tokens"])
    res["vs_i0"] = {}
    if i0:
        for name in conds:
            if name == i0:
                continue
            res["vs_i0"][name] = {k: boot(pp_all[name], common, k, reps, seed, other=pp_all[i0])
                                  for k in ("strict_valid", "valid_ignoring_length", "meter_ok", "rhyme_ok", "bilstm_bayt_ok")}
    res["matched"] = {}
    for name in conds:
        if not name.startswith("I2"):
            continue
        b = res["conditions"][name]["budget"]["gen_tokens"]
        above = next((s for s in samples if res["conditions"][s]["budget"]["gen_tokens"] >= b), None)
        below = next((s for s in reversed(samples) if res["conditions"][s]["budget"]["gen_tokens"] <= b), None)
        res["matched"][name] = {}
        for tag, other in (("I1 with at least this budget", above), ("I1 with at most this budget", below)):
            if other is None:
                res["matched"][name][tag] = None
                continue
            a = [pp_all[name][i]["strict_valid"] for i in common]
            o = [pp_all[other][i]["strict_valid"] for i in common]
            res["matched"][name][tag] = {
                "against": other, "budget_ratio": round(res["conditions"][other]["budget"]["gen_tokens"] / max(1e-9, b), 3),
                **{k: boot(pp_all[name], common, k, reps, seed, other=pp_all[other])
                   for k in ("strict_valid", "valid_ignoring_length", "meter_ok", "rhyme_ok", "bilstm_bayt_ok")},
                "mcnemar_strict_valid": mcnemar(a, o)}
    res["choice"] = choose(res)
    return res


def choose(res: dict) -> dict | None:
    """The pre-registered choice for the test runs, made on the development sweep: among the letters-mode search runs
    whose budget is at most that of best-of-4, the highest strict validity (ties: the smaller budget); best-of-n for the test is the
    smallest n (a power of 2, at most 16) whose budget on the sweep was at least the chosen search run's."""
    conds = res.get("conditions", {})
    bo4 = next((c for n, c in conds.items() if n.startswith("I1 best-of-4 ")), None)
    if bo4 is None:
        return None
    cap = bo4["budget"]["gen_tokens"]
    cands = [(n, c) for n, c in conds.items() if n.startswith("I2") and "binding" not in n and c["budget"]["gen_tokens"] <= cap]
    if not cands:
        return None
    name, c = max(cands, key=lambda nc: (nc[1]["letters"]["strict_valid"][0], -nc[1]["budget"]["gen_tokens"]))
    cfg = (res.get("run_configs") or {}).get(c["run"], {})
    sizes = sorted((cc["n"], cc["budget"]["gen_tokens"]) for n2, cc in conds.items() if n2.startswith(("I0", "I1")))
    n = next((k for k, b in sizes if b >= c["budget"]["gen_tokens"]), 16)
    return {"search_run": c["run"], "search": name, "k": cfg.get("k"), "rounds": cfg.get("rounds"),
            "best_of_n": min(16, n), "budget_cap_best_of_4": cap, "search_budget": c["budget"]["gen_tokens"]}


# ---------------------------------------------------------------------------------------------- render
def _f(t) -> str:
    return m3._fmt(t)


def render(res: dict) -> str:
    out = ["# M4 — verifier-guided decoding against best-of-N", ""]
    if not res.get("conditions"):
        return "\n".join(out + ["No condition has verifier records yet."] + [f"* {n}" for n in res.get("notes", [])])
    out += [f"{res['prompts']:,} prompts shared by every condition; letters mode, margin {res['config']['margin']}; "
            f"95% bootstrap intervals over prompts ({res['config']['reps']} resamples). Budget = generated tokens per "
            "prompt, rejected candidates included.", ""]
    for n in res.get("notes", []):
        out.append(f"* note: {n}")
    if res.get("choice"):
        ch = res["choice"]
        out += [f"**Pre-registered choice for the test runs** (search runs within the budget of best-of-4, highest strict "
                f"validity): {ch['search']} (k = {ch['k']}, R = {ch['rounds']}), against best-of-{ch['best_of_n']}.", ""]
    out += ["## Results", "",
            "| condition | gen tokens / prompt | encoded context tokens / prompt | GPU s / prompt | strictly valid | valid ignoring length | meter | rhyme | exact length | bayts on meter | bayts with no reading | BiLSTM bayts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    conds = sorted(res["conditions"].items(), key=lambda kv: kv[1]["budget"]["gen_tokens"])
    for name, c in conds:
        L, b = c["letters"], c["budget"]
        out.append(f"| {name} | {b['gen_tokens']:,.0f} | {b['prefill_tokens']:,.0f} | {b['gen_seconds']:.2f} | "
                   + " | ".join(_f(L[k]) for k in REPORT_KEYS) + " |")
    out += ["", "**Independent views** (the verifier that selects is also the one that scores above):", "",
            "| condition | Shaer meter conformity | base-meter accuracy | strictly valid, no margin | strictly valid, binding | meter, binding |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, c in conds:
        o = c["official_structural"] or {}
        bd = c["binding"] or {}
        out.append(f"| {name} | {o.get('meter_conformity', '–')} | {o.get('base_meter_accuracy_pct', '–')} | "
                   f"{_f(c['letters_no_margin']['strict_valid'])} | {_f(bd.get('strict_valid'))} | {_f(bd.get('meter_ok'))} |")
    if res.get("matched"):
        out += ["", "## Matched compute: each search run against best-of-N", "",
                "| search run | against | budget ratio (I1 / I2) | strictly valid | valid ignoring length | meter | rhyme | BiLSTM bayts | McNemar (I2 only / I1 only, p) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for name, m in res["matched"].items():
            for tag, x in m.items():
                if not x:
                    out.append(f"| {name} | {tag}: none | | | | | | | |")
                    continue
                mc = x["mcnemar_strict_valid"]
                out.append(f"| {name} | {x['against']} | {x['budget_ratio']} | {_f(x['strict_valid'])} | "
                           f"{_f(x['valid_ignoring_length'])} | {_f(x['meter_ok'])} | {_f(x['rhyme_ok'])} | "
                           f"{_f(x['bilstm_bayt_ok'])} | {mc['a_only']} / {mc['b_only']}, p = {mc['p']:.2g} |")
    if res.get("vs_i0"):
        out += ["", "## Against the single sample (I0), paired, percentage points", "",
                "| condition | strictly valid | valid ignoring length | meter | rhyme | BiLSTM bayts |", "|---|---:|---:|---:|---:|---:|"]
        for name, x in res["vs_i0"].items():
            out.append(f"| {name} | " + " | ".join(_f(x[k]) for k in ("strict_valid", "valid_ignoring_length", "meter_ok",
                                                                         "rhyme_ok", "bilstm_bayt_ok")) + " |")
    out += ["", "## By requested length (strictly valid / valid ignoring length / gen tokens per prompt)", ""]
    buckets = [b for b in m3.BUCKETS if any(b in c["by_length"] for _, c in conds)]
    out += ["| condition | " + " | ".join(f"{b} bayts" for b in buckets) + " |", "|---|" + "---:|" * len(buckets)]
    for name, c in conds:
        cells = []
        for b in buckets:
            x = c["by_length"].get(b)
            cells.append("–" if not x else f"{x['strict_valid']} / {x['valid_ignoring_length']} / {x['gen_tokens']:.0f}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    qk = ("words_per_hemistich", "distinct_1", "distinct_2", "distinct_2_within_poem", "repeated_hemistichs_pct",
          "distinct_rhyme_words_pct", "copied_from_reference_pct", "copied_from_train_pct")
    out += ["", "## Quality and copying", "", "| condition | " + " | ".join(qk) + " |", "|---|" + "---:|" * len(qk)]
    for name, c in conds:
        out.append(f"| {name} | " + " | ".join(str(c["quality"].get(k, "–")) for k in qk) + " |")
    srch = [(n, c) for n, c in conds if c.get("search")]
    if srch:
        out += ["", "## The search", "",
                "| run | hemistichs | first candidate passes | passed | checks / hemistich | rounds / hemistich | sadr passed | ʿajuz passed | poems with a forced hemistich | early end of text / poem | poems with an empty line |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for n, c in srch:
            s = c["search"]
            p = s["passed_by_position_pct"]
            out.append(f"| {n} | {s['hemistichs']:,} | {s['first_candidate_passes_pct']} | {s['passed_pct']} | "
                       f"{s['checks_per_hemistich']} | {s['rounds_per_hemistich']} | {p.get('sadr', '–')} | {p.get('ajuz', '–')} | "
                       f"{s['poems_with_a_forced_hemistich_pct']} | {s['early_end_of_text_per_poem']} | {s['poems_with_an_empty_line']} |")
    out += ["", "## Integrity checks (all must be 0)", "",
            "| condition | missing binding records | missing scored rows | search poems not of the requested length | search poems with no forced hemistich that are not strictly valid | line counts differing from the records |",
            "|---|---:|---:|---:|---:|---:|"]
    dash = lambda v: "–" if v is None else v
    for name, x in res["checks"].items():
        out.append(f"| {name} | {x['missing_binding_records']} | {x['missing_scored_rows']} | {dash(x['length_mismatches'])} | "
                   f"{dash(x['unforced_but_invalid'])} | {x['text_mismatch_with_records']} |")
    return "\n".join(out) + "\n"


def compare(a_path: Path, b_path: Path, reps: int = 1000, seed: int = 7) -> dict:
    """The repository's official metrics of two scored files on the prompts they share (sample 0), with bootstrap
    intervals of the differences a - b: for checking that two generators draw from the same distribution."""
    import numpy as np
    import summarize_structural as ss
    A = {r["id"]: r for r in _jsonl(a_path) if int(r.get("sample_idx", 0) or 0) == 0}
    B = {r["id"]: r for r in _jsonl(b_path) if int(r.get("sample_idx", 0) or 0) == 0}
    ids = sorted(set(A) & set(B))
    out = {"prompts": len(ids), "a": ss.summarise([A[i] for i in ids]), "b": ss.summarise([B[i] for i in ids])}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(reps, len(ids)))
    for key, f in (("meter", lambda r: float(r.get("meter") or 0)),
                   ("exact_count", lambda r: float(int(r.get("parsed_num_lines") or -1) == int(r.get("requested_num_lines") or -2))),
                   ("count_adherence", lambda r: float(r.get("count_adherence") or 0))):
        d = np.array([f(A[i]) - f(B[i]) for i in ids])
        est = d[idx].mean(axis=1)
        lo, hi = np.percentile(est, [2.5, 97.5])
        out[f"diff_{key}"] = (round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--dir", required=True)
    s.add_argument("--train", default="")
    s.add_argument("--reps", type=int, default=1000)
    s.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "compare":
        res = compare(Path(a.a), Path(a.b))
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps(res, ensure_ascii=False, indent=1))
        inside = all(v[1] <= 0 <= v[2] for k, v in res.items() if k.startswith("diff_"))
        print("sample mode vs the reference generator:", "consistent (every interval contains 0)" if inside
              else "an interval excludes 0: look at the numbers above")
        return
    res = summarize(Path(a.dir), Path(a.train) if a.train else None, a.reps, a.seed)
    Path(a.dir, "m4_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    text = render(res)
    Path(a.dir, "m4_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
