#!/usr/bin/env python3
"""Numbers and figures of the paper, computed from the result files (nothing is typed by hand).

Inputs (paths given on the command line):
  --m3       an M3 folder (m3_rescore.py score records of every system, and m3_summary.json)
  --repro    the reproduction folder: e0..e3/summary_drive.json (the baseline notebook's summaries, from Drive) and
             metrics.jsonl (the re-training run's log)
  --m0       M0's m0_summary.json
  --m1       M1's validation JSON on the test split (validate_verifier.py)
Outputs (--out): numbers.tex (one \\newcommand per number), results.json, and the figures (PNG, 400 DPI).

Everything here is descriptive statistics over the stored records: bootstrap intervals resample prompts (5,000
resamples, seed 7), Cohen's kappa compares bayt verdicts, the logistic models use iteratively reweighted least squares.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import m3_rescore as m3  # noqa: E402

import os
REPS, SEED = int(os.environ.get("PAPER_REPS", 5000)), 7
SYSTEMS = [("reference", "Human"), ("e0_released_shaer", "E0 released"), ("e1_shaer_adapter", "E1 re-generated"),
           ("e3_retrained_yehia", "E3 re-trained"), ("e2_yehia_base", "E2 untuned")]
CONTROLS = [("control_wrong_meter", "Wrong-meter control"), ("control_mixed_rhyme", "Mixed-rhyme control")]
COLORS = {"reference": "#2a78d6", "e0_released_shaer": "#eb6834", "e1_shaer_adapter": "#1baf7a",
          "e3_retrained_yehia": "#eda100", "e2_yehia_base": "#e87ba4"}
MARKERS = {"reference": "o", "e0_released_shaer": "s", "e1_shaer_adapter": "^", "e3_retrained_yehia": "D", "e2_yehia_base": "v"}
TAG = {"reference": "Ref", "e0_released_shaer": "Ezero", "e1_shaer_adapter": "Eone", "e2_yehia_base": "Etwo",
       "e3_retrained_yehia": "Ethree", "control_wrong_meter": "Cwm", "control_mixed_rhyme": "Cmr"}
VNAME = {"full": "Full", "no_rhyme": "NoRhyme", "no_length": "NoLength", "any_form": "AnyForm", "no_margin": "NoMargin",
         "margin_1.0": "MarginOne", "margin_3.0": "MarginThree", "binding": "Binding", "base_bilstm": "BaseBilstm"}
LEN = {"1-3": "A", "4-6": "B", "7-10": "C", "11-20": "D"}
PAPER = {"base_meter_accuracy_pct": {"Shaer": 95.17, "Yehia": 26.49}, "meter_conformity": {"Shaer": 0.9064, "Yehia": 0.1494},
         "exact_count_accuracy_pct": {"Shaer": 83.40, "Yehia": 44.47}, "count_adherence": {"Shaer": 0.9792, "Yehia": 0.7680},
         "best_eval_loss": 2.2074, "best_step": 3000, "test_loss": 2.1932}


# ---------------------------------------------------------------------------------------------- helpers
def load(m3dir: Path, name: str, mode: str = "letters") -> dict:
    return {r["id"]: r for r in m3._read_jsonl(m3dir / name / f"{mode}.jsonl") if r.get("sample_idx", 0) == 0 and "error" not in r}


def boot_mean(values: np.ndarray, reps: int = REPS, seed: int = SEED) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    est = values[idx].mean(axis=1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def base_criterion(rec: dict) -> bool:
    """Shaer's own success criterion read off the same poem: every bayt the BiLSTM scored is in the requested meter, and
    the poem has the requested number of hemistichs."""
    bil = [b for b in rec["bilstm"] if b is not None]
    return bool(bil) and all(bil) and rec["count_ok"]


def variants(L: dict, B: dict | None) -> dict:
    """Strict validity with one component of the extended criterion removed or changed (nested comparison)."""
    out = defaultdict(dict)
    for i, r in L.items():
        p15, pinf = m3.poem_metrics(r, 1.5), m3.poem_metrics(r, math.inf)
        p10, p30 = m3.poem_metrics(r, 1.0), m3.poem_metrics(r, 3.0)
        out["full"][i] = p15["strict_valid"]
        out["no_rhyme"][i] = p15["meter_ok"] and r["count_ok"] and r["even"]
        out["no_length"][i] = p15["meter_ok"] and r["rhyme_ok"] and r["even"]
        out["any_form"][i] = p15["meter_ok_any_form"] and r["rhyme_ok"] and r["count_ok"] and r["even"]
        out["no_margin"][i] = pinf["strict_valid"]
        out["margin_1.0"][i] = p10["strict_valid"]
        out["margin_3.0"][i] = p30["strict_valid"]
        out["base_bilstm"][i] = base_criterion(r)
        if B is not None and i in B:
            out["binding"][i] = m3.poem_metrics(B[i], 1.5)["strict_valid"]
    return out


def kappa(a: list[bool], b: list[bool]) -> float:
    a, b = np.array(a, float), np.array(b, float)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def logit_fit(X: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, float]:
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * w[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return beta, ll


def mcnemar_n(delta: float, psi: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Pairs needed by the McNemar test (Connor 1987 normal approximation): delta = p10 - p01, psi = p10 + p01."""
    from scipy.stats import norm
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return int(math.ceil((za * math.sqrt(psi) + zb * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2))


def mcnemar_power(n: int, delta: float, psi: float, alpha: float = 0.05) -> float:
    from scipy.stats import norm
    za = norm.ppf(1 - alpha / 2)
    return float(norm.cdf((math.sqrt(n) * abs(delta) - za * math.sqrt(psi)) / math.sqrt(psi - delta ** 2)))


# ---------------------------------------------------------------------------------------------- main
TERM = {"intercept": "Intercept", "len:4-6": "Length 4--6 bayts", "len:7-10": "Length 7--10 bayts",
        "len:11-20": "Length 11--20 bayts"}
SYS_SHORT = {"reference": "Human", "e1_shaer_adapter": "E1", "e3_retrained_yehia": "E3", "e2_yehia_base": "E2"}
METER_LAT = {"tawil": "al-\\d{T}aw\\={\\i}l", "kamil": "al-K\\={a}mil", "basit": "al-Bas\\={\\i}\\d{t}", "wafir": "al-W\\={a}fir",
             "khafif": "al-Khaf\\={\\i}f", "sari": "al-Sar\\={\\i}`", "mutaqarib": "al-Mutaq\\={a}rib", "ramal": "al-Ramal",
             "rajaz": "al-Rajaz", "munsarih": "al-Munsari\\d{h}"}


def _term(c: str) -> str:
    if c in TERM:
        return TERM[c]
    kind, val = c.split(":", 1)
    if kind == "sys":
        return f"System: {SYS_SHORT.get(val, val)}"
    return f"Meter: {METER_LAT.get(val, val)}"


def write_tables(R: dict, out: Path, systems) -> None:
    """LaTeX rows (booktabs, no rules) for the structural-model and nested-model tables; \\input by main.tex."""
    L = R["logit"]
    lines = []
    for c in L["columns"]:
        cells = [_term(c)]
        for o in ("base", "ext"):
            b = L["models"][o]["coef"][c]; lo, hi = L["models"][o]["ci"][c]
            sig = (lo > 0 or hi < 0)
            est = f"{b:+.2f}"
            cells.append(("\\textbf{" + est + "}") if sig else est)
            cells.append(f"[{lo:+.2f}, {hi:+.2f}]")
            cells.append("--" if c == "intercept" else f"{math.exp(b):.2f}")
        lines.append(" & ".join(cells) + " \\\\")
    for o, lab in (("base", "base"), ("ext", "ext")):
        pass
    mb, me = L["models"]["base"], L["models"]["ext"]
    lines.append("\\midrule")
    lines.append(f"McFadden $R^2$ & \\multicolumn{{3}}{{c}}{{{mb['mcfadden_r2']:.3f}}} & \\multicolumn{{3}}{{c}}{{{me['mcfadden_r2']:.3f}}} \\\\")
    lines.append(f"AIC & \\multicolumn{{3}}{{c}}{{{mb['aic']:,.0f}}} & \\multicolumn{{3}}{{c}}{{{me['aic']:,.0f}}} \\\\")
    lines.append(f"Poems ($n$) / valid share & \\multicolumn{{3}}{{c}}{{{mb['n']:,} / {100*mb['positive_rate']:.1f}\\%}} & "
                 f"\\multicolumn{{3}}{{c}}{{{me['n']:,} / {100*me['positive_rate']:.1f}\\%}} \\\\")
    (out / "tab_logit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    for o, lab in (("base", "Base"), ("ext", "Extended")):
        best_aic = min(r["aic"] for r in L["nested"][o])
        for r in L["nested"][o]:
            aic = f"{r['aic']:,.0f}"
            aic = f"\\textbf{{{aic}}}" if r["aic"] == best_aic else aic
            lr = "--" if "lr" not in r else f"{r['lr']:,.1f} ({r['df']})"
            p = "--" if "p" not in r else ("$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}")
            rows.append(f"{lab if r['step']=='S1' else ''} & {r['step']} {r['label']} & {r['k']} & {r['mcfadden_r2']:.3f} & {aic} & {lr} & {p} \\\\")
        if o == "base":
            rows.append("\\midrule")
    (out / "tab_nested.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", required=True)
    ap.add_argument("--repro", required=True)
    ap.add_argument("--m0", required=True)
    ap.add_argument("--m1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m3dir, repro, out = Path(a.m3), Path(a.repro), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    N: dict = {}          # macro name -> text
    R: dict = {}          # everything, for results.json

    def put(name, value, fmt="{:.2f}"):
        N[name] = fmt.format(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)

    # ------------------------------------------------------------------ reproduction (the baseline notebook)
    rep = {}
    for e in ("e0", "e1", "e2", "e3"):
        p = repro / e / "summary_drive.json"
        if p.exists():
            rep[e] = json.loads(p.read_text())["overall"]
    R["reproduction"] = rep
    for e, o in rep.items():
        E = e.upper().replace("0", "Zero").replace("1", "One").replace("2", "Two").replace("3", "Three")
        put(f"rep{E}Acc", o["base_meter_accuracy_pct"])
        put(f"rep{E}Conf", o["meter_conformity"], "{:.4f}")
        put(f"rep{E}Exact", o["exact_count_accuracy_pct"])
        put(f"rep{E}Adh", o["count_adherence"], "{:.4f}")
    for k, v in PAPER.items():
        if isinstance(v, dict):
            for s, x in v.items():
                put(f"paper{s}{''.join(w.title() for w in k.split('_'))}", x, "{:.4f}" if x < 1.5 else "{:.2f}")
    put("paperBestLoss", PAPER["best_eval_loss"], "{:.4f}")
    put("paperTestLoss", PAPER["test_loss"], "{:.4f}")
    # training log
    steps, ev, tr = [], [], []
    for line in open(repro / "metrics.jsonl"):
        r = json.loads(line)
        if r.get("mode") == "eval" and r.get("eval_loss") is not None:
            ev.append((r["global_step"], r["eval_loss"]))
        if r.get("mode") == "train" and r.get("loss") is not None:
            tr.append((r["global_step"], r["loss"]))
        if r.get("mode") == "test_eval":
            R["test_loss"] = r.get("test_loss")
    ev = sorted(dict(ev).items())
    tr = sorted(dict(tr).items())
    last_step = max(max(x[0] for x in ev), max(x[0] for x in tr))
    # the trainer reloads the best checkpoint at the end and evaluates it again at the last step; that point repeats the
    # best checkpoint's loss and is not a new evaluation, so it is left out of the curve
    if len(ev) > 1 and ev[-1][0] == last_step and abs(ev[-1][1] - min(x[1] for x in ev[:-1])) < 1e-3:
        ev = ev[:-1]
    best = min(ev, key=lambda x: x[1])
    R["training"] = {"eval": ev, "best_step": best[0], "best_eval_loss": best[1], "final_step": tr[-1][0]}
    put("ourBestLoss", best[1], "{:.4f}")
    put("ourBestStep", best[0], "{:d}")
    put("ourSteps", last_step, "{:,}")
    if R.get("test_loss"):
        put("ourTestLoss", R["test_loss"], "{:.4f}")
    put("ourInitLoss", ev[0][1], "{:.4f}")

    # ------------------------------------------------------------------ strict re-scoring (M3)
    summ = json.loads((m3dir / "m3_summary.json").read_text())
    R["m3_reps"] = summ["config"]["bootstrap_reps"]
    have = [(n, lab) for n, lab in SYSTEMS + CONTROLS if (m3dir / n / "letters.jsonl").exists() and n in summ["systems"]]
    R["systems"] = {}
    for n, lab in have:
        s = summ["systems"][n]
        L = s["letters"]
        g = L["margin_1.5"]
        tag = TAG[n]
        R["systems"][n] = {"label": lab, "letters": g, "binding": (s.get("binding") or {}).get("margin_1.5"),
                           "margin_inf": L["margin_inf"], "by_length": L["by_length"], "issues": L["issues_pct"],
                           "vs_reference": L.get("vs_reference"), "by_meter": L["by_meter"],
                           "first_broken_foot": L.get("first_broken_foot_pct"), "bayt_ok_by_position": L["bayt_ok_by_position"],
                           "official": L.get("official_structural")}
        for k in ("strict_valid", "meter_ok", "rhyme_ok", "count_exact", "bayt_ok", "no_reading", "bilstm_bayt_ok"):
            if g.get(k) and g[k][0] == g[k][0]:
                put(f"m{tag}{''.join(w.title() for w in k.split('_'))}", g[k][0])
                put(f"m{tag}{''.join(w.title() for w in k.split('_'))}CI", f"[{g[k][1]:.1f}, {g[k][2]:.1f}]")
        if s.get("binding"):
            bg = s["binding"]["margin_1.5"]
            put(f"m{tag}BindStrict", bg["strict_valid"][0])
            put(f"m{tag}BindBayt", bg["bayt_ok"][0])
        put(f"m{tag}NoMarginStrict", L["margin_inf"]["strict_valid"][0])
        if L.get("vs_reference"):
            v = L["vs_reference"]["strict_valid"]
            put(f"m{tag}VsRefStrict", f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]")
            v = L["vs_reference"]["bilstm_bayt_ok"]
            if v[0] == v[0]:
                put(f"m{tag}VsRefBilstm", f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]")
        for b, x in L["by_length"].items():
            put(f"m{tag}Len{LEN[b]}", x["strict_valid"])

    # ------------------------------------------------------------------ agreement, nested variants, controls
    R["agreement"], R["variants"] = {}, {}
    keys = ["full", "no_rhyme", "no_length", "any_form", "no_margin", "margin_1.0", "margin_3.0", "binding", "base_bilstm"]
    pooled = []
    for n, lab in have:
        L = load(m3dir, n, "letters")
        B = load(m3dir, n, "binding") if (m3dir / n / "binding.jsonl").exists() else None
        ids = sorted(L)
        ver, bil, lb_l, lb_b = [], [], [], []
        for i in ids:
            r = L[i]
            for j, bl in enumerate(r["bilstm"]):
                if bl is not None and j < len(r["target"]["bayt_ok"]):
                    ver.append(r["target"]["bayt_ok"][j])
                    bil.append(bl)
            if B is not None and i in B:
                lb_l += list(r["target"]["bayt_ok"])
                lb_b += list(B[i]["target"]["bayt_ok"])[: len(r["target"]["bayt_ok"])]
                lb_l = lb_l[: len(lb_b)]
        agr = {"bayts": len(ver), "kappa_verifier_bilstm": kappa(ver, bil) if ver else None,
               "agree_pct": 100 * float(np.mean(np.array(ver) == np.array(bil))) if ver else None,
               "bilstm_yes_verifier_no_pct": 100 * float(np.mean([b and not v for v, b in zip(ver, bil)])) if ver else None}
        if B is not None and lb_b:
            agr["kappa_letters_binding"] = kappa(lb_l, lb_b)
        R["agreement"][n] = agr
        V = variants(L, B)
        # of the poems that pass the paper's own criterion (BiLSTM on every bayt, exact length), the share strictly valid
        # McNemar's test against the human poems of the same prompts (strict validity)
        if n == "reference":
            ref_full = V["full"]
        elif n in dict(SYSTEMS):
            common = [i for i in ids if i in ref_full]
            n01 = sum(1 for i in common if not V["full"][i] and ref_full[i])
            n10 = sum(1 for i in common if V["full"][i] and not ref_full[i])
            from scipy.stats import binomtest, chi2 as _chi2
            pval = (binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 < 25
                    else float(_chi2.sf((n01 - n10) ** 2 / (n01 + n10), 1))) if n01 + n10 else 1.0
            R.setdefault("mcnemar_vs_reference", {})[n] = {"n01": n01, "n10": n10, "p": pval}
            put(f"mc{TAG[n]}Chi", (n01 - n10) ** 2 / max(1, n01 + n10), "{:,.1f}")
            put(f"mc{TAG[n]}P", "$<$0.001" if pval < 0.001 else f"{pval:.3f}", "{}")
        passed = [i for i in ids if V["base_bilstm"].get(i)]
        if passed:
            put(f"g{TAG[n]}StrictGivenBase", 100 * sum(bool(V["full"].get(i)) for i in passed) / len(passed))
        R["variants"][n] = {k: boot_mean(np.array([float(V[k][i]) for i in ids if i in V[k]]))
                            for k in keys if V.get(k)}
        tag = TAG[n]
        if agr["kappa_verifier_bilstm"] is not None:
            put(f"k{tag}VB", agr["kappa_verifier_bilstm"], "{:.3f}")
            put(f"k{tag}BYesVNo", agr["bilstm_yes_verifier_no_pct"])
        if agr.get("kappa_letters_binding") is not None:
            put(f"k{tag}LB", agr["kappa_letters_binding"], "{:.3f}")
        for k in keys:
            if k in R["variants"][n]:
                put(f"v{tag}{VNAME[k]}", 100 * R["variants"][n][k][0])
        if n in dict(SYSTEMS) and n != "e2_yehia_base":
            for i in ids:
                r = L[i]
                pooled.append({"system": n, "len": m3.bucket(r["requested_bayts"]), "meter": r["meter"],
                               "ext": float(V["full"][i]), "base": float(V["base_bilstm"][i]), "id": i})

    # ------------------------------------------------------------------ logistic "structural" model
    # validity (Shaer's criterion = base; strict = extended) explained by system, requested length and meter
    if pooled:
        sys_levels = [n for n, _ in SYSTEMS if n != "e2_yehia_base" and any(p["system"] == n for p in pooled)]
        ref_sys = "e0_released_shaer"
        len_levels = ["4-6", "7-10", "11-20"]
        meters = [m for m, _ in sorted(defaultdict(int, {m: sum(1 for p in pooled if p["meter"] == m) for m in
                                                          {p["meter"] for p in pooled}}).items(), key=lambda kv: -kv[1])]
        top_m = [m for m in meters if m != "tawil"][:6]
        cols = ["intercept"] + [f"sys:{s}" for s in sys_levels if s != ref_sys] + [f"len:{b}" for b in len_levels] + [f"meter:{m}" for m in top_m]

        def design(rows):
            X = np.zeros((len(rows), len(cols)))
            for k, p in enumerate(rows):
                X[k, 0] = 1
                for c, name in enumerate(cols[1:], 1):
                    kind, val = name.split(":", 1)
                    X[k, c] = float((kind == "sys" and p["system"] == val) or (kind == "len" and p["len"] == val)
                                    or (kind == "meter" and p["meter"] == val))
            return X
        X = design(pooled)
        models = {}
        prompts = sorted({p["id"] for p in pooled})
        rows_by_prompt = defaultdict(list)
        for k, p in enumerate(pooled):
            rows_by_prompt[p["id"]].append(k)
        rng = np.random.default_rng(SEED)
        for outcome in ("base", "ext"):
            y = np.array([p[outcome] for p in pooled])
            beta, ll = logit_fit(X, y)
            p0 = y.mean()
            ll0 = float(len(y) * (p0 * math.log(p0) + (1 - p0) * math.log(1 - p0)))
            bs = []
            for _ in range(REPS):
                pick = rng.integers(0, len(prompts), len(prompts))
                idx = np.concatenate([rows_by_prompt[prompts[q]] for q in pick])
                b, _ = logit_fit(X[idx], y[idx], iters=25)
                bs.append(b)
            bs = np.array(bs)
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            models[outcome] = {"coef": dict(zip(cols, beta.tolist())), "ci": {c: [float(l), float(h)] for c, l, h in zip(cols, lo, hi)},
                               "loglik": ll, "loglik_null": ll0, "mcfadden_r2": 1 - ll / ll0, "aic": 2 * len(cols) - 2 * ll,
                               "n": int(len(y)), "positive_rate": float(p0)}
        R["logit"] = {"columns": cols, "reference_levels": {"system": ref_sys, "length": "1-3", "meter": "tawil"}, "models": models}
        for outcome, M in models.items():
            O = "Base" if outcome == "base" else "Ext"
            put(f"lr{O}Rsq", M["mcfadden_r2"], "{:.3f}")
            put(f"lr{O}AIC", M["aic"], "{:,.0f}")
            put(f"lr{O}N", M["n"], "{:,}")
        # nested models for each outcome: system only, + requested length, + meter (likelihood-ratio tests)
        from scipy.stats import chi2
        steps = [("S1", "system", [c for c in cols if c == "intercept" or c.startswith("sys:")]),
                 ("S2", "+ length", [c for c in cols if c == "intercept" or c.startswith(("sys:", "len:"))]),
                 ("S3", "+ meter", cols)]
        nested = {}
        for outcome in ("base", "ext"):
            y = np.array([p[outcome] for p in pooled])
            ll0 = models[outcome]["loglik_null"]
            prev, rows_n = None, []
            for key, lab, cs in steps:
                Xi = X[:, [cols.index(c) for c in cs]]
                _, ll = logit_fit(Xi, y)
                row = {"step": key, "label": lab, "k": len(cs), "loglik": ll, "mcfadden_r2": 1 - ll / ll0,
                       "aic": 2 * len(cs) - 2 * ll}
                if prev:
                    lr = 2 * (ll - prev["loglik"]); df = len(cs) - prev["k"]
                    row.update({"lr": lr, "df": df, "p": float(chi2.sf(lr, df))})
                rows_n.append(row); prev = row
            nested[outcome] = rows_n
            O = "Base" if outcome == "base" else "Ext"
            for row in rows_n:
                put(f"nest{O}{ {'S1': 'SOne', 'S2': 'STwo', 'S3': 'SThree'}[row['step']]}Rsq", row["mcfadden_r2"], "{:.3f}")
        R["logit"]["nested"] = nested
        write_tables(R, out, SYSTEMS)

    # ------------------------------------------------------------------ power analysis
    R["power"] = {"alpha": 0.05, "power": 0.80,
                  "n_delta10_psi20": mcnemar_n(0.10, 0.20), "n_delta5_psi10": mcnemar_n(0.05, 0.10),
                  "n_delta5_psi20": mcnemar_n(0.05, 0.20), "power_400_delta10_psi20": mcnemar_power(400, 0.10, 0.20),
                  "power_3481_delta3_psi10": mcnemar_power(3481, 0.03, 0.10), "mde_3481": None}
    for d in np.arange(0.005, 0.2, 0.0025):
        if mcnemar_power(3481, d, 0.10) >= 0.8:
            R["power"]["mde_3481"] = float(d)
            break
    put("powNTen", R["power"]["n_delta10_psi20"], "{:d}")
    put("powNFive", R["power"]["n_delta5_psi10"], "{:d}")
    put("powNFiveWide", R["power"]["n_delta5_psi20"], "{:d}")
    put("powDev", R["power"]["power_400_delta10_psi20"], "{:.3f}")
    put("powMDE", 100 * R["power"]["mde_3481"], "{:.1f}")

    # ------------------------------------------------------------------ M0 and M1
    m0 = json.loads(Path(a.m0).read_text())
    R["m0"] = m0
    m1 = json.loads(Path(a.m1).read_text())
    R["m1"] = m1
    (out / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
    with open(out / "numbers.tex", "w", encoding="utf-8") as f:
        f.write("% generated by prosody-verifier/paper/scripts/analysis.py -- do not edit by hand\n")
        for k in sorted(N):
            f.write(f"\\newcommand{{\\{k}}}{{{N[k]}}}\n")
    figures(R, out)
    print(f"{len(N)} numbers, figures in {out}")


# ---------------------------------------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "savefig.dpi": 400,
                         "figure.dpi": 100, "lines.linewidth": 2})
    return plt


def figures(R: dict, out: Path) -> None:
    plt = _style()
    sysd = R["systems"]
    gen = [n for n, _ in SYSTEMS if n in sysd]
    # 1. the base metric (BiLSTM bayts) against the verifier (bayts on meter; strictly valid poems)
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    measures = [("bilstm_bayt_ok", "BiLSTM bayts\n(base metric)"), ("bayt_ok", "Verifier bayts\n(meter only)"),
                ("strict_valid", "Verifier poems\n(strictly valid)")]
    w = 0.8 / len(gen)
    for k, n in enumerate(gen):
        vals = [sysd[n]["letters"][m][0] if sysd[n]["letters"].get(m) and sysd[n]["letters"][m][0] == sysd[n]["letters"][m][0] else 0
                for m, _ in measures]
        xs = np.arange(len(measures)) + (k - (len(gen) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=COLORS[n], label=sysd[n]["label"], zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.5, color="#52514e")
    ax.set_xticks(range(len(measures)), [l for _, l in measures])
    ax.set_ylabel("percent")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", zorder=0)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(out / "fig_base_vs_verifier.png")
    plt.close(fig)
    # 2. strict validity by requested length
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    buckets = ["1-3", "4-6", "7-10", "11-20"]
    for n in gen:
        ys = [sysd[n]["by_length"].get(b, {}).get("strict_valid", float("nan")) for b in buckets]
        ax.plot(range(4), ys, marker=MARKERS[n], markersize=4, color=COLORS[n], label=sysd[n]["label"])
        ax.text(3.08, ys[-1], f"{ys[-1]:.0f}", fontsize=6, va="center", color="#52514e")
    ax.set_xticks(range(4), [f"{b} bayts" for b in buckets])
    ax.set_ylabel("strictly valid poems (%)")
    ax.set_ylim(-3, 80)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_validity_by_length.png")
    plt.close(fig)
    # 3. training curve
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ev = R["training"]["eval"]
    ax.plot([s for s, _ in ev], [l for _, l in ev], color="#2a78d6", marker="o", markersize=2.5, label="our re-training (eval loss)")
    ax.axhline(PAPER["best_eval_loss"], color="#eb6834", linestyle="--", linewidth=1.2, label="paper: best eval loss")
    ax.scatter([R["training"]["best_step"]], [R["training"]["best_eval_loss"]], s=30, color="#2a78d6", zorder=4)
    ax.annotate(f"best {R['training']['best_eval_loss']:.3f} (step {R['training']['best_step']})",
                (R["training"]["best_step"], R["training"]["best_eval_loss"]), xytext=(8, 14), textcoords="offset points",
                fontsize=6, color="#52514e")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_training_curve.png")
    plt.close(fig)
    # 4. reproduction: paper vs ours
    rep = R["reproduction"]
    if rep:
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
        labs = [("e0", "E0"), ("e1", "E1"), ("e3", "E3"), ("e2", "E2")]
        for ax, (key, title, paper) in zip(axes, (("base_meter_accuracy_pct", "base-meter accuracy (%)", PAPER["base_meter_accuracy_pct"]),
                                                  ("exact_count_accuracy_pct", "exact count accuracy (%)", PAPER["exact_count_accuracy_pct"]))):
            vals = [rep[e][key] if e in rep else 0 for e, _ in labs]
            cols = [COLORS[{"e0": "e0_released_shaer", "e1": "e1_shaer_adapter", "e3": "e3_retrained_yehia", "e2": "e2_yehia_base"}[e]] for e, _ in labs]
            ax.bar(range(4), vals, color=cols, width=0.7, zorder=3)
            ax.axhline(paper["Shaer"], color="#0b0b0b", linestyle="--", linewidth=0.9)
            ax.axhline(paper["Yehia"], color="#898781", linestyle=":", linewidth=0.9)
            for x, v in enumerate(vals):
                ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=5.5, color="#52514e")
            ax.set_xticks(range(4), [l for _, l in labs])
            ax.set_title(title)
            ax.set_ylim(0, 110)
            ax.grid(axis="y", zorder=0)
        axes[0].text(3.4, PAPER["base_meter_accuracy_pct"]["Shaer"] + 2, "paper: Shaer", fontsize=5.5, ha="right")
        axes[0].text(3.4, PAPER["base_meter_accuracy_pct"]["Yehia"] + 2, "paper: untuned", fontsize=5.5, ha="right", color="#52514e")
        fig.tight_layout()
        fig.savefig(out / "fig_reproduction.png")
        plt.close(fig)
    # 5. nested comparison (robustness)
    V = R["variants"]
    order = [("full", "full criterion"), ("no_rhyme", "- rhyme"), ("no_length", "- length"), ("any_form", "- form consistency"),
             ("margin_1.0", "margin 1.0"), ("margin_3.0", "margin 3.0"), ("no_margin", "- margin rule"), ("binding", "binding mode")]
    show = [n for n in gen if n in V] + [n for n, _ in CONTROLS if n in V]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    w = 0.8 / len(show)
    ctrl_colors = {"control_wrong_meter": "#898781", "control_mixed_rhyme": "#52514e"}
    labels = dict(SYSTEMS + CONTROLS)
    for k, n in enumerate(show):
        vals = [100 * V[n][v][0] if v in V[n] else float("nan") for v, _ in order]
        xs = np.arange(len(order)) + (k - (len(show) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=COLORS.get(n, ctrl_colors.get(n)), label=labels[n],
               hatch=("///" if n.startswith("control") else None), edgecolor="#fcfcfb" if not n.startswith("control") else "#fcfcfb",
               linewidth=0.3, zorder=3)
    ax.set_xticks(range(len(order)), [l for _, l in order])
    ax.set_ylabel("strictly valid poems (%)")
    ax.grid(axis="y", zorder=0)
    ax.legend(ncol=4, frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_robustness.png")
    plt.close(fig)
    # 5b. nested structural models: McFadden R^2 per step, both criteria
    if "logit" in R and "nested" in R["logit"]:
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        for o, lab, col, mk in (("base", "base criterion (BiLSTM)", "#eb6834", "s"), ("ext", "extended criterion (strict)", "#2a78d6", "o")):
            rows = R["logit"]["nested"][o]
            xs = range(len(rows))
            ys = [r["mcfadden_r2"] for r in rows]
            ax.plot(xs, ys, marker=mk, markersize=4, linewidth=1.5, color=col, label=lab)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 5 if o == "ext" else -10), textcoords="offset points",
                            ha="center", fontsize=6, color="#52514e")
        ax.set_xticks(range(3), ["S1: system", "S2: + length", "S3: + meter"])
        ax.set_ylabel("McFadden $R^2$")
        lo = min(r["mcfadden_r2"] for o in ("base", "ext") for r in R["logit"]["nested"][o])
        hi = max(r["mcfadden_r2"] for o in ("base", "ext") for r in R["logit"]["nested"][o])
        ax.set_ylim(max(0, lo - 0.05), hi + 0.05)
        ax.set_xlim(-0.3, 2.3)
        ax.grid(axis="y")
        ax.legend(frameon=False, fontsize=6, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "fig_nested.png")
        plt.close(fig)
    # 6. M4 pre-registered ranges (results pending)
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    bon = {1: (5, 14), 2: (11, 17), 4: (15, 28), 8: (18, 40), 16: (20, 51)}
    xs = list(bon)
    ax.fill_between(xs, [bon[x][0] for x in xs], [bon[x][1] for x in xs], color="#eb6834", alpha=0.25, linewidth=0)
    ax.plot(xs, [sum(bon[x]) / 2 for x in xs], color="#eb6834", linestyle="--", linewidth=1.2, label="best-of-N (predicted range)")
    # search runs: vertical bar = predicted strict validity, horizontal bar = predicted budget range (M4 report, 1c)
    for (lo, hi, blo, bhi, ly, lab, col) in ((25, 75, 1.4, 2.5, 52, "search, 4 tries (k=1, R=4)", "#2a78d6"),
                                             (50, 96, 1.5, 3.0, 84, "search, 8 tries (k=1, R=8)", "#1baf7a")):
        bm = (blo * bhi) ** 0.5
        ax.errorbar([bm], [(lo + hi) / 2], yerr=[[(hi - lo) / 2], [(hi - lo) / 2]], xerr=[[bm - blo], [bhi - bm]],
                    fmt="o", color=col, markersize=4, capsize=2, linewidth=1.0)
        ax.text(bhi * 1.08, ly, lab, fontsize=6, va="center", color="#52514e")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("generated tokens per prompt (multiples of one sample)")
    ax.set_ylabel("strictly valid poems (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y")
    ax.set_title("pre-registered predictions; measured results pending", fontsize=6, color="#52514e")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_m4_predictions.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
