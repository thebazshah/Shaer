#!/usr/bin/env python3
"""M3: strict re-scoring of baseline poems with the prosody verifier.

    score      verify every poem of a scored run (rows of evaluation/score_meter_count.py) and write one record per poem
               and reading mode to <out-dir>/<system>/<mode>.jsonl, resumable row by row
    control    build a control "system" from a scored run: wrong_meter (each poem judged against another meter) or
               mixed_rhyme (bayts of two human poems with different rhymes joined), then score it like any system
    summarize  aggregate every system in <dir>: Shaer's own metrics (computed by evaluation/summarize_structural.py,
               unchanged) next to the strict ones, with bootstrap confidence intervals, paired differences, margins,
               breakdowns and the integrity checks; writes m3_summary.json and m3_report.md

Reading modes: "letters" judges the letters only (written vowels ignored; the conservative meter verdict, since a wrong
vowel is not counted), "binding" treats written vowels as binding, as a reader would.

Margins. A record keeps, for every bayt and every form of the requested meter, the cost of the chosen analysis and of
the best rival analysis of each hemistich, and how the bayt was resolved (a permitted ʿarūḍ/ḍarb pair, tadwīr, or no
analysis). Pair choice, tadwīr and the ḍarb/ʿarūḍ consistency check do not depend on the margin, so the meter verdict at
any margin is recomputed exactly from these numbers (`bayt_ok`, checked against the verifier in tests/test_m3.py). The
rhyme verdict uses the readings chosen at the default margin.

Integrity. <system>/<mode>.meta.json pins the verifier source, the lexicon and the record version; a resume with
different code or lexicon starts the file afresh (the old one is kept as .stale) instead of mixing results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))

from prosody import verify as _verify  # noqa: E402
from prosody.lexicon import Lexicon  # noqa: E402
from prosody.meters import METERS, NAME_TO_KEY, SHAER_FORM_TO_KEY, shaer_target  # noqa: E402
from prosody.verify import Verifier  # noqa: E402
from pv_text import rhyme_signatures, split_poem_lines  # noqa: E402

RECORD_VERSION = 1
MODES = ("letters", "binding")
MARGINS = (1.0, 1.5, 3.0, math.inf)
DEFAULT_MARGIN = _verify.MARGIN
MAX_COST = _verify.MAX_COST
SHIPPED_LEXICON = HERE / "data" / "lexicon_train.json.gz"
BUCKETS = ("1-3", "4-6", "7-10", "11-20")


# ---------------------------------------------------------------------------------------------- versions
def _sha(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).name.encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def code_meta(lexicon: Path, mode: str) -> dict:
    return {"record_version": RECORD_VERSION, "mode": mode, "verifier_sha": _sha((HERE / "prosody").glob("*.py")),
            "lexicon_sha": _sha([lexicon]), "margin": DEFAULT_MARGIN, "max_cost": MAX_COST}


# ---------------------------------------------------------------------------------------------- one poem
def bayt_ok(entry: list, margin: float) -> bool:
    """Meter verdict of one bayt at any margin, from its record entry [path, sadr cost, sadr best, ʿajuz cost, ʿajuz best].
    Mirrors Verifier.bayt: a permitted pair is accepted when both chosen analyses cost at most MAX_COST and at most
    `margin` more than the best rival for their position; tadwīr does not use the margin; no analysis never passes."""
    path, sc, sb, ac, ab = entry
    if path == "tadwir":
        return True
    if path != "pair":
        return False
    return sc <= MAX_COST and ac <= MAX_COST and sc - sb <= margin and ac - ab <= margin


def poem_meter_ok(form_rec: dict, margin: float) -> bool:
    """Verifier.poem's meter_ok at any margin: at least one bayt, every bayt accepted, ḍarb/ʿarūḍ form kept."""
    b = form_rec["bayts"]
    return bool(b) and form_rec["consistent"] and all(bayt_ok(e, margin) for e in b)


def _entry(bv) -> list:
    """[path, sadr cost, sadr best rival, ʿajuz cost, ʿajuz best rival] at full precision (no rounding: a verdict can sit
    exactly on the margin)."""
    if bv.path == "pair":
        s, a = bv.sadr, bv.ajuz
        return ["pair", s.match.cost, s.best_any.cost, a.match.cost, a.best_any.cost]
    return [bv.path or "none", None, None, None, None]


def _bilstm(row: dict, n_bayts: int, lines: list) -> tuple[list, int]:
    """Per-bayt BiLSTM verdicts (True/False, None when skipped or missing) and the number of bayts whose text differs
    from the verifier's pairing (an alignment check; it must be 0)."""
    try:
        details = json.loads(row.get("meter_details_json") or "{}").get("per_bayt_details", [])
    except (json.JSONDecodeError, TypeError):
        details = []
    target = row.get("meter_target_used") or row.get("base_meter")
    by_i = {d.get("bayt_index"): d for d in details}
    out, misaligned = [], 0
    for i in range(n_bayts):
        d = by_i.get(i)
        if d is None or d.get("skipped") or not d.get("pred"):
            out.append(None)
            continue
        out.append(d["pred"] == target)
        if d.get("bayt_text") is not None and d["bayt_text"] != f"{lines[2 * i]} // {lines[2 * i + 1]}":
            misaligned += 1
    return out, misaligned


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def score_row(row: dict, v_target: Verifier, v_forms: Verifier) -> dict:
    meter, form = shaer_target(row["base_meter"], row.get("form") or "تام")
    lines = split_poem_lines(row.get("generated_text") or "")
    req = row.get("requested_num_lines") or (int(row.get("requested_bayts") or 0) * 2) or None
    req = int(req) if req else None
    pv = v_target.poem(lines, meter, form, req)
    s = pv.summary()
    rec = {"key": f"{row.get('id')}#{int(row.get('sample_idx', 0) or 0)}", "id": row.get("id"),
           "sample_idx": int(row.get("sample_idx", 0) or 0), "meter": meter, "form": form,
           "requested_lines": req, "lines": len(lines), "requested_bayts": _num(row.get("requested_bayts")),
           "shaer_lines": _num(row.get("parsed_num_lines")), "shaer_meter": _num(row.get("meter")),
           "shaer_count": _num(row.get("count_adherence"))}
    forms = {}
    readable = [False] * len(pv.bayts)
    for fk in METERS[meter].forms:
        fv = pv if fk == form else v_forms.poem(lines, meter, fk, req)
        consistent = not any(i["kind"] in ("darb_changes", "arud_changes") for i in fv.issues)
        forms[fk] = {"bayts": [_entry(b) for b in fv.bayts], "consistent": consistent}
        for i, b in enumerate(fv.bayts):
            readable[i] = readable[i] or b.ok or (b.sadr.match is not None and b.ajuz.match is not None)
    rec["forms"] = forms
    rec["has_reading"] = readable
    rec["bilstm"], rec["misaligned_bayts"] = _bilstm(row, len(pv.bayts), lines)
    rec["target"] = {
        "bayt_ok": [b.ok for b in pv.bayts],
        "reasons": [b.reason for b in pv.bayts],
        "broken_foot": [[(h.nearest or {}).get("first_edit_foot") if not h.ok else None for h in (b.sadr, b.ajuz)]
                        for b in pv.bayts],
    }
    rec["meter_ok"], rec["rhyme_ok"], rec["count_ok"], rec["ok"] = pv.meter_ok, pv.rhyme_ok, pv.count_ok, pv.ok
    rec["even"] = len(lines) % 2 == 0
    rec["rhyme"] = {"rawi": s["rawi"], "refrain": s["refrain"],
                    "issues": sorted({i["kind"] for i in pv.rhyme.get("issues", [])}),
                    "tasri": bool(pv.rhyme.get("tasri"))}
    rec["issues"] = sorted({i["kind"] for i in pv.issues})
    # self-check: the margin recomputation reproduces the verifier's own verdicts at the default margin
    assert [bayt_ok(e, DEFAULT_MARGIN) for e in forms[form]["bayts"]] == rec["target"]["bayt_ok"], rec["key"]
    assert poem_meter_ok(forms[form], DEFAULT_MARGIN) == pv.meter_ok, rec["key"]
    return rec


# ---------------------------------------------------------------------------------------------- score (CLI)
_V: dict = {}


def _init(lex_path: str, mode: str) -> None:
    lex = Lexicon.load(lex_path)
    bare = mode == "letters"
    _V["target"] = Verifier(lex, ignore_diacritics=bare)
    _V["forms"] = Verifier(lex, ignore_diacritics=bare, diagnose=False).share_cache(_V["target"])


def _work(rows: list) -> list:
    out = []
    for r in rows:
        try:
            out.append(score_row(r, _V["target"], _V["forms"]))
        except KeyError as e:            # a meter/form label the tables do not know
            out.append({"key": f"{r.get('id')}#{int(r.get('sample_idx', 0) or 0)}", "id": r.get("id"),
                        "sample_idx": int(r.get("sample_idx", 0) or 0), "error": f"unknown label {e}"})
    return out


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:      # a line cut by an interrupted run: it is re-scored
                    pass
    return out


def score(scored: Path, system: str, out_dir: Path, lexicon: Path, modes, workers: int, limit: int, chunk: int = 10) -> None:
    rows = _read_jsonl(scored)
    rows.sort(key=lambda r: (str(r.get("id")), int(r.get("sample_idx", 0) or 0)))
    if limit:
        firsts = sorted({r["id"] for r in rows})[:limit]
        keep = set(firsts)
        rows = [r for r in rows if r["id"] in keep]
    sysdir = Path(out_dir) / system
    sysdir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        out, meta_path = sysdir / f"{mode}.jsonl", sysdir / f"{mode}.meta.json"
        meta = code_meta(lexicon, mode) | {"source": str(scored), "system": system}
        if out.exists() and meta_path.exists():
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if any(old.get(k) != meta[k] for k in ("record_version", "verifier_sha", "lexicon_sha", "mode", "margin", "max_cost")):
                stale = out.with_suffix(f".stale-{int(time.time())}.jsonl")
                out.rename(stale)
                print(f"  {system}/{mode}: verifier, lexicon or record version changed; old results kept as {stale.name}")
        elif out.exists():
            out.rename(out.with_suffix(f".stale-{int(time.time())}.jsonl"))
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        # keep only complete records (an interrupted write leaves at most one cut line, dropped by _read_jsonl)
        done_recs = _read_jsonl(out)
        done = {r["key"] for r in done_recs}
        if out.exists():
            with out.open("w", encoding="utf-8") as f:
                for r in done_recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        todo = [r for r in rows if f"{r.get('id')}#{int(r.get('sample_idx', 0) or 0)}" not in done]
        print(f"  {system}/{mode}: {len(rows):,} poems, {len(done):,} already scored, {len(todo):,} to score", flush=True)
        if not todo:
            continue
        chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        # spawn, not fork: a parent that already runs threads (numpy, a notebook) can deadlock a forked child
        t0, n = time.time(), 0
        with out.open("a", encoding="utf-8") as f, \
                mp.get_context("spawn").Pool(max(1, workers), initializer=_init, initargs=(str(lexicon), mode)) as pool:
            for k, part in enumerate(pool.imap_unordered(_work, chunks), 1):
                for r in part:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n += len(part)
                if k % 20 == 0 or k == len(chunks):
                    rate = n / max(1e-9, time.time() - t0)
                    print(f"  {system}/{mode}: {len(done) + n:,}/{len(rows):,} poems, {rate:.1f}/s, "
                          f"about {(len(todo) - n) / max(rate, 1e-9) / 60:.0f} min left", flush=True)


# ---------------------------------------------------------------------------------------------- controls
def _has_tam(name: str) -> bool:
    try:
        shaer_target(name, "تام")
        return True
    except KeyError:
        return False


SHAER_METERS = sorted(n for n in NAME_TO_KEY if _has_tam(n))


def control_rows(rows: list[dict], kind: str, seed: int = 11) -> list[dict]:
    """wrong_meter: each poem is judged against another meter (a fixed derangement of the meter labels, form تام).
    mixed_rhyme: for poems of at least 4 bayts, the first half of the bayts is joined with the second half of another
    poem of the same meter and form whose rawī (M0's written-form heuristic, majority over the ʿajuz lines) differs: the
    meter is intact and the rhyme broken, up to the heuristic's errors. BiLSTM fields are dropped."""
    rng = random.Random(seed)
    first = [r for r in rows if int(r.get("sample_idx", 0) or 0) == 0]
    base = {k: v for k, v in [("meter_details_json", None), ("meter", None), ("count_adherence", None),
                                ("meter_target_used", None), ("meter_predicted", None)]}
    out = []
    if kind == "wrong_meter":
        for r in first:
            m = r["base_meter"]
            i = SHAER_METERS.index(m) if m in SHAER_METERS else 0
            other = SHAER_METERS[(i + 1 + rng.randrange(len(SHAER_METERS) - 1)) % len(SHAER_METERS)]
            out.append(r | base | {"base_meter": other, "form": "تام", "control_of_meter": m})
        return out
    if kind == "mixed_rhyme":
        groups = defaultdict(list)
        for r in first:
            lines = split_poem_lines(r.get("generated_text") or "")
            if len(lines) >= 8 and len(lines) % 2 == 0:
                groups[(r["base_meter"], r.get("form"))].append((r, lines))
        def last(ls):        # the poem's rawī by the M0 heuristic (waṣl letters stripped), majority over its ʿajuz lines
            return Counter(rhyme_signatures(x)["rawi"] for x in ls[1::2]).most_common(1)[0][0]
        for _, g in sorted(groups.items()):
            for r, lines in g:
                cands = [(r2, l2) for r2, l2 in g if r2["id"] != r["id"] and last(l2) != last(lines)]
                if not cands:
                    continue
                r2, l2 = cands[rng.randrange(len(cands))]
                h = (len(lines) // 4) * 2
                mixed = lines[:h] + l2[-(len(lines) - h):] if len(l2) >= len(lines) - h else None
                if mixed is None or len(mixed) != len(lines):
                    continue
                out.append(r | base | {"generated_text": "\n".join(mixed), "control_of": [r["id"], r2["id"]]})
        return out
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------- summarize
def bucket(requested_bayts) -> str | None:
    if requested_bayts is None:
        return None
    n = int(requested_bayts)
    return "1-3" if n <= 3 else "4-6" if n <= 6 else "7-10" if n <= 10 else "11-20"


def poem_metrics(rec: dict, margin: float) -> dict:
    """Per-poem values; bayt-level ones are (numerator, denominator) pairs so that they are pooled over bayts."""
    tf = rec["forms"][rec["form"]]
    n = len(tf["bayts"])
    any_form = [any(bayt_ok(fr["bayts"][i], margin) for fr in rec["forms"].values() if i < len(fr["bayts"]))
                for i in range(n)]
    meter_ok = poem_meter_ok(tf, margin)
    meter_any = any(poem_meter_ok(fr, margin) for fr in rec["forms"].values())
    bil = [b for b in rec["bilstm"] if b is not None]
    return {
        "strict_valid": meter_ok and rec["rhyme_ok"] and rec["count_ok"] and rec["even"],
        "meter_ok": meter_ok,
        "meter_ok_any_form": meter_any,
        "rhyme_ok": rec["rhyme_ok"],
        "count_exact": rec["count_ok"],
        "even_lines": rec["even"],
        "bayt_ok": (sum(bayt_ok(e, margin) for e in tf["bayts"]), n),
        "bayt_ok_any_form": (sum(any_form), n),
        "no_reading": (sum(not x for x in rec["has_reading"]), n),
        "bilstm_bayt_ok": (sum(bil), len(bil)),
    }


POEM_KEYS = ("strict_valid", "meter_ok", "meter_ok_any_form", "rhyme_ok", "count_exact", "even_lines")
BAYT_KEYS = ("bayt_ok", "bayt_ok_any_form", "no_reading", "bilstm_bayt_ok")


def _arrays(per_poem: dict, ids: list, key: str):
    import numpy as np
    if key in POEM_KEYS:
        return np.array([float(per_poem[i][key]) for i in ids]), None
    return (np.array([per_poem[i][key][0] for i in ids], dtype=float), np.array([per_poem[i][key][1] for i in ids], dtype=float))


def _stat(values, key):
    """A metric over a list of per-poem values (percent): share of poems, or pooled share of bayts."""
    if key in POEM_KEYS:
        return 100.0 * sum(values) / max(1, len(values))
    num = sum(v[0] for v in values)
    den = sum(v[1] for v in values)
    return 100.0 * num / den if den else float("nan")


_IDX: dict = {}


def _resample_index(n: int, reps: int, seed: int):
    """One matrix of resampled prompt positions per (n, reps, seed), shared by every metric and system of that size, so
    that differences between systems on the same prompts are paired."""
    import numpy as np
    k = (n, reps, seed)
    if k not in _IDX:
        _IDX[k] = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    return _IDX[k]


def bootstrap(per_poem: dict, ids: list, key: str, reps: int, seed: int, other: dict | None = None) -> tuple:
    """Point estimate and 95% percentile interval over prompts (poems resampled with replacement; bayt-level metrics are
    pooled within each resample, i.e. a cluster bootstrap). With `other`, the paired difference self - other on `ids`
    (both resampled with the same indices)."""
    import numpy as np
    n = len(ids)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = _resample_index(n, reps, seed)

    def series(pp):
        v, d = _arrays(pp, ids, key)
        with np.errstate(invalid="ignore", divide="ignore"):
            if d is None:
                return 100.0 * v.mean(), 100.0 * v[idx].mean(axis=1)
            return (100.0 * v.sum() / d.sum() if d.sum() else float("nan")), 100.0 * v[idx].sum(axis=1) / d[idx].sum(axis=1)

    point, est = series(per_poem)
    if other is not None:
        p2, e2 = series(other)
        point, est = point - p2, est - e2
    if point != point:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return (round(float(point), 2), round(float(lo), 2), round(float(hi), 2))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): probability that at least one of k samples drawn from n passes."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_system(sysdir: Path, mode: str) -> tuple[list[dict], dict]:
    recs = _read_jsonl(sysdir / f"{mode}.jsonl")
    meta_p = sysdir / f"{mode}.meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    seen, out = set(), []
    for r in recs:                      # a resumed file cannot hold duplicates, but never count one twice
        if r["key"] not in seen:
            seen.add(r["key"])
            out.append(r)
    return out, meta


def summarize(m3_dir: Path, reps: int = 1000, seed: int = 7, reference: str = "reference") -> dict:
    import summarize_structural as ss        # the repository's own definitions, unchanged
    m3_dir = Path(m3_dir)
    systems = sorted(p.name for p in m3_dir.iterdir() if p.is_dir() and any(p.glob("*.jsonl")))
    res = {"systems": {}, "checks": {}, "config": {"margins": [str(m) for m in MARGINS], "default_margin": DEFAULT_MARGIN,
                                                   "bootstrap_reps": reps, "seed": seed}}
    per_poem_all: dict = {}
    for name in systems:
        sres = res["systems"][name] = {}
        for mode in MODES:
            recs, meta = load_system(m3_dir / name, mode)
            if not recs:
                continue
            errors = [r for r in recs if "error" in r]
            recs = [r for r in recs if "error" not in r]
            first = [r for r in recs if r["sample_idx"] == 0]
            ids = sorted(r["id"] for r in first)
            mres = sres[mode] = {"poems": len(first), "records": len(recs), "errors": len(errors), "meta": meta,
                                 "samples_per_prompt": max(Counter(r["id"] for r in recs).values()) if recs else 0}
            # integrity checks
            mres["checks"] = {
                "line_count_mismatches": sum(1 for r in recs if r["shaer_lines"] is not None and r["shaer_lines"] != r["lines"]),
                "misaligned_bilstm_bayts": sum(r["misaligned_bayts"] for r in recs),
                "default_margin_recomputation_mismatches": sum(
                    poem_meter_ok(r["forms"][r["form"]], DEFAULT_MARGIN) != r["meter_ok"] for r in recs),
            }
            for mg in MARGINS:
                pp = {r["id"]: poem_metrics(r, mg) for r in first}
                per_poem_all[(name, mode, mg)] = pp
                mres[f"margin_{mg}"] = {k: bootstrap(pp, ids, k, reps, seed) for k in POEM_KEYS + BAYT_KEYS}
            pp = per_poem_all[(name, mode, DEFAULT_MARGIN)]
            # cross-check with M1 (compare_bilstm.py): the same rates on the bayts the BiLSTM scored
            sub = [0, 0, 0, 0]
            for r in first:
                for i, bl in enumerate(r["bilstm"]):
                    if bl is None:
                        continue
                    sub[0] += 1
                    sub[1] += bayt_ok(r["forms"][r["form"]]["bayts"][i], DEFAULT_MARGIN)
                    sub[2] += any(bayt_ok(fr["bayts"][i], DEFAULT_MARGIN) for fr in r["forms"].values() if i < len(fr["bayts"]))
                    sub[3] += not r["has_reading"][i]
            if sub[0]:
                mres["m1_crosscheck"] = {"bilstm_scored_bayts": sub[0], "verifier_pct": round(100 * sub[1] / sub[0], 2),
                                         "verifier_any_form_pct": round(100 * sub[2] / sub[0], 2),
                                         "no_reading_pct": round(100 * sub[3] / sub[0], 2)}
            # Shaer's own metrics, by the repository's definitions, on the first sample of each prompt
            rows_like = [{"meter": r["shaer_meter"], "count_adherence": r["shaer_count"]} for r in first]
            if all(r["shaer_meter"] is not None for r in first):
                mres["shaer_meter_conformity"] = round(sum(r["shaer_meter"] for r in first) / len(first), 4)
                mres["shaer_count_adherence"] = round(sum(r["shaer_count"] or 0.0 for r in first) / len(first), 4)
                mres["shaer_exact_count_pct"] = round(100 * sum(r["shaer_lines"] == r["requested_lines"] for r in first) / len(first), 2)
            # rhyme issues (share of poems), length buckets, meters, bayt position, broken foot
            iss = Counter(k for r in first for k in r["rhyme"]["issues"] + r["issues"])
            mres["issues_pct"] = {k: round(100 * v / len(first), 2) for k, v in iss.most_common()}
            mres["refrain_pct"] = round(100 * sum(bool(r["rhyme"]["refrain"]) for r in first) / len(first), 2)
            by_b = defaultdict(list)
            for r in first:
                by_b[bucket(r["requested_bayts"])].append(r["id"])
            mres["by_length"] = {b: {"poems": len(by_b[b]), **{k: _round(_stat([pp[i][k] for i in by_b[b]], k))
                                                               for k in ("strict_valid", "meter_ok", "rhyme_ok", "count_exact",
                                                                         "bayt_ok", "bilstm_bayt_ok")}}
                                 for b in BUCKETS if by_b.get(b)}
            by_m = defaultdict(list)
            for r in first:
                by_m[r["meter"]].append(r["id"])
            mres["by_meter"] = {m: {"poems": len(v), **{k: _round(_stat([pp[i][k] for i in v], k))
                                                        for k in ("strict_valid", "meter_ok", "rhyme_ok", "bayt_ok",
                                                                  "bilstm_bayt_ok")}}
                                for m, v in sorted(by_m.items(), key=lambda kv: -len(kv[1]))}
            pos = defaultdict(lambda: [0, 0])
            for r in first:
                for i, e in enumerate(r["forms"][r["form"]]["bayts"]):
                    p = min(i, 9)
                    pos[p][0] += bayt_ok(e, DEFAULT_MARGIN)
                    pos[p][1] += 1
            mres["bayt_ok_by_position"] = {("10+" if p == 9 else str(p + 1)): round(100 * a / b, 2) for p, (a, b) in sorted(pos.items())}
            foot = Counter(f for r in first for pair in r["target"]["broken_foot"] for f in pair if f is not None)
            tot = sum(foot.values())
            mres["first_broken_foot_pct"] = {str(k): round(100 * v / tot, 2) for k, v in sorted(foot.items())} if tot else {}
            # several samples per prompt: unbiased pass@k for strict validity and for the meter alone
            if mres["samples_per_prompt"] > 1:
                by_id = defaultdict(list)
                for r in recs:
                    by_id[r["id"]].append(r)
                n_s = min(len(v) for v in by_id.values())
                pk = {}
                for k in sorted({1, 2, 4, 8, 16, n_s} & set(range(1, n_s + 1))):
                    for key in ("strict_valid", "meter_ok"):
                        vals = []
                        for v in by_id.values():
                            c = sum(poem_metrics(r, DEFAULT_MARGIN)[key] for r in v[:n_s])
                            vals.append(pass_at_k(n_s, c, k))
                        pk[f"{key}@{k}"] = round(100 * sum(vals) / len(vals), 2)
                mres["pass_at_k"] = pk
                mres["pass_at_k_samples"] = n_s
    # official structural summary of each source file (the repository's function), to set beside the M3 numbers
    for name in systems:
        for mode in MODES:
            m = res["systems"][name].get(mode)
            if m and m.get("meta", {}).get("source") and Path(m["meta"]["source"]).exists():
                ids = {r["id"] for r in load_system(m3_dir / name, mode)[0] if r.get("sample_idx") == 0}
                rows = [r for r in ss.read_jsonl(Path(m["meta"]["source"]))
                        if int(r.get("sample_idx", 0) or 0) == 0 and r.get("id") in ids]
                if rows and all(r.get("meter") not in (None, "") for r in rows):
                    off = ss.summarise(rows)
                    for mode2 in MODES:
                        m2 = res["systems"][name].get(mode2)
                        if not m2:
                            continue
                        m2["official_structural"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in off.items()}
                        # M3's own exact-length rate must be the repository's exact count accuracy on the same poems
                        # (unrounded on both sides: the reported point estimates are rounded to 0.01)
                        pp2 = per_poem_all[(name, mode2, DEFAULT_MARGIN)]
                        own = 100.0 * sum(v["count_exact"] for v in pp2.values()) / len(pp2)
                        m2["checks"]["exact_count_vs_official_pp"] = round(own - off["exact_count_accuracy_pct"], 9)
                break
    # paired differences against the reference poems (same prompts)
    if reference in res["systems"]:
        for name in systems:
            if name == reference:
                continue
            for mode in MODES:
                a, b = per_poem_all.get((name, mode, DEFAULT_MARGIN)), per_poem_all.get((reference, mode, DEFAULT_MARGIN))
                if not a or not b:
                    continue
                ids = sorted(set(a) & set(b))
                res["systems"][name][mode]["vs_reference"] = {
                    "prompts": len(ids), **{k: bootstrap(a, ids, k, reps, seed, other=b)
                                            for k in ("strict_valid", "meter_ok", "rhyme_ok", "count_exact", "bayt_ok",
                                                      "no_reading", "bilstm_bayt_ok")}}
    res["checks"] = {f"{n}/{m}": v["checks"] for n, s in res["systems"].items() for m, v in s.items()}
    return res


def _round(x):
    return None if x != x else round(x, 2)


def _fmt(t) -> str:
    if t is None or t[0] != t[0]:
        return "–"
    return f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"


def render(res: dict) -> str:
    L = ["# M3 — strict re-scoring with the prosody verifier", ""]
    a = L.append
    a(f"Margin {DEFAULT_MARGIN} unless stated; 95% bootstrap intervals over prompts ({res['config']['bootstrap_reps']} "
      "resamples) in brackets. First sample per prompt. *letters*: written vowels ignored (the conservative meter verdict); "
      "*binding*: written vowels count.")
    a("")
    a("**Strictly valid** = every bayt in the requested meter and form (ḍarb/ʿarūḍ form kept), rhyme check passes, exact "
      "number of hemistichs, no odd hemistich. *No reading*: share of bayts with a hemistich that cannot be read in the "
      "requested meter at all (any form, any cost; independent of the margin).")
    a("")
    for mode in MODES:
        rows = [(n, s[mode]) for n, s in res["systems"].items() if mode in s]
        if not rows:
            continue
        a(f"## Poems, {mode}")
        a("")
        a("| system | poems | strictly valid | meter (form) | meter (any form) | rhyme | exact length |")
        a("|---|---:|---:|---:|---:|---:|---:|")
        for n, m in rows:
            g = m[f"margin_{DEFAULT_MARGIN}"]
            a(f"| {n} | {m['poems']:,} | {_fmt(g['strict_valid'])} | {_fmt(g['meter_ok'])} | {_fmt(g['meter_ok_any_form'])} | "
              f"{_fmt(g['rhyme_ok'])} | {_fmt(g['count_exact'])} |")
        a("")
        a(f"## Bayts, {mode}")
        a("")
        a("| system | bayts on meter (verifier) | any form | no reading in the meter | BiLSTM (Shaer metric) |")
        a("|---|---:|---:|---:|---:|")
        for n, m in rows:
            g = m[f"margin_{DEFAULT_MARGIN}"]
            a(f"| {n} | {_fmt(g['bayt_ok'])} | {_fmt(g['bayt_ok_any_form'])} | {_fmt(g['no_reading'])} | {_fmt(g['bilstm_bayt_ok'])} |")
        a("")
        a(f"**Margin sensitivity ({mode})**: strictly valid poems / bayts on meter")
        a("")
        a("| system | " + " | ".join(f"margin {'none' if math.isinf(mg) else mg}" for mg in MARGINS) + " |")
        a("|---|" + "---:|" * len(MARGINS))
        for n, m in rows:
            a(f"| {n} | " + " | ".join(f"{m[f'margin_{mg}']['strict_valid'][0]:.1f} / {m[f'margin_{mg}']['bayt_ok'][0]:.1f}"
                                      for mg in MARGINS) + " |")
        a("")
        diffs = [(n, m["vs_reference"]) for n, m in rows if "vs_reference" in m]
        if diffs:
            a(f"**Paired difference to the human reference poems ({mode})**, percentage points:")
            a("")
            a("| system | prompts | strictly valid | meter | rhyme | exact length | bayts on meter | no reading | BiLSTM bayts |")
            a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for n, d in diffs:
                a(f"| {n} | {d['prompts']:,} | " + " | ".join(_fmt(d[k]) for k in ("strict_valid", "meter_ok", "rhyme_ok",
                                                                                 "count_exact", "bayt_ok", "no_reading",
                                                                                 "bilstm_bayt_ok")) + " |")
            a("")
    mode = "letters"
    rows = [(n, s[mode]) for n, s in res["systems"].items() if mode in s]
    if rows:
        a("## Shaer's own metrics (repository definitions) next to the strict ones (letters)")
        a("")
        a("| system | meter conformity | base-meter accuracy | exact count | count adherence | strictly valid | meter (form) |")
        a("|---|---:|---:|---:|---:|---:|---:|")
        for n, m in rows:
            o = m.get("official_structural")
            g = m[f"margin_{DEFAULT_MARGIN}"]
            if o:
                a(f"| {n} | {o['meter_conformity']} | {o['base_meter_accuracy_pct']:.2f}% | {o['exact_count_accuracy_pct']:.2f}% | "
                  f"{o['count_adherence']} | {g['strict_valid'][0]:.1f}% | {g['meter_ok'][0]:.1f}% |")
        a("")
        a("## By requested length (letters)")
        a("")
        a("| system | bayts | poems | strictly valid | meter | rhyme | exact length | bayts on meter | BiLSTM bayts |")
        a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for n, m in rows:
            for b, v in m["by_length"].items():
                a(f"| {n} | {b} | {v['poems']:,} | " + " | ".join("–" if v[k] is None else str(v[k]) for k in (
                    "strict_valid", "meter_ok", "rhyme_ok", "count_exact", "bayt_ok", "bilstm_bayt_ok")) + " |")
        a("")
        a("## Bayts on meter by position in the poem (letters)")
        a("")
        posk = sorted({p for _, m in rows for p in m["bayt_ok_by_position"]}, key=lambda p: (len(p), p))
        a("| system | " + " | ".join(posk) + " |")
        a("|---|" + "---:|" * len(posk))
        for n, m in rows:
            a(f"| {n} | " + " | ".join(str(m["bayt_ok_by_position"].get(p, "–")) for p in posk) + " |")
        a("")
        a("## Rhyme and form issues (letters, share of poems)")
        a("")
        kinds = sorted({k for _, m in rows for k in m["issues_pct"]})
        a("| system | " + " | ".join(kinds) + " |")
        a("|---|" + "---:|" * len(kinds))
        for n, m in rows:
            a(f"| {n} | " + " | ".join(str(m["issues_pct"].get(k, 0.0)) for k in kinds) + " |")
        a("")
        a("## By meter (letters; strictly valid / bayts on meter / BiLSTM bayts)")
        a("")
        meters = sorted({mk for _, m in rows for mk in m["by_meter"]}, key=lambda mk: -max(m["by_meter"].get(mk, {}).get("poems", 0) for _, m in rows))
        a("| meter | " + " | ".join(n for n, _ in rows) + " |")
        a("|---|" + "---|" * len(rows))
        for mk in meters:
            cells = []
            for _, m in rows:
                v = m["by_meter"].get(mk)
                cells.append(" / ".join("–" if v[k] is None else str(v[k]) for k in ("strict_valid", "bayt_ok", "bilstm_bayt_ok"))
                             + f" ({v['poems']})" if v else "–")
            a(f"| {mk} | " + " | ".join(cells) + " |")
        a("")
        pk = [(n, m) for n, m in rows if "pass_at_k" in m]
        if pk:
            a("## Several samples per prompt (letters, unbiased pass@k)")
            a("")
            for n, m in pk:
                a(f"* {n} ({m['pass_at_k_samples']} samples): " + ", ".join(f"{k} {v}%" for k, v in m["pass_at_k"].items()))
            a("")
    a("## Integrity checks (all must be 0)")
    a("")
    a("| system/mode | line counts differing from Shaer's | BiLSTM bayts misaligned | margin recomputation mismatches | "
      "exact length vs official (pp) | errors |")
    a("|---|---:|---:|---:|---:|---:|")
    for k, c in res["checks"].items():
        n, mode = k.split("/")
        a(f"| {k} | {c['line_count_mismatches']} | {c['misaligned_bilstm_bayts']} | "
          f"{c['default_margin_recomputation_mismatches']} | {c.get('exact_count_vs_official_pp', '–')} | "
          f"{res['systems'][n][mode]['errors']} |")
    a("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--scored", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lexicon", default=str(SHIPPED_LEXICON))
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    p.add_argument("--limit", type=int, default=0, help="first N prompts (by id) only; 0 = all")
    p = sub.add_parser("control")
    p.add_argument("--scored", required=True)
    p.add_argument("--kind", required=True, choices=("wrong_meter", "mixed_rhyme"))
    p.add_argument("--out", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--dir", required=True)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.cmd == "score":
        score(Path(a.scored), a.system, Path(a.out_dir), Path(a.lexicon), [m for m in a.modes.split(",") if m], a.workers, a.limit)
    elif a.cmd == "control":
        rows = control_rows(_read_jsonl(Path(a.scored)), a.kind)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(a.out).with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(a.out)
        print(f"  {a.kind} control: {len(rows):,} poems -> {a.out}")
    else:
        res = summarize(Path(a.dir), a.reps, a.seed)
        d = Path(a.dir)
        (d / "m3_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "m3_report.md").write_text(render(res), encoding="utf-8")
        print(render(res))


if __name__ == "__main__":
    main()
