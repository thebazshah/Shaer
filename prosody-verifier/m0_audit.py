#!/usr/bin/env python3
"""M0 headroom audit: where do the baseline poems fail, and how much room is left?

Works on rows scored by the repository's own evaluator (evaluation/score_meter_count.py, or the batched wrapper
meter_score_batched.py in this folder, which calls the same code), which hold the generated text, the requested length and
the BiLSTM meter classifier's per-bayt predictions (`meter_details_json`). Standard library only.

Three sub-commands:

  reference  turn released test rows (which carry the human `reference_completion`) into rows whose `generated_text` is the
             human poem, so the same evaluator and audit can measure the human ceiling.
  audit      per-poem diagnostics + a summary for one system:
               count    exact / short / long / odd final hemistich, by length bucket
               meter    share of bayts the classifier puts on the target meter, poems with every bayt on target, poems whose
                        bayts get more than one meter, off-target rate by bayt position, by meter
               rhyme    HEURISTIC rhyme-ending consistency across the second hemistichs (see pv_text.rhyme_signatures)
               diacritics  share of letters carrying a haraka
               strict-form (heuristic)  exact count AND every bayt on target AND one rhyme ending
  report     combine the per-system summaries into one table (JSON + Markdown).

The rhyme numbers are heuristics, NOT the verifier: they are only meant to show whether rhyme is a gap worth closing, and must
be read next to the reference poems' numbers (the same heuristic applied to human poems).

Usage:
  python m0_audit.py reference --rows runs/e0/scored.jsonl --out m0/reference/generations.jsonl
  python m0_audit.py audit --scored runs/e0/scored.jsonl --system e0_released_shaer --out-dir m0/e0_released_shaer
  python m0_audit.py report --m0-dir m0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_text as T  # noqa: E402

BUCKETS = ["1-3", "4-6", "7-10", "11-20"]
POSITIONS = [("bayt 1", 0, 0), ("bayts 2-3", 1, 2), ("bayts 4-6", 3, 5), ("bayts 7+", 6, 10 ** 6)]
FULL_COVERAGE = 0.9           # a hemistich counts as "fully diacritised" at or above this share


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def bucket(requested_bayts: int) -> str:
    """Same buckets as evaluation/summarize_structural.py."""
    n = int(requested_bayts)
    return "1-3" if n <= 3 else "4-6" if n <= 6 else "7-10" if n <= 10 else "11-20"


def requested_lines(row: dict) -> int:
    for key in ("requested_num_lines", "sft_num_lines", "requested_lines"):
        if str(row.get(key) or "").strip():
            return int(row[key])
    return 2 * int(row.get("requested_bayts") or 0)


def first_sample_only(rows: list[dict]) -> tuple[list[dict], int]:
    """Runs with several samples per prompt are audited on the first sample, matching the paper's one-poem protocol."""
    samples = {int(r.get("sample_idx", 0) or 0) for r in rows}
    if len(samples) <= 1:
        return rows, len(samples) or 1
    first = min(samples)
    return [r for r in rows if int(r.get("sample_idx", 0) or 0) == first], len(samples)


# ---------------------------------------------------------------------------------------------- per-poem diagnostics
def audit_row(row: dict) -> dict:
    lines = T.split_poem_lines(row.get("generated_text") or row.get("completion") or "")
    want = requested_lines(row)
    n = len(lines)
    req_bayts = int(row.get("requested_bayts") or want // 2)
    out: dict = {
        "id": row.get("id"),
        "base_meter": row.get("base_meter"),
        "form": row.get("form"),
        "length_bucket": row.get("length_bucket") or bucket(req_bayts),
        "requested_lines": want,
        "lines": n,
        "count_delta": n - want,
        "count_status": "exact" if n == want else ("short" if n < want else "long"),
        "odd_tail": n % 2 == 1,
    }

    # meter: the scorer's per-bayt predictions (absent for rows that were never scored)
    details = None
    if row.get("meter_details_json"):
        try:
            details = json.loads(row["meter_details_json"]).get("per_bayt_details")
        except (json.JSONDecodeError, AttributeError):
            details = None
    if details is not None:
        target = str(row.get("meter_target_used") or row.get("base_meter") or "")
        flags = [(d.get("bayt_index", i), d.get("pred") == target)
                 for i, d in enumerate(details) if not d.get("skipped") and d.get("pred")]
        preds = [d.get("pred") for d in details if not d.get("skipped") and d.get("pred")]
        out.update({
            "meter_target": target,
            "meter_bayts_scored": len(flags),
            "meter_bayts_skipped": sum(1 for d in details if d.get("skipped")),
            "meter_bayts_on_target": sum(ok for _, ok in flags),
            "meter_all_on_target": bool(flags) and all(ok for _, ok in flags),
            "meter_distinct_preds": len(set(preds)),
            "meter_majority_pred": Counter(preds).most_common(1)[0][0] if preds else "",
            "meter_flags": [[i, ok] for i, ok in flags],   # [bayt index, on target] for every scored bayt
            "meter_conformity": row.get("meter"),
        })

    # rhyme (heuristic): the second hemistich of each complete bayt; the opening bayt's first hemistich only for tasri'
    second = [lines[i + 1] for i in range(0, n - 1, 2)]
    sigs = [T.rhyme_signatures(x) for x in second]
    for key in ("last_letter", "rawi"):
        c = T.consistency([s[key] for s in sigs])
        out[f"rhyme_{key}_share"] = c["share"]
        out[f"rhyme_{key}_all_same"] = c["all_same"]
        out[f"rhyme_{key}_majority"] = c["majority"]
    rawi_majority = out["rhyme_rawi_majority"]
    vowels = [s["rawi_vowel"] for s in sigs if s["rawi"] == rawi_majority and s["rawi_vowel"]]
    cv = T.consistency(vowels)
    out["rhyme_rawi_vowel_n"] = cv["n"]
    out["rhyme_rawi_vowel_all_same"] = cv["all_same"] if cv["n"] >= 2 else None
    out["rhyme_bayts"] = len(second)
    if n >= 2:
        out["tasri"] = T.rhyme_signatures(lines[0])["rawi"] == T.rhyme_signatures(lines[1])["rawi"]

    # diacritics
    marked = eligible = full = 0
    for x in lines:
        m, e = T.diacritic_coverage(x)
        marked, eligible = marked + m, eligible + e
        full += bool(e) and m / e >= FULL_COVERAGE
    out["diacritic_coverage"] = marked / eligible if eligible else None
    out["hemistichs_fully_diacritised"] = full
    out["hemistichs"] = n

    # strict-form pass (heuristic): what a perfect system would satisfy on every poem
    if details is not None:
        out["strict_form_heuristic"] = (out["count_status"] == "exact" and out["meter_all_on_target"]
                                        and bool(out["rhyme_rawi_all_same"]))
    return out


# ---------------------------------------------------------------------------------------------- summary
def _pct(xs) -> float | None:
    xs = [bool(x) for x in xs if x is not None]
    return round(100.0 * sum(xs) / len(xs), 2) if xs else None


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise(audits: list[dict]) -> dict:
    n = len(audits)
    s: dict = {"poems": n}
    s["count"] = {
        "exact_pct": _pct(a["count_status"] == "exact" for a in audits),
        "short_pct": _pct(a["count_status"] == "short" for a in audits),
        "long_pct": _pct(a["count_status"] == "long" for a in audits),
        "odd_tail_pct": _pct(a["odd_tail"] for a in audits),
        "mean_abs_delta_hemistichs": _mean(abs(a["count_delta"]) for a in audits),
        "delta_histogram": {str(k): v for k, v in sorted(Counter(max(-6, min(6, a["count_delta"])) for a in audits).items())},
        "exact_pct_by_bucket": {b: _pct(a["count_status"] == "exact" for a in audits if a["length_bucket"] == b)
                                for b in BUCKETS},
    }
    scored = [a for a in audits if "meter_bayts_scored" in a]
    if scored:
        tot = sum(a["meter_bayts_scored"] for a in scored)
        pos_rates = {}
        for name, lo, hi in POSITIONS:
            hits = [ok for a in scored for i, ok in a["meter_flags"] if lo <= i <= hi]
            if hits:
                pos_rates[name] = round(100.0 * sum(not ok for ok in hits) / len(hits), 2)
        s["meter"] = {
            "poems_scored": len(scored),
            "conformity_mean": _mean(a.get("meter_conformity") for a in scored),
            "bayts_scored": tot,
            "bayts_on_target_pct": round(100.0 * sum(a["meter_bayts_on_target"] for a in scored) / tot, 2) if tot else None,
            "poems_all_bayts_on_target_pct": _pct(a["meter_all_on_target"] for a in scored),
            "poems_majority_on_target_pct": _pct(a["meter_majority_pred"] == a["meter_target"] for a in scored),
            "poems_more_than_one_meter_pct": _pct(a["meter_distinct_preds"] > 1 for a in scored),
            "off_target_bayt_pct_by_position": pos_rates,
            "poems_all_bayts_on_target_pct_by_meter": {
                m: _pct(a["meter_all_on_target"] for a in scored if a["base_meter"] == m)
                for m in sorted({a["base_meter"] for a in scored})},
        }
        s["strict_form_heuristic_pct"] = _pct(a.get("strict_form_heuristic") for a in scored)
    rh = [a for a in audits if a["rhyme_bayts"] >= 2]
    s["rhyme_heuristic"] = {
        "poems_with_2plus_bayts": len(rh),
        "rawi_all_same_pct": _pct(a["rhyme_rawi_all_same"] for a in rh),
        "rawi_mean_share": _mean(a["rhyme_rawi_share"] for a in rh),
        "last_letter_all_same_pct": _pct(a["rhyme_last_letter_all_same"] for a in rh),
        "rawi_vowel_all_same_pct": _pct(a["rhyme_rawi_vowel_all_same"] for a in rh),
        "rawi_all_same_pct_by_bucket": {b: _pct(a["rhyme_rawi_all_same"] for a in rh if a["length_bucket"] == b)
                                        for b in BUCKETS},
        "tasri_pct": _pct(a.get("tasri") for a in audits),
        # الرجز is often composed as an urjuza (each bayt rhymes internally, not across the poem), so read it separately
        "rawi_all_same_pct_by_meter": {m: _pct(a["rhyme_rawi_all_same"] for a in rh if a["base_meter"] == m)
                                       for m in sorted({a["base_meter"] for a in rh})},
    }
    hem = sum(a["hemistichs"] for a in audits)
    s["diacritics"] = {
        "mean_coverage": _mean(a["diacritic_coverage"] for a in audits),
        "hemistichs_fully_diacritised_pct": round(100.0 * sum(a["hemistichs_fully_diacritised"] for a in audits) / hem, 2)
        if hem else None,
    }
    return s


# ---------------------------------------------------------------------------------------------- commands
def cmd_reference(args) -> None:
    rows = read_jsonl(Path(args.rows))
    keep = ("id", "base_meter", "form", "requested_bayts", "requested_num_lines", "length_bucket", "description")
    out = []
    for r in rows:
        if not r.get("reference_completion"):
            continue
        new = {k: r[k] for k in keep if k in r}
        new["generated_text"] = r["reference_completion"]
        new["system"] = "reference"
        out.append(new)
    n = write_jsonl(Path(args.out), out)
    print(f"reference rows: {n} (of {len(rows)} input rows) -> {args.out}")


def cmd_audit(args) -> None:
    rows, k = first_sample_only(read_jsonl(Path(args.scored)))
    audits = [audit_row(r) for r in rows]
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "audit_rows.jsonl", audits)
    summary = {"system": args.system, "source": str(args.scored), "samples_per_prompt_in_source": k,
               "note": "rhyme figures are written-form heuristics, not the prosody verifier", **summarise(audits)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.system}: audited {len(audits)} poems -> {out_dir}")


ROWS = [  # (label, path into the summary)
    ("poems", ("poems",)),
    ("exact count %", ("count", "exact_pct")),
    ("too short %", ("count", "short_pct")),
    ("too long %", ("count", "long_pct")),
    ("odd final hemistich %", ("count", "odd_tail_pct")),
    ("meter conformity (BiLSTM)", ("meter", "conformity_mean")),
    ("bayts on target %", ("meter", "bayts_on_target_pct")),
    ("poems: every bayt on target %", ("meter", "poems_all_bayts_on_target_pct")),
    ("poems: majority on target %", ("meter", "poems_majority_on_target_pct")),
    ("poems: >1 meter across bayts %", ("meter", "poems_more_than_one_meter_pct")),
    ("rhyme: one rawi (heuristic) %", ("rhyme_heuristic", "rawi_all_same_pct")),
    ("rhyme: one final letter (heuristic) %", ("rhyme_heuristic", "last_letter_all_same_pct")),
    ("rhyme: one rawi vowel (heuristic) %", ("rhyme_heuristic", "rawi_vowel_all_same_pct")),
    ("opening bayt rhymes (tasri') %", ("rhyme_heuristic", "tasri_pct")),
    ("diacritic coverage", ("diacritics", "mean_coverage")),
    ("hemistichs >=90% diacritised %", ("diacritics", "hemistichs_fully_diacritised_pct")),
    ("strict form: count+meter+rhyme (heuristic) %", ("strict_form_heuristic_pct",)),
]


def _get(d: dict, path: tuple):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def cmd_report(args) -> None:
    m0 = Path(args.m0_dir)
    summaries = {}
    for p in sorted(m0.glob("*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        summaries[s["system"]] = s
    order = sorted(summaries, key=lambda k: (k != "reference", k))
    head = "| metric | " + " | ".join(order) + " |\n|---|" + "---:|" * len(order) + "\n"
    body = "".join("| " + label + " | " + " | ".join("" if _get(summaries[k], path) is None else str(_get(summaries[k], path))
                                                    for k in order) + " |\n" for label, path in ROWS)
    md = ("# M0 headroom audit\n\nRhyme rows are written-form heuristics (not the prosody verifier); read them against the "
          "`reference` column (human poems, same heuristic).\n\n" + head + body)
    (m0 / "m0_report.md").write_text(md, encoding="utf-8")
    (m0 / "m0_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--rows", required=True)
    r.add_argument("--out", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--scored", required=True)
    a.add_argument("--system", required=True)
    a.add_argument("--out-dir", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--m0-dir", required=True)
    args = ap.parse_args()
    {"reference": cmd_reference, "audit": cmd_audit, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
