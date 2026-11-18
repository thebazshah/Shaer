#!/usr/bin/env python3
"""Validate the prosody verifier on human poems (positives) and on systematically broken copies of them (negatives).

Human poems from a held-out split (eval to tune, test to report) are taken as correct verse in their labelled meter. The
corpus labels are scraped, so a small share of them is wrong; the numbers below are therefore lower bounds on the true
acceptance rate. For every hemistich:

  acceptance        does the verifier accept it in its labelled meter (and form)?            (want: high)
  wrong meters      in how many of the other meters would it be accepted?                    (want: low)
  identification    is the labelled meter the verifier's best analysis?
For broken copies (negatives) of hemistichs the verifier accepts, the share it now rejects:     (want: high)
  drop_letter   one letter removed from a word        drop_word   one word removed
  add_word      a word from another poem inserted     swap_word   one word replaced by a word of another length
Poem level: meter_ok, rhyme_ok and the rhyme negatives (the last word of one ʿajuz replaced by another poem's rhyme word
with a different rawī: the rhyme check must flag it).

Usage:
  python validate_verifier.py --poems eval.jsonl --lexicon lexicon.json.gz --out report_eval.json [--limit 1000]
  (`--poems` rows: {"verses": [...], "base_meter": ..., "form": ...}; or Shaer rows with `generated_text`)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prosody import arudi  # noqa: E402
from prosody.lexicon import Lexicon  # noqa: E402
from prosody.meters import METERS, shaer_target  # noqa: E402
from prosody.verify import Verifier  # noqa: E402

V: Verifier | None = None          # hemistich scores for every margin (no threshold applied)
VP: Verifier | None = None         # poem-level verdicts with the default thresholds
MARGINS = (0.5, 1.0, 1.5, 2.0, 3.0, 99.0)


def _init(lex_path: str | None, strict: bool):
    global V, VP
    lex = Lexicon.load(lex_path) if lex_path else None
    V = Verifier(lex, strict=strict, margin=99.0, max_cost=99.0, diagnose=False)
    VP = Verifier(lex, strict=strict, diagnose=False)


def load(path: str, limit: int, seed: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            verses = r.get("verses") or [x for x in str(r.get("generated_text") or r.get("reference_completion") or "").split("\n")]
            verses = [v.strip() for v in verses if str(v).strip()]
            try:
                meter, form = shaer_target(r["base_meter"], r.get("form") or "تام")
            except KeyError:
                continue
            if len(verses) >= 2:
                rows.append({"verses": verses, "meter": meter, "form": form})
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def coverage(line: str) -> float:
    units = [u for w in arudi.tokenize(line) for u in arudi._units(w) if u[0] not in "اىآأإ"]
    return sum(1 for _, m in units if m) / max(len(units), 1)


def perturb(line: str, kind: str, rng: random.Random, vocab: list[str]) -> str | None:
    words = line.split()
    if len(words) < 3:
        return None
    i = rng.randrange(len(words) - 1)          # never the rhyme word
    if kind == "drop_letter":
        w = words[i]
        letters = [k for k, ch in enumerate(w) if arudi.LETTER_RE.match(ch)]
        if len(letters) < 3:
            return None
        k = rng.choice(letters[1:])
        words[i] = w[:k] + w[k + 1:]
    elif kind == "drop_word":
        del words[i]
    elif kind == "add_word":
        words.insert(i + 1, rng.choice(vocab))
    elif kind == "swap_word":
        base = len(arudi.strip_marks(words[i]))
        choices = [v for v in rng.sample(vocab, min(50, len(vocab))) if abs(len(arudi.strip_marks(v)) - base) >= 2]
        if not choices:
            return None
        words[i] = choices[0]
    out = " ".join(words)
    return out if out != line else None


def hemistich_scores(text: str, meter: str, form: str, position: str) -> dict:
    """Costs needed for every margin: target cost, best cost in any meter, best other-meter costs."""
    cands, best_any, ms = V._hemistich(text, meter, form, (position,) if position == "ajuz" else ("sadr", "ajuz"))
    other = {}
    for m in ms:
        if m.meter != meter and (m.meter not in other or m.cost < other[m.meter]):
            other[m.meter] = m.cost
    any_form = [m for m in ms if m.meter == meter]
    return {"target": cands[0].cost if cands else None, "best": best_any.cost if best_any else None,
            "target_any_form": any_form[0].cost if any_form else None,
            "best_meter": best_any.meter if best_any else None, "other": other}


def accepted(s: dict, margin: float) -> bool:
    return s["target"] is not None and s["target"] - s["best"] <= margin


def _work(args) -> list[dict]:
    rows, seed, vocab = args
    rng = random.Random(seed)
    out = []
    for r in rows:
        meter, form, verses = r["meter"], r["form"], r["verses"]
        rec = {"meter": meter, "form": form, "hem": [], "neg": [], "poem": None}
        for i, line in enumerate(verses):
            pos = "sadr" if i % 2 == 0 else "ajuz"
            s = hemistich_scores(line, meter, form, pos)
            s["coverage"] = coverage(line)
            s["pos"] = pos
            rec["hem"].append(s)
            if s["target"] is not None:
                for kind in ("drop_letter", "drop_word", "add_word", "swap_word"):
                    p = perturb(line, kind, rng, vocab)
                    if p:
                        ps = hemistich_scores(p, meter, form, pos)
                        rec["neg"].append({"kind": kind, "orig": s, "pert": ps})
        lines = verses[: len(verses) // 2 * 2]
        if len(lines) >= 4:
            pv = VP.poem(lines, meter, form)
            # rhyme negative: one ʿajuz ends with a word of another rawī
            j = 2 * rng.randrange(len(lines) // 2) + 1
            words = lines[j].split()
            rawi = pv.rhyme.get("rawi")
            swap = [w for w in rng.sample(vocab, min(80, len(vocab))) if arudi.strip_marks(w)[-1:] not in (rawi, "", "ا", "ي", "و", "ه", "ة", "ى")]
            neg_ok = None
            if swap and len(words) >= 2:
                broken = list(lines)
                broken[j] = " ".join(words[:-1] + [swap[0]])
                neg_ok = not VP.poem(broken, meter, form).rhyme_ok
            rec["poem"] = {"meter_ok": pv.meter_ok, "rhyme_ok": pv.rhyme_ok, "rhyme_negative_flagged": neg_ok,
                           "rawi": rawi, "issues": [i["kind"] for i in pv.summary()["issues"]]}
        out.append(rec)
    return out


def summarise(recs: list[dict]) -> dict:
    res: dict = {"poems": len(recs)}
    hem = [(r["meter"], h) for r in recs for h in r["hem"]]
    res["hemistichs"] = len(hem)
    curve = {}
    for mg in MARGINS:
        acc = sum(accepted(h, mg) for _, h in hem) / len(hem)
        wrong = sum(sum(1 for c in h["other"].values() if c - h["best"] <= mg) for _, h in hem) / len(hem)
        negs = [n for r in recs for n in r["neg"] if accepted(n["orig"], mg)]
        by_kind = defaultdict(list)
        for n in negs:
            by_kind[n["kind"]].append(not accepted(n["pert"], mg))
        curve[str(mg)] = {"acceptance": round(100 * acc, 2), "wrong_meters_per_hemistich": round(wrong, 3),
                          "negatives_rejected": {k: round(100 * sum(v) / len(v), 2) for k, v in by_kind.items()},
                          "negatives_rejected_all": round(100 * sum(sum(v) for v in by_kind.values()) /
                                                          max(1, sum(len(v) for v in by_kind.values())), 2)}
    res["by_margin"] = curve
    res["acceptance_any_form@1.5"] = round(100 * sum(h["target_any_form"] is not None and h["target_any_form"] - h["best"] <= 1.5
                                                     for _, h in hem) / len(hem), 2)
    res["identification_top1"] = round(100 * sum(h["best_meter"] == m for m, h in hem) / len(hem), 2)
    # which other meters accept a hemistich of meter A (margin 1.5): inherent overlaps such as الكامل with all iḍmār = الرجز
    conf: dict = defaultdict(Counter)
    count_by_meter = Counter(m for m, _ in hem)
    for m, h in hem:
        for other, c in h["other"].items():
            if c - h["best"] <= 1.5:
                conf[m][other] += 1
    res["also_accepted_as@1.5"] = {m: {o: round(100 * c / count_by_meter[m], 1) for o, c in conf[m].most_common(4)}
                                   for m in sorted(conf, key=lambda k: -count_by_meter[k])}
    per_meter = defaultdict(list)
    for m, h in hem:
        per_meter[m].append(h)
    res["by_meter"] = {m: {"n": len(v), "acceptance@1.5": round(100 * sum(accepted(h, 1.5) for h in v) / len(v), 2),
                           "acceptance@inf": round(100 * sum(h["target"] is not None for h in v) / len(v), 2),
                           "top1": round(100 * sum(h["best_meter"] == m for h in v) / len(v), 2)}
                       for m, v in sorted(per_meter.items(), key=lambda kv: -len(kv[1]))}
    cov = defaultdict(list)
    for _, h in hem:
        cov[min(4, int(h["coverage"] * 5))].append(h)
    res["by_diacritic_coverage"] = {f"{20 * b}-{20 * b + 20}%": {
        "n": len(v), "acceptance@1.5": round(100 * sum(accepted(h, 1.5) for h in v) / len(v), 2),
        "wrong_meters@1.5": round(sum(sum(1 for c in h["other"].values() if c - h["best"] <= 1.5) for h in v) / len(v), 3)}
        for b, v in sorted(cov.items())}
    poems = [r["poem"] for r in recs if r["poem"]]
    if poems:
        res["poem_level"] = {
            "poems": len(poems),
            "meter_ok_pct": round(100 * sum(p["meter_ok"] for p in poems) / len(poems), 2),
            "rhyme_ok_pct": round(100 * sum(p["rhyme_ok"] for p in poems) / len(poems), 2),
            "rhyme_negative_flagged_pct": round(100 * sum(bool(p["rhyme_negative_flagged"]) for p in poems
                                                         if p["rhyme_negative_flagged"] is not None) /
                                                max(1, sum(p["rhyme_negative_flagged"] is not None for p in poems)), 2),
            "issue_counts": dict(Counter(i for p in poems for i in set(p["issues"])).most_common()),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poems", required=True)
    ap.add_argument("--lexicon", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--lenient", action="store_true")
    args = ap.parse_args()
    rows = load(args.poems, args.limit, args.seed)
    vocab = sorted({w for r in rows[:2000] for v in r["verses"] for w in v.split() if len(arudi.strip_marks(w)) >= 3})
    print(f"{len(rows):,} poems, vocab {len(vocab):,}", flush=True)
    chunks = [(rows[i:i + 25], args.seed + i, vocab) for i in range(0, len(rows), 25)]
    recs, t0 = [], time.time()
    with mp.Pool(args.workers, initializer=_init, initargs=(args.lexicon or None, not args.lenient)) as pool:
        for k, part in enumerate(pool.imap_unordered(_work, chunks), 1):
            recs.extend(part)
            if k % 10 == 0 or k == len(chunks):
                print(f"  {len(recs):,}/{len(rows):,} poems, {time.time() - t0:.0f}s", flush=True)
    res = summarise(recs)
    res["config"] = {"poems_file": args.poems, "lexicon": args.lexicon, "strict": not args.lenient, "seed": args.seed}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("poems", "hemistichs", "identification_top1", "acceptance_any_form@1.5")},
                     ensure_ascii=False))
    for mg, c in res["by_margin"].items():
        print(f"  margin {mg:>4}: accept {c['acceptance']:5.1f}%  wrong meters/hem {c['wrong_meters_per_hemistich']:.2f}  "
              f"negatives rejected {c['negatives_rejected_all']:5.1f}%  {c['negatives_rejected']}")
    if "poem_level" in res:
        print("  poem level:", res["poem_level"])


if __name__ == "__main__":
    main()
