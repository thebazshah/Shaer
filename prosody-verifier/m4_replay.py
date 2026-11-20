#!/usr/bin/env python3
"""M4 validation without a GPU: the hemistich search run on real lines, then scored by the full M3 pipeline.

The generator replays existing poems: for hemistich j of a prompt it proposes, in this order, line j of each given
source (e.g. the released Shaer poem, then the human poem of the same prompt); a missing line is an empty candidate.
The search keeps the first candidate that adds no defect, exactly as with the model. This checks, on thousands of real
hemistichs (tadwīr, refrains, partial diacritics, wrong lines), that

  * every poem has exactly the requested number of hemistichs;
  * a poem the search completed without a forced hemistich is strictly valid for the evaluation (M3 letters mode,
    margin 1.5): the search verifier must never accept what the evaluation rejects;
  * the accounting (candidates, checks, tokens) adds up.

  python m4_replay.py --source e0.jsonl --source reference.jsonl --limit 500 --out replay.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import m3_rescore as m3  # noqa: E402
from m4_search import Candidate, CheckerPool, PoemState, SearchRunner, SHIPPED_LEXICON  # noqa: E402
from prosody.lexicon import Lexicon  # noqa: E402
from prosody.meters import shaer_target  # noqa: E402
from prosody.verify import Verifier  # noqa: E402
from pv_text import split_poem_lines  # noqa: E402


class ReplayGenerator:
    def __init__(self, sources: list[dict]):
        self.sources = sources            # [{id: [lines]}], in proposal order

    def propose(self, states, k):
        props = []
        for s in states:
            j = len(s.lines)
            texts = [(src.get(s.key) or [""] * (j + 1))[j] if j < len(src.get(s.key) or []) else "" for src in self.sources]
            texts = (texts * k)[:k]
            props.append([Candidate(t, [1] * (len(t.split()) + 1), len(t.split()) + 1) for t in texts])
        return props, {"seconds": [0.0] * len(states), "prefill_tokens": [0] * len(states)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, help="scored JSONL (generated_text per id), in proposal order")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    srcs = [{r["id"]: r for r in m3._read_jsonl(Path(p)) if int(r.get("sample_idx", 0) or 0) == 0} for p in a.source]
    ids = sorted(set.intersection(*(set(s) for s in srcs)))[: a.limit or None]
    gen = ReplayGenerator([{i: split_poem_lines(s[i]["generated_text"]) for i in ids} for s in srcs])
    base = srcs[-1]
    states = []
    for i in ids:
        meter, form = shaer_target(base[i]["base_meter"], base[i].get("form") or "تام")
        states.append(PoemState(i, meter, form, int(base[i]["requested_num_lines"]), [0], 1))
    t0 = time.time()
    pool = CheckerPool(SHIPPED_LEXICON, workers=0)
    SearchRunner(gen, pool, k=len(srcs), rounds=a.rounds).run(states)
    search_s = time.time() - t0
    lex = Lexicon.load(str(SHIPPED_LEXICON))
    vt = Verifier(lex, ignore_diacritics=True)
    vf = Verifier(lex, ignore_diacritics=True, diagnose=False).share_cache(vt)
    res = {"poems": len(states), "search_seconds": round(search_s, 1), "length_errors": 0, "unforced": 0,
           "unforced_but_invalid": [], "strict_valid": 0, "taken_from": [0] * (len(srcs) + 1), "hemistichs": 0,
           "accounting_errors": 0}
    for s in states:
        row = dict(base[s.key], generated_text="\n".join(s.lines), parsed_num_lines=len(s.lines))
        rec = m3.score_row(row, vt, vf)
        ok = m3.poem_metrics(rec, m3.DEFAULT_MARGIN)["strict_valid"]
        res["strict_valid"] += ok
        res["length_errors"] += len(s.lines) != s.requested_lines
        if not s.forced:
            res["unforced"] += 1
            if not ok:
                res["unforced_but_invalid"].append({"id": s.key, "lines": s.lines, "issues": rec["issues"] + rec["rhyme"]["issues"],
                                                    "bayt_ok": rec["target"]["bayt_ok"]})
        for j, line in enumerate(s.lines):
            res["hemistichs"] += 1
            if j in s.forced:
                res["taken_from"][-1] += 1
                continue
            src = next((n for n, g in enumerate(gen.sources) if j < len(g.get(s.key, [])) and g[s.key][j].strip() == line), None)
            res["taken_from"][src if src is not None else -1] += 1
        res["accounting_errors"] += s.candidates != len(s.log) * len(srcs) * 1 and a.rounds == 1
    res["strict_valid_pct"] = round(100 * res["strict_valid"] / len(states), 2)
    res["unforced_pct"] = round(100 * res["unforced"] / len(states), 2)
    res["unforced_but_invalid_count"] = len(res["unforced_but_invalid"])
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "unforced_but_invalid"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
