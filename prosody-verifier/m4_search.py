#!/usr/bin/env python3
"""M4: verifier-guided decoding, the part that does not need a GPU.

Two ways to spend extra samples on one prompt are compared at equal compute:

  I1  best-of-N whole poems    sample N poems with the paper's decoding; keep the first strictly valid one (in sample
                               order), else the one with the fewest defects. I0 is N = 1.
  I2  hemistich search         write the poem one hemistich at a time. For each hemistich sample k candidates in
                               parallel, up to R rounds, and keep the first candidate (in sample order) that adds no new
                               defect to the poem so far; if none does, keep the candidate that adds the fewest. The poem
                               stops after exactly the requested number of hemistichs.

Both are rejection sampling: among the candidates that pass, the one taken is the first the model produced, so the
model's own distribution decides between valid options (the verifier never picks the "most metrical" line).

What "adds no new defect" means. A defect is a failing bayt, or a bayt named by a consistency or rhyme issue (ḍarb or
ʿarūḍ form change, rawī, majrā, ridf, taʾsīs, muṭlaqa/muqayyada, refrain, and īṭāʾ: a rhyme word repeated within 7
bayts, which the evaluation reports only as a warning but the search treats as a defect so it cannot fill a poem with one
rhyme word). A sadr candidate is checked alone: it must scan as a sadr of the requested meter and form (within the margin)
and, after the first bayt, allow the ḍarb already fixed. An ʿajuz candidate completes a bayt, and the poem so far is
verified again (meter, consistency, rhyme): the candidate passes when the defect set of the poem with it is contained in
the defect set without it. The first bayt therefore fixes the rawī and the ḍarb.

The search verifier is the evaluation verifier (letters mode by default: written vowels ignored; binding mode on request)
without the tadwīr search: a bayt
whose word is split across the hemistichs is not accepted during the search (a sadr must scan alone). This only rejects
candidates the evaluation might accept; the final poems are always scored by the full M3 pipeline.

The search itself (SearchRunner) is independent of the language model: a generator only has to propose k candidate
hemistichs for a batch of partial poems (`propose`). m4_decode.py provides the Hugging Face generator; the tests use mock
generators.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prosody import arudi  # noqa: E402
from prosody.lexicon import Lexicon  # noqa: E402
from prosody.meters import METERS, shaer_target  # noqa: E402
from prosody.verify import Verifier  # noqa: E402
from pv_text import split_poem_lines  # noqa: E402

SHIPPED_LEXICON = HERE / "data" / "lexicon_train.json.gz"
NOT_DEFECTS = ("count", "odd_tail")          # length is enforced by the search itself
INF = float("inf")
EMPTY = 10 ** 6       # an empty candidate ranks after every written line (an empty line would shorten the poem)


# ---------------------------------------------------------------------------------------------- the search verifier
class SearchVerifier(Verifier):
    """The evaluation verifier with two changes for speed: bayt verdicts are memoised (a partial poem is verified again
    for every ʿajuz candidate, and its earlier bayts do not change), and the tadwīr search is off (see the module doc)."""

    def __init__(self, lexicon, tadwir: bool = False, **kw):
        kw.setdefault("ignore_diacritics", True)
        kw.setdefault("diagnose", False)
        super().__init__(lexicon, **kw)
        self.tadwir = tadwir
        self._bayt_memo: dict = {}

    def bayt(self, sadr, ajuz, meter, form, first=False):
        key = (sadr, ajuz, meter, form, first)
        hit = self._bayt_memo.get(key)
        if hit is None:
            if len(self._bayt_memo) >= self.cache_limit:
                self._bayt_memo.clear()
            hit = self._bayt_memo[key] = super().bayt(sadr, ajuz, meter, form, first)
        bv, sc, ac = hit
        return dataclasses.replace(bv), sc, ac          # poem() sets .index: never share one object between bayts

    def _tadwir(self, sadr, ajuz, meter, form):
        return super()._tadwir(sadr, ajuz, meter, form) if self.tadwir else None


def defects(pv, ita: bool = True) -> frozenset:
    """The defect set of a (partial) poem verdict: failing bayts and every bayt named by an issue."""
    out = {("bayt", b.index) for b in pv.bayts if not b.ok}
    for issue in pv.issues + pv.rhyme.get("issues", []):
        if issue["kind"] in NOT_DEFECTS or (issue["kind"] == "ita" and not ita):
            continue
        out.update((issue["kind"], i) for i in issue["bayts"])
    return frozenset(out)


@dataclass
class Check:
    ok: bool
    new: int                  # number of new defects (0 when ok)
    cost: float               # scansion cost of the new hemistich(s); inf without an analysis in the meter
    detail: str = ""

    def key(self) -> tuple:
        """Fallback order when no candidate passes: fewer new defects, then a plausible scansion."""
        return (not self.ok, self.new, self.cost)


class PoemChecker:
    """Checks candidate hemistichs for one requested meter and form against the poem written so far."""

    def __init__(self, verifier: SearchVerifier, ita: bool = True):
        self.v = verifier
        self.ita = ita
        self._prefix: dict = {}

    def verdict(self, lines: list, meter: str, form: str):
        return self.v.poem(list(lines), meter, form)

    def prefix(self, lines: list, meter: str, form: str) -> tuple:
        """(defect set, fixed ḍarb form) of the complete bayts in `lines`. The ḍarb is the one of the first bayt that
        passed (None before that)."""
        lines = list(lines[: len(lines) // 2 * 2])
        key = (tuple(lines), meter, form)
        hit = self._prefix.get(key)
        if hit is None:
            if len(self._prefix) > 2_000:
                self._prefix.clear()
            if lines:
                pv = self.verdict(lines, meter, form)
                darb = next((b.darb for b in pv.bayts if b.ok and b.darb), None)
                hit = (defects(pv, self.ita), darb)
            else:
                hit = (frozenset(), None)
            self._prefix[key] = hit
        return hit

    def check_sadr(self, lines: list, cand: str, meter: str, form: str) -> Check:
        if not cand.strip():
            return Check(False, EMPTY, INF, "empty")
        first = len(lines) == 0
        cands, best_any, _ = self.v._hemistich(cand, meter, form, ("sadr", "ajuz") if first else ("sadr",))
        darb = None if first else self.prefix(lines, meter, form)[1]
        if darb is not None:
            pairs = METERS[meter].forms[form].pairs
            cands = [m for m in cands if m.position == "sadr" and darb in pairs.get(m.end, ())]
        if not cands:
            return Check(False, 2, INF, "no analysis in the meter" + (f" pairing with ḍarb {darb}" if darb else ""))
        hv = self.v._judge(cand, cands, best_any, meter, form, "sadr")
        return Check(hv.ok, 0 if hv.ok else 1, cands[0].cost, hv.reason)

    def check_ajuz(self, lines: list, cand: str, meter: str, form: str) -> Check:
        if not cand.strip():
            return Check(False, EMPTY, INF, "empty")
        before = self.prefix(lines[:-1], meter, form)[0]
        pv = self.verdict(lines + [cand], meter, form)
        after = defects(pv, self.ita)
        new = after - before
        b = pv.bayts[-1]
        cost = (b.sadr.match.cost + b.ajuz.match.cost) if (b.sadr.match and b.ajuz.match) else INF
        return Check(not new, len(new), cost, "; ".join(sorted(f"{k}@{i}" for k, i in new)))

    def check(self, lines: list, cand: str, meter: str, form: str) -> Check:
        return (self.check_sadr if len(lines) % 2 == 0 else self.check_ajuz)(lines, cand, meter, form)


# ---------------------------------------------------------------------------------------------- checker workers
_W: dict = {}


def _worker_init(lexicon_path: str, ita: bool, binding: bool = False) -> None:
    _W["checker"] = PoemChecker(SearchVerifier(Lexicon.load(lexicon_path), ignore_diacritics=not binding), ita=ita)


def check_in_order(lines: list, cands: list[str], meter: str, form: str, checker: PoemChecker | None = None):
    """Check candidates in sample order and stop at the first that passes: (index of the passing one or None, checks)."""
    checker = checker or _W["checker"]
    out = []
    for i, c in enumerate(cands):
        r = checker.check(lines, c, meter, form)
        out.append(r)
        if r.ok:
            return i, out
    return None, out


class CheckerPool:
    """W single-process workers; a poem always goes to the same worker, so its scans and bayt verdicts stay cached.
    workers=0 checks in this process (tests, small runs)."""

    def __init__(self, lexicon_path: str | Path = SHIPPED_LEXICON, workers: int = 0, ita: bool = True, binding: bool = False):
        """binding: check with the written vowels binding (as a reader would) instead of the letters alone."""
        self.workers = workers
        if workers:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")          # never fork a process that holds CUDA state
            self.pools = [ProcessPoolExecutor(1, mp_context=ctx, initializer=_worker_init,
                                              initargs=(str(lexicon_path), ita, binding)) for _ in range(workers)]
        else:
            self.local = PoemChecker(SearchVerifier(Lexicon.load(str(lexicon_path)), ignore_diacritics=not binding), ita=ita)

    def run(self, jobs: list[tuple]) -> list:
        """jobs: (affinity, lines, candidate texts, meter, form) -> check_in_order results, in order."""
        if not self.workers:
            return [check_in_order(l, c, m, f, self.local) for _, l, c, m, f in jobs]
        futs = [self.pools[a % self.workers].submit(check_in_order, l, c, m, f) for a, l, c, m, f in jobs]
        return [x.result() for x in futs]

    def close(self) -> None:
        for p in getattr(self, "pools", []):
            p.shutdown(cancel_futures=True)


# ---------------------------------------------------------------------------------------------- search state
@dataclass
class Candidate:
    text: str                 # the hemistich (no newline)
    ids: list                 # the token ids to append to the context when accepted (ending with a line break)
    tokens: int               # tokens the model generated for this candidate (what the budget counts)
    eos: bool = False         # the model ended the poem here


@dataclass
class PoemState:
    key: str
    meter: str
    form: str
    requested_lines: int
    context: list             # prompt ids + accepted completion ids
    prompt_len: int
    lines: list = field(default_factory=list)
    gen_tokens: int = 0       # every generated token, rejected candidates included
    prefill_tokens: int = 0   # context tokens the model encoded (each step re-encodes the context once)
    candidates: int = 0
    checks: int = 0
    rounds: int = 0
    forced: list = field(default_factory=list)      # hemistich indices where no candidate passed
    early_eos: int = 0        # accepted candidates after which the model wanted to stop
    empty_lines: int = 0      # hemistichs where every candidate was empty (the poem then has a line fewer)
    gen_seconds: float = 0.0  # share of the batch's model time
    check_seconds: float = 0.0
    log: list = field(default_factory=list)          # per hemistich: [candidates, rounds, passed]

    @property
    def done(self) -> bool:
        return len(self.lines) >= self.requested_lines


class SearchRunner:
    """Hemistich search over a batch of poems with any generator exposing
    propose(states, k) -> per state a list of k Candidates (and a dict with 'seconds', 'prefill_tokens' per state)."""

    def __init__(self, generator, pool: CheckerPool, k: int, rounds: int):
        assert k >= 1 and rounds >= 1
        self.gen, self.pool, self.k, self.rounds = generator, pool, k, rounds

    def step(self, states: list[PoemState]) -> None:
        active = [s for s in states if not s.done]
        tried = {id(s): [] for s in active}            # (check, candidate) in sample order
        used = {id(s): 0 for s in active}              # rounds this hemistich took
        chosen: dict = {}
        need = list(active)
        for r in range(self.rounds):
            if not need:
                break
            proposals, info = self.gen.propose(need, self.k)
            t0 = time.perf_counter()
            jobs = [(zlib.crc32(s.key.encode()), s.lines, [c.text for c in cands], s.meter, s.form) for s, cands in zip(need, proposals)]
            results = self.pool.run(jobs)
            dt = (time.perf_counter() - t0) / max(1, len(need))
            nxt = []
            for s, cands, (hit, checks), sec, pre in zip(need, proposals, results, info["seconds"], info["prefill_tokens"]):
                s.gen_tokens += sum(c.tokens for c in cands)
                s.candidates += len(cands)
                s.checks += len(checks)
                s.rounds += 1
                used[id(s)] += 1
                s.gen_seconds += sec
                s.prefill_tokens += pre
                s.check_seconds += dt
                tried[id(s)].extend(zip(checks, cands))
                if hit is not None:
                    chosen[id(s)] = (cands[hit], True)
                else:
                    nxt.append(s)
            need = nxt
        for s in need:                                 # nothing passed: the candidate that adds the fewest defects
            pairs = tried[id(s)]
            best = min(range(len(pairs)), key=lambda i: (pairs[i][0].key(), i))
            chosen[id(s)] = (pairs[best][1], False)
        for s in active:
            cand, passed = chosen[id(s)]
            s.log.append([len(tried[id(s)]), used[id(s)], passed])
            if not passed:
                s.forced.append(len(s.lines))
            if cand.eos and len(s.lines) + 1 < s.requested_lines:
                s.early_eos += 1
            if not cand.text.strip():
                s.empty_lines += 1
            s.lines.append(cand.text.strip())
            s.context = s.context + list(cand.ids)

    def run(self, states: list[PoemState], progress=None) -> list[PoemState]:
        while any(not s.done for s in states):
            self.step(states)
            if progress:
                progress(states)
        return states


def state_record(s: PoemState, row: dict, config: dict) -> dict:
    """An output row in the schema of evaluation/generate_shaer_test.py (read by score_meter_count.py), plus the search
    accounting."""
    rec = {k: row.get(k) for k in ("id", "base_meter", "form", "requested_bayts", "requested_num_lines", "length_bucket",
                                   "description", "enhanced_description", "reference_completion", "sft_prompt")}
    if row.get("requested_rawi") is not None:          # M5 prompts that name the rhyme letter
        rec["requested_rawi"] = row["requested_rawi"]
    rec.update({
        "sample_idx": int(row.get("sample_idx", 0) or 0), "generated_text": "\n".join(s.lines),
        "gen_tokens": s.gen_tokens, "finished_with_eos": True,
        "m4": {"config": config, "prefill_tokens": s.prefill_tokens, "candidates": s.candidates, "checks": s.checks,
               "rounds": s.rounds, "forced": s.forced, "early_eos": s.early_eos, "empty_lines": s.empty_lines, "gen_seconds": round(s.gen_seconds, 4),
               "check_seconds": round(s.check_seconds, 4), "log": s.log, "completion_tokens": len(s.context) - s.prompt_len},
    })
    return rec


# ---------------------------------------------------------------------------------------------- best-of-N selection
def poem_rank(verifier: Verifier, row: dict) -> tuple:
    """Rank of one sampled poem for best-of-N (smaller is better): strictly valid first, then the fewest defects counted
    as in the search (failing bayts, issue bayts, īṭāʾ), then the length error."""
    meter, form = shaer_target(row["base_meter"], row.get("form") or "تام")
    lines = split_poem_lines(row.get("generated_text") or "")
    req = int(row.get("requested_num_lines") or 0) or None
    pv = verifier.poem(lines, meter, form, req)
    return (not pv.ok, len(defects(pv)), abs(len(lines) - (req or len(lines))) + (len(lines) % 2))


def best_of_n(samples: list[dict], n: int, verifier: Verifier, ranks: dict | None = None) -> dict:
    """The selected sample among the first n (sample_idx < n) of one prompt; ties go to the earlier sample."""
    pool = sorted((r for r in samples if int(r.get("sample_idx", 0)) < n), key=lambda r: int(r.get("sample_idx", 0)))
    if len(pool) < n:
        raise ValueError(f"{samples[0]['id']}: {len(pool)} samples, {n} needed")
    def rank(r):
        k = (r["id"], int(r.get("sample_idx", 0)))
        if ranks is not None and k in ranks:
            return ranks[k]
        v = poem_rank(verifier, r)
        if ranks is not None:
            ranks[k] = v
        return v
    best = min(pool, key=lambda r: (rank(r), int(r.get("sample_idx", 0))))
    out = dict(best)
    out["m4"] = {"config": {"method": "best_of_n", "n": n}, "selected_sample": int(best.get("sample_idx", 0)),
                 "gen_tokens_total": sum(int(r.get("gen_tokens") or 0) for r in pool),
                 "gen_seconds_total": sum(float((r.get("m4") or {}).get("gen_seconds") or 0) for r in pool)}
    out["gen_tokens"] = out["m4"]["gen_tokens_total"]
    out["sample_idx"] = 0
    return out


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out
