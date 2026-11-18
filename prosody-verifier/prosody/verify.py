"""Poem-level verification: meter (per hemistich, per bayt, across the poem), rhyme, refrain and length.

For a requested meter and form the verifier answers, for every bayt and for the poem:

  meter        does each hemistich scan in the requested meter form? A hemistich passes when an analysis in that form
               exists, it costs at most `max_cost`, and it is not much less plausible than the best analysis in any meter
               (`margin`), so a line is not forced into the meter through an implausible reading.
               The ʿarūḍ / ḍarb pairing of each bayt must be allowed by the form's table; in the first bayt the ʿarūḍ may
               take the ḍarb's form (taṣrīʿ). A bayt whose word is split across the two hemistichs (tadwīr) is scanned whole.
  consistency  the ḍarb form, and the ʿarūḍ form outside the opening bayt, must be the same in every bayt (an ʿilla, once
               used, is binding); zihāfāt may vary.
  rhyme        qafiya.analyse: rawī, majrā, ridf, taʾsīs, waṣl, muṭlaqa/muqayyada, īṭāʾ, refrain.
  length       the number of hemistichs against the request; an odd final hemistich.

Strictness tiers:
  strict    the classical tables; muʿāqaba violations reject; rare zihāfāt allowed but reported
  lenient   also accepts muʿāqaba violations (each costs COST_MUAQABA)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from . import arudi, qafiya
from .meters import METERS, hemistich_analyses, shaer_target
from .scan import Match, scan

MARGIN = 1.5          # target analysis may cost at most this much more than the best analysis in any meter
MAX_COST = 6.0        # absolute cap on the cost of an accepted analysis


@dataclass
class HemistichVerdict:
    text: str
    ok: bool
    match: Match | None
    best_any: Match | None
    reason: str = ""
    nearest: dict | None = None

    def summary(self) -> dict:
        m = self.match
        return {"text": self.text, "ok": self.ok, "reason": self.reason,
                "meter": m and m.meter, "form": m and m.form, "end": m and m.end, "cost": m and round(m.cost, 2),
                "feet": m and m.feet_text, "pattern": m and m.pattern,
                "best_meter": self.best_any and self.best_any.meter,
                "best_cost": self.best_any and round(self.best_any.cost, 2), "nearest": self.nearest}


@dataclass
class BaytVerdict:
    index: int
    sadr: HemistichVerdict
    ajuz: HemistichVerdict
    ok: bool
    arud: str = ""
    darb: str = ""
    tasri: bool = False
    tadwir: bool = False
    reason: str = ""


@dataclass
class PoemVerdict:
    meter: str
    form: str
    bayts: list = field(default_factory=list)
    rhyme: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    lines: int = 0
    requested_lines: int | None = None

    @property
    def meter_ok(self) -> bool:
        return bool(self.bayts) and all(b.ok for b in self.bayts) and not any(i["kind"] in ("darb_changes", "arud_changes")
                                                                               for i in self.issues)

    @property
    def rhyme_ok(self) -> bool:
        return bool(self.rhyme.get("ok"))

    @property
    def count_ok(self) -> bool:
        return self.requested_lines is None or self.lines == self.requested_lines

    @property
    def ok(self) -> bool:
        return self.meter_ok and self.rhyme_ok and self.count_ok and self.lines % 2 == 0

    def summary(self) -> dict:
        nb = len(self.bayts)
        return {
            "meter": self.meter, "form": self.form, "ok": self.ok, "meter_ok": self.meter_ok, "rhyme_ok": self.rhyme_ok,
            "count_ok": self.count_ok, "lines": self.lines, "requested_lines": self.requested_lines,
            "bayts": nb, "bayts_meter_ok": sum(b.ok for b in self.bayts),
            "hemistichs_ok": sum(b.sadr.ok for b in self.bayts) + sum(b.ajuz.ok for b in self.bayts),
            "rawi": self.rhyme.get("rawi"), "refrain": self.rhyme.get("refrain"),
            "issues": self.issues + self.rhyme.get("issues", []),
            "per_bayt": [{"i": b.index, "ok": b.ok, "arud": b.arud, "darb": b.darb, "tasri": b.tasri, "tadwir": b.tadwir,
                          "reason": b.reason, "sadr": b.sadr.summary(), "ajuz": b.ajuz.summary()} for b in self.bayts],
        }


class Verifier:
    def __init__(self, lexicon=None, strict: bool = True, margin: float = MARGIN, max_cost: float = MAX_COST,
                 diagnose: bool = True, ignore_diacritics: bool = False):
        """ignore_diacritics: scan the bare letters only, so the written vowels (which a generator may get wrong) do not
        count; the default treats them as binding, as a reader would."""
        self.lexicon = lexicon
        self.strict = strict
        self.margin = margin
        self.max_cost = max_cost
        self.diagnose = diagnose
        self.ignore_diacritics = ignore_diacritics

    def _text(self, text: str) -> str:
        if not self.ignore_diacritics:
            return text
        return " ".join(arudi.strip_marks(w) for w in arudi.tokenize(text))

    # ---------------------------------------------------------------------------------------------- hemistich
    def _hemistich(self, text: str, meter: str, form: str, positions: tuple, ends: set | None = None) -> tuple:
        ms = scan(self._text(text), strict=self.strict, lexicon=self.lexicon)
        best_any = ms[0] if ms else None
        cands = [m for m in ms if m.meter == meter and m.form == form and m.position in positions
                 and (ends is None or m.end in ends)]
        return cands, best_any, ms

    def _judge(self, text: str, cands: list, best_any: Match | None, meter: str, form: str, position: str) -> HemistichVerdict:
        if not cands:
            near = None
            if self.diagnose:
                from .scan import nearest
                near = nearest(self._text(text), meter, form, position)
            other = best_any and f"scans as {best_any.meter} ({best_any.form})" or "scans in no meter"
            return HemistichVerdict(text, False, None, best_any, f"not in {meter} ({form}); {other}", near)
        m = cands[0]
        if m.cost > self.max_cost:
            return HemistichVerdict(text, False, m, best_any, f"only through implausible readings (cost {m.cost:.1f})")
        if best_any is not None and m.cost - best_any.cost > self.margin:
            return HemistichVerdict(text, False, m, best_any,
                                    f"reads more naturally as {best_any.meter} (cost {best_any.cost:.1f} vs {m.cost:.1f})")
        return HemistichVerdict(text, True, m, best_any)

    # ---------------------------------------------------------------------------------------------- bayt
    def bayt(self, sadr: str, ajuz: str, meter: str, form: str, first: bool = False) -> tuple[BaytVerdict, list, list]:
        f = METERS[meter].forms[form]
        sc, sbest, _ = self._hemistich(sadr, meter, form, ("sadr", "ajuz") if first else ("sadr",))
        ac, abest, _ = self._hemistich(ajuz, meter, form, ("ajuz",))
        # choose the cheapest allowed (ʿarūḍ, ḍarb) pair; taṣrīʿ lets the opening sadr take a ḍarb form
        best = None
        for s in sc:
            for a in ac:
                if s.position == "sadr":
                    allowed = a.end in f.pairs.get(s.end, ())
                    tasri = False
                else:
                    allowed = s.end == a.end and any(a.end in ds for ds in f.pairs.values())
                    tasri = True
                if allowed:
                    c = s.cost + a.cost
                    if best is None or c < best[0]:
                        best = (c, s, a, tasri)
        if best is not None:
            _, s, a, tasri = best
            sv = self._judge(sadr, [s], sbest, meter, form, "sadr")
            av = self._judge(ajuz, [a], abest, meter, form, "ajuz")
            ok = sv.ok and av.ok
            reason = "; ".join(x for x in (sv.reason and "sadr: " + sv.reason, av.reason and "ʿajuz: " + av.reason) if x)
            return BaytVerdict(0, sv, av, ok, s.end, a.end, tasri, False, reason), sc, ac
        sv = self._judge(sadr, sc, sbest, meter, form, "sadr")
        av = self._judge(ajuz, ac, abest, meter, form, "ajuz")
        if sc and ac:
            reason = "the ʿarūḍ and ḍarb forms are not a permitted pair"
            return BaytVerdict(0, sv, av, False, sc[0].end, ac[0].end, False, False, reason), sc, ac
        joined = self._tadwir(sadr, ajuz, meter, form)
        if joined is not None:
            sv2 = HemistichVerdict(sadr, True, joined[0], sbest, "")
            av2 = HemistichVerdict(ajuz, True, joined[1], abest, "")
            return BaytVerdict(0, sv2, av2, True, joined[0].end, joined[1].end, False, True, "tadwīr"), [joined[0]], [joined[1]]
        reason = "; ".join(x for x in (sv.reason and "sadr: " + sv.reason, av.reason and "ʿajuz: " + av.reason) if x)
        return BaytVerdict(0, sv, av, False, "", "", False, False, reason), sc, ac

    def _tadwir(self, sadr: str, ajuz: str, meter: str, form: str):
        """A word split across the hemistichs (الخَل / قِ): join the two halves and look for a split point inside it."""
        sw, aw = arudi.tokenize(self._text(sadr)), arudi.tokenize(self._text(ajuz))
        if not sw or not aw:
            return None
        whole = sw[-1] + aw[0]
        f = METERS[meter].forms[form]
        sadr_pats = {a["pattern"]: a for a in hemistich_analyses(meter, form, "sadr")}
        ajuz_pats = {a["pattern"]: a for a in hemistich_analyses(meter, form, "ajuz")}
        from .scan import candidate_bit_strings
        joined_line = " ".join(sw[:-1] + [whole] + aw[1:])
        for bits, cost in candidate_bit_strings(joined_line, limit=256):
            for k in range(8, len(bits) - 7):
                sp, ap = bits[:k], bits[k:]
                # inside a word the sadr's end is not lengthened, so its final sākin must be real
                if sp in sadr_pats and ap in ajuz_pats and ajuz_pats[ap]["end"] in f.pairs.get(sadr_pats[sp]["end"], ()):
                    s, a = sadr_pats[sp], ajuz_pats[ap]
                    mk = lambda x: Match(meter, form, x["position"], x["end"], x["pattern"], x["feet"], cost, x["rare"],
                                         x["muaqaba"], [])
                    return mk(s), mk(a)
        return None

    # ---------------------------------------------------------------------------------------------- poem
    def poem(self, text_or_lines, meter: str, form: str = "tam", requested_lines: int | None = None,
             refrain: list[str] | None = None) -> PoemVerdict:
        lines = text_or_lines if isinstance(text_or_lines, list) else [x.strip() for x in str(text_or_lines).split("\n")]
        lines = [x for x in lines if x.strip()]
        pv = PoemVerdict(meter, form, lines=len(lines), requested_lines=requested_lines)
        sadr_cands, ajuz_cands = [], []
        for bi in range(len(lines) // 2):
            bv, sc, ac = self.bayt(lines[2 * bi], lines[2 * bi + 1], meter, form, first=bi == 0)
            bv.index = bi
            pv.bayts.append(bv)
            sadr_cands.append(sc)
            ajuz_cands.append(ac)
        if len(lines) % 2:
            pv.issues.append({"kind": "odd_tail", "bayts": [len(lines) // 2], "detail": "a final hemistich without its pair"})
        if pv.requested_lines is not None and pv.lines != pv.requested_lines:
            pv.issues.append({"kind": "count", "bayts": [], "detail": f"{pv.lines} hemistichs, {pv.requested_lines} requested"})
        self._consistency(pv, sadr_cands, ajuz_cands)
        def rhyme_reading(h: HemistichVerdict):
            """The last word's reading for the rhyme analysis: the accepted scansion's, else the most plausible one
            (a reading forced into a meter the line does not follow would distort the rhyme)."""
            m = h.match if (h.ok and h.match is not None) else h.best_any
            return m.choices[-1] if (m is not None and m.choices) else None
        last = [rhyme_reading(b.ajuz) for b in pv.bayts]
        first_reading = rhyme_reading(pv.bayts[0].sadr) if pv.bayts else None
        pv.rhyme = qafiya.analyse([lines[2 * i + 1] for i in range(len(lines) // 2)], last,
                                  sadr_first=lines[0] if lines else None, sadr_first_reading=first_reading, refrain=refrain)
        return pv

    def _consistency(self, pv: PoemVerdict, sadr_cands: list, ajuz_cands: list) -> None:
        """The ḍarb (and, outside the opening bayt, the ʿarūḍ) form must stay the same: pick the form most bayts allow."""
        def most_common_end(cands_per_bayt, skip_first=False):
            votes = Counter()
            for i, cands in enumerate(cands_per_bayt):
                if skip_first and i == 0:
                    continue
                for e in {m.end for m in cands}:
                    votes[e] += 1
            return votes.most_common(1)[0][0] if votes else None
        darb = most_common_end(ajuz_cands)
        if darb:
            bad = [b.index for b, c in zip(pv.bayts, ajuz_cands) if c and darb not in {m.end for m in c}]
            if bad:
                pv.issues.append({"kind": "darb_changes", "bayts": bad, "detail": f"ḍarb form {darb} expected in every bayt"})
        arud = most_common_end([[m for m in c if m.position == "sadr"] for c in sadr_cands], skip_first=True)
        if arud:
            bad = [b.index for b, c in zip(pv.bayts, sadr_cands) if b.index > 0 and c and
                   arud not in {m.end for m in c if m.position == "sadr"}]
            if bad:
                pv.issues.append({"kind": "arud_changes", "bayts": bad, "detail": f"ʿarūḍ form {arud} expected"})


def verify_shaer_row(verifier: Verifier, row: dict) -> PoemVerdict:
    """Verify a Shaer row (generated_text, base_meter, form, requested_num_lines)."""
    meter, form = shaer_target(row["base_meter"], row.get("form") or "تام")
    req = row.get("requested_num_lines")
    return verifier.poem(str(row.get("generated_text") or ""), meter, form, int(req) if req else None)
