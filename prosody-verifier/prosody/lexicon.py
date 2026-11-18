"""A pronunciation prior learned from the corpus: how is each bare word form actually read?

Without diacritics a word allows many readings (كتب: kataba, kutiba, kutub, kutubun, ...), and with every reading equally
cheap almost any line can be forced into several meters. The prior scores each reading by how often the corpus reads that
word that way:

  evidence  every hemistich of the TRAIN split is scanned in its labelled meter; the readings the best scansion chose are
            counted per bare word form. Fully vowelled occurrences contribute too, since they allow only one reading.
            Two rounds: the second round scans with the first round's prior, so the choices become consistent.
  cost      cost(reading) = min(cap, ln((max_count + a) / (count + a))): 0 for the most frequent reading of the word,
            higher for rarer ones, `cap` for readings never seen. Words seen fewer than `min_total` times keep the plain
            costs of arudi.py.

The prior is built from the train split only, so the eval/test poems used to measure the verifier are never seen.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from . import arudi

ALPHA = 0.5


def word_key(word: str) -> str:
    """Bare form used as the lexicon key: no diacritics, hamza-alifs and alif waṣla folded into ا."""
    return arudi._fold(arudi.strip_marks(word))


def reading_key(r: "arudi.WordReading") -> str:
    return f"{r.bits}|{r.final}|{int(r.wasl)}"


class CharModel:
    """How an unvowelled letter is read, given its neighbours: P(bits | letter, previous, next) with back-off.

    Trained from the lexicon (each counted reading is traced back to its letters), it scores the readings of words the
    lexicon has never seen, such as rare words, spelling variants and broken words, so that they do not get every reading
    for free."""

    def __init__(self, tables: dict | None = None):
        self.tables = tables or {"lpn": {}, "ln": {}, "l": {}}

    @staticmethod
    def _ctx(trace: tuple, i: int) -> tuple[str, str, str]:
        f = arudi._fold
        l = f(trace[i][0])
        p = f(trace[i - 1][0]) if i > 0 else "^"
        n = f(trace[i + 1][0]) if i + 1 < len(trace) else "$"
        return l, p, n

    def add(self, trace: tuple, weight: float) -> None:
        for i, (letter, marks, bits) in enumerate(trace):
            if marks:
                continue                                  # only unvowelled letters carry information
            l, p, n = self._ctx(trace, i)
            for name, key in (("lpn", f"{l}{p}{n}"), ("ln", f"{l}{n}"), ("l", l)):
                t = self.tables[name].setdefault(key, {})
                t[bits] = t.get(bits, 0.0) + weight

    def cost(self, trace: tuple) -> float:
        total = 0.0
        for i, (letter, marks, bits) in enumerate(trace):
            if marks:
                continue
            l, p, n = self._ctx(trace, i)
            prob = None
            for name, key, need in (("lpn", f"{l}{p}{n}", 20.0), ("ln", f"{l}{n}", 20.0), ("l", l, 0.0)):
                t = self.tables[name].get(key)
                if t and sum(t.values()) >= need:
                    prob = (t.get(bits, 0.0) + 0.1) / (sum(t.values()) + 0.1 * 4)
                    break
            total += -math.log(prob) if prob else 2.0
        return total


class Lexicon:
    def __init__(self, counts: dict[str, dict[str, float]], cap: float = 3.0, min_total: float = 2.0,
                 chars: CharModel | None = None, char_weight: float = 0.5):
        self.counts = counts
        self.cap = cap
        self.min_total = min_total
        self.chars = chars
        self.char_weight = char_weight

    def _char_reweight(self, readings: list) -> list:
        costs = [self.chars.cost(r.trace) if r.trace else None for r in readings]
        known = [c for c in costs if c is not None]
        if not known:
            return readings
        best = min(known)
        return [arudi.WordReading(r.bits, r.final, r.wasl,
                                  r.cost + min(self.cap, self.char_weight * ((c if c is not None else best) - best)),
                                  r.note, r.trace) for r, c in zip(readings, costs)]

    def reweight(self, word: str, readings: list) -> list:
        entry = self.counts.get(word_key(word))
        if not entry or sum(entry.values()) < self.min_total:
            return self._char_reweight(readings) if self.chars is not None else readings
        total = sum(entry.values())
        top = max(entry.values())
        out = []
        for r in readings:
            c = entry.get(reading_key(r), 0.0)
            prior = min(self.cap, math.log((top + ALPHA) / (c + ALPHA)))
            cost = prior if c > 0 else self.cap + r.cost
            out.append(arudi.WordReading(r.bits, r.final, r.wasl, cost, r.note, r.trace))
        return out

    # ---------------------------------------------------------------------------------------------- persistence
    def train_chars(self) -> "CharModel":
        """Fit the letter model on the lexicon's counted readings (traced back to letters)."""
        cm = CharModel()
        for key, entry in self.counts.items():
            by_key = {reading_key(r): r for r in arudi.word_readings(key, use_lexicon=False)}
            for rk, c in entry.items():
                r = by_key.get(rk)
                if r is not None and r.trace:
                    cm.add(r.trace, c)
        self.chars = cm
        return cm

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        payload = {"meta": meta or {}, "cap": self.cap, "min_total": self.min_total, "counts": self.counts,
                   "chars": self.chars.tables if self.chars is not None else None}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        chars = CharModel(payload["chars"]) if payload.get("chars") else None
        lex = cls(payload["counts"], cap=payload.get("cap", 3.0), min_total=payload.get("min_total", 2.0), chars=chars)
        lex.meta = payload.get("meta", {})
        return lex
