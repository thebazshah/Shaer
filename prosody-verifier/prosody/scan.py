"""Scansion: match a hemistich against every meter pattern, using all readings the text allows.

A hemistich is a sequence of words, each with several possible readings (arudi.word_readings). The scanner searches the
readings and the meter patterns together: a dynamic programme over the words whose state is a node of a trie holding every
hemistich pattern of every meter form (sadr patterns end with an ʿarūḍ form, ʿajuz patterns with a ḍarb form). Word
boundaries are resolved as the search goes (hamzat al-waṣl, the helping vowel, the shortened long vowel, the lengthened final
vowel). The result lists every meter analysis the hemistich can realise, cheapest first; cost counts uncertain reading
choices and rare zihāfāt, so ties between meters are visible instead of hidden.

When nothing matches a requested meter, `nearest` finds the closest pattern of that meter with the fewest letter edits
(change, insert or delete one mutaḥarrik/sākin) and says in which foot the first edit falls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from . import arudi
from .meters import METERS, foot_label, hemistich_analyses

COST_RARE_ZIHAF = 0.3
COST_MUAQABA = 1.0          # a muʿāqaba violation (lenient mode only; strict mode rejects it)


class _Trie:
    def __init__(self):
        self.children: list[dict] = [{}]
        self.terminal: dict[int, list[int]] = {}
        self.mask: list[int] = [0]          # bit set of meter ids reachable from each node

    def add(self, pattern: str, label: int, meter_bit: int = 0) -> None:
        node = 0
        self.mask[0] |= meter_bit
        for ch in pattern:
            nxt = self.children[node].get(ch)
            if nxt is None:
                nxt = len(self.children)
                self.children.append({})
                self.mask.append(0)
                self.children[node][ch] = nxt
            node = nxt
            self.mask[node] |= meter_bit
        self.terminal.setdefault(node, []).append(label)

    def walk(self, node: int, bits: str) -> int:
        for ch in bits:
            node = self.children[node].get(ch, -1)
            if node < 0:
                return -1
        return node


METER_BIT = {k: 1 << i for i, k in enumerate(METERS)}


@dataclass
class _Index:
    analyses: list = field(default_factory=list)
    trie: _Trie = field(default_factory=_Trie)


@lru_cache(maxsize=1)
def index() -> _Index:
    """Every hemistich analysis of every meter form, in one trie (built once, about 2,500 analyses)."""
    idx = _Index()
    for mk, m in METERS.items():
        for fk in m.forms:
            for pos in ("sadr", "ajuz"):
                for a in hemistich_analyses(mk, fk, pos):
                    idx.analyses.append(a)
                    idx.trie.add(a["pattern"], len(idx.analyses) - 1, METER_BIT[mk])
    return idx


@dataclass
class Match:
    meter: str
    form: str
    position: str          # sadr (ʿarūḍ form) or ajuz (ḍarb form)
    end: str               # ʿarūḍ / ḍarb form key
    pattern: str
    feet: list
    cost: float
    rare: int
    muaqaba: list
    readings: list         # per word: the bits it contributed
    choices: list = field(default_factory=list)   # per word: the WordReading chosen

    @property
    def feet_text(self) -> str:
        return " | ".join(foot_label(f) for f in self.feet)


def scan(line: str, strict: bool = True, lexicon=None, meters: tuple | None = None) -> list[Match]:
    """All meter analyses of one hemistich, cheapest first.

    lexicon: a pronunciation prior (prosody.lexicon.Lexicon) that re-weights each word's readings by how the word is
             pronounced in the corpus; without it every reading the spelling allows costs the same.
    meters:  restrict the search to these meter keys (faster; used when verifying a requested meter).
    """
    idx = index()
    trie = idx.trie
    words = arudi.tokenize(line)
    if not words:
        return []
    readings = [arudi.word_readings(w) for w in words]
    if lexicon is not None:
        readings = [lexicon.reweight(w, rs) for w, rs in zip(words, readings)]
    want = sum(METER_BIT[m] for m in meters) if meters else -1
    # state: (trie node, pending final of the previous word) -> (cost, back-pointer)
    states: dict[tuple, tuple] = {(0, None): (0.0, None)}
    for wi, rs in enumerate(readings):
        new: dict[tuple, tuple] = {}
        for (node, pend), (cost, back) in states.items():
            for r in rs:
                if pend is None:
                    prev_opts = [("", 0.0)]
                else:
                    prev_opts = arudi.resolve_final(pend, r.wasl)
                body = ("1" + r.bits) if (wi == 0 and r.wasl) else r.bits     # a line-initial waṣl alif is pronounced
                for prev_bits, pc in prev_opts:
                    n2 = trie.walk(node, prev_bits + body)
                    if n2 < 0 or not (trie.mask[n2] & want):
                        continue
                    key = (n2, r.final)
                    c = cost + pc + r.cost
                    if key not in new or c < new[key][0]:
                        new[key] = (c, (back, wi, prev_bits, body, r))
        states = new
        if not states:
            return []
    results: dict[int, Match] = {}
    for (node, pend), (cost, back) in states.items():
        for end_bits, ec in arudi.resolve_final(pend, None):
            n2 = trie.walk(node, end_bits)
            if n2 < 0 or n2 not in trie.terminal:
                continue
            per_word, choices = _backtrack(back, len(words), end_bits)
            for label in trie.terminal[n2]:
                a = idx.analyses[label]
                if meters and a["meter"] not in meters:
                    continue
                if strict and a["muaqaba"]:
                    continue
                total = cost + ec + COST_RARE_ZIHAF * a["rare"] + (COST_MUAQABA * len(a["muaqaba"]))
                if label not in results or total < results[label].cost:
                    results[label] = Match(a["meter"], a["form"], a["position"], a["end"], a["pattern"], a["feet"], total,
                                           a["rare"], a["muaqaba"], per_word, choices)
    return sorted(results.values(), key=lambda m: (m.cost, m.meter, m.form))


def _backtrack(back, n_words: int, end_bits: str) -> list:
    """Bits contributed by each word on the chosen path (the previous word's final part is credited to that word)."""
    parts = [""] * n_words
    choices = [None] * n_words
    node = back
    while node is not None:
        node, wi, prev_bits, body, r = node
        parts[wi] = body + parts[wi]
        choices[wi] = r
        if wi > 0:
            parts[wi - 1] = parts[wi - 1] + prev_bits
    parts[-1] += end_bits
    return parts, choices


def best_by_meter(matches: list[Match]) -> dict[tuple, Match]:
    """Cheapest match per (meter, form, position, end)."""
    out: dict[tuple, Match] = {}
    for m in matches:
        k = (m.meter, m.form, m.position, m.end)
        if k not in out or m.cost < out[k].cost:
            out[k] = m
    return out


# ---------------------------------------------------------------------------------------------- nearest miss
def candidate_bit_strings(line: str, limit: int = 64) -> list[tuple[str, float]]:
    """The cheapest distinct full readings of a hemistich (bit strings), ignoring meters (for diagnostics)."""
    words = arudi.tokenize(line)
    if not words:
        return []
    beams: list[tuple[float, str, str | None]] = [(0.0, "", None)]
    for wi, w in enumerate(words):
        rs = arudi.word_readings(w)
        nb = []
        for cost, acc, pend in beams:
            for r in rs:
                prev_opts = [("", 0.0)] if pend is None else arudi.resolve_final(pend, r.wasl)
                body = ("1" + r.bits) if (wi == 0 and r.wasl) else r.bits
                for pb, pc in prev_opts:
                    nb.append((cost + pc + r.cost, acc + pb + body, r.final))
        nb.sort(key=lambda x: x[0])
        seen, beams = set(), []
        for b in nb:
            if (b[1], b[2]) not in seen:
                seen.add((b[1], b[2]))
                beams.append(b)
            if len(beams) >= limit * 4:
                break
    out = {}
    for cost, acc, pend in beams:
        for eb, ec in arudi.resolve_final(pend, None):
            s = acc + eb
            if s not in out or cost + ec < out[s]:
                out[s] = cost + ec
    return sorted(out.items(), key=lambda x: x[1])[:limit]


def _edit_distance(a: str, b: str) -> tuple[int, int]:
    """Levenshtein distance and the position in b of the first edit."""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    first = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return prev[-1], first


def nearest(line: str, meter: str, form: str, position: str, limit: int = 64) -> dict | None:
    """Closest pattern of one meter form to any reading of the line: edits, the pattern, and the foot of the first edit."""
    cands = candidate_bit_strings(line, limit=limit)
    if not cands:
        return None
    best = None
    for a in hemistich_analyses(meter, form, position):
        for s, c in cands:
            d, first = _edit_distance(s, a["pattern"])
            key = (d, c)
            if best is None or key < best[0]:
                best = (key, s, a, first)
    (d, c), s, a, first = best
    pos, foot_i = 0, len(a["feet"]) - 1
    for i, f in enumerate(a["feet"]):
        if first < pos + len(f["pattern"]):
            foot_i = i
            break
        pos += len(f["pattern"])
    return {"edits": d, "reading": s, "pattern": a["pattern"], "end": a["end"], "first_edit_foot": foot_i + 1,
            "foot": foot_label(a["feet"][foot_i]), "feet": [foot_label(f) for f in a["feet"]]}
