"""Rhyme (qāfiya) and refrain (radīf) analysis of a poem.

Classical terms (al-Khalīl):
  rawī      the consonant the poem is named by and that every bayt must end on
  majrā     the vowel of the rawī (iqwāʾ: ḍamma alternating with kasra; iṣrāf: fatḥa alternating with either)
  waṣl      the long vowel that lengthens the rawī's vowel, or a hāʾ after the rawī (ـهُ، ـها، ـة in pause)
  khurūj    the long vowel after a waṣl hāʾ (ـهو، ـها، ـهي)
  ridf      a long vowel (ا، و، ي, or a diphthong) right before the rawī; alif alternating with wāw/yāʾ is a flaw
            (sinād al-ridf), wāw with yāʾ is allowed; a ridf in some bayts and not others is a flaw too
  taʾsīs    an alif with exactly one letter (the dakhīl) between it and the rawī; all or none (sinād al-taʾsīs)
  muṭlaqa / muqayyada   the rawī is voweled / has a sukūn; mixing them is a flaw
  īṭāʾ      the same rhyme word repeated (flagged within 7 bayts)

Radīf in the Urdu/Persian sense (a word or phrase repeated after the rhyme in every bayt) is detected separately: when most
bayts end with the same word(s), that ending is the refrain and the rhyme is read on the word before it.

Where the written text leaves it open whether a final ا/و/ي/ى, ه or ة is the rawī or a waṣl, both are considered for each
bayt and the poem-level choice that makes the most bayts agree is taken (for example الهوى / النوى / المنى rhyme on the
alif, while دارِها / جارِها rhyme on the rāʾ).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from . import arudi

ALEF, WAW, YAA, HAA, TA_MARBUTA, ALEF_MAQSURA = "ا", "و", "ي", "ه", "ة", "ى"
MADD = {ALEF: "a", WAW: "u", YAA: "i", ALEF_MAQSURA: "a"}
VOWEL_OF_MARK = {"َ": "a", "ُ": "u", "ِ": "i", "ً": "a", "ٌ": "u", "ٍ": "i"}
HAMZAS = set("ءأإؤئآ")
ITA_WINDOW = 7
CANDIDATE_WEIGHTS = (1.0, 0.6, 0.35)


def fold(letter: str) -> str:
    if letter in HAMZAS:
        return "ء"
    if letter == TA_MARBUTA:
        return "ت"                 # a rawī tāʾ marbūṭa read with its vowel rhymes with ت
    if letter == ALEF_MAQSURA:
        return ALEF
    return letter


def _vowel(marks: str) -> str:
    for m in marks:
        if m in VOWEL_OF_MARK:
            return VOWEL_OF_MARK[m]
    return "0" if "ْ" in marks else ""


@dataclass
class Unit:
    """The rhyme of one bayt."""
    word: str
    letters: list                       # [(letter, marks, bits)] of the rhyme word, with the end-of-line lengthening
    candidates: list = field(default_factory=list)   # [(rawi index, rule)]
    rawi_i: int | None = None
    rule: str = ""

    def _at(self, i):
        return self.letters[i] if 0 <= i < len(self.letters) else None

    @property
    def rawi(self) -> str:
        return fold(self.letters[self.rawi_i][0]) if self.rawi_i is not None else ""

    @property
    def majra(self) -> str:
        """Vowel of the rawī from WRITTEN evidence only: a/u/i from its vowel mark or a written long vowel after it,
        0 from a written sukūn, '' when the text does not say (then nothing is flagged)."""
        if self.rawi_i is None:
            return ""
        l, m, b = self.letters[self.rawi_i]
        v = _vowel(m)
        if v in ("a", "u", "i", "0"):
            return v
        nxt = self._at(self.rawi_i + 1)
        if nxt and nxt[0] in (ALEF, WAW, YAA) and nxt[1] != "virtual" and nxt[2] == "0":
            return MADD[nxt[0]]           # ى is left out: in this corpus it often stands for ي
        return ""

    @property
    def mutlaqa(self) -> bool | None:
        mj = self.majra
        return None if mj == "" else mj != "0"

    @property
    def wasl(self) -> str:
        nxt = self._at(self.rawi_i + 1) if self.rawi_i is not None else None
        if not nxt:
            return ""
        return "haa" if nxt[0] in (HAA, TA_MARBUTA) else "madd"

    @property
    def khuruj(self) -> bool:
        if self.wasl != "haa":
            return False
        k = self._at(self.rawi_i + 2)
        return bool(k and k[0] in MADD)

    @property
    def ridf(self) -> str:
        """'alif', 'wy' (wāw or yāʾ, long vowel or diphthong) or ''."""
        if self.rawi_i is None:
            return ""
        p = self._at(self.rawi_i - 1)
        if not p or p[2] != "0":
            return ""
        if p[0] in (ALEF, ALEF_MAQSURA):
            return "alif"
        if p[0] in (WAW, YAA):
            return "wy"
        return ""

    @property
    def tasis(self) -> bool:
        if self.rawi_i is None or self.ridf:
            return False
        a = self._at(self.rawi_i - 2)
        return bool(a and a[0] == ALEF and a[2] == "0")

    def summary(self) -> dict:
        return {"word": self.word, "rawi": self.rawi, "majra": self.majra, "mutlaqa": self.mutlaqa, "wasl": self.wasl,
                "khuruj": self.khuruj, "ridf": self.ridf, "tasis": self.tasis, "rule": self.rule}


def _letters_with_end(word: str, reading, line_end: bool = True) -> list:
    """Letters of the rhyme word with their bits, plus the lengthening the end of the line adds."""
    trace = list(reading.trace) if reading is not None and reading.trace else []
    if not trace:                              # a lexicon word: take the generic reading with the same bits
        for r in arudi.word_readings(word, use_lexicon=False):
            if reading is None or (r.bits, r.final) == (reading.bits, reading.final):
                trace = list(r.trace)
                break
    if not trace:
        units = arudi._units(word)
        trace = [(l, m, "1") for l, m in units]
    final = reading.final if reading is not None else ""
    if line_end and final in ("1", "n"):       # ishbāʿ: the final vowel becomes a long vowel (the waṣl), marked virtual
        l, m, b = trace[-1]
        v = _vowel(m) or "i"
        trace[-1] = (l, m, "1")
        trace.append(({"a": ALEF, "u": WAW, "i": YAA}.get(v, YAA), "virtual", "0"))
    return trace


def _candidates(letters: list) -> list:
    """Possible rawī positions for one bayt, most classical first."""
    n = len(letters)
    if n == 0:
        return []
    out = []
    last_l, last_m, last_b = letters[-1]
    i = n - 1
    def haa_is_wasl(j: int) -> bool:
        """A hāʾ (pronoun or tāʾ marbūṭa) after a mutaḥarrik is a waṣl; after a sākin it is the rawī (مِنْهُ)."""
        return j >= 1 and letters[j - 1][2].endswith("1")

    if last_l in MADD and last_b == "0" and n >= 2:
        j = n - 2
        haa = letters[j][0] in (HAA, TA_MARBUTA) and n >= 3
        if haa and haa_is_wasl(j):
            out.append((n - 3, "hāʾ waṣl + khurūj"))   # دارِها: rawī ر, waṣl هـ, khurūj ا
        out.append((n - 2, "madd is waṣl"))            # دارِ+ي، منزلا: the madd lengthens the rawī's vowel
        if haa and not haa_is_wasl(j):
            out.append((n - 3, "hāʾ waṣl + khurūj"))
        if last_m != "virtual":                        # only a written long vowel can itself be the rawī (الهوى)
            out.append((n - 1, "alif/madd is the rawī"))
    elif last_l in (HAA, TA_MARBUTA) and n >= 2:
        if haa_is_wasl(n - 1):
            out += [(n - 2, "hāʾ is waṣl"), (n - 1, "hāʾ is the rawī")]
        else:
            out += [(n - 1, "hāʾ is the rawī"), (n - 2, "hāʾ is waṣl")]
    else:
        out.append((i, "final consonant"))
    return [c for c in out if c[0] >= 0]


def detect_refrain(endings: list[list[str]], min_share: float = 0.6) -> list[str]:
    """Words repeated at the end of most bayts (Urdu-style radīf), longest first; [] when there is none."""
    endings = [e for e in endings if e]
    if len(endings) < 2:
        return []
    refrain: list[str] = []
    for k in range(1, 4):
        tails = Counter(tuple(arudi.strip_marks(w) for w in e[-k:]) for e in endings if len(e) > k)
        if not tails:
            break
        tail, c = tails.most_common(1)[0]
        if c / len(endings) >= min_share:
            refrain = list(tail)
        else:
            break
    return refrain


def analyse(ajuz_lines: list[str], last_readings: list, sadr_first: str | None = None, sadr_first_reading=None,
            refrain: list[str] | None = None) -> dict:
    """Rhyme analysis of a poem from its ʿajuz lines and the chosen readings of their last words.

    last_readings[i] is the WordReading the scanner chose for the last word of ajuz_lines[i] (or None when the line did not
    scan). `refrain` forces a required radīf; by default it is detected.
    """
    words_per_line = [arudi.tokenize(x) for x in ajuz_lines]
    detected = detect_refrain(words_per_line)
    use = refrain if refrain is not None else detected
    units: list[Unit | None] = []
    refrain_ok: list[bool] = []
    for words, rd in zip(words_per_line, last_readings):
        if not words:
            units.append(None)
            refrain_ok.append(False)
            continue
        has = bool(use) and [arudi.strip_marks(w) for w in words[-len(use):]] == list(use)
        refrain_ok.append(has)
        rhyme_words = words[:-len(use)] if (use and has) else words
        if not rhyme_words:
            units.append(None)
            continue
        w = rhyme_words[-1]
        reading = rd if not (use and has) else None   # the scanner's choice belongs to the refrain word then
        if reading is None:
            rs = arudi.word_readings(w, use_lexicon=False)
            reading = min(rs, key=lambda r: r.cost) if rs else None
        u = Unit(w, _letters_with_end(w, reading, line_end=not (use and has)))
        u.candidates = _candidates(u.letters)
        units.append(u)

    # poem-level choice of the rawī
    # each bayt votes for its candidates, the classical reading first (weights 1, 0.6, 0.35): دارِها votes ر before ه,
    # منزلا votes ل before ا, while الهوى / المنى / الدجى agree only on the alif
    votes = Counter()
    for u in units:
        if u:
            seen_letters = set()
            for r, (i, rule) in enumerate(u.candidates):
                k = fold(u.letters[i][0])
                if k not in seen_letters:
                    seen_letters.add(k)
                    votes[k] += CANDIDATE_WEIGHTS[min(r, len(CANDIDATE_WEIGHTS) - 1)]
    rawi = max(votes, key=lambda k: (votes[k], k not in MADD)) if votes else ""
    for u in units:
        if not u:
            continue
        pick = next(((i, r) for i, r in u.candidates if fold(u.letters[i][0]) == rawi), u.candidates[0] if u.candidates else (None, ""))
        u.rawi_i, u.rule = pick

    present = [u for u in units if u and u.rawi_i is not None]
    issues: list[dict] = []

    def flag(kind: str, bayts: list[int], detail: str) -> None:
        if bayts:
            issues.append({"kind": kind, "bayts": bayts, "detail": detail})

    flag("rawi", [i for i, u in enumerate(units) if u and u.rawi != rawi], f"rawī {rawi!r} expected")
    mut = Counter(u.mutlaqa for u in present if u.mutlaqa is not None)
    if mut:
        major = mut.most_common(1)[0][0]
        flag("mutlaqa_muqayyada", [i for i, u in enumerate(units) if u and u.mutlaqa is not None and u.mutlaqa != major],
             "voweled and sukūn rawī mixed")
        if major:
            mj = Counter(u.majra for u in present if u.majra in ("a", "u", "i"))
            if mj:
                mv = mj.most_common(1)[0][0]
                bad = [i for i, u in enumerate(units) if u and u.majra in ("a", "u", "i") and u.majra != mv]
                kind = "israf" if mv == "a" or any(units[i].majra == "a" for i in bad) else "iqwa"
                flag(kind, bad, f"rawī vowel {mv!r} expected")
    rf = Counter(u.ridf for u in present)
    if rf:
        major = rf.most_common(1)[0][0]
        if major:
            flag("sinad_ridf", [i for i, u in enumerate(units) if u and u.rawi_i is not None and
                                (not u.ridf or (u.ridf == "alif") != (major == "alif"))], f"ridf {major!r} expected")
        else:
            flag("sinad_ridf", [i for i, u in enumerate(units) if u and u.ridf], "no ridf expected")
    ts = Counter(u.tasis for u in present)
    if ts:
        major = ts.most_common(1)[0][0]
        flag("sinad_tasis", [i for i, u in enumerate(units) if u and u.rawi_i is not None and u.tasis != major],
             "taʾsīs expected" if major else "no taʾsīs expected")
    seen: dict[str, int] = {}
    ita = []
    for i, u in enumerate(units):
        if not u:
            continue
        key = arudi.strip_marks(u.word)
        if key in seen and i - seen[key] <= ITA_WINDOW:
            ita.append(i)
        seen[key] = i
    flag("ita", ita, f"rhyme word repeated within {ITA_WINDOW} bayts")
    if use:
        flag("radif_missing", [i for i, ok in enumerate(refrain_ok) if not ok], f"refrain {' '.join(use)!r} expected")

    tasri = None
    if sadr_first is not None and units and units[0]:
        sw = arudi.tokenize(sadr_first)
        if sw:
            su = Unit(sw[-1], _letters_with_end(sw[-1], sadr_first_reading))
            su.candidates = _candidates(su.letters)
            tasri = any(fold(su.letters[i][0]) == rawi for i, _ in su.candidates)

    return {"rawi": rawi, "refrain": use, "refrain_detected": detected, "tasri": tasri,
            "units": [u.summary() if u else None for u in units], "issues": issues,
            "ok": not any(x["kind"] != "ita" for x in issues)}
