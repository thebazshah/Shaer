"""Rhyme (qāfiya) and refrain (radīf) analysis of a poem.

Classical terms (al-Khalīl):
  rawī      the consonant the poem is named by and that every bayt must end on
  majrā     the vowel of the rawī (iqwāʾ: ḍamma alternating with kasra; iṣrāf: fatḥa alternating with either)
  waṣl      the long vowel that lengthens the rawī's vowel, or a hāʾ after the rawī (ـهُ، ـها، ـة in pause)
  khurūj    the long vowel after a waṣl hāʾ (ـهو، ـها، ـهي)
  ridf      a long vowel (ا، و، ي, or a diphthong) right before the rawī; alif alternating with wāw/yāʾ is a flaw
            (sinād al-ridf), wāw with yāʾ is allowed; a ridf in some bayts and not others is a flaw too
            (wāw/yāʾ: decided from the written word; see Unit.ridf)
  taʾsīs    an alif with exactly one letter (the dakhīl) between it and the rawī; all or none (sinād al-taʾsīs)
  muṭlaqa / muqayyada   the rawī is voweled / has a sukūn; mixing them is a flaw
  īṭāʾ      the same rhyme word repeated (flagged within 7 bayts)

Radīf in the Urdu/Persian sense (a word or phrase repeated after the rhyme in every bayt) is detected separately: when most
bayts end with the same word(s), that ending is the refrain and the rhyme is read on the word before it.

Where the written text leaves it open whether a final ا/و/ي/ى, ه or ة is the rawī or a waṣl, both are considered for each
bayt and the poem-level choice that makes the most bayts agree is taken (for example الهوى / النوى / المنى rhyme on the
alif, while دارِها / جارِها rhyme on the rāʾ). A long vowel after the rhyme consonant and a hāʾ after a voweled letter are
waṣl by rule; they are taken as the rawī only when three bayts in four support that reading (a maqṣūra alif: ى, or ا after a stem
of at most two letters as in دعا / الرضا / هنا; a hāʾ after a long vowel or sukūn as in تراهُ / منهُ). So القمرا / نفرا
(iṭlāq, rawī ر) do not rhyme with هنا / الجنى, nor بعقدِهِ with بسلمِهِ, nor كريمه with قتيله.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from . import arudi

ALEF, WAW, YAA, HAA, TA_MARBUTA, ALEF_MAQSURA = "ا", "و", "ي", "ه", "ة", "ى"
ALEF_MADDA, SHADDA, SUKUN, KASRA, DAMMA = "آ", "ّ", "ْ", "ِ", "ُ"
MADD = {ALEF: "a", WAW: "u", YAA: "i", ALEF_MAQSURA: "a"}
VOWEL_OF_MARK = {"َ": "a", "ُ": "u", "ِ": "i", "ً": "a", "ٌ": "u", "ٍ": "i"}
HAMZAS = set("ءأإؤئآ")
ITA_WINDOW = 7
ALLAH_END = re.compile("لله$")
UNWRITTEN_ALIF = re.compile("(?:ذلك|اولئك|لكن)$")     # ḏālika, ulāʾika, lākin: an alif two letters before the rawī
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
    _written: dict | None = field(default=None, repr=False)
    choice: int = 0                     # which of ridf_options() the poem-level analysis settled on

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

    def written_marks(self, i: int) -> str | None:
        """The marks WRITTEN on the letter at trace index i (None when it cannot be aligned). The trace comes from the
        scanner's reading, which drops letters (the article's alif, alif waṣla) and, in letters mode, every mark; it keeps
        the order of the written letters, so the two are aligned from the end of the word."""
        if self._written is None:
            self._written = _align_written(self.word, self.letters)
        return self._written.get(i, (None, None))[1]

    def written_index(self, i: int) -> int | None:
        """Position in the written word of the letter at trace index i (None when it cannot be aligned)."""
        if self._written is None:
            self._written = _align_written(self.word, self.letters)
        return self._written.get(i, (None, None))[0]

    def ridf_options(self) -> list:
        """The (ridf, taʾsīs) readings the written word allows, the likeliest first; ridf is 'alif', 'wy' (wāw or yāʾ,
        long vowel or diphthong) or ''.

        The scanner's reading only locates the rawī. The letters before it are taken from the WRITTEN word: in letters
        mode the reading of the last word is chosen from bare letters, and at the end of a line the meter rarely decides
        it, so the reading may drop a written alif (كلالها) or make a long vowel a consonant (عظيم / العميم read
        differently in one poem).

        An alif right before the rawī (ا، ى، or آ) is a ridf, and so is the unwritten alif of الله (بالله). A wāw or yāʾ there is a consonant when a vowel or shadda is
        written on it (حَوَر، الأَوَلِ، يَتَرَيَّعُ), when it is the word's first letter (the wa- of ولي), when a sākin
        alif precedes it (مداويا: it is then the dakhīl of a taʾsīs), when the letter before it has a sukūn (أَحْوَر), or
        when the rawī is itself a long vowel (الهوى). It is a ridf when it has a sukūn, or when the letter before it has the
        matching vowel (ḍamma before wāw, kasra before yāʾ: مَسْرُور، حَبِيب). Otherwise it is open, ridf first (most such
        words have one): it votes for a ridf, but takes the poem's majority reading and is never flagged itself."""
        if self.rawi_i is None:
            return [("", False)]
        units = arudi._units(self.word)
        k = self.written_index(self.rawi_i)
        if k is None or k < 1:
            return [("", False)]
        pl, pm = units[k - 1]
        ql, qm = units[k - 2] if k >= 2 else ("", "")
        # a taʾsīs alif is pronounced: an alif waṣla or the article's alif (باسمه، والتي) is dropped by every reading, so
        # a written ا counts only where the reading has it; ذلك، أولئك، لكن have one that is not written
        read = {w for w, _ in (self._written or {}).values()}
        alif_q = (ql == ALEF_MADDA or (ql == ALEF and k - 2 in read)
                  or bool(UNWRITTEN_ALIF.search("".join(arudi._fold(l) for l, _ in units[:k + 1]))))
        if pl in (ALEF, ALEF_MAQSURA, ALEF_MADDA):
            return [("alif", False)]
        if units[k][0] == HAA and pl == ql == "ل" and ALLAH_END.search(arudi.strip_marks(self.word)):
            return [("alif", False)]           # الله، لله، بالله: the alif of -lāh is not written
        if pl in (WAW, YAA):
            r = self.letters[self.rawi_i]
            own = {YAA: KASRA, WAW: DAMMA}[pl]
            voweled = SHADDA in pm or any(m in VOWEL_OF_MARK and m != own for m in pm)
            if voweled or k - 1 == 0 or alif_q or SUKUN in qm or (r[0] in MADD and r[2] == "0"):
                return [("", alif_q)]
            if SUKUN in pm or own in qm:
                return [("wy", False)]
            return [("wy", False), ("", False)]
        return [("", alif_q)]

    @property
    def ridf(self) -> str:
        return self.ridf_options()[self.choice][0]

    @property
    def tasis(self) -> bool:
        return self.ridf_options()[self.choice][1]

    def summary(self) -> dict:
        return {"word": self.word, "rawi": self.rawi, "majra": self.majra, "mutlaqa": self.mutlaqa, "wasl": self.wasl,
                "khuruj": self.khuruj, "ridf": self.ridf, "tasis": self.tasis, "rule": self.rule}


def _align_written(word: str, letters: list) -> dict:
    """{trace index: (position in the written word, marks written on that letter)}.

    The reading's trace can drop written letters (the article's alif, alif waṣla, an alif a bare-letter reading lost) and
    add some (the end-of-line lengthening, marked virtual, or a long vowel a reading inserted), so the two are aligned
    as a longest common subsequence of letters, preferring matches nearest the end of the word (where the rhyme is)."""
    units = arudi._units(word)
    tr = [(j, l) for j, (l, m, b) in enumerate(letters) if m != "virtual"]
    n, m = len(tr), len(units)
    # best[i][k]: LCS length of tr[i:] and units[k:]
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for k in range(m - 1, -1, -1):
            same = fold(tr[i][1]) == fold(units[k][0])
            best[i][k] = max(best[i + 1][k], best[i][k + 1], best[i + 1][k + 1] + 1 if same else 0)
    # read the alignment back from the end so that ties go to matches nearest the end of the word
    end = [[0] * (m + 1) for _ in range(n + 1)]   # LCS of tr[:i] and units[:k]
    for i in range(1, n + 1):
        for k in range(1, m + 1):
            same = fold(tr[i - 1][1]) == fold(units[k - 1][0])
            end[i][k] = max(end[i - 1][k], end[i][k - 1], end[i - 1][k - 1] + 1 if same else 0)
    out: dict = {}
    i, k = n, m
    while i > 0 and k > 0:
        if fold(tr[i - 1][1]) == fold(units[k - 1][0]) and end[i][k] == end[i - 1][k - 1] + 1:
            out[tr[i - 1][0]] = (k - 1, units[k - 1][1])
            i, k = i - 1, k - 1
        elif end[i - 1][k] >= end[i][k - 1]:
            i -= 1
        else:
            k -= 1
    return out


def _written_voweled(marks: str | None) -> bool:
    """A vowel or shadda is written on the letter: it is mutaḥarrik (the second half of a geminate is)."""
    return bool(marks) and (SHADDA in marks or any(m in VOWEL_OF_MARK for m in marks))


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


def _haa_is_wasl(letters: list, j: int, written: dict | None = None) -> bool:
    """A hāʾ (pronoun or tāʾ marbūṭa) after a mutaḥarrik is a waṣl; after a sākin it is the rawī (مِنْهُ). A sukūn written
    on the letter before decides even when the reading was taken from the bare letters."""
    if j < 1:
        return False
    marks = (written or {}).get(j - 1, (0, None))[1]
    if marks and SUKUN in marks:
        return False
    prev = letters[j - 1][0]
    if prev in (ALEF, ALEF_MAQSURA, ALEF_MADDA) or (prev == WAW and letters[j][0] == HAA and marks is not None
                                                     and not _written_voweled(marks)):
        return False                   # تراهُ، أضاعوه: a written long vowel before the hāʾ, whatever a bare-letter reading
                                       # says (ـيه is left to the reading: -īhi in فيه, -iyah in هاويه)
    return letters[j - 1][2].endswith("1") or _written_voweled(marks)


def _maqsura_like(word: str) -> bool:
    """The word's final alif can itself be the rawī (alif maqṣūra): written ى, or ا after a stem of at most two letters
    (دعا، الرضا، هنا، العلا; the article and the conjunctions و/ف, and بـ/كـ/لـ before the article, are not counted).
    A longer stem, a shadda before the alif or a tanwīn marks the alif of iṭlāq (القمرا، ثيابا، حبّا، أمرًا), a waṣl."""
    units = arudi._units(word)
    if not units:
        return False
    last = units[-1][0]
    if last == ALEF_MAQSURA:
        return True
    if last != ALEF or len(units) < 2:
        return False
    if any("ً" in m for _, m in units[-2:]) or SHADDA in units[-2][1] or units[-2][0] == HAA:
        return False                   # ـها: the alif is the khurūj after a hāʾ waṣl
    return len(_stem(word)) <= 2


def _stem(word: str, drop_last: int = 1) -> str:
    """The written letters of the word without its last `drop_last` letters, the article and the conjunctions و/ف (and
    بـ/كـ/لـ before the article)."""
    stem = "".join(l for l, _ in arudi._units(word)[:-drop_last])
    stem = re.sub("^[وف](?=..)", "", stem)
    return re.sub("^(?:[بكل]?ال|لل)(?=..)", "", stem)


def _weak(u: "Unit", i: int) -> bool:
    """Is taking letter i of the rhyme word as the rawī only a latitude (a reading that holds when most bayts share it)?
    True for a long vowel (sākin ا/و/ي/ى) other than a maqṣūra alif ending the word, and for a hāʾ (or a tāʾ marbūṭa read
    as hāʾ) after a mutaḥarrik, unless the hāʾ is a root letter as far as the written word shows (الله، الشبه، العمه)."""
    l, m, b = u.letters[i]
    if l == TA_MARBUTA and i < len(u.letters) - 1:
        return False                   # read with its vowel (ـةِ + the lengthening) it is a tāʾ: a consonant
    if l in (HAA, TA_MARBUTA):
        if not _haa_is_wasl(u.letters, i, u._written):
            return False
        bare = arudi.strip_marks(u.word)
        return not (l == HAA and bare.endswith(HAA) and (ALLAH_END.search(bare) or len(_stem(u.word)) == 2))
    if l in MADD and b == "0" and m != "virtual":
        if _written_voweled(u.written_marks(i)):
            return False               # a shadda or vowel is written on it: a consonant (حَيّ، عَلِيِّ)
        k, units = u.written_index(i), arudi._units(u.word)
        if (l in (WAW, YAA) and k is not None and k == len(units) - 2 and not units[k][1]
                and (units[-1][0] == TA_MARBUTA or (units[-1][0] == HAA and not _written_voweled(units[-1][1])))):
            return False               # ـية, and unmarked ـيه / ـوه: -iyyah / -uwwah as much as -īh / -ūh (الزكيه، البريه);
                                       # a vowel on the hāʾ (معانيهِ) shows the pronoun after a long vowel
        return not (i == len(u.letters) - 1 and l in (ALEF, ALEF_MAQSURA) and _maqsura_like(u.word))
    return False


def _candidates(letters: list, written: dict | None = None) -> list:
    """Possible rawī positions for one bayt, most classical first. `written` ({trace index: written marks}) lets a vowel
    or shadda written before a hāʾ count even when the reading was taken from the bare letters (سَنِيَّه: rawī ي)."""
    n = len(letters)
    if n == 0:
        return []
    out = []
    last_l, last_m, last_b = letters[-1]
    i = n - 1
    def haa_is_wasl(j: int) -> bool:
        return _haa_is_wasl(letters, j, written)

    if last_l in MADD and last_b == "0" and n >= 2:
        j = n - 2
        # a bare-letter reading can make the WRITTEN final long vowel a voweled consonant and lengthen that
        # (وتقصيري read …ri-yi + ī): a written unvoweled و/ي/ا ending the word after the rhyme consonant is the waṣl
        wj = (written or {}).get(j)
        if (last_m == "virtual" and n >= 3 and letters[j][0] in (WAW, YAA, ALEF) and wj is not None
                and wj[0] == max(v[0] for v in written.values()) and not _written_voweled(wj[1])
                and letters[j - 1][0] not in MADD):
            out.append((n - 3, "written madd is waṣl"))
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
        u._written = _align_written(u.word, u.letters)
        u.candidates = _candidates(u.letters, u._written)
        units.append(u)

    # poem-level choice of the rawī: the letter the most bayts can take as a strong reading, then as any allowed reading,
    # then the weighted vote (each bayt votes for its
    # candidates, the classical reading first: دارِها votes ر before ه, منزلا votes ل before ا), then a consonant.
    # A weak reading (see _weak) counts only when it is a strong reading in at least three bayts of four: الدجى / الثرى /
    # المنى / المبتدا rhyme on the alif, القمرا / الجنى / هنا and الصبا / ذكرى / ذكرا do not.
    live = [u for u in units if u and u.candidates]
    strong = Counter()
    for u in live:
        strong.update({fold(u.letters[i][0]) for i, _ in u.candidates if not _weak(u, i)})
    allowed_weak = {k for k, c in strong.items() if 4 * c >= 3 * len(live)}
    votes, cover = Counter(), Counter()
    for u in live:
        eff = [(i, r) for i, r in u.candidates if not _weak(u, i) or fold(u.letters[i][0]) in allowed_weak]
        u.candidates = eff or u.candidates[:1]
        seen_letters = set()
        for r, (i, rule) in enumerate(u.candidates):
            k = fold(u.letters[i][0])
            if k not in seen_letters:
                seen_letters.add(k)
                cover[k] += 1
                votes[k] += CANDIDATE_WEIGHTS[min(r, len(CANDIDATE_WEIGHTS) - 1)]
    rawi = max(votes, key=lambda k: (strong[k], cover[k], votes[k], k not in MADD)) if votes else ""
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
    # ridf: the majority of the likeliest readings sets the poem's ridf; each bayt takes a reading that agrees with it
    # when its written form allows one, and is flagged only when none does (wāw alternating with yāʾ is allowed)
    rf = Counter(u.ridf_options()[0][0] for u in present)
    if rf:
        major = rf.most_common(1)[0][0]
        bad = []
        for i, u in enumerate(units):
            if not (u and u.rawi_i is not None):
                continue
            ok = [k for k, (r, _) in enumerate(u.ridf_options()) if r == major]
            u.choice = ok[0] if ok else 0
            if not ok:
                bad.append(i)
        flag("sinad_ridf", bad, f"ridf {major!r} expected" if major else "no ridf expected")
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
            su.candidates = _candidates(su.letters, _align_written(su.word, su.letters))
            tasri = any(fold(su.letters[i][0]) == rawi for i, _ in su.candidates)

    return {"rawi": rawi, "refrain": use, "refrain_detected": detected, "tasri": tasri,
            "units": [u.summary() if u else None for u in units], "issues": issues,
            "ok": not any(x["kind"] != "ita" for x in issues)}
