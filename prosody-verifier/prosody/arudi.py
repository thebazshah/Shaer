"""Prosodic writing (al-kitāba al-ʿarūḍiyya): Arabic text -> the mutaḥarrik (1) / sākin (0) choices it allows.

Scansion counts what is pronounced, not what is written. The rules applied here:

* a letter with fatḥa/ḍamma/kasra is 1, with sukūn 0; shadda doubles the letter (0 then 1); tanwīn adds a sākin nūn (10)
* long vowels: ا and ى are 0; و and ي are 0 when they lengthen the vowel before them, 1 or 0 as consonants
* hamzat al-waṣl (the alif of ال and of words such as اسم، ابن، استفعل) is silent inside a line and pronounced (1) at its
  start; after a sākin the next word's waṣl gives that sākin a helping vowel (0 -> 1), and a long vowel before it is
  shortened (dropped): في البيت = fil-bayt
* the lām of ال is 0 before a moon letter and silent before a sun letter, which is doubled instead (الشمس = ash-shams)
* hidden long vowels and silent letters of a small lexicon (هذا = hādhā, لكن = lākin, الله = allāh, أولئك = ulāʾika,
  عمرو = ʿamr, ...), and the dagger alif (ـٰ)
* the silent alif after the plural wāw (كتبوا) and after tanwīn al-fatḥ (كتابًا)
* the end of a hemistich: a final short vowel is lengthened (ishbāʿ: 1 -> 10), tanwīn is read as that lengthening, and a
  final tāʾ marbūṭa may be paused on (0)

A letter without diacritics can usually be read several ways, so every word yields a list of readings (WordReading). The
choices are constrained by Arabic phonotactics: a word cannot start with a sākin, a long vowel needs a vowel before it, and
two sākin letters do not meet inside a word (except in a final pause). Written diacritics are binding: the scanner trusts
them, so a wrong written vowel makes a line fail, as it should.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

FATHA, DAMMA, KASRA, SUKUN, SHADDA = "َ", "ُ", "ِ", "ْ", "ّ"
FATHATAN, DAMMATAN, KASRATAN = "ً", "ٌ", "ٍ"
DAGGER_ALIF = "ٰ"
VOWELS = {FATHA, DAMMA, KASRA}
TANWIN = {FATHATAN, DAMMATAN, KASRATAN}
MARKS = VOWELS | TANWIN | {SUKUN, SHADDA, DAGGER_ALIF, "ٓ", "ٔ", "ٕ"}
TATWEEL = "ـ"

ALEF, ALEF_WASL, ALEF_MADDA, ALEF_HAMZA_A, ALEF_HAMZA_B = "ا", "ٱ", "آ", "أ", "إ"
ALEF_MAQSURA, TA_MARBUTA, WAW, YAA, LAM, HAA = "ى", "ة", "و", "ي", "ل", "ه"
LETTER_RE = re.compile(r"[ء-غف-يٱ]")
SUN_LETTERS = set("تثدذرزسشصضطظلن")
PROCLITICS = set("وفبكل")
# words that start with a bare alif carrying hamzat qaṭʿ often written without the hamza
QAT_BARE = {"الى", "الا", "الي", "ان", "انا", "انت", "انتم", "انتما", "انتن", "انه", "انها", "انهم", "اذا", "اذ", "او", "اما",
            "ام", "اي", "اين", "اني", "انني", "اذن"}

# cost of reading choices the text does not decide (lower = more likely); written diacritics are never overridden
COST_QAT_FOR_BARE_ALIF = 0.3
COST_RARE_READING = 0.3
COST_IMPLICIT_SHADDA = 0.3
COST_FALLBACK = 3.0


@dataclass(frozen=True)
class WordReading:
    bits: str            # pattern of the word, without its boundary-sensitive final part (see `final`)
    final: str           # the final part: "1" short vowel, "0" sākin consonant, "m" long vowel, "n" tanwīn,
                         # "p" a tāʾ marbūṭa paused on (only at the end of a hemistich), "" none
    wasl: bool           # the word starts with hamzat al-waṣl (its alif is not in `bits`)
    cost: float
    note: str = ""
    trace: tuple = ()    # ((letter, marks, bits), ...) for every pronounced letter, in order (generic readings only)


def _units(word: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ch in word:
        if ch in MARKS:
            if out:
                out[-1] = (out[-1][0], out[-1][1] + ch)
        elif LETTER_RE.match(ch):
            out.append((ch, ""))
    return out


def tokenize(line: str) -> list[str]:
    """Arabic words of a line (NFC, tatweel removed, everything that is not a letter or a mark is a separator)."""
    text = unicodedata.normalize("NFC", str(line or "")).replace(TATWEEL, "")
    words, cur = [], ""
    for ch in text:
        if ch in MARKS or LETTER_RE.match(ch):
            cur += ch
        else:
            if cur:
                words.append(cur)
            cur = ""
    if cur:
        words.append(cur)
    return [w for w in words if _units(w)]


def strip_marks(word: str) -> str:
    return "".join(l for l, _ in _units(word))


# ---------------------------------------------------------------------------------------------- lexicon
def _lexicon() -> dict[str, list[tuple[str, str, bool]]]:
    """Undiacritised form -> readings (bits, final, wasl). Hidden long vowels and silent letters. `bits` excludes the
    boundary-sensitive final part (see WordReading); a waṣl word's bits exclude its alif."""
    f3 = ("1", "0", "n")
    f2 = ("1", "0")
    return {
        "هذا": [("101", "m", False)],                                   # hā-dhā
        "هذي": [("101", "m", False)],                                   # hā-dhī
        "هذه": [("101", "1", False), ("1011", "m", False)],             # hā-dhi-hi / hā-dhi-hī
        "هذان": [("1010", "1", False)],                                 # hā-dhā-ni
        "هذين": [("1010", "1", False)],                                 # hā-dhay-ni
        "هؤلاء": [("10110", "1", False)],                               # hā-ʾu-lā-ʾi
        "ذلك": [("101", "1", False)],                                   # dhā-li-ka
        "ذلكم": [("1011", f, False) for f in f2],                       # dhā-li-kum / -kumu
        "ذلكما": [("10111", "m", False)],                               # dhā-li-ku-mā
        "لكن": [("101", "0", False), ("1010", "1", False)],             # lā-kin / lā-kin-na
        "لكنما": [("101011", "m", False)],                              # lā-kin-na-mā
        "هكذا": [("1011", "m", False)],                                 # hā-ka-dhā
        "الله": [("010", f, True) for f in f3],                         # (a)l-lā-h
        "تالله": [("1010", f, False) for f in f2],                      # tal-lā-h
        "لله": [("1010", f, False) for f in f2],                        # lil-lā-h
        "اللهم": [("01010", "1", True)],                                # (a)l-lā-hum-ma
        "اله": [("110", f, False) for f in f3],                         # ʾi-lā-h
        "الاله": [("0110", f, True) for f in f3],                       # (a)l-ʾi-lā-h
        "الرحمن": [("01010", f, True) for f in f2],                     # (a)r-raḥ-mā-n
        "رحمن": [("1010", f, False) for f in f2],                       # raḥ-mā-n
        "طه": [("101", "m", False)],                                    # ṭā-hā
        "داود": [("1010", f, False) for f in f3],                       # dā-wū-d
        "طاوس": [("1010", f, False) for f in f3],                       # ṭā-wū-s
        "اولئك": [("1101", "1", False)],                                # ʾu-lāʾ-i-ka
        "اولاء": [("110", "1", False)],                                 # ʾu-lā-ʾi
        "اولو": [("11", "m", False)],                                   # ʾu-lū
        "اولي": [("11", "m", False)],                                   # ʾu-lī
        "اولات": [("110", f, False) for f in f3],                       # ʾu-lā-t
        "عمرو": [("10", f, False) for f in f3],                         # ʿam-r (the wāw is silent)
        "مائة": [("11", f, False) for f in f3],                         # mi-ʾa (the alif is silent)
        "السموات": [("011010", f, True) for f in f3],                   # (a)s-sa-mā-wā-t
        "سموات": [("11010", f, False) for f in f3],                     # sa-mā-wā-t
        "انا": [("1", "1", False), ("11", "m", False)],                 # ʾa-na / ʾa-nā
        # relative pronouns: the article's lām and the word's lām are written once (الّذي = al-la-dhī)
        "الذي": [("011", "m", True)], "التي": [("011", "m", True)],     # (a)l-la-dhī / (a)l-la-tī
        "الذين": [("0110", "1", True)],                                 # (a)l-la-dhī-na
        "اللذان": [("0110", "1", True)], "اللتان": [("0110", "1", True)],
        "اللذين": [("0110", "1", True)], "اللتين": [("0110", "1", True)],
        "اللاتي": [("0101", "m", True)], "اللائي": [("0101", "m", True)], "اللواتي": [("01101", "m", True)],
        "للذي": [("1011", "m", False)], "للتي": [("1011", "m", False)],   # li-l-la-dhī
        "للذين": [("10110", "1", False)],
    }


LEXICON = _lexicon()


def _fold(word_plain: str) -> str:
    return (word_plain.replace(ALEF_HAMZA_A, ALEF).replace(ALEF_HAMZA_B, ALEF).replace(ALEF_MADDA, ALEF)
            .replace(ALEF_WASL, ALEF))


def _lexicon_readings(word: str) -> list[WordReading] | None:
    """Lexicon match, allowing up to two proclitics (و ف ب ك ل) in front."""
    raw = strip_marks(word).replace(ALEF_WASL, ALEF)
    plain = _fold(raw)
    for k in range(0, 3):
        pre, rest = plain[:k], (plain if k == 0 else raw)[k:]   # behind a proclitic the word must match as written
        if any(c not in PROCLITICS for c in pre):
            break
        if k and rest.startswith(ALEF) and not rest.startswith(ALEF + LAM):
            continue                                  # hamza words (أنا، إله، أولئك) behind a proclitic: كانا is not كَأنا
        entries = LEXICON.get(rest)
        if entries is None:
            continue
        out = []
        for bits, final, wasl in entries:
            if pre:                                   # a proclitic carries a vowel; a waṣl after it is silent
                out.append(WordReading("1" * len(pre) + bits, final, False, 0.0, f"lexicon:{rest}"))
            else:
                out.append(WordReading(bits, final, wasl, 0.0, f"lexicon:{rest}"))
        return out
    return None


# ---------------------------------------------------------------------------------------------- generic readings
def _letter_options(letter: str, marks: str, is_last: bool) -> list[tuple[str, str, float]]:
    """(bits, kind, cost) for one ordinary letter; kind is used to resolve the word ending: v vowel, s sukun, n tanwin."""
    vowel = any(m in VOWELS for m in marks)
    tanwin = any(m in TANWIN for m in marks)
    sukun = SUKUN in marks
    shadda = SHADDA in marks
    dagger = DAGGER_ALIF in marks
    pre = "0" if shadda else ""
    if tanwin:
        return [(pre + "10", "n", 0.0)]
    if vowel:
        return [(pre + "1" + ("0" if dagger else ""), "v" if not dagger else "m", 0.0)]
    if sukun:
        return [(pre + "0", "s", 0.0)]
    if dagger:
        return [(pre + "10", "m", 0.0)]
    if shadda:                                        # shadda without a vowel: the second half still carries one
        opts = [(pre + "1", "v", 0.0)]
        if is_last:
            opts.append((pre + "10", "n", 0.1))
        return opts
    opts = [("1", "v", 0.0), ("0", "s", 0.0)]
    if is_last:
        opts.append(("10", "n", 0.1))
    return opts


def _article(units: list[tuple[str, str]]) -> tuple[int, int, bool] | None:
    """(proclitic letters, index of the article's lām, whether the article's alif is written) for a word with ال.
    Handles ال، وال، بال، فال، كال، وبال، ... and لِ + ال written لل (the alif dropped)."""
    raw = "".join(l.replace(ALEF_WASL, ALEF) for l, _ in units)      # hamza-alifs are NOT folded: أَلْقَتْ is no article
    if raw in QAT_BARE:
        return None
    plain = raw
    for k in range(0, 3):
        prefix, rest = plain[:k], plain[k:]
        if any(c not in PROCLITICS for c in prefix):
            break
        if rest.startswith(ALEF + LAM) and len(rest) > 2:
            return k, k + 1, True
        if k >= 1 and prefix.endswith(LAM) and rest.startswith(LAM) and len(rest) > 1:
            return k, k, False
    return None


def word_readings(word: str, use_lexicon: bool = True) -> list[WordReading]:
    """All readings of one written word (diacritised, partly diacritised or bare). `use_lexicon=False` skips the small
    lexicon of irregular spellings (used to get letter traces)."""
    lex = _lexicon_readings(word) if use_lexicon else None
    if lex is not None:
        return lex
    units = _units(word)
    if not units:
        return []
    n = len(units)
    # tanwīn al-fatḥ written on the final alif (كتاباً) belongs to the letter before it
    if n >= 2 and units[-1][0] in (ALEF, ALEF_MAQSURA) and FATHATAN in units[-1][1]:
        units[-2] = (units[-2][0], units[-2][1] + units[-1][1])      # كُلاًّ، هوىً: the marks belong to the letter before
        units[-1] = (units[-1][0], "")
    if n >= 2 and units[-1][0] == ALEF_MAQSURA and FATHATAN in units[-2][1]:
        units[-1] = (units[-1][0], "")
    letters = [l for l, _ in units]
    plain = "".join(_fold(l) for l in letters)

    skip: set[int] = set()                 # silent letters
    forced: dict[int, list] = {}           # letter index -> its only options
    wasl = alt_qat = False

    # --- the start of the word: article, hamzat al-waṣl, hamzat qaṭʿ written without the hamza
    art = _article(units)
    if art is not None:
        k, lam_i, alif_written = art
        if alif_written:
            skip.add(k)                    # the article's alif: waṣl (silent) or silent after a proclitic's vowel
            wasl = k == 0
        nxt = lam_i + 1
        lam_marks = units[lam_i][1]
        if SHADDA in lam_marks:            # الَّذي, اللَّيل written with the shadda on the lām: the lām itself is doubled
            pass
        elif _fold(letters[nxt]) in SUN_LETTERS:
            skip.add(lam_i)                # assimilated lām; the sun letter is doubled instead
            if SHADDA not in units[nxt][1]:
                units[nxt] = (letters[nxt], units[nxt][1] + SHADDA)
        else:
            forced[lam_i] = [("0", "s", 0.0)]
    elif letters[0] in (ALEF, ALEF_WASL) and n > 1:
        m0 = units[0][1]
        bare_plain = "".join(_fold(l) for l in letters)
        if bare_plain in QAT_BARE or FATHA in m0 or DAMMA in m0:
            forced[0] = [("1", "v", 0.0)]  # an unwritten hamzat qaṭʿ (اِذا = إذا, اَنت = أنت)
        else:
            wasl = True                    # bare ا or اِ: hamzat al-waṣl (اِعمَلوا، ابن) ...
            alt_qat = letters[0] == ALEF   # ... or, in this corpus, an unwritten hamza with kasra (اِنّي = إنّي)
            skip.add(0)

    # --- silent letters at the end
    if n >= 3 and letters[-1] == ALEF and letters[-2] == WAW and not any(m in VOWELS | TANWIN for m in units[-2][1]):
        skip.add(n - 1)                    # the alif after the plural wāw: كتبوا
    if n >= 2 and letters[-1] in (ALEF, ALEF_MAQSURA) and FATHATAN in units[-2][1]:
        skip.add(n - 1)                    # كتابًا، هدىً: the alif carries no sound

    idxs = [i for i in range(n) if i not in skip]
    if not idxs:
        return []
    last_i = idxs[-1]

    def pos_of(i: int) -> int:
        return idxs.index(i)

    def options(i: int) -> list[tuple[str, str, float]]:
        letter, marks = units[i]
        is_last = i == last_i
        if i in forced:
            return forced[i]
        if letter == ALEF_MADDA:
            return [("10", "m", 0.0)]                                 # ʾā
        if letter in (ALEF_HAMZA_A, ALEF_HAMZA_B) and i == idxs[0]:
            return [("1", "v", 0.0)]                                  # hamzat qaṭʿ at the start always has a vowel
        if letter == ALEF and art is not None and i == art[1] + 1:
            return [("1", "v", 0.0)]      # an alif right after the article's lām is a hamza: الاِخيار = al-ʾakhyār
        if letter in (ALEF, ALEF_WASL) or letter == ALEF_MAQSURA:
            return [("0", "m", 0.0)]      # long ā (a fatḥa typed on it, as in لاَ, belongs to the letter before)
        if letter == TA_MARBUTA and not any(m in VOWELS | TANWIN | {SUKUN} for m in marks):
            return [("1", "v", 0.0), ("10", "n", 0.1), ("0", "p", 0.0)] if is_last else [("1", "v", 0.0)]
        if SHADDA not in marks and SUKUN not in marks and (pos_of(i) > 0 or wasl) and letter not in (ALEF, ALEF_WASL, ALEF_MAQSURA, TA_MARBUTA, ALEF_MADDA):
            # an unwritten shadda: the consonant may be doubled (common in partly vowelled text: مجربة = مُجَرَّبة,
            # حبٍّ written حبٍ, البريّات)
            base = options_plain(i)
            gem = [(b, k, c + COST_IMPLICIT_SHADDA) for b, k, c in _letter_options(letter, marks + SHADDA, is_last)]
            return base + gem
        return options_plain(i)

    def options_plain(i: int) -> list[tuple[str, str, float]]:
        letter, marks = units[i]
        is_last = i == last_i
        if letter in (WAW, YAA) and not marks:
            prev = units[idxs[idxs.index(i) - 1]][1] if idxs.index(i) > 0 else ""
            lengthens = (letter == YAA and KASRA in prev) or (letter == WAW and DAMMA in prev)
            madd = ("0", "m" if is_last else "s", 0.0)
            consonant = ("1", "v", COST_RARE_READING if lengthens else 0.0)
            return [madd, consonant]
        return _letter_options(letter, marks, is_last)

    readings: list[WordReading] = []
    pronoun_mim = n >= 3 and letters[-1] == "\u0645" and letters[-2] in ("\u0647", "\u0643", "\u062A") \
        and SUKUN not in units[-1][1]           # ـهم ـكم ـتم: mīm al-jamʿ may be lengthened (هُمُو)

    def last_options(i: int, acc: str) -> list[tuple[str, str, float]]:
        """Options of the last letter that depend on the letter before it."""
        letter, marks = units[i]
        if letter == HAA and n >= 2 and SUKUN not in marks and not any(m in TANWIN for m in marks) and FATHA not in marks:
            after_vowel = bool(acc) and acc[-1] == "1"
            # the pronoun hāʾ is lengthened after a vowel (لَهُ = lahū) and normally short after a sākin (مِنْهُ)
            short = ("1", "v", 0.2 if after_vowel else 0.0)
            long = ("10", "m", 0.0 if after_vowel else COST_RARE_READING)
            return [short, long] + ([("0", "s", 0.2)] if not marks else [])
        opts = options(i)
        if pronoun_mim:
            opts = opts + [("10", "m", 0.1)]
        return opts

    def dfs(pos: int, acc: str, cost: float, kind: str, trace: tuple = ()) -> None:
        if pos == len(idxs):
            if kind == "n":
                readings.append(WordReading(acc[:-2], "n", wasl, cost, "", trace))
            else:
                readings.append(WordReading(acc[:-1], {"v": "1", "s": "0", "m": "m", "p": "p"}[kind], wasl, cost, "", trace))
            return
        last = pos == len(idxs) - 1
        for bits, k, c in (last_options(idxs[pos], acc) if last else options(idxs[pos])):
            new = acc + bits
            if pos == 0 and new[0] != ("0" if wasl else "1"):
                continue           # no word starts on a sākin; a waṣl word starts (after its silent alif) on one
            if "00" in (new[:-1] if last else new):
                continue           # two sākin letters do not meet inside a word (a final pause is judged later)
            if k == "m" and bits == "0" and (not acc or acc[-1] != "1"):
                continue           # a long vowel needs a vowel before it
            dfs(pos + 1, new, cost + c, k, trace + ((units[idxs[pos]][0], units[idxs[pos]][1], bits),))

    dfs(0, "", 0.0, "")
    # (c) و / ف + a bare alif may be the proclitic followed by a silent hamzat al-waṣl (فامْدُدْ، واستغنم): then the next
    #     letter is sākin
    if art is None and n > 3 and letters[0] in ("\u0648", "\u0641") and letters[1] == ALEF and \
            not any(m in (DAMMA,) for m in units[1][1]):
        saved = list(readings)
        readings.clear()
        idxs[:] = [i for i in idxs if i != 1]
        forced[2] = [("0", "s", 0.0)] if not any(m in VOWELS | TANWIN for m in units[2][1]) and SHADDA not in units[2][1] else forced.get(2, [])
        if forced[2]:
            dfs(0, "", 0.2, "")
        readings[:] = saved + [WordReading(r.bits, r.final, r.wasl, r.cost, "silent alif after a proclitic", r.trace)
                               for r in readings]
    if alt_qat:
        readings += [WordReading("1" + r.bits, r.final, False, r.cost + COST_QAT_FOR_BARE_ALIF, "bare alif read as hamza",
                                 ((ALEF, "", "1"),) + r.trace) for r in list(readings)]
    if not readings:
        # nothing satisfies the constraints (a typo, an unusual spelling): fall back to reading every letter as written,
        # with unmarked letters voweled, at a high cost so the word only fits a meter when nothing else does
        bits = "".join("0" if (SUKUN in m or l in (ALEF, ALEF_MAQSURA)) else "1" for i, (l, m) in enumerate(units)
                       if i not in skip)
        if bits:
            readings.append(WordReading(bits[:-1], "1" if bits[-1] == "1" else "0", False, COST_FALLBACK, "fallback"))
    best: dict[tuple, WordReading] = {}
    for r in readings:
        key = (r.bits, r.final, r.wasl)
        if key not in best or r.cost < best[key].cost:
            best[key] = r
    return list(best.values())


# ---------------------------------------------------------------------------------------------- boundaries
def resolve_final(final: str, next_wasl: bool | None) -> list[tuple[str, float]]:
    """Bits of a word's final part given what follows: next_wasl True (a waṣl word), False (another word) or None (the
    end of the hemistich)."""
    if final == "":
        return [("", 0.0)]
    if final == "p":                              # tāʾ marbūṭa paused on as a sākin hāʾ: only at the end of a hemistich
        return [("0", 0.0)] if next_wasl is None else []
    if next_wasl is None:                         # end of the hemistich
        return {"1": [("10", 0.0)],                 # ishbāʿ of the final short vowel
                "0": [("0", 0.0)],
                "m": [("0", 0.0)],
                "n": [("10", 0.0)]}[final]          # tanwīn at the end is read as the lengthened vowel
    if next_wasl:
        return {"1": [("1", 0.0)],
                "0": [("1", 0.0)],                  # helping vowel on the sākin before the waṣl
                "m": [("", 0.0)],                   # the long vowel is shortened before the waṣl's sākin
                "n": [("11", 0.0)]}[final]          # the tanwīn's nūn takes a kasra
    return {"1": [("1", 0.0)], "0": [("0", 0.0)], "m": [("0", 0.0)], "n": [("10", 0.0)]}[final]
