"""Arabic text helpers shared by the prosody-verifier code (standard library only).

Everything here is a *heuristic* used for the M0 headroom audit. It is NOT the prosody verifier: the verifier (M1-M2) will
scan meter and qafiya with explicit prosodic rules. The rhyme functions below only look at the written end of a hemistich, so
they approximate the rawi (the rhyme consonant) and are calibrated on the human reference poems before any conclusion is drawn.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------------------------- character classes
HARAKAT = set(chr(c) for c in range(0x064B, 0x0653))           # tanwin (3), fatha, damma, kasra, shadda, sukun
TANWIN = {"ً": "a", "ٌ": "u", "ٍ": "i"}           # fathatan, dammatan, kasratan -> vowel
VOWEL_MARKS = {"َ": "a", "ُ": "u", "ِ": "i", "ْ": "0"}   # fatha, damma, kasra, sukun ("0" = no vowel)
SHADDA = "ّ"
OTHER_MARKS = set(chr(c) for c in range(0x0653, 0x0660)) | {"ٰ"}   # maddah/hamza above-below, superscript alef, ...
MARKS = HARAKAT | OTHER_MARKS
TATWEEL = "ـ"
ARABIC_LETTER_RE = re.compile(r"[ء-غف-يٱ]")
ALEF_FORMS = {"ا", "أ", "إ", "آ", "ٱ"}   # ا أ إ آ ٱ
ALEF_MAQSURA = "ى"                                           # ى
TA_MARBUTA = "ة"                                             # ة
HAA = "ه"                                                    # ه
WAW, YAA, ALEF = "و", "ي", "ا"
HAMZA_FORMS = {"ء", "ؤ", "ئ"}                      # ء ؤ ئ (أ إ are folded into alef)
MADD_VOWEL = {ALEF: "a", WAW: "u", YAA: "i"}                     # a long-vowel letter after the rawi lengthens its vowel
# letters that normally carry no haraka of their own, so they are left out of the diacritic-coverage count
UNMARKED_BY_DESIGN = ALEF_FORMS | {ALEF_MAQSURA}


# ---------------------------------------------------------------------------------------------- line splitting
def normalize_whitespace(text: str) -> str:
    """Identical to grpo/rewards/common.py::normalize_whitespace (the scorer's parser), so line counts match it exactly."""
    text = str(text or "")
    text = text.replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[  ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_poem_lines(text: str) -> list[str]:
    """Identical to grpo/rewards/common.py::split_poem_lines: one non-empty line = one hemistich."""
    text = normalize_whitespace(text)
    return [x for x in (line.strip() for line in text.split("\n")) if x]


# ---------------------------------------------------------------------------------------------- units
def units(text: str) -> list[tuple[str, str]]:
    """Split text into (letter, marks) pairs. Non-letters (spaces, punctuation, digits) become ("", "") separators."""
    out: list[tuple[str, str]] = []
    for ch in unicodedata.normalize("NFC", str(text or "")):
        if ch == TATWEEL:
            continue
        if ch in MARKS:
            if out and out[-1][0]:
                out[-1] = (out[-1][0], out[-1][1] + ch)
            continue
        if ARABIC_LETTER_RE.match(ch):
            out.append((ch, ""))
        elif not out or out[-1][0]:
            out.append(("", ""))
    return out


def words(text: str) -> list[list[tuple[str, str]]]:
    """Arabic words of a line as lists of (letter, marks)."""
    result, cur = [], []
    for letter, marks in units(text):
        if letter:
            cur.append((letter, marks))
        elif cur:
            result.append(cur)
            cur = []
    if cur:
        result.append(cur)
    return result


def fold_letter(letter: str) -> str:
    """Spelling variants that do not change the rhyme consonant: alef forms -> ا, ى -> ا, hamza seats -> ء, ة -> ه."""
    if letter in ALEF_FORMS or letter == ALEF_MAQSURA:
        return ALEF
    if letter in HAMZA_FORMS:
        return "ء"
    if letter == TA_MARBUTA:
        return HAA
    return letter


def vowel_of(marks: str) -> str:
    """The vowel written on a letter: 'a', 'u', 'i', '0' (sukun) or '' (unmarked). Tanwin counts as its vowel."""
    for m in marks:
        if m in TANWIN:
            return TANWIN[m]
    for m in marks:
        if m in VOWEL_MARKS:
            return VOWEL_MARKS[m]
    return ""


# ---------------------------------------------------------------------------------------------- diacritics
def diacritic_coverage(text: str) -> tuple[int, int]:
    """(letters carrying at least one haraka, letters that normally can carry one). Alef forms and ى are excluded."""
    marked = eligible = 0
    for letter, marks in units(text):
        if not letter or letter in UNMARKED_BY_DESIGN:
            continue
        eligible += 1
        if any(m in HARAKAT for m in marks):
            marked += 1
    return marked, eligible


# ---------------------------------------------------------------------------------------------- rhyme heuristics
def rhyme_signatures(line: str) -> dict[str, str]:
    """Approximate rhyme keys for the end of one hemistich.

    last_letter  the final letter of the last word, spelling variants folded.
    rawi         the final letter after removing a trailing long-vowel letter (wasl / ishba': ا و ي ى, and the silent
                 alif after a plural waw) and then a trailing ه (treated as the wasl haa of a pronoun or a ta marbuta in
                 pause). This is the classical position of the rawi in most poems; it is wrong for poems whose rawi really
                 is ه, which the reference calibration exposes.
    rawi_vowel   the vowel of that rawi: written on it, or implied by the long-vowel letter after it ('' when unknown).
    """
    ws = words(line)
    if not ws:
        return {"last_letter": "", "rawi": "", "rawi_vowel": ""}
    w = [(fold_letter(l), m) for l, m in ws[-1]]
    last_letter = w[-1][0]
    implied = ""
    if len(w) > 2 and w[-1][0] == ALEF and w[-2][0] == WAW and not vowel_of(w[-2][1]):
        w = w[:-1]                     # the silent alif after a plural waw (رَحِمُوا): the waw is the long vowel, not the alif
    if len(w) > 1 and w[-1][0] in MADD_VOWEL and vowel_of(w[-1][1]) not in ("a", "u", "i"):   # a voweled و/ي is a consonant
        implied = MADD_VOWEL[w[-1][0]]
        w = w[:-1]
    if len(w) > 1 and w[-1][0] == HAA:
        implied = ""
        w = w[:-1]
    rawi_letter, rawi_marks = w[-1]
    return {"last_letter": last_letter, "rawi": rawi_letter, "rawi_vowel": vowel_of(rawi_marks) or implied}


def consistency(keys: list[str]) -> dict:
    """Share of non-empty keys equal to the most common one, and whether all are equal."""
    keys = [k for k in keys if k]
    if not keys:
        return {"n": 0, "majority": "", "share": None, "all_same": None}
    majority, count = Counter(keys).most_common(1)[0]
    return {"n": len(keys), "majority": majority, "share": count / len(keys), "all_same": count == len(keys)}
