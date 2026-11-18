"""The base feet (tafāʿīl) and the operations that derive their variants.

A foot is built from the classical units:
  SK  sabab khafīf   "10"   (mutaḥarrik + sākin)            e.g. لُنْ
  ST  sabab thaqīl   "11"   (two mutaḥarrik)                 e.g. مُتَ
  WM  watad majmūʿ   "110"  (two mutaḥarrik + sākin)         e.g. فَعُو
  WF  watad mafrūq   "101"  (mutaḥarrik + sākin + mutaḥarrik) e.g. لَاتُ

Zihāfāt (changes to the second letter of a sabab; allowed in any foot position unless a meter forbids them) are defined
by the ordinal of the letter they touch, as in the classical books:
  single:  khabn   drop the 2nd letter (sākin)        ṭayy  drop the 4th (sākin)       qabḍ  drop the 5th (sākin)
           kaff    drop the 7th (sākin)              iḍmār silence the 2nd (mutaḥarrik) ʿaṣb  silence the 5th (mutaḥarrik)
           waqṣ    drop the 2nd (mutaḥarrik)          ʿaql  drop the 5th (mutaḥarrik)
  double:  khabl = khabn+ṭayy   khazl = iḍmār+ṭayy   shakl = khabn+kaff   naqṣ = ʿaṣb+kaff
A zihāf is valid only when the letter it touches is the second letter of a sabab; `apply_zihaf` checks this, so a zihāf can
never touch a watad.

ʿIlal (changes at the end of the last foot of a hemistich, the ʿarūḍ or ḍarb; once used they hold for the whole poem):
  by reduction: ḥadhf (drop the final sabab khafīf), qaṭf (ḥadhf + ʿaṣb), qaṭʿ (drop the sākin of a final watad majmūʿ and
  silence the letter before it), qaṣr (drop the sākin of a final sabab khafīf and silence its first letter), ḥadhadh (drop
  a final watad majmūʿ), ṣalm (drop a final watad mafrūq), kashf (drop the last letter of a final watad mafrūq), waqf
  (silence the last letter of a final watad mafrūq), batr (ḥadhf + qaṭʿ), tashʿīth (drop a letter of the watad of
  فاعلاتن / فاعلن: فالاتن / فالن)
  by addition: tarfīl (add a sabab khafīf after a final watad majmūʿ), tadhyīl (add a sākin after a final watad majmūʿ),
  tasbīgh (add a sākin to a final sabab khafīf)
"""
from __future__ import annotations

from dataclasses import dataclass

UNIT_PATTERN = {"SK": "10", "ST": "11", "WM": "110", "WF": "101"}


@dataclass(frozen=True)
class Letter:
    bit: str              # "1" mutaḥarrik, "0" sākin
    unit: int             # index of the unit in the foot
    unit_kind: str        # SK / ST / WM / WF
    pos_in_unit: int      # 0-based


@dataclass(frozen=True)
class Foot:
    key: str
    name: str             # Arabic name of the sound foot
    units: tuple          # unit kinds

    @property
    def letters(self) -> tuple:
        out = []
        for u, kind in enumerate(self.units):
            for p, bit in enumerate(UNIT_PATTERN[kind]):
                out.append(Letter(bit, u, kind, p))
        return tuple(out)

    @property
    def pattern(self) -> str:
        return "".join(UNIT_PATTERN[k] for k in self.units)


FEET = {f.key: f for f in (
    Foot("faulun", "فَعُولُنْ", ("WM", "SK")),
    Foot("failun", "فَاعِلُنْ", ("SK", "WM")),
    Foot("mafailun", "مَفَاعِيلُنْ", ("WM", "SK", "SK")),
    Foot("mustafilun", "مُسْتَفْعِلُنْ", ("SK", "SK", "WM")),
    Foot("mustafi_lun", "مُسْتَفْعِ لُنْ", ("SK", "WF", "SK")),      # watad mafrūq (الخفيف، المجتث)
    Foot("failatun", "فَاعِلَاتُنْ", ("SK", "WM", "SK")),
    Foot("fai_latun", "فَاعِ لَاتُنْ", ("WF", "SK", "SK")),         # watad mafrūq (المضارع)
    Foot("mufaalatun", "مُفَاعَلَتُنْ", ("WM", "ST", "SK")),
    Foot("mutafailun", "مُتَفَاعِلُنْ", ("ST", "SK", "WM")),
    Foot("mafulatu", "مَفْعُولَاتُ", ("SK", "SK", "WF")),
)}

# ---------------------------------------------------------------------------------------------- zihāfāt
# (ordinal of the touched letter, action, required bit of that letter)
_SINGLE = {
    "khabn": (2, "drop", "0"), "tayy": (4, "drop", "0"), "qabd": (5, "drop", "0"), "kaff": (7, "drop", "0"),
    "idmar": (2, "silence", "1"), "asb": (5, "silence", "1"), "waqs": (2, "drop", "1"), "aql": (5, "drop", "1"),
}
ZIHAF_PARTS = {k: (k,) for k in _SINGLE}
ZIHAF_PARTS.update({"khabl": ("khabn", "tayy"), "khazl": ("idmar", "tayy"), "shakl": ("khabn", "kaff"),
                    "naqs": ("asb", "kaff")})
ZIHAF_NAMES = {"khabn": "خبن", "tayy": "طي", "qabd": "قبض", "kaff": "كف", "idmar": "إضمار", "asb": "عصب", "waqs": "وقص",
               "aql": "عقل", "khabl": "خبل", "khazl": "خزل", "shakl": "شكل", "naqs": "نقص"}


class InvalidOperation(ValueError):
    pass


def apply_zihaf(foot: Foot, zihaf: str) -> str:
    """Pattern of `foot` after `zihaf` (a single or double zihāf). Raises InvalidOperation when the zihāf would touch
    anything other than the second letter of a sabab, or a letter of the wrong kind."""
    letters = list(foot.letters)
    bits = [l.bit for l in letters]
    drop = set()
    for part in ZIHAF_PARTS[zihaf]:
        ordinal, action, need = _SINGLE[part]
        if ordinal > len(letters):
            raise InvalidOperation(f"{zihaf} on {foot.key}: the foot has no letter {ordinal}")
        letter = letters[ordinal - 1]
        if letter.unit_kind not in ("SK", "ST") or letter.pos_in_unit != 1:
            raise InvalidOperation(f"{zihaf} on {foot.key}: letter {ordinal} is not the second letter of a sabab")
        if letter.bit != need:
            raise InvalidOperation(f"{zihaf} on {foot.key}: letter {ordinal} is {'sākin' if letter.bit == '0' else 'mutaḥarrik'}")
        if action == "drop":
            drop.add(ordinal - 1)
        else:
            bits[ordinal - 1] = "0"
    return "".join(b for i, b in enumerate(bits) if i not in drop)


# ---------------------------------------------------------------------------------------------- ʿilal
ILLA_NAMES = {"hadhf": "حذف", "qatf": "قطف", "qat": "قطع", "qasr": "قصر", "hadhadh": "حذذ", "salm": "صلم", "kashf": "كشف",
              "waqf": "وقف", "batr": "بتر", "tasheeth": "تشعيث", "tarfil": "ترفيل", "tadhyil": "تذييل", "tasbigh": "تسبيغ"}


def _ends(foot: Foot, kind: str) -> bool:
    return foot.units[-1] == kind


def apply_illa(foot: Foot, illa: str, pattern: str | None = None) -> str:
    """Pattern of the last foot of a hemistich after an ʿilla. `pattern` is the foot's pattern after any zihāf applied
    first (the ʿilla acts on the end of the foot, which a zihāf never touches except through the cases checked here)."""
    p = foot.pattern if pattern is None else pattern
    if illa == "hadhf":            # drop the final sabab khafīf
        if not _ends(foot, "SK"):
            raise InvalidOperation(f"ḥadhf needs a final sabab khafīf ({foot.key})")
        return p[:-2]
    if illa == "qatf":             # ḥadhf + ʿaṣb (مفاعلتن -> مفاعلْ)
        if foot.key != "mufaalatun":
            raise InvalidOperation("qaṭf applies to مفاعلتن")
        return apply_zihaf(foot, "asb")[:-2]
    if illa == "qat":              # final watad majmūʿ 110 -> 10
        if not _ends(foot, "WM") or not p.endswith("110"):
            raise InvalidOperation(f"qaṭʿ needs a final watad majmūʿ ({foot.key})")
        return p[:-3] + "10"
    if illa == "qasr":             # final sabab khafīf 10 -> 0
        if not _ends(foot, "SK") or not p.endswith("10"):
            raise InvalidOperation(f"qaṣr needs a final sabab khafīf ({foot.key})")
        return p[:-2] + "0"
    if illa == "hadhadh":          # drop the final watad majmūʿ
        if not _ends(foot, "WM"):
            raise InvalidOperation(f"ḥadhadh needs a final watad majmūʿ ({foot.key})")
        return p[:-3]
    if illa == "salm":             # drop the final watad mafrūq
        if not _ends(foot, "WF"):
            raise InvalidOperation(f"ṣalm needs a final watad mafrūq ({foot.key})")
        return p[:-3]
    if illa == "kashf":            # drop the final mutaḥarrik of a watad mafrūq
        if not _ends(foot, "WF"):
            raise InvalidOperation(f"kashf needs a final watad mafrūq ({foot.key})")
        return p[:-1]
    if illa == "waqf":             # silence the final mutaḥarrik of a watad mafrūq
        if not _ends(foot, "WF"):
            raise InvalidOperation(f"waqf needs a final watad mafrūq ({foot.key})")
        return p[:-1] + "0"
    if illa == "batr":             # ḥadhf then qaṭʿ (فعولن -> فعْ, فاعلاتن -> فاعلْ)
        if not _ends(foot, "SK") or foot.units[-2] != "WM":
            raise InvalidOperation(f"batr needs watad majmūʿ + sabab khafīf at the end ({foot.key})")
        q = p[:-2]
        return q[:-3] + "10"
    if illa == "tasheeth":         # فاعلاتن -> فالاتن, فاعلن -> فالن: drop one letter of the watad (pattern-wise the 1 of "11")
        if foot.key not in ("failatun", "failun"):
            raise InvalidOperation("tashʿīth applies to فاعلاتن and فاعلن")
        i = p.find("110")
        return p[:i] + p[i + 1:]
    if illa == "tarfil":           # add a sabab khafīf after a final watad majmūʿ
        if not _ends(foot, "WM"):
            raise InvalidOperation(f"tarfīl needs a final watad majmūʿ ({foot.key})")
        return p + "10"
    if illa == "tadhyil":          # add a sākin after a final watad majmūʿ
        if not _ends(foot, "WM"):
            raise InvalidOperation(f"tadhyīl needs a final watad majmūʿ ({foot.key})")
        return p + "0"
    if illa == "tasbigh":          # add a sākin to a final sabab khafīf
        if not _ends(foot, "SK"):
            raise InvalidOperation(f"tasbīgh needs a final sabab khafīf ({foot.key})")
        return p + "0"
    raise InvalidOperation(f"unknown ʿilla {illa}")


def variant_pattern(foot_key: str, zihaf: str | None = None, illa: str | None = None) -> str:
    """Pattern of a foot variant: optional zihāf first, then optional ʿilla."""
    foot = FEET[foot_key]
    p = apply_zihaf(foot, zihaf) if zihaf else foot.pattern
    if illa:
        if illa == "qatf" or illa == "tasheeth":
            if zihaf:
                raise InvalidOperation(f"{illa} with an extra zihāf is not modelled")
            return apply_illa(foot, illa)
        p = apply_illa(foot, illa, p)
    return p


def variant_name(foot_key: str, zihaf: str | None = None, illa: str | None = None) -> str:
    parts = [FEET[foot_key].name]
    if zihaf:
        parts.append(ZIHAF_NAMES[zihaf])
    if illa:
        parts.append(ILLA_NAMES[illa])
    return " + ".join(parts)
