"""The rule tables against the textbook equalities (each derived foot equals a named classical foot).

Run: python -m pytest prosody-verifier/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prosody.feet import FEET, InvalidOperation, apply_illa, apply_zihaf, variant_pattern  # noqa: E402
from prosody.meters import METERS, SHAER_FORM_TO_KEY, hemistich_analyses, shaer_target  # noqa: E402

# Named feet, in 1/0 notation (1 = mutaḥarrik, 0 = sākin), written out letter by letter in the comments.
NAMED = {
    "فعولن": "11010",      # فَ عُ و لُ نْ
    "فعولُ": "1101",
    "فاعلن": "10110",      # فَا عِ لُ نْ
    "فعِلن": "1110",
    "فعْلن": "1010",
    "مفاعيلن": "1101010",
    "مفاعلن": "110110",
    "مفاعيلُ": "110101",
    "مستفعلن": "1010110",
    "مفتعلن": "101110",
    "فعِلتن": "11110",
    "مفعولن": "101010",
    "فاعلاتن": "1011010",
    "فعِلاتن": "111010",
    "فاعلاتُ": "101101",
    "فعِلاتُ": "11101",
    "مفاعلتن": "1101110",
    "متفاعلن": "1110110",
    "مفعولاتُ": "1010101",
    "فعولاتُ": "110101",
}

ZIHAF_CASES = [  # (foot, zihāf, expected named foot)
    ("faulun", "qabd", "فعولُ"),
    ("failun", "khabn", "فعِلن"),
    ("mafailun", "qabd", "مفاعلن"), ("mafailun", "kaff", "مفاعيلُ"),
    ("mustafilun", "khabn", "مفاعلن"), ("mustafilun", "tayy", "مفتعلن"), ("mustafilun", "khabl", "فعِلتن"),
    ("mustafi_lun", "khabn", "مفاعلن"),
    ("failatun", "khabn", "فعِلاتن"), ("failatun", "kaff", "فاعلاتُ"), ("failatun", "shakl", "فعِلاتُ"),
    ("mufaalatun", "asb", "مفاعيلن"), ("mufaalatun", "aql", "مفاعلن"), ("mufaalatun", "naqs", "مفاعيلُ"),
    ("mutafailun", "idmar", "مستفعلن"), ("mutafailun", "waqs", "مفاعلن"), ("mutafailun", "khazl", "مفتعلن"),
    ("mafulatu", "khabn", "فعولاتُ"), ("mafulatu", "tayy", "فاعلاتُ"), ("mafulatu", "khabl", "فعِلاتُ"),
]


def test_base_feet_patterns():
    for key, named in [("faulun", "فعولن"), ("failun", "فاعلن"), ("mafailun", "مفاعيلن"), ("mustafilun", "مستفعلن"),
                       ("failatun", "فاعلاتن"), ("mufaalatun", "مفاعلتن"), ("mutafailun", "متفاعلن"),
                       ("mafulatu", "مفعولاتُ")]:
        assert FEET[key].pattern == NAMED[named], key
    # the watad-mafrūq feet have the same letters as their majmūʿ twins; only the zihāfāt they admit differ
    assert FEET["mustafi_lun"].pattern == NAMED["مستفعلن"]
    assert FEET["fai_latun"].pattern == NAMED["فاعلاتن"]


@pytest.mark.parametrize("foot,zihaf,named", ZIHAF_CASES)
def test_zihaf_gives_named_foot(foot, zihaf, named):
    assert apply_zihaf(FEET[foot], zihaf) == NAMED[named]


def test_mustafi_lun_kaff_and_shakl():
    assert apply_zihaf(FEET["mustafi_lun"], "kaff") == "101011"      # مستفعِ لُ
    assert apply_zihaf(FEET["mustafi_lun"], "shakl") == "11011"      # متفعِ لُ


@pytest.mark.parametrize("foot,zihaf", [
    ("faulun", "khabn"),          # the 2nd letter of فعولن is in the watad
    ("failatun", "tayy"),         # the 4th letter of فاعلاتن is in the watad
    ("failatun", "qabd"),         # the 5th letter of فاعلاتن is in the watad
    ("mustafi_lun", "tayy"),      # the 4th letter of مستفع لن is in the watad mafrūq: no ṭayy in الخفيف / المجتث
    ("mustafilun", "kaff"),       # مستفعلن has no 7th sākin of a sabab
    ("mutafailun", "khabn"),      # the 2nd letter of متفاعلن is mutaḥarrik
    ("mufaalatun", "qabd"),       # the 5th letter of مفاعلتن is mutaḥarrik
    ("mafailun", "asb"),          # the 5th letter of مفاعيلن is sākin
])
def test_zihaf_never_touches_a_watad_or_wrong_letter(foot, zihaf):
    with pytest.raises(InvalidOperation):
        apply_zihaf(FEET[foot], zihaf)


ILLA_CASES = [  # (foot, zihāf first, ʿilla, expected)
    ("faulun", None, "hadhf", "110"),            # فعو = فَعَلْ
    ("faulun", None, "qasr", "1100"),            # فعولْ
    ("faulun", None, "batr", "10"),              # فعْ
    ("mafailun", None, "hadhf", "11010"),        # مفاعي = فعولن
    ("failatun", None, "hadhf", "10110"),        # فاعلا = فاعلن
    ("failatun", "khabn", "hadhf", "1110"),      # فعِلا = فعِلن
    ("failatun", None, "qasr", "101100"),        # فاعلاتْ = فاعلانْ
    ("failatun", None, "tasbigh", "10110100"),   # فاعلاتانْ
    ("failatun", None, "batr", "1010"),          # فاعلْ = فعْلن
    ("failatun", None, "tasheeth", "101010"),    # فالاتن = مفعولن
    ("failun", None, "qat", "1010"),             # فاعلْ = فعْلن
    ("failun", None, "tadhyil", "101100"),       # فاعلانْ
    ("failun", None, "tarfil", "1011010"),       # فاعلاتن
    ("failun", None, "tasheeth", "1010"),        # فالن = فعْلن
    ("mustafilun", None, "qat", "101010"),       # مستفعلْ = مفعولن
    ("mustafilun", "khabn", "qat", "11010"),     # متفعلْ = فعولن (مخلع البسيط)
    ("mustafilun", None, "tadhyil", "10101100"), # مستفعلانْ
    ("mustafi_lun", "khabn", "qasr", "11010"),   # متفعْ = فعولن (مجزوء الخفيف)
    ("mutafailun", None, "qat", "111010"),       # متفاعلْ = فعِلاتن
    ("mutafailun", "idmar", "qat", "101010"),    # متْفاعلْ = مفعولن
    ("mutafailun", None, "hadhadh", "1110"),     # متفا = فعِلن
    ("mutafailun", "idmar", "hadhadh", "1010"),  # متْفا = فعْلن
    ("mutafailun", None, "tarfil", "111011010"), # متفاعلاتن
    ("mutafailun", None, "tadhyil", "11101100"), # متفاعلانْ
    ("mufaalatun", None, "qatf", "11010"),       # مفاعلْ = فعولن
    ("mafulatu", None, "kashf", "101010"),       # مفعولا = مفعولن
    ("mafulatu", None, "waqf", "1010100"),       # مفعولاتْ = مفعولانْ
    ("mafulatu", None, "salm", "1010"),          # مفعو = فعْلن
    ("mafulatu", "tayy", "kashf", "10110"),      # مفعُلا = فاعلن (عروض السريع)
    ("mafulatu", "tayy", "waqf", "101100"),      # مفعُلاتْ = فاعلانْ
    ("mafulatu", "khabl", "kashf", "1110"),      # معُلا = فعِلن
]


@pytest.mark.parametrize("foot,zihaf,illa,expected", ILLA_CASES)
def test_illa_patterns(foot, zihaf, illa, expected):
    assert variant_pattern(foot, zihaf, illa) == expected


@pytest.mark.parametrize("foot,illa", [("mustafilun", "hadhf"), ("faulun", "qat"), ("failatun", "hadhadh"),
                                       ("mutafailun", "kashf"), ("mafailun", "tarfil"), ("failun", "tasbigh")])
def test_illa_needs_the_right_ending(foot, illa):
    with pytest.raises(InvalidOperation):
        apply_illa(FEET[foot], illa)


# ---------------------------------------------------------------------------------------------- meter tables
CANONICAL = {  # the classical canonical realisation of one hemistich (sadr), foot by foot
    ("tawil", "tam"): ["فعولن", "مفاعيلن", "فعولن", "مفاعلن"],
    ("madid", "tam"): ["فاعلاتن", "فاعلن", "فاعلاتن"],
    ("basit", "tam"): ["مستفعلن", "فاعلن", "مستفعلن", "فعِلن"],
    ("basit", "mukhalla"): ["مستفعلن", "فاعلن", "فعولن"],
    ("wafir", "tam"): ["مفاعلتن", "مفاعلتن", "فعولن"],
    ("kamil", "tam"): ["متفاعلن", "متفاعلن", "متفاعلن"],
    ("hazaj", "tam"): ["مفاعيلن", "مفاعيلن"],
    ("rajaz", "tam"): ["مستفعلن", "مستفعلن", "مستفعلن"],
    ("ramal", "tam"): ["فاعلاتن", "فاعلاتن", "فاعلن"],
    ("sari", "tam"): ["مستفعلن", "مستفعلن", "فاعلن"],
    ("munsarih", "tam"): ["مستفعلن", "فاعلاتُ", "مستفعلن"],
    ("khafif", "tam"): ["فاعلاتن", "مستفعلن", "فاعلاتن"],
    ("mujtath", "tam"): ["مستفعلن", "فاعلاتن"],
    ("mutaqarib", "tam"): ["فعولن", "فعولن", "فعولن", "فعولن"],
}


@pytest.mark.parametrize("key,feet", list(CANONICAL.items()))
def test_canonical_sadr_is_generated(key, feet):
    patterns = {a["pattern"] for a in hemistich_analyses(*key, "sadr")}
    assert "".join(NAMED[f] for f in feet) in patterns


def test_every_shaer_label_maps_to_a_form():
    shaer_labels = [("الطويل", "تام"), ("الكامل", "تام"), ("البسيط", "تام"), ("الوافر", "تام"), ("الخفيف", "تام"),
                    ("السريع", "تام"), ("المتقارب", "تام"), ("الرجز", "تام"), ("الرمل", "تام"), ("المنسرح", "تام"),
                    ("المجتث", "تام"), ("الكامل", "مجزوء"), ("الرمل", "مجزوء"), ("البسيط", "مخلع"), ("الهزج", "تام"),
                    ("الرجز", "مجزوء"), ("المديد", "تام"), ("الكامل", "أحذ"), ("الخفيف", "مجزوء"), ("الوافر", "مجزوء"),
                    ("البسيط", "مجزوء")]
    assert len(shaer_labels) == 21
    for meter, form in shaer_labels:
        mk, fk = shaer_target(meter, form)
        assert fk in METERS[mk].forms
    assert set(SHAER_FORM_TO_KEY) == {"تام", "مجزوء", "مخلع", "أحذ"}


def test_every_end_form_is_used_and_every_pair_exists():
    for m in METERS.values():
        for f in m.forms.values():
            arud_keys, darb_keys = {a.key for a in f.aruds}, {d.key for d in f.darbs}
            assert set(f.pairs) <= arud_keys, (m.key, f.key)
            for a, ds in f.pairs.items():
                assert set(ds) <= darb_keys, (m.key, f.key, a)
            for e in f.aruds + f.darbs:
                assert e.variants(), (m.key, f.key, e.key)      # every table entry is a valid operation
            assert len(f.hashw_zihaf) == len(f.hashw) - 1


def test_hemistichs_end_on_a_sakin():
    """A hemistich never ends on a mutaḥarrik (the final vowel is lengthened in recitation)."""
    for m in METERS.values():
        for f in m.forms:
            for pos in ("sadr", "ajuz"):
                for a in hemistich_analyses(m.key, f, pos):
                    assert a["pattern"].endswith("0"), (m.key, f, pos, a["pattern"])


def test_no_two_sakins_in_a_row_except_at_the_end():
    """Inside a hemistich two sākin letters never meet (iltiqāʾ al-sākinayn); only the pause at the end allows it."""
    for m in METERS.values():
        for f in m.forms:
            for pos in ("sadr", "ajuz"):
                for a in hemistich_analyses(m.key, f, pos):
                    assert "00" not in a["pattern"][:-1], (m.key, f, pos, a["pattern"])
