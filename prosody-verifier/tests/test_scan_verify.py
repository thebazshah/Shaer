"""Prosodic writing, scansion, rhyme and poem verification on hand-checked cases.

Run: python -m pytest prosody-verifier/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prosody import arudi  # noqa: E402
from prosody.qafiya import analyse  # noqa: E402
from prosody.scan import nearest, scan  # noqa: E402
from prosody.verify import Verifier  # noqa: E402


def readings(word):
    return {(r.bits, r.final, r.wasl) for r in arudi.word_readings(word)}


# ---------------------------------------------------------------------------------------------- prosodic writing
@pytest.mark.parametrize("word,expected", [
    ("قِفَا", ("11", "m", False)),             # qi-fā
    ("نَبْكِ", ("10", "1", False)),            # nab-ki
    ("حَبِيبٍ", ("110", "n", False)),          # ḥa-bī-bin
    ("اللِّوَى", ("011", "m", True)),          # (a)l-li-wā: sun letter doubled, waṣl
    ("الْقَمَرِ", ("0110", "1", True)),        # (a)l-qa-ma-ri: moon letter, lām sākin
    ("كِتَابًا", ("110", "n", False)),          # ki-tā-ban: the alif after tanwīn al-fatḥ is silent
    ("كَتَبُوا", ("111", "m", False)),          # ka-ta-bū: the alif after the plural wāw is silent
    ("هوىً", ("1", "n", False)),               # ha-wan: tanwīn written on ى
    ("لَهُ", ("11", "m", False)),              # la-hū: the pronoun hāʾ after a vowel is lengthened
    ("مِنْهُ", ("10", "1", False)),            # min-hu: short after a sākin
    ("هذا", ("101", "m", False)),              # hā-dhā (hidden long vowel)
    ("لكن", ("101", "0", False)),              # lā-kin
    ("الله", ("010", "1", True)),              # (a)l-lā-hu
    ("والله", ("1010", "1", False)),           # wal-lā-hu
    ("الذي", ("011", "m", True)),              # (a)l-la-dhī
    ("عمرو", ("10", "1", False)),              # ʿam-ru: the wāw is silent
    ("أَلْقَتْ", ("101", "0", False)),          # ʾal-qat: a hamza-alif is not the article
    ("فامدد", ("101", "0", False)),            # fam-dud: silent waṣl after the proclitic
])
def test_word_reading(word, expected):
    assert expected in readings(word)


def test_boundary_rules():
    # helping vowel: a sākin before a waṣl gets a vowel; a long vowel before it is shortened
    assert arudi.resolve_final("0", True) == [("1", 0.0)]
    assert arudi.resolve_final("m", True) == [("", 0.0)]
    # end of the hemistich: the final vowel and the tanwīn are read as a long vowel
    assert arudi.resolve_final("1", None) == [("10", 0.0)]
    assert arudi.resolve_final("n", None) == [("10", 0.0)]
    assert arudi.resolve_final("p", False) == []            # a paused tāʾ marbūṭa only at the end


# ---------------------------------------------------------------------------------------------- scansion
MNEMONICS = {  # al-Ṣafī al-Ḥillī's keys to the meters (the sadr of each)
    "tawil": "طويلٌ له دون البحور فضائلُ",
    "madid": "لمديد الشعر عندي صفاتُ",
    "basit": "إن البسيط لديه يُبسط الأملُ",
    "wafir": "بحور الشعر وافرها جميلُ",
    "kamil": "كمل الجمال من البحور الكاملُ",
    "hazaj": "على الأهزاج تسهيلُ",
    "rajaz": "في أبحر الأرجاز بحرٌ يسهلُ",
    "ramal": "رمل الأبحر ترويه الثقاتُ",
    "sari": "بحرٌ سريعٌ ما له ساحلُ",
    "munsarih": "منسرحٌ فيه يُضرب المثلُ",
    "khafif": "يا خفيفاً خفّت به الحركاتُ",
    "mudari": "تُعَدّ المضارعاتُ",
    "muqtadab": "اقتضب كما سألوا",
    "mujtath": "إن جُثّت الحركاتُ",
    "mutaqarib": "عن المتقارب قال الخليلُ",
    "mutadarik": "حركاتُ المُحدَث تنتقلُ",
}


@pytest.mark.parametrize("meter,line", list(MNEMONICS.items()))
def test_mnemonic_verse_scans_in_its_meter(meter, line):
    assert meter in {m.meter for m in scan(line)}


@pytest.mark.parametrize("line,meter", [
    ("فعولن مفاعيلن فعولن مفاعلن", "tawil"),
    ("مستفعلن فاعلن مستفعلن فعلن", "basit"),
    ("متفاعلن متفاعلن متفاعلن", "kamil"),
    ("فاعلاتن فاعلاتن فاعلن", "ramal"),
    ("مفاعلتن مفاعلتن فعولن", "wafir"),
    ("فاعلاتن مستفع لن فاعلاتن", "khafif"),
])
def test_foot_names_scan_as_their_meter(line, meter):
    assert meter in {m.meter for m in scan(line)}


def test_famous_lines():
    assert scan("قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ", meters=("tawil",))
    assert scan("على قدر أهل العزم تأتي العزائم", meters=("tawil",))
    assert scan("أَخِي جَاوَزَ الظَّالِمُونَ الْمَدَى", meters=("mutaqarib",))


def test_fully_vowelled_text_is_binding():
    # a written sukūn where the meter needs a vowel: the line must fail
    assert not scan("قِفَا نَبْكِ مِنْ ذِكْرَى حَبْيبٍ وَمَنْزِلِ", meters=("tawil",))


def test_nearest_points_at_the_broken_foot():
    n = nearest("قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلٍ مِنْ", "tawil", "tam", "sadr")
    assert n is not None and n["edits"] >= 1


# ---------------------------------------------------------------------------------------------- rhyme
@pytest.mark.parametrize("ends,rawi,issue", [
    (["مِن دُونِها لا تَحتويهِ دِيارُ", "وَبِتُربِها تَتبرك الأَبرارُ", "وَالقُطب مَركَزَها بِها المُختارُ"], "ر", None),
    (["كريمِ", "سليمِ", "غنيمِ", "شامِتِ", "عظيمِ"], "م", "rawi"),
    (["منزلا", "معقلا", "مرسلا"], "ل", None),
    (["الهوى", "النوى", "المنى", "الدجى"], "ا", None),          # al-maqṣūra: the alif is the rawī
    (["دارِها", "جارِها", "نارِها"], "ر", None),               # hāʾ waṣl + khurūj
    (["مِنْهُ", "عَنْهُ", "فيهِ"], "ه", "iqwa"),               # hāʾ after a sākin is the rawī; ḍamma vs kasra = iqwāʾ
    (["دارُ", "نارُ", "نورُ"], "ر", "sinad_ridf"),             # ridf alif alternating with wāw
    (["الكواكِبُ", "المراكِبُ", "الحبائِبُ", "القلوبُ"], "ب", "sinad_tasis"),
    # M3 audit of human poems: ridf and taʾsīs read from the written word
    (["المآقي", "باقِ", "احتراقِ"], "ق", None),                # the ridf alif written inside آ
    (["الحذرِ", "الغِيَرِ", "النظرِ", "القمرِ"], "ر", None),    # a voweled yāʾ is no ridf
    (["يَتَرَيَّعُ", "يجمَعُ", "المطمَعُ"], "ع", None),          # nor a geminated one
    (["الأَوَلِ", "الأملِ", "العملِ"], "ل", None),
    (["ولي", "فقلي"], "ل", None),                               # the wa- of ولي is never sākin
    (["مداويا", "راجيا", "واقيا"], "ي", None),                  # alif + wāw + rawī: the wāw is the dakhīl of a taʾsīs
    (["سَنِيَّه", "بَهِيَّه", "عَلِيَّه"], "ي", None),          # hāʾ after a geminate yāʾ is the waṣl
    (["الزاهي", "ناهي", "باللهِ"], "ه", None),                 # the unwritten alif of الله is a ridf
    (["كَذَلِكا", "مُشارِكا", "المَسالِكا"], "ك", None),        # ... and that of ذلك a taʾsīs alif
    (["عظيم", "العميم", "نديم"], "م", None),
    (["قلب", "حبيب", "قريب", "لهيب"], "ب", "sinad_ridf"),      # an open yāʾ votes for a ridf: قلب is flagged
    (["باب", "حبيب", "قريب", "لهيب"], "ب", "sinad_ridf"),      # alif against yāʾ
    # M3 controls: a long vowel or a hāʾ after a voweled letter is the rawī only when most bayts support it
    (["الهوى", "النوى", "الجوى", "المنى"], "ا", None),          # maqṣūra although three share the wāw
    (["الدجى", "الثرى", "المنى", "المبتدا", "فتى"], "ا", None),  # المبتدا joins a maqṣūra poem
    (["دعا", "الرضا", "العلا", "هنا"], "ا", None),             # ا after a stem of two letters can be the rawī
    (["القمرا", "نفرا", "الجنى", "هنا"], "ر", "rawi"),        # iṭlāq (rawī ر) against maqṣūra
    (["الخطابا", "الصغارا"], "ب", "rawi"),                    # iṭlāq alif on two consonants
    (["حبّا", "أمرا"], "ب", "rawi"),
    (["القمرا", "نفرا", "السفرا", "سحرا"], "ر", None),
    (["صدِّهِ", "عقدِهِ", "جفنِهِ", "سلمِهِ"], "د", "rawi"),     # hāʾ after a mutaḥarrik is the waṣl
    (["صدِّهِ", "عقدِهِ", "بعدِهِ", "وعدِهِ"], "د", None),
    (["تراهُ", "هواهُ", "رضاهُ", "سواهُ"], "ه", None),          # ... after a long vowel it is the rawī
    (["خربه", "معمره", "حنظله", "كريمه"], "ب", "rawi"),       # tāʾ marbūṭa after different letters
    (["رحمه", "نعمه", "حكمه", "ظلمه"], "م", None),
])
def test_rhyme_cases(ends, rawi, issue):
    r = analyse(ends, [None] * len(ends))
    assert r["rawi"] == rawi
    kinds = {i["kind"] for i in r["issues"]}
    assert (issue in kinds) if issue else not kinds


def test_rhyme_letters_come_from_the_written_word_not_the_reading():
    """In letters mode the scanner reads bare letters; a reading that drops a written alif or turns the written final
    yāʾ into a voweled consonant must not change the rhyme (cases from the M3 audit)."""
    from prosody import arudi, qafiya
    def bogus(word, bits_of):
        # a reading of the bare word whose trace drops / mis-reads letters as bits_of says
        r = min(arudi.word_readings(arudi.strip_marks(word), use_lexicon=False), key=lambda r: r.cost)
        return arudi.WordReading(r.bits, r.final, r.wasl, r.cost, r.note, tuple(bits_of))
    kalal = bogus("كَلالُها", [("ك", "", "1"), ("ل", "", "1"), ("ل", "", "0"), ("ه", "", "1"), ("ا", "", "0")])
    r = analyse(["كَلالُها", "ظِلالُها", "غَزالُها"], [kalal, None, None])
    assert r["rawi"] == "ل" and not r["issues"]
    taq = bogus("وتقصيري", [("و", "", "1"), ("ت", "", "1"), ("ق", "", "1"), ("ص", "", "0"), ("ي", "", "1"),
                            ("ر", "", "1"), ("ي", "", "1")])
    taq = arudi.WordReading(taq.bits, "1", taq.wasl, taq.cost, taq.note, taq.trace)
    r = analyse(["وتقصيري", "مأمور", "مهجور", "مستور"], [taq, None, None, None])
    assert r["rawi"] == "ر" and not r["issues"]


def test_refrain_is_detected_and_rhyme_read_before_it():
    ends = ["تعال إلى الدارِ يا صاحبي", "وهات الكؤوس يا صاحبي", "وغن لنا الليل يا صاحبي"]
    r = analyse(ends, [None] * 3)
    assert r["refrain"] == ["يا", "صاحبي"]
    assert "rawi" in {i["kind"] for i in r["issues"]}        # الدار / الكؤوس / الليل do not rhyme
    r2 = analyse(ends[:2] + ["وغن لنا الليل يا صاحبتي"], [None] * 3, refrain=["يا", "صاحبي"])
    assert "radif_missing" in {i["kind"] for i in r2["issues"]}


# ---------------------------------------------------------------------------------------------- poem
MUALLAQA = """قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ
بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ
فَتُوضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا
لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"""


def test_poem_passes_and_reports():
    r = Verifier().poem(MUALLAQA, "tawil", "tam", requested_lines=4)
    assert r.ok and r.meter_ok and r.rhyme_ok and r.count_ok
    assert r.rhyme["rawi"] == "ل"
    assert r.bayts[0].arud == "A_qabd"


def test_poem_failures_are_located():
    v = Verifier()
    broken = MUALLAQA.replace("لِمَا نَسَجَتْهَا", "لِمَا")                  # a hemistich missing two feet' worth
    r = v.poem(broken, "tawil", "tam", requested_lines=4)
    assert not r.meter_ok and not r.bayts[1].ok and r.bayts[0].ok
    r2 = v.poem(MUALLAQA, "kamil", "tam")                                     # wrong meter requested
    assert not r2.meter_ok
    r3 = v.poem(MUALLAQA, "tawil", "tam", requested_lines=6)                  # wrong length
    assert not r3.count_ok and not r3.ok


# ---------------------------------------------------------------------------------------------- later fixes
def test_tasheeth_is_a_free_variant_not_a_binding_darb():
    from prosody.meters import METERS
    f = METERS["khafif"].forms["tam"]
    assert "D_tasheeth" not in {d.key for d in f.darbs}
    pats = {v["pattern"] for v in f.end("D_sahih").variants()}
    assert "101010" in pats and "1011010" in pats and "111010" in pats      # مفعولن، فاعلاتن، فعلاتن in one ḍarb


@pytest.mark.parametrize("word,expected", [
    ("اِذا", ("11", "m", False)),          # اِ = an unwritten hamza (إذا) in this corpus
    ("اِعمَلوا", ("011", "m", True)),      # ... or hamzat al-waṣl
    ("الاِخيار", ("01010", "0", True)),    # an alif after the article is a hamza: al-ʾakhyār
    ("اتِقادُ", ("0110", "1", True)),      # ittiqād: unwritten shadda right after the waṣl
    ("كُلاًّ", ("10", "n", False)),         # marks typed on the final alif belong to the lām
    ("كانا", ("101", "m", False)),          # not ك + أنا
])
def test_corpus_spelling_conventions(word, expected):
    assert expected in readings(word)


def test_every_word_gets_a_reading():
    for w in ["قْقْقْ", "ءءء", "اْ", "ىى"]:
        assert arudi.word_readings(w) or not arudi._units(w)


def test_ignore_diacritics_judges_letters_only():
    wrong_vowels = MUALLAQA.replace("حَبِيبٍ", "حُبْيُبٍ")
    assert not Verifier().poem(wrong_vowels, "tawil", "tam").meter_ok
    assert Verifier(ignore_diacritics=True).poem(wrong_vowels, "tawil", "tam").meter_ok


def test_shipped_lexicon_loads_and_reweights():
    from prosody.lexicon import Lexicon
    lex = Lexicon.load(Path(__file__).resolve().parents[1] / "data" / "lexicon_train.json.gz")
    assert len(lex.counts) > 100_000 and lex.chars is not None
    rs = lex.reweight("حبيب", arudi.word_readings("حبيب"))
    best = min(rs, key=lambda r: r.cost)
    assert (best.bits, best.final) in {("110", "1"), ("110", "n"), ("110", "0")}      # ḥa-bī-b
    assert Verifier(lex).poem(MUALLAQA, "tawil", "tam").ok


def test_arud_change_and_israf_are_flagged():
    poem = """وَلَكُم شَفِيعٌ عِنْدَ كُلِّ عَدُوِّهِ
يَوْمَ الشَّفِيعِ وَفِي الحَشَا شَفَّانَهْ
لَكَ فِي المَجَرَّةِ مَنْزِلٌ وَلَكَمْ
تَسْمُو بِهِ الأَوْهَامُ مِنْ أَعْلَانِهْ"""
    r = Verifier().poem(poem, "kamil", "tam")
    assert r.bayts[0].ok
    assert not r.bayts[1].ok                       # outside the opening bayt a ḥadhdhāʾ ʿarūḍ cannot pair with a maqṭūʿ ḍarb
    kinds = {i["kind"] for i in r.rhyme["issues"]}
    assert "israf" in kinds                        # أَعْلَانِهْ (kasra) against شَفَّانَهْ (fatḥa)


def test_margin_rival_must_fit_the_same_position():
    """A sadr outside the opening bayt is not rejected because a pause reading of it would fit a ḍarb better
    (found in the test split: shaer_test_002014, bayt 9)."""
    from prosody.lexicon import Lexicon
    lex = Lexicon.load(Path(__file__).resolve().parents[1] / "data" / "lexicon_train.json.gz")
    v = Verifier(lex, ignore_diacritics=True)
    sadr, ajuz = "رضي الحَبيب عَن الحَبيب فسله", "سل الضَمير تعلق الإِلهاما"
    bv, sc, _ = v.bayt(sadr, ajuz, "kamil", "tam", first=False)
    assert bv.ok, bv.reason
    assert all(m.position == "sadr" for m in sc)
    assert bv.sadr.best_any is None or bv.sadr.best_any.position == "sadr"


def test_compare_bilstm_reason_kinds():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compare_bilstm import reason_kinds
    assert reason_kinds("sadr: not in kamil (tam); scans in no meter; ʿajuz: reads more naturally as rajaz (cost 0.0 vs 3.0)") == [
        "sadr: no reading in any meter", "ajuz: in the meter, but another meter is far more natural"]
    assert reason_kinds("ʿajuz: not in kamil (tam); scans as rajaz (tam)") == ["ajuz: scans only in another meter or form"]
    assert reason_kinds("the ʿarūḍ and ḍarb forms are not a permitted pair") == [
        "bayt: the ʿarūḍ and ḍarb forms are not a permitted pair"]


def test_tadwir_with_the_split_word_written_whole_in_the_sadr():
    """رُبَّما تَجزَعُ النُفوسُ مِنَ الأَمـ / ـرِ لَهُ فَرجَةٌ كَحَلِّ العِقالِ, with الامر written whole in the sadr (corpus layout,
    shaer_test test split); a line break misplaced between whole words is not tadwīr."""
    v = Verifier()
    bv, _, _ = v.bayt("ربما تجزع النفوس من الامر", "له فرجة كحل العقال", "khafif", "tam")
    assert bv.ok and bv.tadwir
    assert bv.sadr.match.pattern.endswith("0")
    bv, _, _ = v.bayt("قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ", "وَمَنْزِلِ بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ", "tawil", "tam")
    assert not bv.ok and not bv.tadwir
