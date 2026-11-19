"""M3 strict re-scoring: margin recomputation, record fields, statistics, controls, resume and integrity checks."""
import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import m3_rescore as m3  # noqa: E402
from prosody.lexicon import Lexicon  # noqa: E402
from prosody.meters import METERS, shaer_target  # noqa: E402
from prosody.verify import Verifier  # noqa: E402

LEX = Lexicon.load(HERE / "data" / "lexicon_train.json.gz")

MUALLAQA = """قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ
بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ
فَتُوضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا
لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"""

# poems that exercise every path of Verifier.bayt: permitted pair (accepted and rejected by the margin), a pair that is not
# permitted, tadwīr, no analysis; plus corpus lines whose verdict flips between margins
CASES = [
    (MUALLAQA, "الطويل", "تام"),
    (MUALLAQA.replace("حَبِيبٍ", "حُبْيُبٍ"), "الطويل", "تام"),
    (MUALLAQA.replace("وَمَنْزِلِ", ""), "الطويل", "تام"),
    ("ربما تجزع النفوس من الامر\nله فرجة كحل العقال", "الخفيف", "تام"),
    ("رضي الحَبيب عَن الحَبيب فسله\nسل الضَمير تعلق الإِلهاما\nما كانَ أقبحَ أن نُضافَ إليكُمُ\nإذ نِلتُمُ مِنا بِنَا تَعريفا",
     "الكامل", "تام"),
    ("""وَلَكُم شَفِيعٌ عِنْدَ كُلِّ عَدُوِّهِ
يَوْمَ الشَّفِيعِ وَفِي الحَشَا شَفَّانَهْ
لَكَ فِي المَجَرَّةِ مَنْزِلٌ وَلَكَمْ
تَسْمُو بِهِ الأَوْهَامُ مِنْ أَعْلَانِهْ""", "الكامل", "تام"),
    ("هجرانيَ الجامحُ مَن لَم يَكُن\nيدري بما في القلب من شجن", "السريع", "تام"),
    ("وَجَدِّيَ رَسولُ اللَهِ أَفضَلُ مَن مَشى\nوَأَكرَمُ مَن يُرجى إِذا الدَهرُ أَعضَلا", "الطويل", "تام"),
    ("لنا مغنٍّ ان شدا\nتدفننا ثلوجهُ", "الرجز", "تام"),
]


def row(text, base, form, i=0, sample=0, req=None):
    lines = m3.split_poem_lines(text)
    target = base
    details = [{"bayt_index": k, "bayt_text": f"{lines[2 * k]} // {lines[2 * k + 1]}", "skipped": False, "pred": target}
               for k in range(len(lines) // 2)]
    return {"id": f"t{i:03d}", "sample_idx": sample, "base_meter": base, "form": form, "generated_text": text,
            "requested_num_lines": req if req is not None else len(lines), "requested_bayts": (req or len(lines)) // 2,
            "parsed_num_lines": len(lines), "meter": 0.9, "count_adherence": 1.0, "meter_target_used": target,
            "meter_predicted": target, "meter_details_json": json.dumps({"per_bayt_details": details}, ensure_ascii=False)}


def verifiers(bare):
    v = Verifier(LEX, ignore_diacritics=bare)
    return v, Verifier(LEX, ignore_diacritics=bare, diagnose=False).share_cache(v)


def test_margin_recomputation_reproduces_the_verifier_at_every_margin():
    paths = set()
    for bare in (False, True):
        vt, vf = verifiers(bare)
        for text, base, form in CASES:
            rec = m3.score_row(row(text, base, form), vt, vf)
            meter, _ = shaer_target(base, form)
            lines = m3.split_poem_lines(text)
            for fk in METERS[meter].forms:
                fr = rec["forms"][fk]
                paths.update(e[0] for e in fr["bayts"])
                for mg in (0.5, 1.0, 3.0, math.inf):
                    ref = Verifier(LEX, ignore_diacritics=bare, margin=mg, diagnose=False).poem(lines, meter, fk, len(lines))
                    assert [m3.bayt_ok(e, mg) for e in fr["bayts"]] == [b.ok for b in ref.bayts], (text[:20], fk, mg, bare)
                    assert m3.poem_meter_ok(fr, mg) == ref.meter_ok, (text[:20], fk, mg, bare)
    assert {"pair", "tadwir", "none"} <= paths          # every resolution path was exercised


def test_record_fields_and_line_counting():
    vt, vf = verifiers(True)
    text = MUALLAQA.replace("\n", "\r\n") + "\n\n\t\n"          # Shaer's parser: \r, blank and whitespace-only lines
    rec = m3.score_row(row(text, "الطويل", "تام", req=4), vt, vf)
    assert rec["lines"] == 4 and rec["requested_lines"] == 4 and rec["count_ok"] and rec["even"]
    assert rec["meter_ok"] and rec["rhyme_ok"] and rec["ok"] and rec["misaligned_bayts"] == 0
    assert rec["bilstm"] == [True, True] and rec["has_reading"] == [True, True]
    odd = m3.score_row(row(MUALLAQA + "\nوَقَفْتُ بِهَا", "الطويل", "تام", req=4), vt, vf)
    assert odd["lines"] == 5 and not odd["count_ok"] and not odd["even"] and not odd["ok"]
    bad = row(MUALLAQA, "الطويل", "تام")
    d = json.loads(bad["meter_details_json"])
    d["per_bayt_details"][1]["bayt_text"] = "something else"
    d["per_bayt_details"][0]["skipped"] = True
    bad["meter_details_json"] = json.dumps(d, ensure_ascii=False)
    rec = m3.score_row(bad, vt, vf)
    assert rec["misaligned_bayts"] == 1 and rec["bilstm"] == [None, True]


def test_poem_metrics_pool_bayts_and_bootstrap_pairs():
    pp = {"a": {"strict_valid": True, "bayt_ok": (1, 1)}, "b": {"strict_valid": False, "bayt_ok": (0, 3)}}
    ids = ["a", "b"]
    assert m3.bootstrap(pp, ids, "strict_valid", 200, 1)[0] == 50.0
    assert m3.bootstrap(pp, ids, "bayt_ok", 200, 1)[0] == 25.0          # pooled 1/4, not the mean of ratios (50)
    lo_hi = m3.bootstrap(pp, ids, "bayt_ok", 200, 1)
    assert lo_hi[1] <= lo_hi[0] <= lo_hi[2]
    assert m3.bootstrap(pp, ids, "bayt_ok", 200, 1, other=pp) == (0.0, 0.0, 0.0)     # paired: identical systems differ by 0
    assert m3.bootstrap(pp, ids, "bayt_ok", 200, 1) == m3.bootstrap(pp, ids, "bayt_ok", 200, 1)   # deterministic


def test_pass_at_k_matches_enumeration():
    for n in range(1, 7):
        for c in range(n + 1):
            for k in range(1, n + 1):
                draws = list(itertools.combinations([1] * c + [0] * (n - c), k))
                exact = sum(any(d) for d in draws) / len(draws)
                assert abs(m3.pass_at_k(n, c, k) - exact) < 1e-12


def test_controls():
    rows = [row(MUALLAQA, "الطويل", "تام", i=i) for i in range(3)] + [row(MUALLAQA, "الكامل", "تام", i=9)]
    wm = m3.control_rows(rows, "wrong_meter")
    assert len(wm) == 4 and all(r["base_meter"] != r["control_of_meter"] and r["form"] == "تام" for r in wm)
    assert all(shaer_target(r["base_meter"], "تام") for r in wm) and all(r["meter_details_json"] is None for r in wm)
    a = "\n".join(["قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ", "بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"] * 4)
    same = "\n".join(["أَلا عِمْ صَباحاً أَيُّها الطَلَلُ البالي", "وَهَل يَعِمَن مَن كانَ في العُصُرِ الخالي"] * 4)
    assert m3.control_rows([row(a, "الطويل", "تام", i=1), row(same, "الطويل", "تام", i=2)], "mixed_rhyme") == []  # both in لِ
    b = "\n".join(["لِخَولَةَ أَطلالٌ بِبُرقَةِ ثَهمَدِ", "تَلوحُ كَباقي الوَشمِ في ظاهِرِ اليَدِ"] * 4)
    mixed = m3.control_rows([row(a, "الطويل", "تام", i=1), row(b, "الطويل", "تام", i=2)], "mixed_rhyme")
    assert len(mixed) == 2
    ls = m3.split_poem_lines(mixed[0]["generated_text"])
    assert len(ls) == 8 and ls[:4] == m3.split_poem_lines(a)[:4] and ls[4:] == m3.split_poem_lines(b)[4:]
    vt, vf = verifiers(True)
    rec = m3.score_row(mixed[0], vt, vf)
    assert not rec["rhyme_ok"] and "rawi" in rec["rhyme"]["issues"]


def _rows_file(tmp_path, n=6):
    rows = [row(t, b, f, i=i) for i, (t, b, f) in enumerate(CASES[:n])]
    src = tmp_path / "scored.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return src


def _sorted(path):
    return sorted(Path(path).read_text(encoding="utf-8").splitlines())


def test_score_is_deterministic_resumable_and_pinned(tmp_path):
    src = _rows_file(tmp_path)
    lex = m3.SHIPPED_LEXICON
    m3.score(src, "s", tmp_path / "a", lex, ["letters"], workers=1, limit=0, chunk=1)
    m3.score(src, "s", tmp_path / "b", lex, ["letters"], workers=3, limit=0, chunk=2)
    fa, fb = tmp_path / "a" / "s" / "letters.jsonl", tmp_path / "b" / "s" / "letters.jsonl"
    assert _sorted(fa) == _sorted(fb) and len(_sorted(fa)) == 6             # worker count does not change results
    # interrupted run: keep 3 records and half of a fourth, then resume
    lines = fa.read_text(encoding="utf-8").splitlines()
    fa.write_text("\n".join(lines[:3]) + "\n" + lines[3][: len(lines[3]) // 2], encoding="utf-8")
    m3.score(src, "s", tmp_path / "a", lex, ["letters"], workers=2, limit=0)
    assert _sorted(fa) == _sorted(fb)
    m3.score(src, "s", tmp_path / "a", lex, ["letters"], workers=2, limit=0)  # nothing left: unchanged
    assert _sorted(fa) == _sorted(fb)
    # a different verifier version must not be mixed in
    meta = json.loads((tmp_path / "a" / "s" / "letters.meta.json").read_text())
    meta["verifier_sha"] = "0" * 16
    (tmp_path / "a" / "s" / "letters.meta.json").write_text(json.dumps(meta))
    m3.score(src, "s", tmp_path / "a", lex, ["letters"], workers=1, limit=0)
    assert list((tmp_path / "a" / "s").glob("letters.stale-*.jsonl")) and _sorted(fa) == _sorted(fb)


def test_summarize_checks_and_official_numbers(tmp_path):
    src = _rows_file(tmp_path)
    for name in ("reference", "sys"):
        m3.score(src, name, tmp_path / "m3", m3.SHIPPED_LEXICON, list(m3.MODES), workers=2, limit=0)
    res = m3.summarize(tmp_path / "m3", reps=100)
    for k, c in res["checks"].items():
        assert c["line_count_mismatches"] == 0 and c["misaligned_bilstm_bayts"] == 0, k
        assert c["default_margin_recomputation_mismatches"] == 0 and c["exact_count_vs_official_pp"] == 0, k
    s = res["systems"]["sys"]["letters"]
    assert s["official_structural"]["n"] == 6 and s["official_structural"]["meter_conformity"] == 0.9
    assert s["vs_reference"]["strict_valid"] == (0.0, 0.0, 0.0)            # same poems: paired difference 0
    g = s["margin_1.5"]
    assert g["bilstm_bayt_ok"][0] == 100.0 and 0 <= g["strict_valid"][0] <= g["meter_ok"][0]
    assert s["margin_1.0"]["bayt_ok"][0] <= g["bayt_ok"][0] <= s["margin_3.0"]["bayt_ok"][0] <= s["margin_inf"]["bayt_ok"][0]
    assert "## Integrity checks" in m3.render(res)
