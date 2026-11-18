"""Tests for the M0 helpers. Run: python -m pytest prosody-verifier/tests -q"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import m0_audit as A  # noqa: E402
import pv_text as T  # noqa: E402


def test_line_split_matches_the_repository_scorer():
    sys.path.insert(0, str(HERE.parent / "grpo"))
    try:
        from rewards.common import split_poem_lines  # type: ignore
    except Exception as exc:  # the scorer's module needs the full stack (datasets, httpx, ...)
        pytest.skip(f"repository scorer not importable here: {exc}")
    samples = ["a\n\n\nb\r\nc\t d", "  سطر أول  \n\nسطر ثان\n", "", "x  y\n\n\n\nz"]
    for s in samples:
        assert T.split_poem_lines(s) == split_poem_lines(s)


def test_rhyme_rawi_strips_long_vowel_and_pronoun_haa():
    # دِيارُ / الأَمصارُ: rawi ر, vowel u
    assert T.rhyme_signatures("مِن دُونِها لا تَحتويهِ دِيارُ") == {"last_letter": "ر", "rawi": "ر", "rawi_vowel": "u"}
    # ...سَمَا : trailing alef is the wasl/ishba' letter -> rawi م, implied vowel a
    assert T.rhyme_signatures("بلغ السَّمَا")["rawi"] == "م"
    assert T.rhyme_signatures("بلغ السَّمَا")["rawi_vowel"] == "a"
    # مَكَانَهْ / البَرَاكَهْ : the haa is wasl, the rawi differs (ن vs ك)
    assert T.rhyme_signatures("مَكَانَهْ")["rawi"] == "ن"
    assert T.rhyme_signatures("البَرَاكَهْ")["rawi"] == "ك"
    # دارُها : strip alef then haa -> ر
    assert T.rhyme_signatures("في دارُها")["rawi"] == "ر"
    # plural waw + silent alif: رَحَموا rhymes on م
    assert T.rhyme_signatures("رَحَموا")["rawi"] == "م"
    assert T.rhyme_signatures("رَحَموا")["rawi_vowel"] == "u"
    # a voweled final waw is a consonant, not a long vowel
    assert T.rhyme_signatures("دَلْوُ")["rawi"] == "و"


def test_fold_letter_variants():
    assert T.fold_letter("أ") == "ا" and T.fold_letter("ى") == "ا"
    assert T.fold_letter("ؤ") == "ء" and T.fold_letter("ة") == "ه"


def test_diacritic_coverage_ignores_alef():
    assert T.diacritic_coverage("كَتَبَ") == (3, 3)
    assert T.diacritic_coverage("كتاب") == (0, 3)          # ا is excluded


def _row(text: str, lines: int, preds: list[str], target: str = "الكامل") -> dict:
    details = [{"bayt_index": i, "pred": p, "skipped": False} for i, p in enumerate(preds)]
    return {"id": "x", "base_meter": target, "form": "تام", "requested_bayts": lines // 2, "requested_num_lines": lines,
            "generated_text": text, "meter": 0.5, "meter_target_used": target,
            "meter_details_json": json.dumps({"per_bayt_details": details}, ensure_ascii=False)}


def test_audit_row_count_meter_rhyme():
    text = "أ ب\nديارُ\nج د\nالأَمصارُ\nه و\nمَكَانَهْ"
    a = A.audit_row(_row(text, 6, ["الكامل", "الكامل", "الطويل"]))
    assert a["count_status"] == "exact" and a["lines"] == 6
    assert a["meter_bayts_on_target"] == 2 and not a["meter_all_on_target"] and a["meter_distinct_preds"] == 2
    assert a["meter_flags"] == [[0, True], [1, True], [2, False]]
    assert a["rhyme_rawi_majority"] == "ر" and a["rhyme_rawi_all_same"] is False
    assert a["strict_form_heuristic"] is False

    short = A.audit_row(_row("a\nb\nc", 6, ["الكامل"]))
    assert short["count_status"] == "short" and short["odd_tail"]


def test_summary_position_rates_use_scored_bayts_only():
    rows = [_row("a\nb\nc\nd", 4, ["الكامل", "الطويل"]), _row("a\nb\nc\nd", 4, ["الكامل", "الكامل"])]
    s = A.summarise([A.audit_row(r) for r in rows])
    assert s["meter"]["off_target_bayt_pct_by_position"] == {"bayt 1": 0.0, "bayts 2-3": 50.0}
    assert s["meter"]["poems_all_bayts_on_target_pct"] == 50.0


def test_first_sample_only():
    rows = [{"id": "a", "sample_idx": 0}, {"id": "a", "sample_idx": 1}, {"id": "b", "sample_idx": 0}]
    kept, k = A.first_sample_only(rows)
    assert k == 2 and [r["id"] for r in kept] == ["a", "b"]
