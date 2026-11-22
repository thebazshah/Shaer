"""M5 (training data for idea 2, its gate and summary) and M6 (quality study): prompts, labels, data sets, the 2c gate,
the summaries, the blinded human sheet, versioning and the training configuration.

Run: python -m pytest prosody-verifier/tests -q
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "colab"))
sys.path.insert(0, str(HERE))
import m5_data as D  # noqa: E402

KAMIL = ["يا أَيُّها الرَجُلُ المُعَلِّمُ غَيرَهُ", "هَلّا لِنَفسِكَ كانَ ذا التَعليمُ",
         "اِبدَأ بِنَفسِكَ فَاِنهَها عَن غَيِّها", "فَإِذا اِنتَهَت عَنهُ فَأَنتَ حَكيمُ"]
FOREIGN = "قِفا نَبكِ مِن ذِكرى حَبيبٍ وَمَنزِلِ"


def prompt(n_lines=4, meter="الكامل"):
    return ("<s> [INST] <<SYS>>\nأنت شاعر عربي تكتب الشعر العمودي الكلاسيكي.\n<</SYS>>\n\n"
            f"البحر الأساسي: {meter}\nالصيغة: تام\nعدد الأشطر المطلوب: {n_lines}\nالموضوع: الحكمة والتعليم.\n\n"
            f"اكتب {n_lines} أشطارًا ملتزمة بصيغة تام من بحر {meter} دون أي شرح إضافي. [/INST]")


def row(i, lines, meter="الكامل"):
    p, comp = prompt(len(lines), meter), "\n".join(lines)
    return {"id": i, "sft_prompt": p, "sft_completion": comp, "sft_full_text": p + " " + comp + " </s>",
            "sft_num_lines": len(lines), "requested_bayts": len(lines) // 2, "base_meter": meter, "form": "تام",
            "length_bucket": "1-3", "description": "حكمة", "enhanced_description": "قصيدة في الحكمة"}


GOOD, BAD = KAMIL, KAMIL[:3] + [FOREIGN]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    root = tmp_path_factory.mktemp("m5")
    src = root / "src"
    src.mkdir()
    splits = {"train": [row(i, GOOD if i % 2 == 0 else BAD) for i in range(6)],
              "eval": [row(i, GOOD) for i in range(2)], "test": [row(i, GOOD) for i in range(3)]}
    for s, rows in splits.items():
        pq.write_table(pa.Table.from_pylist(rows), src / f"{s}.parquet")
    D.relabel(str(src), root / "labels", str(D.SHIPPED_LEXICON), workers=1)
    return root, src, splits


# ---------------------------------------------------------------------------------------------- prompts and labels
def test_rawi_line_goes_before_the_topic_and_the_training_text_follows():
    p = prompt()
    q = D.add_rawi(p, "م")
    assert "عدد الأشطر المطلوب: 4\nحرف الروي: م\nالموضوع:" in q and q.replace("حرف الروي: م\n", "") == p
    r = D.with_prompt(row(0, GOOD), q)
    assert r["sft_full_text"] == q + " " + "\n".join(GOOD) + " </s>" and r["sft_completion"] == "\n".join(GOOD)
    r2 = D.with_prompt(row(0, GOOD), q, "\n".join(BAD))
    assert r2["sft_full_text"] == q + " " + "\n".join(BAD) + " </s>" and r2["sft_completion"] == "\n".join(BAD)
    with pytest.raises(ValueError):
        D.add_rawi("no topic line here", "م")


def test_labels(corpus):
    root, _, splits = corpus
    labels = D.load_labels(root / "labels")
    assert {s: len(v) for s, v in labels.items()} == {s: len(r) for s, r in splits.items()}
    good, bad = labels["train"][0], labels["train"][1]
    assert good["rawi"] == "م" and good["strict_valid"] and good["rhyme_ok"] and good["meter_ok_no_margin"]
    assert not bad["strict_valid"] and not (bad["meter_ok_no_margin"] and bad["rhyme_ok"])
    D.relabel(str(root / "src"), root / "labels", str(D.SHIPPED_LEXICON), workers=1)      # resumes: nothing added
    assert len(D.read_jsonl(root / "labels" / "train.jsonl")) == 6


def test_build_2a_2b(corpus):
    root, src, splits = corpus
    d2a = D.build("2a", str(src), root / "labels", root / "data")
    assert D.build("2a", str(src), root / "labels", root / "data") == d2a                # same inputs: reused
    data = D.load_source(str(d2a))
    for s in D.SPLITS:
        assert len(data[s]) == len(splits[s])
        for r in data[s]:
            assert "حرف الروي: م" in r["sft_prompt"] and r["requested_rawi"] == "م"
            assert r["sft_full_text"].startswith(r["sft_prompt"] + " ") and r["sft_full_text"].endswith(" </s>")
    d2b = D.build("2b", str(src), root / "labels", root / "data")
    b = D.load_source(str(d2b))
    assert len(b["train"]) == 3 and len(b["eval"]) == 2 and len(b["test"]) == 3       # only the valid train poems
    man = json.loads((d2b / "manifest.json").read_text())
    assert man["train_kept"] == {"before": 6, "after": 3, "rule": man["train_kept"]["rule"]}
    assert d2a.name.startswith("2a-") and (d2a / "README.md").exists() and (d2a / "_COMPLETE").exists()


def test_built_folder_loads_with_datasets(corpus):
    datasets = pytest.importorskip("datasets")
    root, src, _ = corpus
    d2a = D.build("2a", str(src), root / "labels", root / "data")
    ds = datasets.load_dataset(str(d2a))
    assert set(ds.keys()) == {"train", "eval", "test"} and len(ds["eval"]) == 2


# ---------------------------------------------------------------------------------------------- the 2c gate
def _gen(i, lines, forced=(), rawi="م"):
    return {"id": f"shaer_train_{i + 1:06d}", "sample_idx": 0, "base_meter": "الكامل", "form": "تام",
            "requested_num_lines": 4, "generated_text": "\n".join(lines), "m4": {"forced": list(forced), "empty_lines": 0}}


def test_gate_reasons():
    from prosody.lexicon import Lexicon
    from prosody.verify import Verifier
    v = Verifier(Lexicon.load(str(D.SHIPPED_LEXICON)), ignore_diacritics=True, diagnose=False)
    assert D.gate(_gen(0, GOOD), "م", v, set()) == ""
    assert D.gate(_gen(0, GOOD, forced=[3]), "م", v, set()) == "forced hemistich"
    assert D.gate(_gen(0, GOOD[:2]), "م", v, set()) == "length"
    assert D.gate(_gen(0, BAD), "م", v, set()) == "not strictly valid"
    assert D.gate(_gen(0, GOOD), "ر", v, set()) == "another rawī"
    assert D.gate(_gen(0, GOOD), "م", v, D.train_line_keys([{"sft_completion": GOOD[1]}])).startswith("hemistich copied")


def test_accept_and_build_2c(corpus):
    root, src, _ = corpus
    d2a = D.build("2a", str(src), root / "labels", root / "data")
    gens = root / "gens.jsonl"
    poem2 = ["يا أَيُّها الرَجُلُ المُعَلِّمُ غَيرَهُ", "هَلّا لِنَفسِكَ كانَ ذا التَعليمُ",
             "وَإِذا مَشَيتَ فَكُن لِقَومِكَ قُدوَةً", "فَإِذا اِنتَهَت عَنهُ فَأَنتَ حَكيمُ"]
    rows = [_gen(0, GOOD), _gen(1, BAD), _gen(2, poem2)]
    gens.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out = root / "accepted.jsonl"
    s = D.accept(gens, str(src), d2a, str(D.SHIPPED_LEXICON), out)
    # every train poem's lines count as the corpus: GOOD and poem2 reuse its lines, so both are refused as copies
    assert s["poems"] == 3 and s["accepted"] == 0
    assert s["reasons"]["hemistich copied from the training poems"] == 2 and s["reasons"]["not strictly valid"] == 1
    out.write_text(json.dumps({"index": 2, "generated_text": "\n".join(poem2), "requested_rawi": "م"}, ensure_ascii=False) + "\n")
    d2c = D.build("2c_r1", str(src), root / "labels", root / "data", base_dir=d2a, accepted=out, mix=2.0)
    c = D.load_source(str(d2c))
    new = [r for r in c["train"] if r.get("m5_source") == "self_generated"]
    assert len(new) == 1 and len(c["train"]) == 3 and new[0]["sft_completion"] == "\n".join(poem2)
    assert new[0]["sft_full_text"] == new[0]["sft_prompt"] + " " + "\n".join(poem2) + " </s>"
    assert "حرف الروي: م" in new[0]["sft_prompt"] and len(c["eval"]) == 2


# ---------------------------------------------------------------------------------------------- the M5 summary
def _runs_fixture(tmp_path, systems: dict):
    import m3_rescore as m3
    base = {"base_meter": "الكامل", "form": "تام", "requested_bayts": 2, "requested_num_lines": 4,
            "reference_completion": "\n".join(GOOD), "parsed_num_lines": 4, "meter": 0.9, "count_adherence": 1.0,
            "meter_target_used": "الكامل", "meter_predicted": "الكامل"}

    def details(text):
        ls = text.split("\n")
        return json.dumps({"per_bayt_details": [{"bayt_index": k, "bayt_text": f"{ls[2 * k]} // {ls[2 * k + 1]}",
                                                 "skipped": False, "pred": "الكامل"} for k in range(2)]}, ensure_ascii=False)
    for name, poems in systems.items():
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        rows = [dict(base, id=f"shaer_test_{i + 1:06d}", sample_idx=0, generated_text="\n".join(p), gen_tokens=50,
                     meter_details_json=details("\n".join(p)), **({"requested_rawi": "م"} if name.startswith("m5_") else {}))
                for i, p in enumerate(poems)]
        for f in ("generations.jsonl", "scored.jsonl"):
            (d / f).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        m3.score(d / "scored.jsonl", name, tmp_path / "m3", m3.SHIPPED_LEXICON, list(m3.MODES), workers=1, limit=0)
    return tmp_path


def test_m5_summary(tmp_path):
    import m5_report as R
    d = _runs_fixture(tmp_path, {"e3_retrained": [BAD, GOOD, BAD], "m5_2a": [GOOD, GOOD, BAD]})
    labels = tmp_path / "test_labels.jsonl"
    labels.write_text("".join(json.dumps({"index": i, "rawi": "م"}) + "\n" for i in range(3)))
    res = R.summarize(d, labels, reps=50)
    assert res["baseline"] == "e3_retrained" and res["prompts"] == 3
    a, b = res["systems"]["m5_2a"], res["systems"]["e3_retrained"]
    assert round(a["letters"]["strict_valid"][0], 1) == 66.7 and round(b["letters"]["strict_valid"][0], 1) == 33.3
    assert a["rawi_as_requested"][0] == 100.0 and a["asked_for_rawi"] == 3 and b["asked_for_rawi"] == 0
    x = res["vs_baseline"]["m5_2a"]
    assert round(x["strict_valid"][0], 1) == 33.3 and x["mcnemar_strict_valid"]["a_only"] == 1
    assert res["checks"]["m5_2a"]["m5_prompts_without_rawi"] == 0
    assert "## Paired differences" in R.render(res)


# ---------------------------------------------------------------------------------------------- M6
def test_m6_collect_human_sheet_and_scores(tmp_path):
    import m6_quality as Q
    m5 = _runs_fixture(tmp_path / "m5", {"e3_retrained": [BAD, GOOD, BAD], "m5_2a": [GOOD, GOOD, BAD]})
    m6 = tmp_path / "m6"
    found = Q.collect(m6, None, m5)
    assert set(found) == {"e3_retrained", "m5_2a", "human"}
    sheet, key_path = Q.human_sheet(m6, ["human", "e3_retrained", "m5_2a"], prompts=2, seed=1)
    text = sheet.read_text(encoding="utf-8-sig")
    assert not any(n in text for n in ("e3_retrained", "m5_2a", "human"))                # blinded
    key = json.loads(key_path.read_text())
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        items = list(csv.DictReader(f))
    assert len(items) == 6 and {key["items"][r["item"]]["system"] for r in items} == {"human", "e3_retrained", "m5_2a"}
    filled = []
    for k, rater in enumerate(("a", "b")):
        p = m6 / f"rater_{rater}.csv"
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(items[0]))
            w.writeheader()
            for r in items:
                score = {"human": 5, "m5_2a": 4, "e3_retrained": 2}[key["items"][r["item"]]["system"]] - k
                w.writerow({**r, "overall": score})
        filled.append(p)
    hs = Q.human_score(filled, key_path, m6 / "human" / "human_scores.json")
    assert hs["systems"]["human"]["overall"]["mean"] == 4.5 and hs["systems"]["e3_retrained"]["overall"]["mean"] == 1.5
    # a judge run for every system (the per-poem file run_llm_judge.py writes)
    for n, base_score in (("human", 5), ("e3_retrained", 3), ("m5_2a", 4)):
        jd = m6 / "judge" / n
        jd.mkdir(parents=True)
        rows = [{"judge_source_id": f"shaer_test_{i + 1:06d}", **{m: base_score for m in Q.JUDGE_METRICS}} for i in range(3)]
        (jd / "judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = Q.summarize(m6, "e3_retrained", reps=50)
    assert res["judged_prompts"] == 3 and res["systems"]["m5_2a"]["judge"]["meaning"][0] == 4.0
    assert res["vs_baseline"]["m5_2a"]["judge"]["fluency"][0] == 1.0
    assert round(res["systems"]["m5_2a"]["strictly_valid_pct"], 1) == 66.7 and res["systems"]["human"]["strictly_valid_pct"] == 100.0
    assert res["systems"]["human"]["human"]["overall"]["mean"] == 4.5
    assert "## Blinded human ratings" in Q.render(res)


# ---------------------------------------------------------------------------------------------- notebook plumbing
def test_version_check_and_training_config(tmp_path, capsys):
    import pv_colab as pv
    import shaer_colab as sc
    pv.check_version(pv.__version__)
    assert "works with" in capsys.readouterr().out
    pv.check_version("0.0.1")
    assert "TOO OLD" in capsys.readouterr().out
    cfg = sc.make_config(base_model=sc.YEHIA, drive_base=str(tmp_path / "MyDrive"))
    p = pv.prosody_sc_config(cfg, "prosody_m5_2a")
    t = pv.TrainConfig(**{f.name: getattr(p, f.name) for f in __import__("dataclasses").fields(p)},
                       extra_env={"SFT_DATASET_ID": "/data/2a-x"})
    env = t.env(SFT_RESUME_MODE="auto")
    assert env["SFT_DATASET_ID"] == "/data/2a-x" and env["SFT_RESUME_MODE"] == "auto" and env["SFT_PUSH_TO_HUB"] == "0"
    assert t.run_dir == tmp_path / "MyDrive" / "shaer-repro-yehia-prosody" / "sft_outputs" / "train" / "prosody_m5_2a"


def test_m4_selection_file_names():
    from m4_report import slug
    assert slug("I1 best-of-4 [i1_n16]") == "i1_n16__i1_best_of_4"
    assert slug("I2 k=1 R=8 [i2_k1_r8]") == "i2_k1_r8__i2_k_1_r_8"
