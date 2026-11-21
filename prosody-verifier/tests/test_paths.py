"""This notebook never writes into the baseline's Drive folder (its E3 adapter and its generations).

Run: python -m pytest prosody-verifier/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "colab"))
sys.path.insert(0, str(HERE))
import pv_colab as pv  # noqa: E402
import shaer_colab as sc  # noqa: E402


@pytest.fixture()
def baseline(tmp_path):
    cfg = sc.make_config(base_model=sc.YEHIA, drive_base=str(tmp_path / "MyDrive"))
    cfg.repo_dir = str(tmp_path / "Shaer")
    (tmp_path / "Shaer" / "sft").mkdir(parents=True)
    # the baseline's fine-tuned adapter and E3 generations
    (cfg.run_dir / "checkpoint-4200").mkdir(parents=True)
    (cfg.run_dir / "run_summary.json").write_text("{}")
    (cfg.runs / "e3").mkdir(parents=True)
    (cfg.runs / "e3" / "generations.jsonl").write_text("baseline\n")
    return cfg


def test_baseline_layout_is_what_the_notebook_assumes(baseline, tmp_path):
    root = tmp_path / "MyDrive" / "shaer-repro-yehia"
    assert baseline.base_model == "Navid-AI/Yehia-7B-preview" and baseline.run_name == "paper_repro_a100"
    assert baseline.run_dir == root / "sft_outputs" / "train" / "paper_repro_a100"
    assert baseline.runs == root / "runs" and baseline.local_run_dir.name == "paper_repro_a100"


def test_every_output_of_this_notebook_is_in_the_sibling(baseline, tmp_path):
    root = tmp_path / "MyDrive" / "shaer-repro-yehia"
    cfg = pv.prosody_sc_config(baseline, "prosody_selftrain_r1")
    sib = tmp_path / "MyDrive" / "shaer-repro-yehia-prosody"
    assert Path(cfg.drive_root) == sib and sib.parent == root.parent              # adjacent to the baseline's folder
    assert cfg.run_dir == sib / "sft_outputs" / "train" / "prosody_selftrain_r1"
    for p in (cfg.runs, cfg.logs, cfg.sft_outputs, cfg.run_dir, cfg.runs / "e3"):
        assert not pv._inside(p, root), p
    assert cfg.local_run_dir != baseline.local_run_dir
    assert cfg.base_model == baseline.base_model and cfg.hf_token == baseline.hf_token
    assert pv.baseline_root(cfg) == root and pv.baseline_root(baseline) == root
    again = pv.prosody_sc_config(cfg, "prosody_other")                                # from the sibling config too
    assert Path(again.drive_root) == sib and again.run_name == "prosody_other"


@pytest.mark.parametrize("name", ["paper_repro_a100", "main", "prosody", "prosody_/x", "", "../prosody_x"])
def test_run_names_that_could_be_a_baseline_run_are_refused(baseline, name):
    with pytest.raises(ValueError):
        pv.prosody_sc_config(baseline, name)


def test_a_name_already_used_in_the_baseline_is_refused(baseline):
    (baseline.run_dir.parent / "prosody_taken").mkdir()
    with pytest.raises(ValueError):
        pv.prosody_sc_config(baseline, "prosody_taken")
    baseline.run_name = "prosody_same"                       # the baseline's own RUN_NAME, even before its folder exists
    with pytest.raises(ValueError):
        pv.prosody_sc_config(baseline, "prosody_same")


def test_make_config_moves_the_repository_links_off_the_baseline(baseline, tmp_path):
    repo = Path(baseline.repo_dir)
    sc.sync_repo(baseline)                                   # what the notebook's old section 2 did
    assert (repo / "sft" / "outputs").resolve() == baseline.sft_outputs.resolve()
    pvc = pv.make_config(baseline, drive_base=str(tmp_path / "MyDrive"))
    sib = tmp_path / "MyDrive" / "shaer-repro-yehia-prosody"
    assert Path(pvc.sc_cfg.drive_root) == sib
    assert (repo / "sft" / "outputs").resolve() == (sib / "sft_outputs").resolve()
    assert (repo / "runs").resolve() == (sib / "runs").resolve()
    # the baseline's adapter and generations are untouched
    assert (baseline.run_dir / "checkpoint-4200").is_dir() and (baseline.run_dir / "run_summary.json").exists()
    assert (baseline.runs / "e3" / "generations.jsonl").read_text() == "baseline\n"
    cfg = pv.training_config(pvc, "prosody_selftrain_r1")
    assert cfg.run_dir.parent == sib / "sft_outputs" / "train"
    assert pvc.root == tmp_path / "MyDrive" / "shaer-prosody"
