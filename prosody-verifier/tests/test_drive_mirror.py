"""The Drive mirror of the prosody notebook (pv_colab._phase): every step works on the VM's disk and its results are
uploaded to Drive, verified, while it runs and when it ends or is stopped; a new session copies them back.

Drive is a plain folder here (sc._LocalFolderAPI rooted at "/"), the same stand-in the baseline's uploader tests use."""
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "colab"))
import shaer_colab as sc  # noqa: E402
import pv_colab as pv  # noqa: E402


@pytest.fixture
def pvc(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_MIRROR", "1")
    monkeypatch.setenv("PV_LOCAL_ROOT", str(tmp_path / "vm"))
    monkeypatch.setattr(pv, "_API_FACTORY", lambda: sc._LocalFolderAPI("/"))
    monkeypatch.setattr(sc, "flush_drive", lambda remount=True: True)
    monkeypatch.setattr(pv, "_ACTIVE", None)
    monkeypatch.setattr(pv, "_ON_DRIVE", {})
    cfg = sc.make_config(base_model=sc.YEHIA, drive_base=str(tmp_path / "MyDrive"))
    return pv.make_config(cfg, drive_base=str(tmp_path / "MyDrive"))


def new_session(pvc):
    """A new Colab VM: its disk is empty and the Python process is new; Drive keeps what was uploaded."""
    shutil.rmtree(pvc.root.parent, ignore_errors=True)
    pv._ON_DRIVE.clear()


def test_root_is_on_the_vm_and_drive_root_on_drive(pvc, tmp_path):
    assert pvc.mirrored
    assert pvc.root == tmp_path / "vm" / "shaer-prosody"
    assert pvc.drive_root == tmp_path / "MyDrive" / "shaer-prosody"


def test_a_step_writes_on_the_vm_and_its_results_reach_drive(pvc):
    @pv._phase("m3")
    def step(p):
        out = p.root / "m3" / "reference" / "letters.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"id": 1}\n{"id": 2}\n')
        (p.root / "m3" / "m3_report.md").write_text("# report\n")
        assert not (p.drive_root / "m3" / "reference" / "letters.jsonl").exists()   # nothing goes through the mount
    step(pvc)
    assert (pvc.drive_root / "m3" / "reference" / "letters.jsonl").read_text() == '{"id": 1}\n{"id": 2}\n'
    assert (pvc.drive_root / "m3" / "m3_report.md").read_text() == "# report\n"


def test_only_complete_jsonl_lines_are_uploaded(pvc):
    @pv._phase("m4")
    def step(p):
        out = p.root / "m4" / "eval" / "runs" / "i1_n4" / "generations.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"id": 1}\n{"id": 2}\n{"id": 3, "generated_te')          # a row being written
    step(pvc)
    assert (pvc.drive_root / "m4" / "eval" / "runs" / "i1_n4" / "generations.jsonl").read_text() == '{"id": 1}\n{"id": 2}\n'


def test_results_reach_drive_while_the_step_runs(pvc, monkeypatch):
    monkeypatch.setattr(pv, "SYNC_EVERY", 0.2)
    drive_file = pvc.drive_root / "m5" / "labels" / "train.jsonl"
    seen = threading.Event()

    @pv._phase("m5/labels")
    def step(p):
        out = p.root / "m5" / "labels" / "train.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"i": 1}\n')
        old = time.time() - 10
        os.utime(out, (old, old))                    # not "still being written"
        for _ in range(100):
            if drive_file.exists():
                seen.set()
                break
            time.sleep(0.05)
    step(pvc)
    assert seen.is_set(), "the background upload did not run during the step"


def test_a_new_session_resumes_from_what_is_on_drive(pvc):
    @pv._phase("m3")
    def first(p):
        f = p.root / "m3" / "e0" / "letters.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(f'{{"id": {i}}}\n' for i in range(5)))
    first(pvc)
    new_session(pvc)
    assert not (pvc.root / "m3").exists()

    @pv._phase("m3")
    def second(p):
        f = p.root / "m3" / "e0" / "letters.jsonl"
        assert f.read_text().count("\n") == 5         # restored before the step runs: it resumes after row 5
        with f.open("a") as h:
            h.write('{"id": 5}\n')
    second(pvc)
    assert (pvc.drive_root / "m3" / "e0" / "letters.jsonl").read_text().count("\n") == 6


def test_the_vm_copy_wins_within_a_session(pvc):
    (pvc.drive_root / "m6").mkdir(parents=True)
    (pvc.drive_root / "m6" / "notes.json").write_text("old")
    (pvc.root / "m6").mkdir(parents=True)
    (pvc.root / "m6" / "notes.json").write_text("new")
    pv._phase("m6")(lambda p: None)(pvc)
    assert (pvc.drive_root / "m6" / "notes.json").read_text() == "new"


def test_removed_and_renamed_files_are_mirrored(pvc):
    @pv._phase("m3")
    def write(p):
        (p.root / "m3" / "e1").mkdir(parents=True, exist_ok=True)
        (p.root / "m3" / "e1" / "letters.jsonl").write_text('{"id": 1}\n')
    write(pvc)

    @pv._phase("m3")
    def set_aside(p):                                # what m3_rescore does with records of another verifier version
        f = p.root / "m3" / "e1" / "letters.jsonl"
        f.rename(f.with_suffix(".stale-1.jsonl"))
    set_aside(pvc)
    names = sorted(x.name for x in (pvc.drive_root / "m3" / "e1").iterdir())
    assert names == ["letters.stale-1.jsonl"]


def test_a_stop_uploads_first_and_still_stops_run_all(pvc, capsys):
    @pv._phase("m4")
    def step(p):
        f = p.root / "m4" / "test" / "runs" / "i2_k1_r8" / "generations.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('{"id": 1}\n')
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        step(pvc)
    assert (pvc.drive_root / "m4" / "test" / "runs" / "i2_k1_r8" / "generations.jsonl").exists()
    out = capsys.readouterr().out
    assert "everything it wrote is on Drive" in out and "Run all stops here" in out


def test_a_step_called_by_another_shares_the_mirror(pvc):
    calls = []

    @pv._phase("m5/data")
    def inner(p):
        calls.append(pv._ACTIVE)
        (p.root / "m5" / "data").mkdir(parents=True, exist_ok=True)
        (p.root / "m5" / "data" / "2a.path").write_text("x")

    @pv._phase("m5/variants.json")
    def outer(p):
        calls.append(pv._ACTIVE)
        (p.root / "m5").mkdir(parents=True, exist_ok=True)
        (p.root / "m5" / "variants.json").write_text("{}")
        inner(p)
    outer(pvc)
    assert calls[0] is calls[1] and calls[0] is not None
    assert pv._ACTIVE is None
    assert (pvc.drive_root / "m5" / "data" / "2a.path").exists() and (pvc.drive_root / "m5" / "variants.json").exists()


def test_a_failed_upload_is_reported_not_hidden(pvc, monkeypatch, capsys):
    class Broken(sc._LocalFolderAPI):
        def upload(self, *a, **k):
            raise OSError("quota")
    monkeypatch.setattr(pv, "_API_FACTORY", lambda: Broken("/"))
    monkeypatch.setattr(pv.time, "sleep", lambda s: None)

    @pv._phase("m6")
    def step(p):
        (p.root / "m6").mkdir(parents=True, exist_ok=True)
        (p.root / "m6" / "m6_summary.json").write_text("{}")
    step(pvc)
    assert "NOT EVERYTHING IS ON DRIVE" in capsys.readouterr().out


def test_a_recorded_path_is_found_on_the_vm_or_on_drive(pvc):
    data_drive = pvc.drive_root / "m5" / "data" / "2a-abc"
    data_drive.mkdir(parents=True)
    (data_drive / "train.parquet").write_text("p")

    @pv._phase("m5/variants.json")
    def step(p):
        local = pv._need(p, data_drive)               # recorded on Drive by an older version
        assert local == p.root / "m5" / "data" / "2a-abc" and (local / "train.parquet").exists()
        assert pv._need(p, local) == local            # recorded on the VM
    step(pvc)


def test_without_the_mirror_everything_is_on_drive_directly(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_MIRROR", "0")
    cfg = sc.make_config(base_model=sc.YEHIA, drive_base=str(tmp_path / "MyDrive"))
    p = pv.make_config(cfg, drive_base=str(tmp_path / "MyDrive"))
    assert not p.mirrored and p.root == p.drive_root
    assert pv._phase("m3")(lambda q: 7)(p) == 7


def test_an_unchanged_file_is_not_uploaded_again(pvc, monkeypatch):
    uploads = []
    real = sc._LocalFolderAPI.upload

    def counting(self, parent, name, path, file_id=None):
        uploads.append(name)
        return real(self, parent, name, path, file_id)
    monkeypatch.setattr(sc._LocalFolderAPI, "upload", counting)

    @pv._phase("m3")
    def step(p):
        (p.root / "m3").mkdir(parents=True, exist_ok=True)
        f = p.root / "m3" / "a.jsonl"
        if not f.exists():
            f.write_text('{"id": 1}\n')
    step(pvc)
    step(pvc)                                        # a second step of the same session: nothing changed
    assert uploads == ["a.jsonl"]


def test_notebook_steps_end_to_end_with_real_scripts(pvc, monkeypatch):
    """A new session: the M5 test runs of an earlier one are only on Drive. pv.m6_collect and pv.m6_human_sheet (real
    scripts, as subprocesses) find them on the VM and their outputs end up on Drive."""
    import dataclasses
    from test_m5_m6 import BAD, GOOD, _runs_fixture
    prefix = Path(sys.executable).resolve().parent.parent
    monkeypatch.setattr(pvc, "sc_cfg", dataclasses.replace(pvc.sc_cfg, venv_dir=str(prefix)))
    _runs_fixture(pvc.drive_root / "m5" / "test", {"e3_retrained": [BAD, GOOD, BAD], "m5_2a": [GOOD, GOOD, BAD]})
    new_session(pvc)
    pv.m6_collect(pvc)
    assert (pvc.root / "m5" / "test" / "runs" / "m5_2a" / "scored.jsonl").exists()        # restored for the step
    sheet = pv.m6_human_sheet(pvc, ("human", "e3_retrained", "m5_2a"), prompts=2)
    assert sheet == pvc.root / "m6" / "human" / "rating_sheet.csv"
    on_drive = {p.relative_to(pvc.drive_root / "m6").as_posix() for p in (pvc.drive_root / "m6").rglob("*") if p.is_file()}
    on_vm = {p.relative_to(pvc.root / "m6").as_posix() for p in (pvc.root / "m6").rglob("*") if p.is_file()}
    assert on_vm and on_vm == on_drive
    for rel in on_vm:
        assert (pvc.root / "m6" / rel).read_bytes() == (pvc.drive_root / "m6" / rel).read_bytes(), rel


def test_a_stuck_script_is_restarted_instead_of_ending_the_cell(monkeypatch, capsys):
    monkeypatch.setattr(pv.time, "sleep", lambda s: None)
    calls = []

    def start():
        calls.append(1)
        if len(calls) < 3:
            raise sc.StepStalled("M3 e1 was silent and idle for 15:00")
        return 0
    assert pv._resuming(start, "M3 e1") == 0 and len(calls) == 3
    assert "restarting M3 e1 automatically" in capsys.readouterr().out


def test_a_crashed_generation_continues_from_the_poems_it_saved(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pv.time, "sleep", lambda s: None)
    out = tmp_path / "generations.jsonl"
    calls = []

    def start():
        calls.append(1)
        with out.open("a") as f:
            f.write('{"id": %d}\n' % len(calls))
        if len(calls) < 4:                      # crashes three times, each time after saving a poem
            raise RuntimeError("CUDA error")
    pv._resuming_generation(start, out, "M4 test i1_n1")
    assert len(calls) == 4 and "continues from the 1 rows" in capsys.readouterr().out


def test_a_generation_that_never_progresses_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(pv.time, "sleep", lambda s: None)
    calls = []

    def start():
        calls.append(1)
        raise RuntimeError("bug")
    with pytest.raises(RuntimeError):
        pv._resuming_generation(start, tmp_path / "g.jsonl", "x")
    assert len(calls) == 3


def test_the_runtime_is_released_only_when_everything_is_on_drive(pvc, monkeypatch, capsys):
    import types
    released = []
    colab = types.ModuleType("google.colab"); colab.runtime = types.SimpleNamespace(unassign=lambda: released.append(1))
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    (pvc.root / "m4").mkdir(parents=True)
    (pvc.root / "m4" / "a.json").write_text("{}")
    pv.release_runtime(pvc)
    assert released == [1] and (pvc.drive_root / "m4" / "a.json").exists()
    monkeypatch.setattr(pv, "sync_to_drive", lambda p, scope=".": False)
    pv.release_runtime(pvc)
    assert released == [1] and "NOT releasing" in capsys.readouterr().out


def test_training_restarts_after_a_crash_but_not_forever(tmp_path, monkeypatch, capsys):
    """sc.start_training's foreground loop (used by M5): a crash that is not out-of-memory restarts it from the checkpoint."""
    class Up:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def finish(self): return True
    cfg = sc.make_config(base_model=sc.YEHIA, drive_base=str(tmp_path / "MyDrive"))
    monkeypatch.setattr(sc, "_CheckpointUploader", Up)
    monkeypatch.setattr(sc, "flush_drive", lambda remount=True: True)
    monkeypatch.setattr(sc, "code_version", lambda c: "test")
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    monkeypatch.setattr(sc, "_pid_file", lambda c: tmp_path / "pid")
    monkeypatch.setattr(sc, "_log_has_oom", lambda *a, **k: False)
    log = tmp_path / "train.log"
    calls = []

    def fake_run(*a, **k):
        calls.append(1)
        if len(calls) <= 2:
            raise RuntimeError("training exited with code 1")
    monkeypatch.setattr(sc, "run", fake_run)
    sc._train_foreground(cfg, ["x"], {}, log, True, 3, "mount")
    assert len(calls) == 3 and "restart 2/3" in capsys.readouterr().out
    calls.clear()
    monkeypatch.setattr(sc, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        sc._train_foreground(cfg, ["x"], {}, log, True, 1, "mount")


def test_scoring_resumes_after_the_rows_already_scored(tmp_path):
    import json
    import meter_score_batched as msb
    rows = [{"id": f"p{i}", "sample_idx": 0} for i in range(5)]
    out = tmp_path / "scored.jsonl"
    out.write_text("".join(json.dumps({**r, "meter": 1}) + "\n" for r in rows[:3]) + '{"id": "p3", "sampl')   # cut line
    assert len(msb.resume_prefix(out, rows)) == 3
    assert msb.resume_prefix(out, rows[1:]) == []                       # not the beginning of this input: start again
    assert msb.resume_prefix(tmp_path / "none.jsonl", rows) == []


def test_labels_reader_skips_a_cut_line(tmp_path):
    import json
    import m5_data
    src = {s: [] for s in m5_data.SPLITS}
    out = tmp_path / "labels"
    out.mkdir()
    (out / "train.jsonl").write_text(json.dumps({"index": 0}) + "\n" + '{"index": 1, "ra')
    kept = m5_data.read_jsonl(out / "train.jsonl")
    assert [r["index"] for r in kept] == [0]


def test_the_judge_is_restarted_within_one_budget(pvc, monkeypatch, tmp_path, capsys):
    import json
    monkeypatch.setattr(pv.time, "sleep", lambda s: None)
    wd = tmp_path / "workers"
    wd.mkdir()
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[cmd.index("--max-cost-usd") + 1])
        with (wd / "worker_0.jsonl").open("a") as f:
            f.write(json.dumps({"judge_total_cost_usd": 1.0}) + "\n")
        return 3 if len(calls) < 3 else 0
    monkeypatch.setattr(sc, "run", fake_run)
    code = pv._judge_with_retries(pvc, "m5_2a", ["j.py", "--max-cost-usd", "10"], wd, 10.0)
    assert code == 0 and calls == ["10.0000", "9.0000", "8.0000"]        # every attempt gets what is left of the budget


def test_a_filled_human_study_is_never_overwritten(tmp_path):
    import json
    import m6_quality as Q
    sysdir = tmp_path / "systems"
    sysdir.mkdir()
    for name, text in (("a", "x\ny"), ("b", "z\nw")):
        (sysdir / f"{name}.jsonl").write_text("".join(
            json.dumps({"id": f"p{i}", "generated_text": text, "base_meter": "الطويل", "form": "تام"}) + "\n" for i in range(5)))
    sheet, key = Q.human_sheet(tmp_path, ["a", "b"], prompts=3)
    before = (sheet.read_text(encoding="utf-8-sig"), key.read_text())
    Q.human_sheet(tmp_path, ["a", "b"], prompts=3)                           # run again: unchanged
    assert (sheet.read_text(encoding="utf-8-sig"), key.read_text()) == before
    (tmp_path / "human" / "filled").mkdir()
    (tmp_path / "human" / "filled" / "rater1.csv").write_text("x")
    with pytest.raises(ValueError):
        Q.human_sheet(tmp_path, ["a", "b"], prompts=2)                       # different prompts: would break the item codes
    assert key.read_text() == before[1]
