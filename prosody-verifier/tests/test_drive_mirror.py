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
