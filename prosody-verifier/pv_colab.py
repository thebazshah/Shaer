"""Notebook helpers for the prosody-verifier research (called from prosody_main.ipynb; plain Python, no shell).

It reuses the baseline's Colab module (colab/shaer_colab.py, unmodified) for what already exists there: the isolated
environment (/content/venv), the watched subprocess runner (progress bars, heartbeat, stall detection) and the baseline's
Drive folders. What is new lives in this folder and writes to its own Drive folder:

  MyDrive/shaer-prosody/
      m1/splits/{train,eval,test}.jsonl              the Shaer data splits (lexicon from train, tuning on eval, report on test)
      m1/lexicon_train.json.gz                       a rebuilt pronunciation lexicon (the repository ships one)
      m1/validation_{eval,test}.json                 verifier validation on human poems and broken copies
      m1/bilstm_{e0,reference}[_bare].json           agreement with the BiLSTM meter classifier
      m1/m1_report.md                                the combined M1 tables
      m0/sources/e0/{generations,scored}.jsonl        the released Shaer poems, if the baseline's E0 is not on Drive
      m0/reference/{generations,scored}.jsonl          the human test poems, scored by the same evaluator
      m0/<system>/audit_rows.jsonl, summary.json      per-system headroom audit
      m0/m0_report.md, m0_summary.json                 the combined table
      logs/

Baseline outputs are only read, from where the baseline notebook wrote them (MyDrive/shaer-repro-yehia/runs/...).

Saving to Drive (every phase, M0-M6). On Colab the scripts write to a mirror of that folder on the VM's own disk
(/content/pv_local/shaer-prosody = pvc.root), never through the Drive mount, whose background upload can fall far behind
(a re-run then resumed from an older point than the log had shown). A phase first copies the files it reads and resumes
from back from Drive (when the VM does not have them, e.g. in a new session); while it runs, a background thread uploads
every changed file to MyDrive/shaer-prosody (= pvc.drive_root) through the Drive API and checks its MD5 with Drive's copy
(JSONL files up to their last complete line), about every minute; when the phase finishes or is stopped, a final verified
upload runs before anything else (and the stop then halts Run all). Training runs (M5) use the baseline's own mechanism
(sc.start_training: checkpoints on the VM's disk, verified upload, restore on a new session).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time

import shaer_colab as sc

__version__ = "1.2.0"                    # bump on every change of the prosody code (and the notebook's version with it)
__updated__ = "2026-11-25 UTC"
NOTEBOOK_REQUIRED = "1.0.0"              # the oldest notebook version whose cells still match this code

HERE = pathlib.Path(__file__).resolve().parent
DRIVE_FOLDER = "shaer-prosody"
RELEASED_TEST_ROWS = 3481
# baseline run folders -> system names used in the M0 tables
SYSTEM_NAMES = {"e0": "e0_released_shaer", "e1": "e1_shaer_adapter", "e2": "e2_yehia_base", "e3": "e3_retrained_yehia"}
BASELINE_FOLDER = "shaer-repro-yehia"      # the baseline notebook's Drive folder for Navid-AI/Yehia-7B-preview


@dataclasses.dataclass
class PVConfig:
    sc_cfg: "sc.Config"
    drive_base: str

    @property
    def drive_root(self) -> pathlib.Path:
        """This notebook's folder on Google Drive (MyDrive/shaer-prosody)."""
        return pathlib.Path(self.drive_base, DRIVE_FOLDER)

    @property
    def mirrored(self) -> bool:
        """On Colab (Drive under /content/drive) the work happens in a mirror on the VM's disk; PV_MIRROR=0 turns it off,
        PV_MIRROR=1 forces it (tests)."""
        flag = os.environ.get("PV_MIRROR", "")
        return flag == "1" or (flag != "0" and str(self.drive_base).startswith("/content/drive"))

    @property
    def root(self) -> pathlib.Path:
        """Where every phase reads and writes: the VM-disk mirror when mirrored (kept in sync with drive_root), else Drive."""
        if self.mirrored:
            return pathlib.Path(os.environ.get("PV_LOCAL_ROOT", "/content/pv_local"), DRIVE_FOLDER)
        return self.drive_root

    @property
    def m0(self) -> pathlib.Path:
        return self.root / "m0"

    @property
    def logs(self) -> pathlib.Path:
        return self.root / "logs"

    @property
    def py(self) -> str:
        return self.sc_cfg.py

    @property
    def repo(self) -> pathlib.Path:
        return pathlib.Path(self.sc_cfg.repo_dir)

    def run(self, args: list, label: str, **kw) -> None:
        """A watched subprocess in the isolated environment (progress, heartbeat, stall detection). The stop button stops
        the script too, so nothing keeps writing after the final upload of the phase."""
        kw.setdefault("stop_on_interrupt", True)
        _resuming(lambda: sc.run([self.py, *args], label=label, env=self.sc_cfg.env(), stall_after=self.sc_cfg.stall_seconds, **kw),
                  label)


MAX_STALL_RESTARTS = 20                  # a stuck script is restarted this many times in a row before the step gives up


def _resuming(start, label: str):
    """Run `start()` (a watched subprocess); when it hangs (Colab sometimes does) the watcher has stopped it, so start it
    again: every script continues from the files it already wrote. The cell keeps running and keeps using the runtime,
    instead of ending with an error while the runtime still burns compute units."""
    for attempt in range(MAX_STALL_RESTARTS + 1):
        try:
            return start()
        except sc.StepStalled as exc:
            if attempt >= MAX_STALL_RESTARTS:
                raise
            print(f"⚠ {exc}\n  restarting {label} automatically, it continues from what is saved "
                  f"(restart {attempt + 1}/{MAX_STALL_RESTARTS}) ...", flush=True)
            time.sleep(10)


# ---------------------------------------------------------------------------------------------- never write into the baseline
# The baseline notebook keeps its fine-tuned adapter and its generations in MyDrive/shaer-repro-yehia/:
#   sft_outputs/train/<RUN_NAME>/checkpoint-*, run_summary.json   (the E3 adapter; RUN_NAME is paper_repro_a100 by default)
#   runs/e0 ... runs/e3/{generations,scored}.jsonl, summary.json    (sc.run_generation(cfg, "e3") writes runs/e3)
# This notebook only reads those folders. Anything it trains or generates through the baseline's machinery (sc.start_training,
# sc.run_generation, the repository's sft/outputs and runs/ links) goes to the sibling MyDrive/shaer-repro-yehia-prosody/,
# with the same layout, under run names that start with "prosody_".
SIBLING_SUFFIX = "-prosody"
TRAIN_PREFIX = "prosody_"


def baseline_root(sc_cfg: "sc.Config") -> pathlib.Path:
    """The baseline's Drive folder (MyDrive/shaer-repro-yehia), whether sc_cfg is the baseline's config or the sibling's."""
    root = pathlib.Path(sc_cfg.drive_root)
    return root.with_name(root.name[: -len(SIBLING_SUFFIX)]) if root.name.endswith(SIBLING_SUFFIX) else root


def _inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    path, root = pathlib.Path(os.path.abspath(path)), pathlib.Path(os.path.abspath(root))
    return path == root or root in path.parents


def prosody_sc_config(sc_cfg: "sc.Config", run_name: str = f"{TRAIN_PREFIX}main") -> "sc.Config":
    """A copy of the baseline's configuration whose every output folder (runs/, logs/, sft_outputs/, the training run
    folder and its local copy on the VM) is in the sibling MyDrive/shaer-repro-yehia-prosody/, beside the baseline's.
    Refuses run names that do not start with "prosody_" or that a baseline run already uses."""
    if not re.fullmatch(rf"{TRAIN_PREFIX}[A-Za-z0-9_.-]+", run_name or ""):
        raise ValueError(f"run name {run_name!r}: a run of this notebook must be named {TRAIN_PREFIX}<something> "
                         "(letters, digits, _ . -), so it can never be taken for a baseline run")
    base = baseline_root(sc_cfg)
    sibling = base.with_name(base.name + SIBLING_SUFFIX)
    if (base / "sft_outputs" / "train" / run_name).exists() or (base / "runs" / run_name).exists():
        raise ValueError(f"run name {run_name!r} is already used in the baseline folder {base}: choose another")
    cfg = dataclasses.replace(sc_cfg, drive_root=str(sibling), run_name=run_name)
    sc._require_drive(cfg.drive_root)
    for sub in (cfg.runs, cfg.logs, cfg.sft_outputs):
        sub.mkdir(parents=True, exist_ok=True)
    for p in (cfg.runs, cfg.logs, cfg.sft_outputs, cfg.run_dir):
        if _inside(p, base):
            raise RuntimeError(f"refusing a configuration that writes into the baseline folder: {p}")
    # the VM-local training folder is named after the run only (/content/sft_local/train/<run name>), so a run name the
    # baseline itself uses would share it when both train on the same runtime
    if run_name == sc_cfg.run_name and not _inside(pathlib.Path(sc_cfg.drive_root), sibling):
        raise ValueError(f"run name {run_name!r} is the baseline's own RUN_NAME: choose another")
    return cfg


def training_config(pv: "PVConfig", run_name: str) -> "sc.Config":
    """The configuration for a fine-tuning run of this research (idea 2 and later): sc.start_training(training_config(pvc,
    "prosody_selftrain_r1")) writes its checkpoints to MyDrive/shaer-repro-yehia-prosody/sft_outputs/train/<run_name>/,
    never into the baseline's sft_outputs, and sc.run_generation(..., "e3") with it writes runs/e3 of the sibling."""
    cfg = prosody_sc_config(pv.sc_cfg, run_name)
    sc.sync_repo(cfg)                    # the repository's sft/outputs and runs/ links now point at the sibling
    print(f"training run {run_name!r}: checkpoints -> {cfg.run_dir}")
    return cfg


# ---------------------------------------------------------------------------------------------- saving to Drive (all phases)
SYNC_EVERY = 60                     # seconds between upload rounds while a phase runs
_SKIP_SUFFIXES = (".uploading", ".partial", ".tmp")


def _drive_api():
    """One authenticated Drive API client per session (Google's consent prompt appears once); without it, the mirror is
    copied through the Drive mount and Drive is flushed at the end of each phase (sc._LocalFolderAPI)."""
    global _API
    if "_API" not in globals():
        try:
            _API = sc._DriveAPI()
            print("  Drive: results are uploaded through the Drive API and verified (MD5)")
        except Exception as exc:
            _API = sc._LocalFolderAPI()
            print(f"  ⚠ Drive API not available ({exc.__class__.__name__}: {str(exc)[:200]}): copying through the Drive "
                  "mount instead, then flushing Drive at the end of each step (not verified). For the verified upload run "
                  "from google.colab import auth; auth.authenticate_user()  and approve Google's prompt.")
    return _API


_API_FACTORY = _drive_api           # tests replace it
_ON_DRIVE: dict[str, dict] = {}     # drive_root -> {relative path: (size, mtime_ns) of the VM file whose content Drive has}


class _Mirror:
    """Keeps pvc.drive_root on Drive in step with pvc.root on the VM's disk, for the folders and files a phase uses."""

    def __init__(self, pv: "PVConfig", api) -> None:
        self.pv, self.api = pv, api
        self.local, self.drive = pv.root, pv.drive_root
        self.scopes: list[str] = []
        self.known = _ON_DRIVE.setdefault(str(self.drive), {})   # kept for the whole session, across steps
        self.last_upload: dict[str, float] = {}
        self.folder_ids: dict[str, str] = {}
        self.listing: dict[str, dict] = {}          # relative folder -> {name: Drive entry}
        self.snap = self.local.parent / ".pv_upload_snapshots"
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.thread: threading.Thread | None = None
        self.uploaded = 0
        self.errors = 0

    # -- which part of the tree this phase uses; copy it back from Drive where the VM does not have it
    def add_scopes(self, scopes) -> None:
        with self.lock:
            for rel in scopes:
                rel = str(pathlib.PurePosixPath(rel))
                if rel in self.scopes:
                    continue
                self.scopes.append(rel)
                self._restore(rel)

    def _restore(self, rel: str) -> None:
        src = self.drive / rel
        if not src.exists():
            return
        files = [src] if src.is_file() else [f for f in src.rglob("*") if f.is_file() and not f.name.endswith(_SKIP_SUFFIXES)]
        copied, size, t0 = 0, 0, time.time()
        for f in files:
            r = f.relative_to(self.drive).as_posix()
            dst = self.local / r
            if dst.exists():                         # the VM's copy is the newer one (everything is written here);
                continue                             # it is uploaded unless Drive already has exactly it
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            self.known[r] = self._sig(dst)
            copied, size = copied + 1, size + dst.stat().st_size
        if copied:
            print(f"  Drive → VM: {rel}: {copied} file(s), {size / 2**20:.1f} MB in {sc._fmt(time.time() - t0)}", flush=True)

    @staticmethod
    def _sig(path: pathlib.Path) -> tuple:
        st = path.stat()
        return (st.st_size, st.st_mtime_ns)

    # -- Drive side
    def _folder(self, rel_dir: str) -> str:
        if rel_dir in self.folder_ids:
            return self.folder_ids[rel_dir]
        if hasattr(self.api, "folder_for"):          # through the mount (and tests): the path itself
            fid = self.api.folder_for(self.drive / rel_dir if rel_dir != "." else self.drive)
        elif rel_dir == ".":
            base = pathlib.Path(self.drive)
            rel = base.relative_to("/content/drive/MyDrive")      # the Drive API addresses folders from My Drive
            fid = "root"
            for part in rel.parts:
                match = [f for f in self.api.children(fid) if f["name"] == part and f["mimeType"] == sc._FOLDER]
                fid = match[0]["id"] if match else self.api.create_folder(fid, part)
        else:
            parent_rel = pathlib.PurePosixPath(rel_dir).parent.as_posix()
            parent = self._folder(parent_rel)
            name = pathlib.PurePosixPath(rel_dir).name
            entry = self._entries(parent_rel, parent).get(name)
            fid = entry["id"] if entry and entry["mimeType"] == sc._FOLDER else self.api.create_folder(parent, name)
        self.folder_ids[rel_dir] = fid
        return fid

    def _entries(self, rel_dir: str, fid: str, refresh: bool = False) -> dict:
        if refresh or rel_dir not in self.listing:
            self.listing[rel_dir] = {f["name"]: f for f in self.api.children(fid)}
        return self.listing[rel_dir]

    def _upload(self, rel: str, final: bool) -> bool:
        path = self.local / rel
        sig = self._sig(path)
        if self.known.get(rel) == sig:
            return False
        now = time.time()
        if not final:
            if now - sig[1] / 1e9 < 2:                   # still being written
                return False
            if now - self.last_upload.get(rel, 0) < SYNC_EVERY * max(1, sig[0] // (64 * 2**20)):
                return False                         # large, growing files: not every round
        snap = self.snap / rel
        snap.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":                  # only complete lines: Drive never holds a half-written row
            with open(path, "rb") as f:
                data = f.read()
            snap.write_bytes(data[: data.rfind(b"\n") + 1])
        else:
            shutil.copyfile(path, snap)
        rel_dir = pathlib.PurePosixPath(rel).parent.as_posix()
        parent = self._folder(rel_dir)
        name = pathlib.PurePosixPath(rel).name
        entry = None if hasattr(self.api, "folder_for") else self._entries(rel_dir, parent).get(name)
        want = sc._md5(snap)
        if entry and str(entry.get("md5Checksum", "")) == want:  # Drive has exactly this already (e.g. a restarted kernel)
            snap.unlink(missing_ok=True)
            self.known[rel], self.last_upload[rel] = sig, now
            return False
        resp = self.api.upload(parent, name, snap, file_id=entry["id"] if entry else None)
        if str(resp.get("md5Checksum", "")) != want or int(resp.get("size", -1)) != snap.stat().st_size:
            raise RuntimeError(f"{rel}: Drive's copy does not match (size {resp.get('size')} vs {snap.stat().st_size})")
        if not entry and not hasattr(self.api, "folder_for"):
            self._entries(rel_dir, parent)[name] = {"id": resp["id"], "name": name, "mimeType": "application/octet-stream"}
        snap.unlink(missing_ok=True)
        self.known[rel], self.last_upload[rel] = sig, now
        self.uploaded += 1
        return True

    def _trash(self, rel: str) -> None:
        rel_dir = pathlib.PurePosixPath(rel).parent.as_posix()
        name = pathlib.PurePosixPath(rel).name
        if hasattr(self.api, "folder_for"):
            target = self.drive / rel
            if target.exists():
                self.api.trash(str(target))
        else:
            entry = self._entries(rel_dir, self._folder(rel_dir), refresh=True).get(name)
            if entry:
                self.api.trash(entry["id"])
                self.listing[rel_dir].pop(name, None)
        self.known.pop(rel, None)

    def sync_once(self, final: bool = False) -> int:
        with self.lock:
            n = 0
            present = set()
            for rel in self.scopes:
                base = self.local / rel
                files = [base] if base.is_file() else (
                    [f for f in base.rglob("*") if f.is_file() and not f.is_symlink()] if base.is_dir() else [])
                for f in files:
                    if f.name.endswith(_SKIP_SUFFIXES):
                        continue
                    r = f.relative_to(self.local).as_posix()
                    present.add(r)
                    try:
                        n += self._upload(r, final)
                    except FileNotFoundError:        # renamed or removed while being copied: picked up next round
                        continue
            for r in [r for r in self.known if r not in present and any(r == s or r.startswith(s + "/") for s in self.scopes)]:
                self._trash(r)                       # removed or renamed on the VM (e.g. a .stale file set aside)
            return n

    # -- the background loop
    def start(self) -> None:
        def loop():
            while not self.stop_event.wait(SYNC_EVERY):
                try:
                    self.sync_once()
                    self.errors = 0
                except Exception as exc:
                    self.errors += 1
                    print(f"  ⚠ Drive upload failed ({exc.__class__.__name__}: {str(exc)[:160]}); retrying in {SYNC_EVERY} s",
                          flush=True)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def finish(self) -> bool:
        """Stop the loop and upload everything that changed, verified; through the mount (then flush) if the API fails."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        for attempt in range(3):
            try:
                n = self.sync_once(final=True)
                break
            except Exception as exc:
                print(f"  ⚠ final Drive upload failed ({exc.__class__.__name__}: {str(exc)[:160]})", flush=True)
                if attempt < 2:
                    time.sleep(10)
        else:
            if isinstance(self.api, sc._LocalFolderAPI):
                print(f"  ✖ NOT EVERYTHING IS ON DRIVE: the newest results are on this VM ({self.local}). Do not end the "
                      "session: run the cell again, or pv.sync_to_drive(pvc).", flush=True)
                return False
            print("  copying through the Drive mount instead ...", flush=True)
            self.api, self.folder_ids, self.listing = sc._LocalFolderAPI(), {}, {}
            self.known.clear()                       # not known to be on Drive any more: copy everything in scope
            return self.finish()
        if isinstance(self.api, sc._LocalFolderAPI):
            sc.flush_drive()                         # the mount uploads lazily: wait until it has
        print(f"  ☁ Drive: everything this step wrote is on Drive ({self.uploaded} upload(s), the last {n})", flush=True)
        return True


_ACTIVE: _Mirror | None = None


def _phase(*scopes: str):
    """Run a notebook step inside the Drive mirror: copy `scopes` (paths under pvc.root) from Drive to the VM where it
    lacks them, upload what changes in the background, and on the way out (finished, failed or stopped) upload the rest,
    verified. A stop then still propagates, so Runtime > Run all halts at the stopped cell."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(pv: "PVConfig", *args, **kwargs):
            global _ACTIVE
            if not pv.mirrored:
                return fn(pv, *args, **kwargs)
            sc._require_drive(pv.drive_root)
            if _ACTIVE is not None:                  # a step called by another step: same mirror, more scopes
                _ACTIVE.add_scopes(scopes)
                return fn(pv, *args, **kwargs)
            mirror = _Mirror(pv, _API_FACTORY())
            mirror.add_scopes(scopes)
            mirror.start()
            _ACTIVE = mirror
            stopped = False
            try:
                return fn(pv, *args, **kwargs)
            except KeyboardInterrupt:
                stopped = True
                print("■ stopped: uploading what this step wrote to Drive before anything else ...", flush=True)
                raise
            finally:
                _ACTIVE = None
                ok = mirror.finish()
                if stopped:
                    print("■ " + ("everything it wrote is on Drive" if ok else "NOT everything is on Drive (see above)")
                          + ". Run all stops here; run the cell again to resume.", flush=True)
        return wrapper
    return deco


def _need(pv: "PVConfig", path) -> pathlib.Path:
    """A path a step reads (e.g. a training set recorded in variants.json): on this VM when mirrored (copied from Drive if
    missing), and the same path whether it was recorded on Drive or on the VM."""
    path = pathlib.Path(path)
    for base in (pv.drive_root, pv.root):
        if _inside(path, base) and path != base:
            rel = pathlib.Path(os.path.abspath(path)).relative_to(os.path.abspath(base)).as_posix()
            if pv.mirrored and _ACTIVE is not None:
                _ACTIVE.add_scopes([rel])
            local = pv.root / rel
            return local if local.exists() or pv.mirrored else pv.drive_root / rel
    return path


def sync_to_drive(pv: "PVConfig", scope: str = ".") -> None:
    """Upload everything under pvc.root (or one sub-folder) that Drive does not have yet, verified. Normally not needed:
    every step does it by itself; this is for an interrupted final upload."""
    if not pv.mirrored:
        print("results are written to Drive directly (no mirror on this machine)")
        return
    mirror = _Mirror(pv, _API_FACTORY())
    mirror.scopes = [scope]
    mirror.finish()


def check_version(notebook_version: str) -> None:
    """Say whether this notebook copy works with the downloaded code. Code changes arrive with section 2 (git pull and
    reload); only a change of the notebook's cells needs the notebook reopened."""
    def key(v: str) -> tuple:
        return tuple(int(x) for x in re.findall(r"\d+", v)[:3])
    if key(notebook_version) >= key(NOTEBOOK_REQUIRED):
        print(f"✔ notebook v{notebook_version} works with the prosody code v{__version__} (updated {__updated__})")
    else:
        print(f"⚠ NOTEBOOK TOO OLD: this notebook is v{notebook_version}, the prosody code v{__version__} needs notebook "
              f"v{NOTEBOOK_REQUIRED} or newer (its cells changed). Reopen it from GitHub (the link in "
              "prosody-verifier/README.md); everything on Drive is kept.")


def make_config(sc_cfg: "sc.Config", drive_base: str = "/content/drive/MyDrive") -> PVConfig:
    """This notebook's configuration. If sc_cfg is the baseline's own (as from sc.make_config), it is replaced by the
    sibling copy (prosody_sc_config) and the repository's output links are pointed at the sibling, so nothing this
    notebook runs can write into the baseline's adapter or generations."""
    if sc_cfg.base_model != sc.YEHIA:
        raise ValueError(f"this research uses {sc.YEHIA} (the base model of the released Shaer adapter): set BASE_MODEL to "
                         f"it in the configuration cell (not {sc_cfg.base_model!r})")
    if not pathlib.Path(sc_cfg.drive_root).name.endswith(SIBLING_SUFFIX):
        sc_cfg = prosody_sc_config(sc_cfg)
        if pathlib.Path(sc_cfg.repo_dir).exists():
            sc.sync_repo(sc_cfg)
    pv = PVConfig(sc_cfg=sc_cfg, drive_base=drive_base)
    for d in (pv.m0, pv.logs):
        d.mkdir(parents=True, exist_ok=True)
    print(f"prosody-verifier code v{__version__} | outputs ->", pv.root)
    print(f"baseline folder (read only) -> {baseline_root(sc_cfg)}; anything trained or generated with the baseline's "
          f"tools -> {sc_cfg.drive_root}")
    return pv


def runtime_info() -> None:
    """Say which runtime this is and what it is good for (compute-unit costs differ a lot between runtimes)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    print("runtime:", out or "CPU only (no GPU)")
    print("  M0 needs no GPU: a CPU runtime is enough and costs the fewest compute units.\n"
          "  Idea 1's generation runs (later) need a GPU: an L4 (24 GB) works with smaller batches; an A100 is faster but\n"
          "  costs more units per hour (Colab shows the rate under 'Change runtime type').")


def _lines(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def _need_env(pv: PVConfig) -> None:
    if not pathlib.Path(pv.py).exists():
        raise RuntimeError(f"{pv.py} is missing: run the installation cell (section 3) first.")


def _score(pv: PVConfig, gens: pathlib.Path, scored: pathlib.Path, label: str) -> pathlib.Path:
    """Score rows with the repository's evaluator (batched wrapper; same results). Skipped when already complete."""
    total = _lines(gens)
    if total and _lines(scored) >= total:
        print(f"{label}: already scored ({total:,} rows) -> {scored}")
        return scored
    pv.run([HERE / "meter_score_batched.py", "--input-jsonl", gens, "--output-jsonl", scored], label=label)
    return scored


def baseline_runs(pv: PVConfig) -> dict[str, pathlib.Path]:
    """Complete scored baseline runs found on Drive: {system name: scored.jsonl}. Incomplete runs are listed and skipped."""
    found: dict[str, pathlib.Path] = {}
    runs = pathlib.Path(pv.drive_base, BASELINE_FOLDER, "runs")
    for step, name in SYSTEM_NAMES.items():
        scored, gens = runs / step / "scored.jsonl", runs / step / "generations.jsonl"
        if not scored.exists():
            continue
        n_scored, n_gens = _lines(scored), _lines(gens)
        if n_gens and n_scored < n_gens:
            print(f"  skipping {name}: scored.jsonl has {n_scored:,} of {n_gens:,} rows (finish the baseline's scoring first)")
            continue
        found[name] = scored
    return found


def _released_scored(pv: PVConfig, runs: dict[str, pathlib.Path]) -> pathlib.Path:
    """The released Shaer poems with scores: the baseline's E0 if complete, else exported and scored here."""
    if "e0_released_shaer" in runs and _lines(runs["e0_released_shaer"]) >= RELEASED_TEST_ROWS:
        return runs["e0_released_shaer"]
    src = pv.m0 / "sources" / "e0"
    gens = src / "generations.jsonl"
    if _lines(gens) < RELEASED_TEST_ROWS:
        pv.run([pv.repo / "evaluation" / "export_released_generations.py", "--out", gens], label="export of released poems")
    return _score(pv, gens, src / "scored.jsonl", "scoring the released Shaer poems")


@_phase('m0')
def run_m0(pv: PVConfig) -> None:
    """M0 headroom audit over every complete baseline run on Drive plus the human reference poems."""
    _need_env(pv)
    t0 = sc._start("M0 headroom audit", "a few minutes, also on a CPU runtime")
    runs = baseline_runs(pv)
    runs["e0_released_shaer"] = _released_scored(pv, runs)
    ref = pv.m0 / "reference"
    if _lines(ref / "generations.jsonl") < RELEASED_TEST_ROWS:
        pv.run([HERE / "m0_audit.py", "reference", "--rows", runs["e0_released_shaer"], "--out", ref / "generations.jsonl"],
               label="reference rows")
    systems = {"reference": _score(pv, ref / "generations.jsonl", ref / "scored.jsonl", "scoring the human reference poems"),
               **runs}
    print("systems audited:", ", ".join(systems))
    for name, scored in systems.items():
        pv.run([HERE / "m0_audit.py", "audit", "--scored", scored, "--system", name, "--out-dir", pv.m0 / name],
               label=f"audit {name}")
    pv.run([HERE / "m0_audit.py", "report", "--m0-dir", pv.m0], label="M0 report", quiet=True)
    sc._finish(t0, f"M0 (tables in {pv.m0 / 'm0_report.md'})")


@_phase('m0')
def show_m0(pv: PVConfig) -> None:
    """Show the combined M0 table (rendered as Markdown in Colab)."""
    report = pv.m0 / "m0_report.md"
    if not report.exists():
        print("no M0 report yet: run pv.run_m0(pvc) first")
        return
    text = report.read_text(encoding="utf-8")
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:
        print(text)


@_phase('m0')
def m0_summary(pv: PVConfig) -> dict:
    """The combined M0 summary as a dict (one entry per system)."""
    return json.loads((pv.m0 / "m0_summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------- M1: the verifier
SHIPPED_LEXICON = HERE / "data" / "lexicon_train.json.gz"


def m1_dir(pv: PVConfig) -> pathlib.Path:
    d = pv.root / "m1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def m1_tests(pv: PVConfig) -> None:
    """The unit tests of the rule tables, the prosodic writing, the scanner, the rhyme check and the poem verifier."""
    _need_env(pv)
    pv.run(["-m", "pytest", HERE / "tests", "-q", "-p", "no:cacheprovider"], label="M1 tests")


@_phase('m1/splits')
def m1_prepare_splits(pv: PVConfig) -> pathlib.Path:
    """Download the Shaer train / eval / test splits as small JSONL files on Drive (skipped when present)."""
    _need_env(pv)
    out = m1_dir(pv) / "splits"
    if all(_lines(out / f"{s}.jsonl") for s in ("train", "eval", "test")):
        print("splits already on Drive:", out)
        return out
    pv.run([HERE / "prepare_splits.py", "--out-dir", out], label="downloading the Shaer splits")
    return out


@_phase('m1')
def m1_lexicon(pv: PVConfig, rebuild: bool = False, workers: int = 0) -> pathlib.Path:
    """The pronunciation lexicon: the one shipped in the repository, or (rebuild=True) a new one built from the train
    split on Drive (about 15 minutes on 4 CPU cores, longer on a 2-core Colab CPU runtime)."""
    if not rebuild:
        print("using the lexicon shipped with the code:", SHIPPED_LEXICON)
        return SHIPPED_LEXICON
    splits = m1_prepare_splits(pv)
    out = m1_dir(pv) / "lexicon_train.json.gz"
    args = [HERE / "build_lexicon.py", "--train", splits / "train.jsonl", "--out", out]
    if workers:
        args += ["--workers", str(workers)]
    pv.run(args, label="building the pronunciation lexicon")
    return out


@_phase('m1')
def m1_validate(pv: PVConfig, split: str = "eval", limit: int = 1000, lexicon: pathlib.Path | None = None) -> pathlib.Path:
    """Validate the verifier on human poems of a split and on broken copies of them (limit=0: the whole split)."""
    _need_env(pv)
    splits = m1_prepare_splits(pv)
    out = m1_dir(pv) / f"validation_{split}.json"
    t0 = sc._start(f"M1 validation on the {split} split", "about 10 minutes per 1,000 poems on a 2-core CPU runtime")
    pv.run([HERE / "validate_verifier.py", "--poems", splits / f"{split}.jsonl", "--lexicon", lexicon or SHIPPED_LEXICON,
            "--out", out, "--limit", str(limit)], label=f"validation ({split})")
    sc._finish(t0, f"validation ({split}) -> {out}")
    return out


@_phase('m0', 'm1')
def m1_compare_bilstm(pv: PVConfig, limit: int = 1000, lexicon: pathlib.Path | None = None) -> None:
    """Bayt-by-bayt agreement with the BiLSTM classifier on the released Shaer poems and on the human reference poems,
    with written diacritics binding and ignored. Needs M0 (it reuses M0's scored files)."""
    _need_env(pv)
    runs = baseline_runs(pv)
    e0 = runs.get("e0_released_shaer") or (pv.m0 / "sources" / "e0" / "scored.jsonl")
    ref = pv.m0 / "reference" / "scored.jsonl"
    if not (_lines(e0) and _lines(ref)):
        raise RuntimeError("run M0 first (pv.run_m0(pvc)): it scores the released and the reference poems")
    t0 = sc._start("M1 agreement with the BiLSTM classifier", "a few minutes per 1,000 poems and mode")
    for name, src in (("e0", e0), ("reference", ref)):
        for bare in (False, True):
            out = m1_dir(pv) / f"bilstm_{name}{'_bare' if bare else ''}.json"
            args = [HERE / "compare_bilstm.py", "--scored", src, "--lexicon", lexicon or SHIPPED_LEXICON, "--out", out,
                    "--limit", str(limit)] + (["--ignore-diacritics"] if bare else [])
            pv.run(args, label=f"agreement {name}{' (diacritics ignored)' if bare else ''}")
    sc._finish(t0, "BiLSTM agreement")


@_phase('m1')
def m1_report(pv: PVConfig) -> str:
    """Combine the M1 results on Drive into one Markdown table (also written to m1/m1_report.md)."""
    d = m1_dir(pv)
    out = ["# M1 verifier validation", ""]
    for split in ("eval", "test"):
        p = d / f"validation_{split}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["by_margin"]["1.5"]
        out += [f"## Human poems, {split} split ({r['poems']:,} poems, {r['hemistichs']:,} hemistichs)", "",
                "| measure | value |", "|---|---:|",
                f"| hemistichs accepted in their labelled meter and form | {c['acceptance']}% |",
                f"| ... in their labelled meter, any form | {r.get('acceptance_any_form@1.5')}% |",
                f"| other meters also accepting a hemistich (mean) | {c['wrong_meters_per_hemistich']} |",
                f"| labelled meter = best analysis | {r['identification_top1']}% |",
                f"| broken copies rejected (all kinds) | {c['negatives_rejected_all']}% |"]
        out += [f"| ... {k} | {v}% |" for k, v in c["negatives_rejected"].items()]
        if "poem_level" in r:
            pl = r["poem_level"]
            out += [f"| poems: every bayt accepted | {pl['meter_ok_pct']}% |", f"| poems: rhyme passes | {pl['rhyme_ok_pct']}% |",
                    f"| planted rhyme breaks flagged | {pl['rhyme_negative_flagged_pct']}% |"]
        out.append("")
    rows, kinds = [], []
    for name in ("e0", "e0_bare", "reference", "reference_bare"):
        p = d / f"bilstm_{name}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            v = r["verifier_any_form"]
            nr = r.get("no_reading_in_meter", {})
            rows.append(f"| {name} | {r['bayts']:,} | {v['bilstm_ok_pct']}% | {v['verifier_ok_pct']}% | {v['kappa']} | "
                        f"{nr.get('bilstm_accepted_bayts_pct', '-')}% |")
            kinds.append((name, r.get("rejection_kinds_of_bilstm_accepted", {})))
    if rows:
        out += ["## Bayts on the requested meter: BiLSTM vs verifier", "",
                "Last column: share of the bayts the BiLSTM accepts that have a hemistich with no reading at all in the "
                "requested meter (any form, any cost), so it does not depend on the margin rule.", "",
                "| poems | bayts | BiLSTM | verifier (any form) | kappa | BiLSTM-accepted, no reading in the meter |",
                "|---|---:|---:|---:|---:|---:|"] + rows + [""]
        for name, k in kinds:
            if k:
                out += [f"Why the verifier rejects bayts the BiLSTM accepts ({name}, {k.get('bayts', 0):,} bayts; a bayt "
                        "can have two reasons): " + "; ".join(f"{a} {b}%" for a, b in k.items() if a != "bayts"), ""]
    text = "\n".join(out)
    (d / "m1_report.md").write_text(text, encoding="utf-8")
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:
        print(text)
    return text


# ---------------------------------------------------------------------------------------------- M3
def m3_dir(pv: PVConfig) -> pathlib.Path:
    d = pv.root / "m3"
    d.mkdir(parents=True, exist_ok=True)
    return d


def m3_systems(pv: PVConfig) -> dict[str, pathlib.Path]:
    """What M3 re-scores: the human reference poems and the released Shaer poems (both prepared by M0), and every
    complete baseline run on Drive (E1-E3)."""
    runs = baseline_runs(pv)
    e0 = runs.pop("e0_released_shaer", None) or (pv.m0 / "sources" / "e0" / "scored.jsonl")
    ref = pv.m0 / "reference" / "scored.jsonl"
    if not (_lines(e0) and _lines(ref)):
        raise RuntimeError("run M0 first (pv.run_m0(pvc)): it prepares and scores the reference and the released poems")
    return {"reference": ref, "e0_released_shaer": e0, **runs}


@_phase('m0', 'm3')
def m3_rescore(pv: PVConfig, limit: int = 0, modes: tuple = ("letters", "binding"), controls: bool = True,
               workers: int = 0) -> None:
    """Re-score every system with the verifier (both reading modes), then the two controls built from the reference
    poems. Every finished poem is written to Drive at once: re-running the cell resumes where it stopped. Records made
    by another version of the verifier or lexicon are set aside (.stale), never mixed in. limit=N: the first N prompts
    of each system only (a quick look); limit=0: all 3,481."""
    _need_env(pv)
    out = m3_dir(pv)
    systems = m3_systems(pv)
    print("systems:", ", ".join(systems))
    cores = workers or os.cpu_count() or 1
    t0 = sc._start("M3 strict re-scoring", f"{cores} worker processes (one per CPU core); roughly 30-60 minutes per system and "
                   "reading mode on a 2-core runtime, about 1/cores of that with more cores; resumable")
    extra = ["--workers", str(workers)] if workers else []
    for name, scored in systems.items():
        pv.run([HERE / "m3_rescore.py", "score", "--scored", scored, "--system", name, "--out-dir", out,
                "--modes", ",".join(modes), "--limit", str(limit), *extra], label=f"M3 {name}")
    if controls:
        for kind in ("wrong_meter", "mixed_rhyme"):
            src = out / "controls" / f"{kind}.jsonl"
            if not _lines(src):
                pv.run([HERE / "m3_rescore.py", "control", "--scored", systems["reference"], "--kind", kind, "--out", src],
                       label=f"control {kind}")
            pv.run([HERE / "m3_rescore.py", "score", "--scored", src, "--system", f"control_{kind}", "--out-dir", out,
                    "--modes", "letters", "--limit", str(limit), *extra], label=f"M3 control {kind}")
    sc._finish(t0, f"M3 re-scoring -> {out}")


@_phase('m3')
def m3_report(pv: PVConfig, reps: int = 1000) -> str:
    """Summarise every re-scored system (Shaer's own metrics next to the strict ones, with bootstrap intervals,
    margins, breakdowns, controls and integrity checks): m3/m3_summary.json and m3/m3_report.md on Drive."""
    _need_env(pv)
    out = m3_dir(pv)
    pv.run([HERE / "m3_rescore.py", "summarize", "--dir", out, "--reps", str(reps)], label="M3 summary", quiet=True)
    text = (out / "m3_report.md").read_text(encoding="utf-8")
    summary = json.loads((out / "m3_summary.json").read_text(encoding="utf-8"))
    bad = {k: v for k, v in summary["checks"].items()
           if any(v.get(c) not in (0, 0.0, None) for c in ("line_count_mismatches", "misaligned_bilstm_bayts",
                                                           "default_margin_recomputation_mismatches",
                                                           "exact_count_vs_official_pp"))}
    print("integrity checks:", "all 0" if not bad else f"FAILED for {', '.join(bad)} (see the last table)")
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:
        print(text)
    return text


# ---------------------------------------------------------------------------------------------- M4
# The development sweep (eval split, a seeded subset) and the final test runs. Run names encode the condition:
#   i1_n<N>          N whole poems per prompt (I0 = the first sample, I1 = best-of-n for n = 2, 4, ... N)
#   i2_k<k>_r<R>     the hemistich search with k candidates per round and up to R rounds (suffix _b: checked in binding mode)
M4_DEV_SUBSET = 400
M4_DEV_SAMPLES = 16
M4_DEV_SEARCH = ((1, 4), (1, 8), (4, 1), (4, 2), (8, 1), (4, 2, "binding"))   # (k, R[, check mode])


def m4_dir(pv: PVConfig, split: str) -> pathlib.Path:
    d = pv.root / "m4" / split
    (d / "runs").mkdir(parents=True, exist_ok=True)
    return d


def _need_gpu(what: str) -> None:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    if "GPU" not in out:
        raise RuntimeError(f"{what} needs a GPU runtime (A100 preferred, L4 works): Runtime > Change runtime type")


@_phase('m4')
def m4_generate(pv: PVConfig, split: str, run: str, *, subset: int = 0, num_samples: int = 1, k: int = 4,
                rounds: int = 1, batch_size: str = "auto", extra: list | None = None, folder: str = "") -> pathlib.Path:
    """One M4 generation run with the released Shaer adapter on Yehia-7B-preview (resumable: finished poems are on Drive)."""
    cfg = pv.sc_cfg
    _need_env(pv)
    _need_gpu("M4 generation")
    out = m4_dir(pv, folder or split) / "runs" / run / "generations.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "sample" if run.startswith("i1") else "search"
    args = [HERE / "m4_decode.py", mode, "--split", split, "--subset", str(subset), "--out", out,
            "--base-model", cfg.base_model, "--batch-size", str(batch_size)]
    args += ["--num-samples", str(num_samples)] if mode == "sample" else ["--k", str(k), "--rounds", str(rounds)]
    if mode == "search" and run.endswith("_b"):
        args += ["--check", "binding"]
    pv.run([*args, *(extra or [])], label=f"M4 {split} {run}")
    return out


@_phase('m4/eval')
def m4_dev(pv: PVConfig, subset: int = M4_DEV_SUBSET, samples: int = M4_DEV_SAMPLES, search=M4_DEV_SEARCH,
           only: tuple | None = None) -> None:
    """The development sweep on the eval split: one sample run (I0 and best-of-2 ... best-of-`samples`) and the search
    configurations, all on the same seeded subset of prompts. Resumable; `only=("i2_k4_r1",)` runs some of them."""
    runs = [(f"i1_n{samples}", dict(num_samples=samples))]
    for k, r, *check in search:
        runs.append((f"i2_k{k}_r{r}" + ("_b" if check and check[0] == "binding" else ""), dict(k=k, rounds=r)))
    t0 = sc._start(f"M4 development sweep: {len(runs)} runs on {subset} eval prompts",
                   "the sample run is the long one (about 2 h on an A100 for 16 samples per prompt); each search run "
                   "takes minutes to tens of minutes; every finished poem is on Drive")
    for name, kw in runs:
        if only and name not in only:
            continue
        m4_generate(pv, "eval", name, subset=subset, **kw)
    sc._finish(t0, "M4 development sweep")


@_phase('m4/test')
def m4_test(pv: PVConfig, k: int, rounds: int, samples: int) -> None:
    """The final runs on the full test split (3,481 prompts): the chosen search configuration and best-of-`samples`.
    Choose them on the development sweep first (m4_report(pvc, "eval") prints the choice its rule makes)."""
    m4_generate(pv, "test", f"i2_k{k}_r{rounds}", k=k, rounds=rounds)
    m4_generate(pv, "test", f"i1_n{samples}", num_samples=samples)


@_phase('m4/eval')
def m4_selfcheck(pv: PVConfig, prompts: int = 8) -> dict:
    """Before the long runs (a few minutes on the GPU): on the real model with the adapter, the shared-cache logits against
    a full forward pass, and greedy decoding one hemistich at a time against one-shot greedy decoding (they must agree;
    bf16 can make a rare greedy tie break differently). Uses a few eval prompts."""
    cfg = pv.sc_cfg
    _need_env(pv)
    _need_gpu("the M4 self-check")
    out = m4_dir(pv, "eval") / "selfcheck.json"
    pv.run([HERE / "m4_decode.py", "selfcheck", "--split", "eval", "--subset", str(prompts), "--out", out,
            "--base-model", cfg.base_model], label="M4 self-check")
    res = json.loads(out.read_text())
    ok = res["argmax_agreement_pct"] == 100.0 and res["greedy_identical_pct"] >= 75.0 and res["logit_max_abs_diff"] < 0.5
    print("self-check:", "PASSED" if ok else "FAILED: do not start the long runs; send me selfcheck.json", res)
    return res


@_phase('m4/samplecheck')
def m4_check_sampling(pv: PVConfig, subset: int = 300) -> None:
    """Optional: the sample mode of m4_decode.py is the paper's decoding. One sample on a seeded subset of test prompts,
    compared with the baseline's E1 poems (generate_shaer_test.py) on the same prompts: the official metrics must agree
    within the sampling noise shown."""
    e1 = pathlib.Path(pv.drive_base, "shaer-repro-yehia", "runs", "e1", "scored.jsonl")
    if not e1.exists():
        raise RuntimeError(f"no E1 poems at {e1}: run E1 in the baseline notebook first")
    d = m4_dir(pv, "samplecheck")
    gens = m4_generate(pv, "test", "i1_n1", subset=subset, num_samples=1, folder="samplecheck")
    scored = gens.parent / "scored.jsonl"
    _score(pv, gens, scored, "scoring the sample-mode poems")
    pv.run([HERE / "m4_report.py", "compare", "--a", scored, "--b", e1, "--out", d / "compare.json"],
           label="sample mode vs E1", quiet=False)


@_phase('m4')
def m4_test_from_choice(pv: PVConfig) -> None:
    """The final test runs chosen by the pre-registered rule on the development sweep (m4/eval/m4_summary.json)."""
    p = m4_dir(pv, "eval") / "m4_summary.json"
    if not p.exists():
        raise RuntimeError("summarise the development sweep first: pv.m4_score(pvc, 'eval'); pv.m4_report(pvc, 'eval')")
    ch = json.loads(p.read_text()).get("choice")
    if not ch:
        raise RuntimeError("the development summary made no choice (no search run within the budget of best-of-4)")
    print(f"test runs: search k={ch['k']} R={ch['rounds']} and best-of-{ch['best_of_n']}")
    m4_test(pv, ch["k"], ch["rounds"], ch["best_of_n"])


def _score_runs(pv: PVConfig, d: pathlib.Path, label: str, workers: int = 0) -> None:
    """Score every run folder under d/runs: the repository's scorer (BiLSTM meter, line count; CPU) and the M3 verifier
    in both reading modes, into d/m3. Finished rows are kept, so it can run while generation goes on."""
    extra = ["--workers", str(workers)] if workers else []
    for run in sorted(p for p in (d / "runs").iterdir() if (p / "generations.jsonl").exists()):
        gens = run / "generations.jsonl"
        meta = run / "generations.meta.json"
        finished = (meta.exists() and json.loads(meta.read_text()).get("runs")) or not meta.exists()
        if not finished:
            print(f"{run.name}: generation has not finished yet; scoring what is there")
        scored = run / "scored.jsonl"
        if _lines(scored) < _lines(gens):
            scored.unlink(missing_ok=True)          # the scorer rewrites its file: score the whole run again
            _score(pv, gens, scored, f"scoring {run.name}")
        pv.run([HERE / "m3_rescore.py", "score", "--scored", scored, "--system", run.name, "--out-dir", d / "m3", *extra],
               label=f"{label} verifier {run.name}")


@_phase('m4')
def m4_score(pv: PVConfig, split: str, workers: int = 0) -> None:
    """Score every M4 run of a split: the repository's scorer (BiLSTM meter, line count; CPU) and the M3 verifier in both
    reading modes. A CPU runtime is enough (and cheaper); finished rows are kept, so it can run while generation goes on."""
    _need_env(pv)
    _score_runs(pv, m4_dir(pv, split), "M4", workers)


@_phase('m4', 'm1/splits/train.jsonl')
def m4_report(pv: PVConfig, split: str, reps: int = 1000) -> str:
    """Summarise the M4 runs of a split (m4/<split>/m4_report.md and m4_summary.json on Drive)."""
    _need_env(pv)
    d = m4_dir(pv, split)
    train = pv.root / "m1" / "splits" / "train.jsonl"
    args = [HERE / "m4_report.py", "summarize", "--dir", d, "--reps", str(reps)]
    if train.exists():
        args += ["--train", train]
    pv.run(args, label=f"M4 summary {split}", quiet=True)
    text = (d / "m4_report.md").read_text(encoding="utf-8")
    summary = json.loads((d / "m4_summary.json").read_text(encoding="utf-8"))
    bad = {k: v for k, v in summary.get("checks", {}).items() if any(x not in (0, None) for x in v.values())}
    print("integrity checks:", "all 0" if not bad else f"FAILED for {', '.join(bad)} (see the last table)")
    if split == "eval" and summary.get("choice"):
        print("pre-registered choice for the test runs:", summary["choice"])
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:
        print(text)
    return text


# ---------------------------------------------------------------------------------------------- M5 (idea 2: self-training)
# Folders (Drive):
#   shaer-prosody/m5/labels/{train,eval,test}.jsonl      the verifier's labels of the corpus (rawī, meter, rhyme)
#   shaer-prosody/m5/data/<variant>-<hash>/              training sets (train / eval / test Parquet) for sft/train_sft.py
#   shaer-prosody/m5/variants.json                      variant -> its data folder, training run, starting adapter
#   shaer-prosody/m5/test/runs/<system>/               single test samples of each system, scored like M4
#   shaer-prosody/m5/selftrain/r<n>/                    2c: the search run on train prompts, the accepted poems
#   shaer-repro-yehia-prosody/sft_outputs/train/prosody_m5_<variant>/   the checkpoints (never the baseline's folder)
M5_SOURCE = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
GRPO_ADAPTER = ("Shaer-AI/Shaer-adapters-grpo-short1k-no-trio-v2", "best-checkpoint")   # the authors' best GRPO stage
M5_BASELINES = {"e1_shaer_adapter": "e1_released", "e3_retrained_yehia": "e3_retrained"}


@dataclasses.dataclass
class TrainConfig(sc.Config):
    """The baseline's training configuration with extra environment variables for sft/train_sft.py (its data set, the
    adapter to start from, the number of epochs): sc.start_training builds the subprocess environment from cfg.env()."""
    extra_env: dict = dataclasses.field(default_factory=dict)

    def env(self, **extra: str) -> dict:
        return super().env(**{**self.extra_env, **extra})


def m5_dir(pv: PVConfig) -> pathlib.Path:
    d = pv.root / "m5"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _variants(pv: PVConfig) -> dict:
    p = m5_dir(pv) / "variants.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _variant_update(pv: PVConfig, variant: str, **fields) -> dict:
    vs = _variants(pv)
    vs[variant] = {**vs.get(variant, {}), **{k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in fields.items()}}
    (m5_dir(pv) / "variants.json").write_text(json.dumps(vs, ensure_ascii=False, indent=1))
    return vs[variant]


def _variant(pv: PVConfig, variant: str) -> dict:
    v = _variants(pv).get(variant)
    if not v:
        raise RuntimeError(f"no M5 variant {variant!r} yet: build it first (pv.m5_build(pvc, {variant!r}))")
    if v.get("data"):                    # recorded on Drive or on the VM in an earlier session: the copy this step can use
        v = {**v, "data": str(_need(pv, v["data"]))}
    return v


@_phase('m5/labels', 'm5/variants.json')
def m5_relabel(pv: PVConfig, workers: int = 0) -> pathlib.Path:
    """The verifier's labels of every poem of the corpus (CPU; about 1-2 h with 12 CPUs, longer on a 2-CPU runtime;
    resumable). Labels made by another verifier or lexicon are set aside and made again."""
    _need_env(pv)
    out = m5_dir(pv) / "labels"
    t0 = sc._start("M5: labelling the corpus with the verifier", "about 116k poems; every finished chunk is on Drive")
    pv.run([HERE / "m5_data.py", "relabel", "--source", M5_SOURCE, "--out", out,
            *(["--workers", str(workers)] if workers else [])], label="M5 relabel")
    sc._finish(t0, f"M5 labels -> {out}")
    return out


@_phase('m5/labels', 'm5/data', 'm5/variants.json')
def m5_build(pv: PVConfig, variant: str, base: str = "", accepted: pathlib.Path | None = None, mix: float = 1.0) -> pathlib.Path:
    """Build a training set: "2a" (the rawī in every prompt), "2b" (2a on verifier-clean poems), or a 2c round (use
    m5_selftrain, which calls this). Returns its folder; a set already built from the same inputs is reused."""
    _need_env(pv)
    record = m5_dir(pv) / "data" / f"{variant}.path"
    record.parent.mkdir(parents=True, exist_ok=True)
    args = [HERE / "m5_data.py", "build", "--variant", variant, "--source", M5_SOURCE, "--labels", m5_dir(pv) / "labels",
            "--out-root", m5_dir(pv) / "data", "--record", record]
    if base:
        args += ["--base", _variant(pv, base)["data"]]
    if accepted:
        args += ["--accepted", accepted, "--mix", str(mix)]
    pv.run(args, label=f"M5 build {variant}")
    data = pathlib.Path(record.read_text().strip())
    _variant_update(pv, variant, data=data, root=_variant(pv, base).get("root", base) if base else variant)
    print(f"M5 {variant}: {data}")
    return data


@_phase('m5/variants.json')
def m5_train(pv: PVConfig, variant: str, epochs: float | None = None, init_from: pathlib.Path | str | None = None,
             gradient_checkpointing: bool = False, save_minutes: float = 10, eval_batch_size="auto") -> None:
    """Fine-tune a variant with the baseline's training recipe and runner (sc.start_training: QLoRA on Yehia-7B-preview,
    live progress, resume, verified checkpoints on Drive). Checkpoints go to
    MyDrive/shaer-repro-yehia-prosody/sft_outputs/train/prosody_m5_<variant>/, a resume checkpoint about every 10
    minutes. Re-running the cell resumes. epochs: None = the recipe's (2); init_from: an adapter folder to continue."""
    _need_gpu("M5 training")
    v = _variant(pv, variant)
    cfg = training_config(pv, f"{TRAIN_PREFIX}m5_{variant.replace('-', '_')}")
    extra = {"SFT_DATASET_ID": str(v["data"])}
    if init_from:
        extra.update(SFT_INIT_ADAPTER_PATH=str(init_from), SFT_INIT_FROM_ADAPTER="1")
    if epochs:
        extra["SFT_NUM_TRAIN_EPOCHS"] = str(epochs)
    tcfg = TrainConfig(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}, extra_env=extra)
    for p in (tcfg.run_dir, tcfg.runs, tcfg.sft_outputs):       # the guard of prosody_sc_config, once more
        if _inside(p, baseline_root(tcfg)):
            raise RuntimeError(f"refusing to train into the baseline folder: {p}")
    _variant_update(pv, variant, run_name=tcfg.run_name, run_dir=tcfg.run_dir, init_from=str(init_from or ""),
                    epochs=epochs)
    print(f"M5 {variant}: data {v['data']}\n  checkpoints -> {tcfg.run_dir}" + (f"\n  starting from {init_from}" if init_from else ""))
    sc.checkpoint_report(tcfg)
    sc.start_training(tcfg, gradient_checkpointing=gradient_checkpointing, save_minutes=save_minutes,
                      eval_batch_size=eval_batch_size)


@_phase('m5/variants.json')
def m5_best_checkpoint(pv: PVConfig, variant: str) -> pathlib.Path:
    """The best checkpoint (lowest eval loss) of a finished M5 training run."""
    v = _variant(pv, variant)
    run_dir = pathlib.Path(v.get("run_dir") or "")
    summary = run_dir / "run_summary.json"
    if not summary.exists():
        raise RuntimeError(f"M5 {variant}: training has not finished (no {summary}); run pv.m5_train(pvc, {variant!r})")
    best = json.loads(summary.read_text()).get("best_eval_checkpoint") or ""
    for cand in (pathlib.Path(best), run_dir / pathlib.Path(best).name):
        if best and cand.is_dir():
            return cand
    raise RuntimeError(f"M5 {variant}: the best checkpoint {best!r} is not in {run_dir}")


@_phase('m5/variants.json')
def m5_generate(pv: PVConfig, system: str, split: str = "test", subset: int = 0) -> pathlib.Path:
    """One sample per prompt for a system: a trained variant ("2a", "2b", "2c_r1", ...: its best checkpoint, prompts with
    the rawī) or "grpo_v2" (the authors' best GRPO adapter, the paper's prompts). GPU; resumable."""
    _need_env(pv)
    _need_gpu("M5 generation")
    if system == "grpo_v2":
        name, data, adapter = "grpo_v2", M5_SOURCE, ["--adapter-repo", GRPO_ADAPTER[0], "--adapter-subfolder", GRPO_ADAPTER[1]]
    else:
        name, data = f"m5_{system}", _variant(pv, system)["data"]
        adapter = ["--adapter-path", m5_best_checkpoint(pv, system)]
    _need(pv, m5_dir(pv) / split)                # resume from the poems already on Drive
    out = m5_dir(pv) / split / "runs" / name / "generations.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    pv.run([HERE / "m4_decode.py", "sample", "--split", split, "--subset", str(subset), "--num-samples", "1", "--out", out,
            "--base-model", pv.sc_cfg.base_model, "--dataset-id", data, *adapter], label=f"M5 generation {name}")
    return out


@_phase('m5/test')
def m5_import_baselines(pv: PVConfig) -> list[str]:
    """Copy the baseline's finished E1 (released adapter) and E3 (re-trained) test poems into m5/test/runs, read only
    from the baseline's folder, so M5 compares against them on the same prompts without generating them again."""
    import shutil
    done = []
    for name, scored in baseline_runs(pv).items():
        if name not in M5_BASELINES:
            continue
        dst = m5_dir(pv) / "test" / "runs" / M5_BASELINES[name]
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("generations.jsonl", "scored.jsonl"):
            src = scored.parent / f
            if src.exists() and (not (dst / f).exists() or _lines(dst / f) != _lines(src)):
                shutil.copyfile(src, dst / f)
        done.append(M5_BASELINES[name])
    print("baselines in m5/test/runs:", ", ".join(done) or "none found (run E1 / E3 in the baseline notebook)")
    return done


@_phase()
def m5_score(pv: PVConfig, split: str = "test", workers: int = 0) -> None:
    """Score the M5 runs like M4's: the repository's scorer and the verifier (both modes). CPU is enough."""
    _need_env(pv)
    _score_runs(pv, _need(pv, m5_dir(pv) / split), "M5", workers)


@_phase('m5/test', 'm5/labels/test.jsonl', 'm1/splits/train.jsonl')
def m5_report(pv: PVConfig, baseline: str = "", reps: int = 1000) -> str:
    """Single samples of every M5 system against the baseline (default: E3 if present, else E1), paired on the test
    prompts: m5/test/m5_report.md and m5_summary.json."""
    _need_env(pv)
    d = m5_dir(pv) / "test"
    args = [HERE / "m5_report.py", "summarize", "--dir", d, "--labels", m5_dir(pv) / "labels" / "test.jsonl", "--reps", str(reps)]
    if baseline:
        args += ["--baseline", baseline]
    train = pv.root / "m1" / "splits" / "train.jsonl"
    if train.exists():
        args += ["--train", train]
    pv.run(args, label="M5 summary", quiet=True)
    return _show(d / "m5_report.md")


@_phase('m5/variants.json')
def m5_selftrain(pv: PVConfig, round_no: int, base: str, prompts: int = 4000, k: int = 1, rounds: int = 8,
                 mix: float = 1.0, epochs: float = 1.0) -> None:
    """One round of 2c (rejection-sampling fine-tuning). With the adapter of `base` ("2a", "2b", or the previous round,
    e.g. "2c_r1"): (1) the hemistich search on a seeded sample of train prompts (GPU); (2) the quality gate (CPU);
    (3) the training set: the accepted poems mixed with as many original train rows; (4) one epoch of training that
    starts from `base`'s adapter (GPU). Each step resumes; a finished step is skipped."""
    _need_env(pv)
    variant = f"2c_r{round_no}"
    root = _variant(pv, base).get("root", base)
    root_data = _variant(pv, root)["data"]
    adapter = m5_best_checkpoint(pv, base)
    d = _need(pv, m5_dir(pv) / "selftrain" / f"r{round_no}")
    gens = d / "runs" / "search" / "generations.jsonl"
    gens.parent.mkdir(parents=True, exist_ok=True)
    print(f"2c round {round_no}: prompts from {root} ({root_data}), adapter of {base} ({adapter})")
    if _lines(gens) < prompts:
        _need_gpu("the 2c search")
        pv.run([HERE / "m4_decode.py", "search", "--split", "train", "--subset", str(prompts), "--subset-seed",
                str(2026 + round_no), "--k", str(k), "--rounds", str(rounds), "--out", gens, "--base-model",
                pv.sc_cfg.base_model, "--dataset-id", root_data, "--adapter-path", adapter], label=f"2c r{round_no} search")
    accepted = d / "accepted.jsonl"
    pv.run([HERE / "m5_data.py", "accept", "--generations", gens, "--source", M5_SOURCE, "--base", root_data,
            "--out", accepted], label=f"2c r{round_no} gate")
    summary = json.loads(accepted.with_suffix(".summary.json").read_text())
    print(f"  accepted {summary['accepted']:,} of {summary['poems']:,} poems: {summary['reasons']}")
    if summary["accepted"] == 0:
        raise RuntimeError("no poem passed the gate: nothing to train on")
    m5_build(pv, variant, base=root, accepted=accepted, mix=mix)
    _variant_update(pv, variant, root=root, started_from=base)
    m5_train(pv, variant, epochs=epochs, init_from=adapter)


# ---------------------------------------------------------------------------------------------- M6 (quality study)
def m6_dir(pv: PVConfig) -> pathlib.Path:
    d = pv.root / "m6"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _show(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:
        print(text)
    return text


@_phase('m6', 'm4/test', 'm5/test')
def m6_collect(pv: PVConfig, m4_conditions: tuple = ()) -> None:
    """Gather the final systems (one poem per test prompt): the M4 test conditions (all, or those whose file name
    contains one of m4_conditions, e.g. ("i0_single", "best_of_2", "i2_k1_r8")), every M5 test run, and the human
    poems of the same prompts. Summarise M4's test runs first (pv.m4_report(pvc, "test")): it writes the selections."""
    _need_env(pv)
    args = [HERE / "m6_quality.py", "collect", "--out", m6_dir(pv), "--m4", m4_dir(pv, "test"), "--m5", m5_dir(pv) / "test"]
    if m4_conditions:
        args += ["--m4-conditions", ",".join(m4_conditions)]
    pv.run(args, label="M6 collect", quiet=False)


@_phase('m6')
def m6_judge(pv: PVConfig, names: tuple = (), limit_rows: int = 300, dry_run: bool = False, max_cost_usd: float = 0.0,
             workers: int = 6) -> None:
    """The paper's LLM judge (five metrics, 1-5; paid, through OpenRouter: one poem is five calls) on the same
    `limit_rows` prompts of every collected system (or of `names`). Start with dry_run=True (no call), then a small
    limit_rows; finished poems are kept, so a larger limit later only adds poems. max_cost_usd stops each system's run."""
    _need_env(pv)
    d = m6_dir(pv)
    pv.run([HERE / "m6_quality.py", "judge-input", "--out", d, *(["--names", ",".join(names)] if names else [])],
           label="M6 judge input", quiet=True)
    for p in sorted((d / "judge_input").glob("*.jsonl")):
        if names and p.stem not in names:
            continue
        cmd = [pv.repo / "evaluation" / "run_llm_judge.py", "--generations", p, "--out-dir", d / "judge" / p.stem,
               "--key", p.stem, "--workers", str(workers), "--limit-rows", str(limit_rows)]
        if dry_run:
            cmd.append("--dry-run")
        if max_cost_usd:
            cmd += ["--max-cost-usd", str(max_cost_usd)]
        code = _resuming(lambda: sc.run([pv.py, "-u", *cmd], label=f"M6 judge {p.stem}", cwd=pv.repo, env=pv.sc_cfg.env(),
                                        check=False, drop=(), stall_after=pv.sc_cfg.stall_seconds, stop_on_interrupt=True),
                         f"M6 judge {p.stem}")
        if code == 4:
            print(f"■ {p.stem}: stopped at the cost limit; run again with a higher max_cost_usd to continue")
        elif code not in (0,):
            print(f"■ {p.stem}: the judge did not finish (exit {code}); run the cell again, it resumes")
        if dry_run:
            break


@_phase('m6')
def m6_human_sheet(pv: PVConfig, names: tuple, prompts: int = 40) -> pathlib.Path:
    """A blinded rating sheet (m6/human/rating_sheet.csv; open it in Google Sheets) for `prompts` prompts x the given
    systems, and its key (rating_key.json; keep it from the raters). Give each rater a copy; put the filled copies in
    m6/human/filled/ as CSV."""
    _need_env(pv)
    pv.run([HERE / "m6_quality.py", "human-sheet", "--out", m6_dir(pv), "--names", ",".join(names), "--prompts",
            str(prompts)], label="M6 human sheet", quiet=False)
    (m6_dir(pv) / "human" / "filled").mkdir(parents=True, exist_ok=True)
    return m6_dir(pv) / "human" / "rating_sheet.csv"


@_phase('m6')
def m6_human_score(pv: PVConfig) -> None:
    """Unblind and summarise every filled sheet in m6/human/filled/*.csv."""
    _need_env(pv)
    sheets = sorted((m6_dir(pv) / "human" / "filled").glob("*.csv"))
    if not sheets:
        raise RuntimeError(f"no filled sheet in {m6_dir(pv) / 'human' / 'filled'} (download each rater's copy as CSV there)")
    pv.run([HERE / "m6_quality.py", "human-score", "--out", m6_dir(pv), *[x for s in sheets for x in ("--sheet", s)]],
           label="M6 human ratings", quiet=False)


@_phase('m6', 'm1/splits/train.jsonl')
def m6_report(pv: PVConfig, baseline: str = "", reps: int = 1000) -> str:
    """The quality table of the final systems: judge metrics, strict validity, copying and diversity, human ratings;
    paired differences to the baseline. m6/m6_report.md and m6_summary.json."""
    _need_env(pv)
    d = m6_dir(pv)
    args = [HERE / "m6_quality.py", "summarize", "--out", d, "--reps", str(reps)]
    if baseline:
        args += ["--baseline", baseline]
    train = pv.root / "m1" / "splits" / "train.jsonl"
    if train.exists():
        args += ["--train", train]
    pv.run(args, label="M6 summary", quiet=True)
    return _show(d / "m6_report.md")


def verify_poem(text: str, base_meter: str, form: str = "تام", requested_lines: int | None = None,
                ignore_diacritics: bool = False) -> dict:
    """Verify one poem in this Python process (the verifier needs only the standard library) and print a readable report.
    base_meter / form use the Shaer labels, e.g. verify_poem(poem, "الطويل")."""
    import sys
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from prosody.lexicon import Lexicon
    from prosody.meters import shaer_target
    from prosody.verify import Verifier
    global _VERIFIER
    if "_VERIFIER" not in globals() or _VERIFIER[0] != ignore_diacritics:
        _VERIFIER = (ignore_diacritics, Verifier(Lexicon.load(SHIPPED_LEXICON), ignore_diacritics=ignore_diacritics))
    meter, fk = shaer_target(base_meter, form)
    r = _VERIFIER[1].poem(text, meter, fk, requested_lines).summary()
    print(f"meter {base_meter} ({form}): {'OK' if r['meter_ok'] else 'FAILS'} ({r['bayts_meter_ok']}/{r['bayts']} bayts) | "
          f"rhyme: {'OK' if r['rhyme_ok'] else 'FAILS'} (rawī {r['rawi']!r}"
          + (f", refrain {' '.join(r['refrain'])!r}" if r["refrain"] else "") + ")"
          + (f" | length: {'OK' if r['count_ok'] else 'FAILS'} ({r['lines']}/{r['requested_lines']})" if requested_lines else ""))
    for b in r["per_bayt"]:
        mark = "✔" if b["ok"] else "✘"
        print(f" {mark} bayt {b['i'] + 1}: {b['sadr']['text']}  *  {b['ajuz']['text']}")
        for h in ("sadr", "ajuz"):
            x = b[h]
            if x["ok"]:
                print(f"      {h}: {x['feet']}")
            else:
                n = x.get("nearest") or {}
                print(f"      {h}: {x['reason']}" + (f"; nearest pattern {n.get('edits')} edit(s) away, first in foot "
                                                       f"{n.get('first_edit_foot')} ({n.get('foot')})" if n else ""))
    for i in r["issues"]:
        print(f"   issue: {i['kind']} in bayts {[k + 1 for k in i['bayts']]}: {i['detail']}")
    return r
