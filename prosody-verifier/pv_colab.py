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

Checkpoint policy for later training runs (none in M0 or idea 1): save about every 10 minutes of wall-clock time, keep the
two most recent checkpoints plus the best one.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import subprocess

import shaer_colab as sc

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
    def root(self) -> pathlib.Path:
        return pathlib.Path(self.drive_base, DRIVE_FOLDER)

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
        """A watched subprocess in the isolated environment (progress, heartbeat, stall detection)."""
        sc.run([self.py, *args], label=label, env=self.sc_cfg.env(), stall_after=self.sc_cfg.stall_seconds, **kw)


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
    print("prosody-verifier outputs ->", pv.root)
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


def m1_prepare_splits(pv: PVConfig) -> pathlib.Path:
    """Download the Shaer train / eval / test splits as small JSONL files on Drive (skipped when present)."""
    _need_env(pv)
    out = m1_dir(pv) / "splits"
    if all(_lines(out / f"{s}.jsonl") for s in ("train", "eval", "test")):
        print("splits already on Drive:", out)
        return out
    pv.run([HERE / "prepare_splits.py", "--out-dir", out], label="downloading the Shaer splits")
    return out


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
    t0 = sc._start("M3 strict re-scoring", "roughly 30-60 minutes per system and reading mode on a 2-core CPU runtime; "
                   "resumable")
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


def m4_test(pv: PVConfig, k: int, rounds: int, samples: int) -> None:
    """The final runs on the full test split (3,481 prompts): the chosen search configuration and best-of-`samples`.
    Choose them on the development sweep first (m4_report(pvc, "eval") prints the choice its rule makes)."""
    m4_generate(pv, "test", f"i2_k{k}_r{rounds}", k=k, rounds=rounds)
    m4_generate(pv, "test", f"i1_n{samples}", num_samples=samples)


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


def m4_score(pv: PVConfig, split: str, workers: int = 0) -> None:
    """Score every M4 run of a split: the repository's scorer (BiLSTM meter, line count; CPU) and the M3 verifier in both
    reading modes. A CPU runtime is enough (and cheaper); finished rows are kept, so it can run while generation goes on."""
    _need_env(pv)
    d = m4_dir(pv, split)
    extra = ["--workers", str(workers)] if workers else []
    for run in sorted(p for p in (d / "runs").iterdir() if (p / "generations.jsonl").exists()):
        gens = run / "generations.jsonl"
        meta = run / "generations.meta.json"
        finished = meta.exists() and json.loads(meta.read_text()).get("runs")
        if not finished:
            print(f"{run.name}: generation has not finished yet; scoring what is there")
        scored = run / "scored.jsonl"
        if _lines(scored) < _lines(gens):
            scored.unlink(missing_ok=True)          # the scorer rewrites its file: score the whole run again
            _score(pv, gens, scored, f"scoring {run.name}")
        pv.run([HERE / "m3_rescore.py", "score", "--scored", scored, "--system", run.name, "--out-dir", d / "m3", *extra],
               label=f"M4 verifier {run.name}")


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
