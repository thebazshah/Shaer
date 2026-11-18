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

Baseline outputs are only read, from where the baseline notebook wrote them (MyDrive/shaer-repro-<model>/runs/...).

Checkpoint policy for later training runs (none in M0 or idea 1): save about every 10 minutes of wall-clock time, keep the
two most recent checkpoints plus the best one.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess

import shaer_colab as sc

HERE = pathlib.Path(__file__).resolve().parent
DRIVE_FOLDER = "shaer-prosody"
RELEASED_TEST_ROWS = 3481
# baseline run folders -> system names used in the M0 tables
SYSTEM_NAMES = {("yehia", "e0"): "e0_released_shaer", ("yehia", "e1"): "e1_shaer_adapter", ("yehia", "e2"): "e2_yehia_base",
                ("yehia", "e3"): "e3_retrained_yehia", ("allam", "e1"): "e1_adapter_on_allam", ("allam", "e2"): "e2_allam_base",
                ("allam", "e3"): "e3_retrained_allam"}


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


def make_config(sc_cfg: "sc.Config", drive_base: str = "/content/drive/MyDrive") -> PVConfig:
    pv = PVConfig(sc_cfg=sc_cfg, drive_base=drive_base)
    for d in (pv.m0, pv.logs):
        d.mkdir(parents=True, exist_ok=True)
    print("prosody-verifier outputs ->", pv.root)
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
    for tag in ("yehia", "allam"):
        runs = pathlib.Path(pv.drive_base, f"shaer-repro-{tag}", "runs")
        for step in ("e0", "e1", "e2", "e3"):
            name = SYSTEM_NAMES.get((tag, step))
            scored, gens = runs / step / "scored.jsonl", runs / step / "generations.jsonl"
            if not name or not scored.exists():
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
    rows = []
    for name in ("e0", "e0_bare", "reference", "reference_bare"):
        p = d / f"bilstm_{name}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            v = r["verifier_any_form"]
            rows.append(f"| {name} | {r['bayts']:,} | {v['bilstm_ok_pct']}% | {v['verifier_ok_pct']}% | {v['kappa']} |")
    if rows:
        out += ["## Bayts on the requested meter: BiLSTM vs verifier", "",
                "| poems | bayts | BiLSTM | verifier | kappa |", "|---|---:|---:|---:|---:|"] + rows + [""]
    text = "\n".join(out)
    (d / "m1_report.md").write_text(text, encoding="utf-8")
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
