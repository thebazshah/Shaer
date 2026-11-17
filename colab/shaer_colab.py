"""Everything the Colab notebook does, as plain Python functions (no shell).

The notebook imports this module and calls one function per step. The repository's training / generation / scoring scripts
run in their own isolated Python environment (`/content/venv`, with the repository's pinned package versions), so they are
started as subprocesses of that environment's interpreter. Commands are always argument lists, never shell strings.

Every long step reports progress and is watched, because Colab sometimes hangs:
  * library progress bars (downloads, tokenising, loading shards) are shown as throttled bar lines,
  * a heartbeat line every ~20 s says how long the step has run, how long it has been silent, and what the GPU is doing,
  * a step that is silent AND idle (GPU below 10 % and almost no CPU) for `cfg.stall_minutes` is stopped with a clear message;
    re-running the cell resumes where it left off,
  * training runs in the background and `watch_training` shows a live bar, loss, ETA and GPU, and can restart a hung run.

Flow selection (`make_config`):
  BASE_MODEL == "Navid-AI/Yehia-7B-preview"          -> paper flow        E0, E1, E2, E3 as in the paper
  BASE_MODEL == "ALLaM-AI/ALLaM-7B-Instruct-preview" -> substitute flow   E1 skipped (the released adapter was trained on Yehia),
                                                        E2/E3 measure ALLaM; the paper's rows are labelled Yehia-based
  anything else                                      -> ValueError
"""
from __future__ import annotations

import collections
import dataclasses
import json
import os
import pathlib
import queue
import re
import shutil
import signal
import subprocess
import threading
import time

YEHIA = "Navid-AI/Yehia-7B-preview"
ALLAM = ("ALLaM-AI/ALLaM-7B-Instruct-preview", "humain-ai/ALLaM-7B-Instruct-preview")
PAPER_NUMBERS = {  # base-meter accuracy %, meter conformity, exact count %, count adherence (paper Tables 3, 20, 21)
    "Shaer": (95.17, 0.9064, 83.40, 0.9792),
    "Yehia-7B": (26.49, 0.1494, 44.47, 0.7680),
}
METER_MODEL_ID = "Shaer-AI/4BiLSTM-meter-classification-pytorch"


# ------------------------------------------------------------------------------------------------ configuration
@dataclasses.dataclass
class Config:
    base_model: str
    paper_flow: bool
    tag: str
    run_e1: bool
    force_e1: bool
    run_name: str
    num_samples: int
    gen_batch_size: str
    drive_root: str
    repo_url: str
    branch: str
    hf_token: str = ""
    stall_minutes: float = 15          # a step silent AND idle for this long is treated as stuck
    repo_dir: str = "/content/Shaer"
    venv_dir: str = "/content/venv"

    @property
    def py(self) -> str:
        return str(pathlib.Path(self.venv_dir, "bin", "python"))

    @property
    def runs(self) -> pathlib.Path:
        return pathlib.Path(self.drive_root, "runs")

    @property
    def logs(self) -> pathlib.Path:
        return pathlib.Path(self.drive_root, "logs")

    @property
    def sft_outputs(self) -> pathlib.Path:
        return pathlib.Path(self.drive_root, "sft_outputs")

    @property
    def run_dir(self) -> pathlib.Path:
        return self.sft_outputs / "train" / self.run_name

    @property
    def stall_seconds(self) -> int:
        return int(self.stall_minutes * 60)

    def env(self, **extra: str) -> dict:
        """Environment for the subprocesses (the Hugging Face token never goes on a command line)."""
        e = dict(os.environ)
        e.update(HF_TOKEN=self.hf_token, BASE_MODEL=self.base_model, SFT_BASE_MODEL_ID=self.base_model,  # train_sft.py reads the latter
                 METER_MODEL_ID=METER_MODEL_ID, SFT_PUSH_TO_HUB="0",   # never create or push to a Hugging Face repo
                 PAPER_FLOW="1" if self.paper_flow else "0", PYTHONUNBUFFERED="1")
        e.update(extra)
        return e


def make_config(*, base_model: str, drive_base: str = "/content/drive/MyDrive", run_name: str = "paper_repro_a100",
                num_samples: int = 1, gen_batch_size="auto", force_e1: bool = False, stall_minutes: float = 15,
                repo_url: str = "https://github.com/thebazshah/Shaer.git", branch: str = "dev", hf_token: str = "") -> Config:
    """Pick the flow from the base model. Exactly two models are supported."""
    if base_model == YEHIA:
        # PAPER FLOW: the setup of the Shaer paper. E0, E1 (released adapter), E2 (untuned Yehia) and E3 (re-training) all
        # apply, and the results are compared with the paper's Yehia-based numbers.
        paper_flow, tag, run_e1 = True, "yehia", True
        print("PAPER FLOW: Yehia-7B-preview. E0, E1, E2, E3 run as in the paper.")
    elif base_model in ALLAM:
        # SUBSTITUTE FLOW: ALLaM is the model Yehia was fine-tuned from (same architecture, byte-identical tokenizer). The
        # released adapter was trained on Yehia's weights, so E1 is skipped; E2/E3 measure ALLaM, not Yehia.
        paper_flow, tag, run_e1 = False, "allam", bool(force_e1)
        print("SUBSTITUTE FLOW: ALLaM-7B-Instruct-preview. "
              + ("E1 is FORCED (exploratory: the adapter was trained on Yehia)." if run_e1
                 else "E1 is skipped (the released adapter was trained on Yehia).")
              + " E2/E3 are NOT the paper's Yehia numbers.")
    else:
        raise ValueError(f"unsupported BASE_MODEL {base_model!r}: use {YEHIA!r} or {ALLAM[0]!r}")
    cfg = Config(base_model=base_model, paper_flow=paper_flow, tag=tag, run_e1=run_e1, force_e1=bool(force_e1), run_name=run_name,
                 num_samples=int(num_samples), gen_batch_size=str(gen_batch_size), stall_minutes=float(stall_minutes),
                 drive_root=str(pathlib.Path(drive_base, f"shaer-repro-{tag}")), repo_url=repo_url, branch=branch, hf_token=hf_token)
    for sub in (cfg.runs, cfg.logs, cfg.sft_outputs):
        sub.mkdir(parents=True, exist_ok=True)
    print("base model:", base_model, "| Drive folder:", cfg.drive_root, f"| a silent, idle step is stopped after {stall_minutes} min")
    return cfg


# ------------------------------------------------------------------------------------------------ small helpers
def _fmt(seconds: float) -> str:
    m, s = divmod(int(max(seconds, 0)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _bar(fraction: float, width: int = 20) -> str:
    n = int(round(min(max(fraction, 0.0), 1.0) * width))
    return "█" * n + "░" * (width - n)


def _start(title: str, expect: str | None = None) -> float:
    print(f"\n▶ {title}" + (f"   (expect {expect})" if expect else ""), flush=True)
    return time.time()


def _finish(t0: float, title: str) -> None:
    print(f"✔ {title}: done in {_fmt(time.time() - t0)}", flush=True)


def _gpu_stats() -> tuple | None:
    """(utilisation %, memory used MiB, memory total MiB) of the first GPU, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split("\n")[0]
        util, used, total = (int(x.strip()) for x in out.split(","))
        return util, used, total
    except Exception:
        return None


def _gpu_text(stats: tuple | None) -> str:
    return "no GPU info" if not stats else f"GPU {stats[0]}% busy, {stats[1] / 1024:.1f}/{stats[2] / 1024:.1f} GB"


def _cpu_seconds(pid: int) -> float | None:
    """CPU time used so far by a process (utime + stime), from /proc."""
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def _count_lines(path: pathlib.Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# library progress bars (tqdm): "desc:  37%|███▋      | 27000/109070 [03:41<07:20, 311 examples/s]"
_TQDM = re.compile(r"^\s*(?P<desc>[^\r\n]*?)\s*:?\s*(?P<pct>\d{1,3})%\|[^|]*\|\s*(?P<n>[\d.,]+\s?[A-Za-z]*)/(?P<total>[\d.,]+\s?[A-Za-z]*)"
                   r"\s*\[(?P<el>[^<\]]*)(?:<(?P<eta>[^,\]]*))?")


class _Bars:
    """Turn the flood of tqdm updates into a few readable bar lines (first, every 10 %, every 30 s, and the end)."""

    def __init__(self) -> None:
        self.state: dict = {}

    def render(self, line: str) -> str | None:
        m = _TQDM.match(line)
        if not m:
            return None
        desc = (m["desc"] or "progress").strip()[-30:] or "progress"
        pct = int(m["pct"])
        last_pct, last_t = self.state.get(desc, (-1, 0.0))
        if pct < last_pct:
            last_pct = -1                       # a new bar that reuses the description
        now = time.time()
        if pct == 100 and last_pct == 100:
            return ""
        if not (last_pct < 0 or pct == 100 or pct >= last_pct + 10 or (pct > last_pct and now - last_t >= 30)):
            return ""                           # swallowed
        self.state[desc] = (pct, now)
        eta = f"  ETA {m['eta'].strip()}" if m["eta"] and m["eta"].strip() not in ("?", "") else ""
        return f"  {desc:<30} [{_bar(pct / 100)}] {pct:3d}%  {m['n'].strip()}/{m['total'].strip()}  elapsed {m['el'].strip()}{eta}"


# ------------------------------------------------------------------------------------------------ running commands
class StepStalled(RuntimeError):
    """A step was silent and idle for too long and was stopped."""


def run(cmd: list, *, label: str = "step", cwd=None, env=None, quiet: bool = False, check: bool = True, tail: int | None = None,
        drop: tuple = ("warn",), stall_after: int | None = None, heartbeat: int = 20, progress=None) -> int:
    """Run a command (no shell), stream its output into the cell, and watch it.

    Shown as it is produced: normal lines, plus tqdm bars as throttled bar lines. Hidden: lines containing a word of `drop`.
    While the command is silent a heartbeat line is printed every `heartbeat` seconds (elapsed, silence, GPU). `progress`, if
    given, returns (done, total) and is shown as a bar at each heartbeat (used for steps that print nothing but write a file).
    If the command makes no progress for `stall_after` seconds AND looks idle (GPU < 10 % and almost no CPU), it is stopped and
    StepStalled is raised. Raises RuntimeError on a non-zero exit when `check`.
    """
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:                  # text mode: \r and \n both end a line, so progress arrives as it happens
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    bars, shown = _Bars(), (collections.deque(maxlen=tail) if tail else None)
    start = last_activity = time.time()
    last_cpu, last_tick, last_progress = _cpu_seconds(proc.pid), start, None
    while True:
        try:
            raw = lines.get(timeout=heartbeat)
        except queue.Empty:
            now = time.time()
            if progress:
                try:
                    done = progress()
                except Exception:
                    done = None
                if done and done[0] != last_progress:
                    last_progress, last_activity = done[0], now           # the output file is growing: not stuck
                    print(f"  … {label}  [{_bar(done[0] / max(done[1], 1))}] {100 * done[0] / max(done[1], 1):5.1f}%  "
                          f"{done[0]:,}/{done[1]:,}  elapsed {_fmt(now - start)}", flush=True)
                    continue
            idle_for = now - last_activity
            gpu, cpu = _gpu_stats(), _cpu_seconds(proc.pid)
            cpu_share = ((cpu - last_cpu) / max(now - last_tick, 1e-9)) if (cpu is not None and last_cpu is not None) else None
            last_cpu, last_tick = cpu, now
            busy = (gpu is not None and gpu[0] >= 10) or (cpu_share is not None and cpu_share >= 0.2)
            print(f"  … {label}: running {_fmt(now - start)} | silent for {_fmt(idle_for)} | {_gpu_text(gpu)}"
                  + ("" if cpu_share is None else f" | CPU {100 * cpu_share:.0f}%") + (" | busy" if busy else " | idle"), flush=True)
            if stall_after and idle_for >= stall_after:
                if busy:
                    print(f"    (no output for {_fmt(idle_for)}, but the process is busy, so it is not treated as stuck)", flush=True)
                    continue
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise StepStalled(f"{label} was silent and idle for {_fmt(idle_for)} and looks stuck (Colab sometimes hangs). "
                                  "I stopped it. Run this cell again: it resumes where it left off "
                                  "(generation continues from the file on Drive, training from the last checkpoint).")
            continue
        if raw is None:
            break
        last_activity = time.time()              # any output, even a hidden line, shows the process is alive
        low = raw.lower()
        if quiet or not raw.strip() or any(d in low for d in drop):
            continue
        bar_line = bars.render(raw)
        if bar_line is not None:
            if bar_line:
                print(bar_line, flush=True)
            continue
        if shown is not None:
            shown.append(raw)
        else:
            print(raw, end="", flush=True)
    code = proc.wait()
    if shown:
        print("".join(shown), end="", flush=True)
    if check and code != 0:
        raise RuntimeError(f"{label} failed (exit code {code}): {' '.join(str(c) for c in cmd)}")
    return code


# ------------------------------------------------------------------------------------------------ code + environment
def sync_repo(cfg: Config) -> None:
    """Point the repository's output folders at Google Drive, so checkpoints and results survive a recycled runtime."""
    repo = pathlib.Path(cfg.repo_dir)
    for link, target in ((repo / "sft" / "outputs", cfg.sft_outputs), (repo / "runs", cfg.runs)):
        assert str(link).startswith(str(repo) + os.sep), link          # never touch anything outside the repository folder
        if link.is_symlink():
            link.unlink()
        elif link.exists():                     # a real folder (e.g. results of an earlier run on this VM): keep it, never delete it
            aside = link.with_name(f"{link.name}.local-{int(time.time())}")
            link.rename(aside)
            print(f"note: {link} already existed as a real folder on this VM; moved it to {aside} (nothing was deleted)")
        link.symlink_to(target, target_is_directory=True)
    head = subprocess.run(["git", "-C", str(repo), "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
    print("code:", head, "| outputs ->", cfg.drive_root)


def check_environment(cfg: Config) -> None:
    """Print package versions and the GPU, and say which experiments this GPU supports."""
    t0 = _start("checking the training environment and the GPU", "a few seconds")
    run([cfg.py, pathlib.Path(cfg.repo_dir, "colab", "env_check.py")], label="environment check", env=cfg.env(), stall_after=300)
    run([cfg.py, "-m", "bitsandbytes"], label="bitsandbytes check", env=cfg.env(), tail=5, check=False, stall_after=300)
    _finish(t0, "environment check")


def check_hf_access(cfg: Config) -> None:
    """Fail early (before any long step) if the base model or its slow tokenizer cannot be loaded."""
    t0 = _start(f"checking access to {cfg.base_model} and loading its tokenizer", "under a minute")
    run([cfg.py, pathlib.Path(cfg.repo_dir, "colab", "hf_access_check.py")], label="access check", env=cfg.env(),
        stall_after=cfg.stall_seconds)
    _finish(t0, "access check")


# ------------------------------------------------------------------------------------------------ experiments
def _score_and_summarise(cfg: Config, key: str) -> None:
    repo = pathlib.Path(cfg.repo_dir)
    generations, scored = cfg.runs / key / "generations.jsonl", cfg.runs / key / "scored.jsonl"
    total = _count_lines(generations)
    t0 = _start(f"scoring {total:,} poems with the repository's evaluator (CPU, deterministic)", "a few minutes")
    # The evaluator writes its output file row by row, so the file's line count is a progress bar.
    run([cfg.py, "score_meter_count.py", "--input-jsonl", f"../runs/{key}/generations.jsonl",
         "--output-jsonl", f"../runs/{key}/scored.jsonl"], label=f"scoring {key.upper()}", cwd=repo / "evaluation",
        env=cfg.env(CUDA_VISIBLE_DEVICES=""), stall_after=cfg.stall_seconds, heartbeat=10,
        progress=lambda: (_count_lines(scored), total))
    run([cfg.py, "evaluation/summarize_structural.py", "--scored", f"runs/{key}/scored.jsonl", "--out", f"runs/{key}/summary.json"],
        label="summary", cwd=repo, env=cfg.env(), quiet=True, stall_after=300)
    _finish(t0, f"scoring {key.upper()}")


def run_e0(cfg: Config) -> None:
    """E0: re-score the authors' released poems with the repository's own evaluator."""
    t0 = _start("E0: exporting the authors' released poems and re-scoring them", "about 3-5 minutes")
    run([cfg.py, "evaluation/export_released_generations.py", "--out", "runs/e0/generations.jsonl"], label="export of released poems",
        cwd=cfg.repo_dir, env=cfg.env(), stall_after=cfg.stall_seconds)
    _score_and_summarise(cfg, "e0")
    _finish(t0, f"E0 (results in {cfg.runs / 'e0'})")


def run_generation(cfg: Config, step: str, extra_args: list | None = None) -> None:
    """E1 (released adapter), E2 (untuned base model) or E3 (adapter trained here): generate, score, summarise."""
    extra_args = list(extra_args or [])
    if step == "e1":
        if cfg.paper_flow:
            adapter_args, what = [], "the released Shaer adapter on Yehia-7B-preview"
        elif cfg.run_e1:
            adapter_args, what = ["--allow-adapter-mismatch"], f"the released adapter on {cfg.base_model} (EXPLORATORY, not the paper's E1)"
        else:
            print(f"E1 skipped: the released adapter was trained on Yehia-7B-preview, so it is not valid on {cfg.base_model}.")
            print("(set FORCE_E1 = True in the configuration cell for an exploratory run)")
            return
    elif step == "e2":
        adapter_args, what = ["--no-adapter"], f"the untuned {cfg.base_model}"
    elif step == "e3":
        summary = cfg.run_dir / "run_summary.json"
        if not summary.exists():
            raise RuntimeError("training has not finished (no run_summary.json yet): run the training steps first")
        best = json.loads(summary.read_text())["best_eval_checkpoint"]
        if not best or not pathlib.Path(best).is_dir():
            raise RuntimeError(f"checkpoint folder is missing: {best}")
        adapter_args, what = ["--adapter-path", best], f"the adapter trained here ({pathlib.Path(best).name})"
    else:
        raise ValueError(f"unknown step {step!r} (use 'e1', 'e2' or 'e3')")
    t0 = _start(f"{step.upper()}: generating the test poems with {what}",
                "model download + load first (about 5-10 min on a fresh runtime), then the progress bar with an ETA")
    run([cfg.py, "-u", "evaluation/generate_shaer_test.py", *adapter_args, "--base-model", cfg.base_model,
         "--out", f"runs/{step}/generations.jsonl", "--num-samples", str(cfg.num_samples), "--batch-size", cfg.gen_batch_size,
         *extra_args], label=f"{step.upper()} generation", cwd=cfg.repo_dir, env=cfg.env(), stall_after=cfg.stall_seconds)
    _score_and_summarise(cfg, step)
    _finish(t0, f"{step.upper()} (results in {cfg.runs / step})")


# ------------------------------------------------------------------------------------------------ E3: training
def train_smoke_test(cfg: Config) -> None:
    """Train 6 steps with the real 4-bit configuration and report the measured peak GPU memory."""
    t0 = _start("smoke test: 6 training steps with the real 4-bit setup", "about 10-15 min, mostly the 15 GB model download and loading")
    samples, stop = [], threading.Event()

    def sample() -> None:
        while not stop.is_set():
            stats = _gpu_stats()
            if stats:
                samples.append(stats[1])
            stop.wait(2)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        run([cfg.py, "-u", "train_sft.py", "--mode", "sanity", "--run_name", f"sanity_{cfg.run_name}"], label="smoke test",
            cwd=pathlib.Path(cfg.repo_dir, "sft"), env=cfg.env(), stall_after=cfg.stall_seconds)
    finally:
        stop.set()
        thread.join(timeout=5)
    stats = _gpu_stats()
    if samples and stats:
        print(f"peak GPU memory during the smoke test: {max(samples)} MiB of {stats[2]} MiB")
    _finish(t0, "smoke test")


def _pid_file(cfg: Config) -> pathlib.Path:
    return cfg.logs / f"train_{cfg.run_name}.pid"


def _training_pid(cfg: Config) -> int | None:
    """Pid of the running training process of this run, or None (the pid file lives on Drive, so check the process itself)."""
    try:
        pid = int(_pid_file(cfg).read_text())
        cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    return pid if b"train_sft.py" in cmdline and cfg.run_name.encode() in cmdline else None


def start_training(cfg: Config, gradient_checkpointing: bool = True) -> None:
    """Start the full training run in the background; it logs to Drive and resumes from the newest complete checkpoint."""
    if _training_pid(cfg):
        print("training is already running: use sc.watch_training(cfg) to follow it")
        return
    if (cfg.run_dir / "run_summary.json").exists():
        print(f"run '{cfg.run_name}' already finished (run_summary.json exists): go to the generation step. "
              "Use a new RUN_NAME to train again.")
        return
    env = cfg.env(SFT_RESUME_MODE="auto")        # resumes from the newest complete checkpoint of this RUN_NAME on Drive
    if not gradient_checkpointing:
        env["SFT_GRADIENT_CHECKPOINTING"] = "0"  # faster, uses more memory (see the notebook text before using it on 40 GB)
    log = cfg.logs / f"train_{cfg.run_name}.log"
    with open(log, "ab") as handle:
        proc = subprocess.Popen([cfg.py, "-u", "train_sft.py", "--mode", "train", "--run_name", cfg.run_name],
                                cwd=pathlib.Path(cfg.repo_dir, "sft"), env=env, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)      # survives the notebook cell, and a closed browser tab
    _pid_file(cfg).write_text(str(proc.pid))
    resumed = sorted(p.name for p in cfg.run_dir.glob("checkpoint-*")) if cfg.run_dir.exists() else []
    print(f"▶ training started (pid {proc.pid})" + (f", resuming from the newest complete of {resumed}" if resumed else ""))
    print(f"  log: {log}\n  follow it live with: sc.watch_training(cfg)")


def stop_training(cfg: Config) -> None:
    """Stop the background training process (checkpoints are kept; start_training resumes from the newest complete one)."""
    pid = _training_pid(cfg)
    if not pid:
        print("training is not running")
        return
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    for _ in range(20):
        time.sleep(1)
        if not _training_pid(cfg):
            break
    else:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    print("training stopped")


_TRAIN_TQDM = re.compile(r"(\d{1,3})%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<\]]*)<([^,\]]*),\s*([^\]]*)\]")


def _training_progress(cfg: Config) -> dict:
    """Progress of the background run, read from its log and metrics file."""
    info = dict(step=None, total=None, eta=None, rate=None, loss=None, lr=None, epoch=None, eval_loss=None, log_age=None, message=None)
    log = cfg.logs / f"train_{cfg.run_name}.log"
    text = ""
    if log.exists():
        info["log_age"] = time.time() - log.stat().st_mtime
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 40000))
            text = f.read().decode("utf-8", "replace")
        plain = [l.strip() for l in re.split(r"[\r\n]+", text) if l.strip() and "warn" not in l.lower() and "%|" not in l]
        info["message"] = plain[-1][:150] if plain else None
    # total optimiser steps = steps per epoch x epochs, from the effective config and the logged (step, epoch)
    rows = []
    metrics = cfg.run_dir / "metrics.jsonl"
    if metrics.exists():
        with open(metrics, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 30000))
            for raw in f.read().decode("utf-8", "replace").splitlines()[1:]:
                try:
                    rows.append(json.loads(raw))
                except ValueError:
                    pass
    train_rows = [r for r in rows if r.get("mode") == "train" and "loss" in r]
    eval_rows = [r for r in rows if r.get("mode") == "eval" and "eval_loss" in r]
    if train_rows:
        last = train_rows[-1]
        info.update(step=int(last["global_step"]), loss=last.get("loss"), lr=last.get("learning_rate"), epoch=last.get("epoch"))
        try:
            conf = json.loads((cfg.run_dir / "effective_runtime_config.json").read_text())
            epochs, max_steps = float(conf.get("num_train_epochs", 0)), int(conf.get("max_steps", -1))
            if max_steps > 0:
                info["total"] = max_steps
            elif last.get("epoch"):
                info["total"] = int(round(last["global_step"] / float(last["epoch"]) * epochs))
        except (OSError, ValueError, KeyError):
            pass
    if eval_rows:
        info["eval_loss"] = eval_rows[-1]["eval_loss"]
    for m in reversed(list(_TRAIN_TQDM.finditer(text))):         # the per-step bar of the Trainer (updates every step)
        step, total = int(m.group(2)), int(m.group(3))
        if info["total"] is None or abs(total - info["total"]) <= 1:
            info.update(step=max(step, info["step"] or 0), total=total, eta=m.group(5).strip(), rate=m.group(6).strip())
            break
    return info


def _render_training(info: dict, alive: bool, finished: bool, gpu: tuple | None) -> str:
    state = "FINISHED" if finished else ("RUNNING" if alive else "NOT running")
    out = [f"training: {state} | {_gpu_text(gpu)}"]
    if info["step"] is not None and info["total"]:
        frac = info["step"] / info["total"]
        out.append(f"  [{_bar(frac, 30)}] {100 * frac:5.1f}%  step {info['step']:,}/{info['total']:,}"
                   + (f"  ETA {info['eta']}" if info["eta"] else "") + (f"  ({info['rate']})" if info["rate"] else ""))
    elif info["message"]:
        out.append(f"  starting up: {info['message']}")
    if info["loss"] is not None:
        out.append(f"  train loss {info['loss']:.4f}" + (f" | lr {info['lr']:.2e}" if info["lr"] else "")
                   + (f" | epoch {info['epoch']:.2f}" if info["epoch"] else "")
                   + (f" | last eval loss {info['eval_loss']:.4f}" if info["eval_loss"] else ""))
    if info["log_age"] is not None:
        out.append(f"  log last updated {_fmt(info['log_age'])} ago")
    return "\n".join(out)


def training_status(cfg: Config) -> None:
    """One-shot status: running or not, progress bar, loss, ETA, GPU, last log line, checkpoints."""
    info, alive = _training_progress(cfg), _training_pid(cfg) is not None
    print(_render_training(info, alive, (cfg.run_dir / "run_summary.json").exists(), _gpu_stats()))
    print("  checkpoints:", sorted(p.name for p in cfg.run_dir.glob("checkpoint-*")) if cfg.run_dir.exists() else "none yet")


def watch_training(cfg: Config, every: int = 20, auto_restart: bool = False, max_restarts: int = 3) -> None:
    """Live view of the background run; returns when training finishes, dies, or looks stuck.

    Stopping this cell (the stop button) is safe: training keeps running in the background. Stuck means: the log has not changed
    for `cfg.stall_minutes` and the GPU is idle. With `auto_restart=True` a stuck run is stopped and restarted from its last
    checkpoint (up to `max_restarts` times), which is what you want when Colab hangs while you are away.
    """
    try:
        from IPython.display import clear_output
    except Exception:
        clear_output = None
    began, restarts = time.time(), 0
    try:
        while True:
            info, alive = _training_progress(cfg), _training_pid(cfg) is not None
            finished, gpu = (cfg.run_dir / "run_summary.json").exists(), _gpu_stats()
            if clear_output:
                clear_output(wait=True)
            print(_render_training(info, alive, finished, gpu))
            if finished:
                print("✔ training finished: run the generation step (7c)")
                return
            if not alive:
                print("✖ the training process is not running. Run sc.start_training(cfg): it resumes from the newest complete checkpoint.")
                return
            # Silence is measured from the newer of the last log write and this process's own start (the pid file is written at
            # start), so a freshly restarted process is not judged by the old, hung process's log.
            try:
                since_start = time.time() - _pid_file(cfg).stat().st_mtime
            except OSError:
                since_start = time.time() - began
            age = since_start if info["log_age"] is None else min(info["log_age"], since_start)
            if age >= cfg.stall_seconds and not (gpu and gpu[0] >= 10):
                print(f"⚠ no log output for {_fmt(age)} and " + ("the GPU is idle" if gpu else "no GPU activity could be read")
                      + ": the run looks stuck (Colab sometimes hangs).")
                if auto_restart and restarts < max_restarts:
                    restarts += 1
                    print(f"  restarting from the last checkpoint (restart {restarts}/{max_restarts}) ...")
                    stop_training(cfg)
                    start_training(cfg)
                    began = time.time()
                else:
                    print("  Run sc.stop_training(cfg) and then sc.start_training(cfg) to resume from the last checkpoint.")
                    return
            time.sleep(every)
    except KeyboardInterrupt:
        print("stopped watching; training continues in the background (sc.watch_training(cfg) to look again)")


# ------------------------------------------------------------------------------------------------ results
def results_table(cfg: Config) -> str:
    """Your runs against the paper, as a markdown table (also saved to Drive as results.md)."""
    suffix = "" if cfg.paper_flow else ", Yehia-based"       # the paper's rows are Yehia numbers even when we run the substitute
    base = cfg.base_model.split("/")[-1]

    def row(name: str, s: dict) -> str:
        return (f"| {name} | {s['base_meter_accuracy_pct']:.2f} | {s['meter_conformity']:.4f} | "
                f"{s['exact_count_accuracy_pct']:.2f} | {s['count_adherence']:.4f} |")

    lines = ["| System | Base-meter acc. (%) | Meter conformity | Exact count (%) | Count adherence |", "|---|---:|---:|---:|---:|"]
    for n, (a, b, c, d) in PAPER_NUMBERS.items():
        lines.append(f"| {n} (paper{suffix}) | {a:.2f} | {b:.4f} | {c:.2f} | {d:.4f} |")
    if cfg.paper_flow:
        steps = [("e0", "E0 released poems, re-scored"), ("e1", "E1 released adapter"), ("e2", "E2 untuned Yehia-7B"),
                 ("e3", "E3 re-trained here")]
    else:
        steps = [("e0", "E0 released poems, re-scored"), ("e1", f"E1 EXPLORATORY: Yehia adapter on {base}"),
                 ("e2", f"E2 untuned {base} (substitute baseline)"), ("e3", f"E3 re-trained on {base}")]
    for key, name in steps:
        f = cfg.runs / key / "summary.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if "overall" in d:
            lines.append(row(name, d["overall"]))
        else:
            lines.append(row(name + " (first sample)", d["first_sample"]))
            lines.append(row(name + f" (best of {d['samples_per_prompt']} by meter score)", d["best_of_k_by_meter"]))
    table = "\n".join(lines)
    if not cfg.paper_flow:
        table += f"\n\nSubstitute base model ({base}): E2/E3 are not comparable one-to-one with the paper's Yehia-based numbers."
    (cfg.runs / "results.md").write_text(table + "\n")
    print(table)
    print("\nBase-meter accuracy here = first-bayt prediction, the definition that reproduces the paper's own tables;")
    print("the majority-vote variant described in the paper's text is in each summary.json (base_meter_accuracy_majority_pct).")
    return table


def plot_training_curves(cfg: Config) -> None:
    """Training curves from the newest complete checkpoint's trainer_state.json, saved to Drive as training_curves.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    states = sorted(cfg.run_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        print("no checkpoint yet")
        return
    history = json.loads(states[-1].read_text())["log_history"]
    train = [(x["step"], x["loss"]) for x in history if "loss" in x]
    evals = [(x["step"], x["eval_loss"]) for x in history if "eval_loss" in x]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if train:
        ax.plot(*zip(*train), lw=0.8, label="train loss")
    if evals:
        ax.plot(*zip(*evals), "o-", ms=3, label="eval loss")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy")
    ax.grid(alpha=0.3)
    ax.legend()
    out = cfg.runs / "training_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)
    if evals:
        print("best eval loss so far:", min(e[1] for e in evals), "| paper: best 2.2074 at step 3000, test 2.1932")
    try:                                         # show the figure in the notebook
        from IPython.display import Image, display
        display(Image(str(out)))
    except Exception:
        pass
