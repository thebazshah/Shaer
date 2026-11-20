"""Everything the Colab notebook does, as plain Python functions (no shell).

The notebook imports this module and calls one function per step. The repository's training / generation / scoring scripts
run in their own isolated Python environment (`/content/venv`, with the repository's pinned package versions), so they are
started as subprocesses of that environment's interpreter. Commands are always argument lists, never shell strings.

Every long step reports progress and is watched, because Colab sometimes hangs:
  * library progress bars (downloads, tokenising, loading shards) are shown as throttled bar lines,
  * a heartbeat line every ~20 s says how long the step has run, how long it has been silent, and what the GPU is doing,
  * a step that is silent AND idle (GPU below 10 % and almost no CPU) for `cfg.stall_minutes` is stopped with a clear message;
    re-running the cell resumes where it left off,
  * training runs in the notebook cell with live progress (or detached with background=True, followed by `watch_training`).

Flow selection (`make_config`):
  BASE_MODEL == "Navid-AI/Yehia-7B-preview"          -> paper flow        E0, E1, E2, E3 as in the paper
  BASE_MODEL == "ALLaM-AI/ALLaM-7B-Instruct-preview" -> substitute flow   E1 skipped (the released adapter was trained on Yehia),
                                                        E2/E3 measure ALLaM; the paper's rows are labelled Yehia-based
  anything else                                      -> ValueError
"""
from __future__ import annotations

# Bump both whenever this module or the notebook changes, and set the same values at the top of reproduce_shaer_a100.ipynb
# (title line and NOTEBOOK_VERSION in section 1): the notebook compares them and warns when they differ.
__version__ = "1.13.0"
__updated__ = "2026-11-20 22:17 UTC"

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
PAPER_JUDGE_MODEL = "qwen/qwen3-235b-a22b-2507"           # the judge of the paper (strict v2 prompts, temperature 0)
PAPER_JUDGE_SCORES = {  # description adherence, meaning, fluency, coherence, poeticness (evaluation/results2.md)
    "Shaer": (4.75, 3.72, 4.31, 3.69, 3.82),
    "Yehia-7B": (4.42, 3.75, 4.21, 3.82, 3.73),
}
JUDGE_METRICS = ("description_adherence", "meaning", "fluency", "coherence", "poeticness")


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
    openrouter_key: str = ""           # only for the LLM judge (E4); never printed
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
        if self.openrouter_key:
            e["OPENROUTER_API_KEY"] = self.openrouter_key
        e.update(extra)
        return e


def make_config(*, base_model: str, drive_base: str = "/content/drive/MyDrive", run_name: str = "paper_repro_a100",
                num_samples: int = 1, gen_batch_size="auto", force_e1: bool = False, stall_minutes: float = 15,
                repo_url: str = "https://github.com/thebazshah/Shaer.git", branch: str = "dev", hf_token: str = "",
                openrouter_key: str = "") -> Config:
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
                 num_samples=int(num_samples), gen_batch_size=str(gen_batch_size), stall_minutes=float(stall_minutes), openrouter_key=openrouter_key,
                 drive_root=str(pathlib.Path(drive_base, f"shaer-repro-{tag}")), repo_url=repo_url, branch=branch, hf_token=hf_token)
    _require_drive(cfg.drive_root)
    for sub in (cfg.runs, cfg.logs, cfg.sft_outputs):
        sub.mkdir(parents=True, exist_ok=True)
    print("base model:", base_model, "| Drive folder:", cfg.drive_root, f"| a silent, idle step is stopped after {stall_minutes} min")
    return cfg


def _require_drive(path) -> None:
    """Refuse to work on a Drive path while Google Drive is not mounted.

    Unmounted, /content/drive is an ordinary empty folder on the VM: writing there silently puts checkpoints on local disk (lost
    with the session), and a training run finds none of the checkpoints that are on Drive and starts again at step 0.
    """
    if str(path).startswith("/content/drive") and not os.path.ismount("/content/drive"):
        raise RuntimeError("Google Drive is NOT mounted (sc.flush_drive() or a dropped connection unmounts it). Run section 1 "
                           "(drive.mount) again, then section 2. If drive.mount says the mountpoint already contains files (written while "
                           "Drive was unmounted), use Runtime > Disconnect and delete runtime, then run sections 1-3 on the new one.")


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


def _gpu_name() -> str:
    """Name of the first GPU ("NVIDIA A100-SXM4-40GB", "NVIDIA L4", ...), or "" without one."""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.split("\n")[0].strip()
    except Exception:
        return ""


def _gpu_text(stats: tuple | None) -> str:
    return "no GPU info" if not stats else f"GPU {stats[0]}% busy, {stats[1] / 1024:.1f}/{stats[2] / 1024:.1f} GB"


def _cpu_seconds(pid: int) -> float | None:
    """CPU time used so far by a process (utime + stime), from /proc."""
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def _gpu_jobs() -> list:
    """Other processes of this project that are using the GPU: training runs and generation runs, as (pid, description).

    A process counts only if one of its ARGUMENTS is the script (so a shell or editor that merely mentions the name does not).
    """
    jobs = []
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        scripts = {os.path.basename(a) for a in args[1:]}
        if "train_sft.py" in scripts and "--mode" in args:
            name = args[args.index("--run_name") + 1] if "--run_name" in args and args.index("--run_name") + 1 < len(args) else None
            jobs.append((int(proc.name), "training" + (f" (run '{name}')" if name else "")))
        elif "generate_shaer_test.py" in scripts:
            jobs.append((int(proc.name), "generation"))
    return jobs


def _warn_if_gpu_shared(doing: str) -> None:
    """Two jobs on one GPU make both slower, and 'auto' batch size shrinks because it only sees the memory left over."""
    jobs = _gpu_jobs()
    if not jobs:
        return
    stats = _gpu_stats()
    print(f"⚠ another job is already using this GPU: " + ", ".join(f"{what} [pid {pid}]" for pid, what in jobs)
          + (f" ({stats[1] / 1024:.1f} GB in use, {stats[0]}% busy)" if stats else "") + ".")
    print(f"  Running {doing} at the same time makes BOTH slower (the GPU is time-shared) and the generation batch smaller, because the "
          "automatic batch size only sees the memory that is left. Let one finish first, or stop the other one; a stopped "
          "training run resumes from its last checkpoint with sc.start_training(cfg) (use the config of that run: same BASE_MODEL "
          "and RUN_NAME).", flush=True)


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
    """Turn the flood of tqdm updates into a few readable bar lines (first, every 10 %, every 30 s, and the end).

    Rate and ETA are measured here from the updates themselves (median of the recent per-update rates), not taken from tqdm:
    after a resume the Trainer's bar jumps from 0 to the resumed step in one update, which makes tqdm's average rate far too fast
    and its ETA grow for hours. The median ignores that one jump; until three updates are seen no ETA is shown.
    """

    def __init__(self) -> None:
        self.state: dict = {}
        self.samples: dict = {}

    @staticmethod
    def _number(text: str) -> float | None:
        m = re.match(r"\s*([\d.,]+)\s*([kKMG]?)", text or "")
        if not m:
            return None
        value = float(m.group(1).replace(",", ""))
        return value * {"": 1, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]

    def _measured(self, desc: str, n: float | None, total: float | None, now: float) -> str:
        if n is None:
            return ""
        samples = self.samples.setdefault(desc, collections.deque(maxlen=40))
        if samples and n < samples[-1][1]:
            samples.clear()                         # a new bar that reuses the description
        if not samples or n > samples[-1][1]:
            samples.append((now, n))
        rates = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(samples, list(samples)[1:]) if b[0] > a[0]]
        if len(rates) < 3:
            return ""
        rate = sorted(rates)[len(rates) // 2]       # items per second
        if rate <= 0:
            return ""
        speed = f"{1 / rate:.1f} s/it" if rate < 1 else f"{rate:,.1f} it/s"
        eta = f"  ETA {_fmt((total - n) / rate)}" if total and total > n else ""
        return f"  ({speed}){eta}"

    def render(self, line: str) -> str | None:
        m = _TQDM.match(line)
        if not m:
            return None
        desc = (m["desc"] or "progress").strip()[-30:] or "progress"
        pct = int(m["pct"])
        measured = self._measured(desc, self._number(m["n"]), self._number(m["total"]), time.time())   # every update counts
        last_pct, last_t, last_n = self.state.get(desc, (-1, 0.0, None))
        if pct < last_pct:
            last_pct = -1                       # a new bar that reuses the description
        now = time.time()
        if pct == 100 and last_pct == 100:
            return ""
        # also when the count moved and 30 s passed: on long bars 1 % can take many minutes (1 % of 4,236 steps on an L4 is ~13 min)
        if not (last_pct < 0 or pct == 100 or pct >= last_pct + 10 or (m["n"] != last_n and now - last_t >= 30)):
            return ""                           # swallowed
        self.state[desc] = (pct, now, m["n"])
        return f"  {desc:<30} [{_bar(pct / 100)}] {pct:3d}%  {m['n'].strip()}/{m['total'].strip()}  elapsed {m['el'].strip()}{measured}"


# ------------------------------------------------------------------------------------------------ running commands
class StepStalled(RuntimeError):
    """A step was silent and idle for too long and was stopped."""


def run(cmd: list, *, label: str = "step", cwd=None, env=None, quiet: bool = False, check: bool = True, tail: int | None = None,
        drop: tuple = ("warn",), stall_after: int | None = None, heartbeat: int = 20, progress=None, log_path=None,
        on_start=None, stop_on_interrupt: bool = False) -> int:
    """Run a command (no shell), stream its output into the cell, and watch it.

    Shown as it is produced: normal lines, plus tqdm bars as throttled bar lines. Hidden: lines containing a word of `drop`.
    While the command is silent a heartbeat line is printed every `heartbeat` seconds (elapsed, silence, GPU). `progress`, if
    given, returns (done, total) and is shown as a bar at each heartbeat (used for steps that print nothing but write a file).
    If the command makes no progress for `stall_after` seconds AND looks idle (GPU < 10 % and almost no CPU), it is stopped and
    StepStalled is raised. Raises RuntimeError on a non-zero exit when `check`.
    `log_path`: also append every output line, unfiltered, to this file. `on_start(proc)`: called right after the process starts.
    `stop_on_interrupt`: when the cell is stopped (KeyboardInterrupt), stop the command too (SIGTERM, then SIGKILL) instead of
    leaving it running.
    """
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=stop_on_interrupt)
    if on_start:
        on_start(proc)
    lines: queue.Queue = queue.Queue()
    log_handle = open(log_path, "a", encoding="utf-8", errors="replace") if log_path else None

    def pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:                  # text mode: \r and \n both end a line, so progress arrives as it happens
            if log_handle:
                log_handle.write(raw)
                log_handle.flush()
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    try:
        return _watch_process(proc, lines, cmd=cmd, label=label, quiet=quiet, check=check, tail=tail, drop=drop,
                              stall_after=stall_after, heartbeat=heartbeat, progress=progress)
    except KeyboardInterrupt:
        if stop_on_interrupt and proc.poll() is None:
            print(f"\n■ stopping {label} ...", flush=True)
            _stop_process_group(proc)
        raise
    finally:
        if log_handle:
            log_handle.close()


def _stop_process_group(proc: subprocess.Popen, grace: int = 30) -> None:
    """SIGTERM the process and its children (it runs in its own session), SIGKILL whatever is left after `grace` seconds."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def _watch_process(proc, lines, *, cmd, label, quiet, check, tail, drop, stall_after, heartbeat, progress) -> int:
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
                if os.getpgid(proc.pid) != os.getpgid(0):
                    _stop_process_group(proc, grace=15)
                else:
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
    _require_drive(cfg.drive_root)
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
    print(code_version(cfg), "| outputs ->", cfg.drive_root)


def code_version(cfg: Config) -> str:
    """One line naming the code that is running: module version and date, git commit, branch and commit time."""
    head = subprocess.run(["git", "-C", str(cfg.repo_dir), "log", "-1", "--format=%h %cd %s", "--date=format-local:%Y-%m-%d %H:%M UTC"],
                          capture_output=True, text=True, env={**os.environ, "TZ": "UTC"}).stdout.strip()
    return f"code: v{__version__} (updated {__updated__}) | branch {cfg.branch} | commit {head or 'unknown'}"


def check_version(notebook_version: str) -> None:
    """Warn loudly when the notebook copy and the downloaded code are not the same version."""
    if notebook_version == __version__:
        print(f"✔ notebook v{notebook_version} matches the code v{__version__} (updated {__updated__})")
    else:
        print(f"⚠ VERSION MISMATCH: this notebook is v{notebook_version}, the downloaded code is v{__version__} (updated {__updated__}).\n"
              "  Open the latest reproduce_shaer_a100.ipynb from GitHub (branch dev), or re-run this cell if the code is older.")


def _colab_runtime_info() -> list[str]:
    """The Colab runtime itself (the notebook's own Python, not the training environment): runtime, OS, Python, GPU driver."""
    import platform
    import sys
    lines = []
    tag = os.environ.get("COLAB_RELEASE_TAG", "")
    os_name = ""
    try:
        for line in pathlib.Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                os_name = line.split("=", 1)[1].strip('"')
    except OSError:
        pass
    lines.append(f"Colab runtime: {tag or 'release tag not available'} | {os_name or platform.platform()} | "
                 f"notebook Python {platform.python_version()}")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                                                     timeout=20).stdout)
        if smi:
            lines.append(f"GPU driver: {smi.splitlines()[0]} | driver supports CUDA up to {cuda.group(1) if cuda else '?'}")
    except (OSError, subprocess.TimeoutExpired):
        lines.append("GPU driver: nvidia-smi not available (no GPU runtime?)")
    return lines


def check_environment(cfg: Config) -> None:
    """Print the Colab runtime (OS, Python, GPU driver), the training environment's Python and package versions (checked against
    requirements-colab.txt) and the GPU, and say which experiments this GPU supports."""
    t0 = _start("checking the training environment and the GPU", "a few seconds")
    print(code_version(cfg))
    for line in _colab_runtime_info():
        print(line)
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
    _warn_if_gpu_shared(f"{step.upper()} generation")
    t0 = _start(f"{step.upper()}: generating the test poems with {what}",
                "model download + load first (about 5-10 min on a fresh runtime), then the progress bar with an ETA")
    run([cfg.py, "-u", "evaluation/generate_shaer_test.py", *adapter_args, "--base-model", cfg.base_model,
         "--out", f"runs/{step}/generations.jsonl", "--num-samples", str(cfg.num_samples), "--batch-size", cfg.gen_batch_size,
         *extra_args], label=f"{step.upper()} generation", cwd=cfg.repo_dir, env=cfg.env(), stall_after=cfg.stall_seconds)
    _score_and_summarise(cfg, step)
    _finish(t0, f"{step.upper()} (results in {cfg.runs / step})")


def run_judge(cfg: Config, step: str = "e3", *, limit_rows: int = 0, workers: int = 6, all_samples: bool = False,
              judge_model: str = PAPER_JUDGE_MODEL, max_cost_usd: float = 0.0, dry_run: bool = False) -> None:
    """E4: judge the poems of a generation step ("e1", "e2" or "e3") with the repository's LLM judge (five metrics, 1-5).

    Uses the paper's judge (`qwen/qwen3-235b-a22b-2507`, strict v2 prompts, temperature 0) through OpenRouter, which is a PAID API:
    one poem costs five calls. Needs the Colab secret OPENROUTER_API_KEY. Recommended order: `dry_run=True` (plan and the exact prompt,
    no call), then `limit_rows=10` (a trial), then the full run. Safe to stop and run again: finished poems are kept, failed ones are
    judged again. Only the poem and its description are sent, never the reference poem. `max_cost_usd` stops the run at a budget
    (the API-reported cost). By default only sample 0 of each prompt is judged (`all_samples=True` for all).
    """
    generations = cfg.runs / step / "generations.jsonl"
    if not generations.exists():
        raise RuntimeError(f"no poems to judge: {generations} does not exist. Run the {step.upper()} generation step first.")
    if not dry_run and not cfg.openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it as a Colab secret named OPENROUTER_API_KEY (key icon in the left sidebar, "
                           "'Notebook access' on), then run the configuration cell again.")
    out_dir = cfg.runs / "e4" / step
    cmd = [cfg.py, "-u", "evaluation/run_llm_judge.py", "--generations", generations, "--out-dir", out_dir, "--key", step,
           "--judge-model", judge_model, "--workers", workers]
    if limit_rows:
        cmd += ["--limit-rows", limit_rows]
    if all_samples:
        cmd += ["--all-samples"]
    if max_cost_usd:
        cmd += ["--max-cost-usd", max_cost_usd]
    if dry_run:
        cmd += ["--dry-run"]
    if judge_model != PAPER_JUDGE_MODEL:
        print(f"note: {judge_model} is not the paper's judge ({PAPER_JUDGE_MODEL}); its scores are not comparable with the paper's.")
    t0 = _start(f"E4: {'previewing' if dry_run else 'judging'} the {step.upper()} poems with {judge_model}",
                None if dry_run else "a bar with rows, calls, rate, ETA and the API-reported cost")
    code = run(cmd, label="E4 judge", cwd=cfg.repo_dir, env=cfg.env(), check=False, drop=(), stall_after=cfg.stall_seconds)
    if code == 0:
        _finish(t0, f"E4 {step.upper()} judge" + ("" if dry_run else f" (results in {out_dir})"))
    elif code == 4:
        print("■ stopped at the cost limit. Run this cell again with a higher max_cost_usd to continue; finished poems are kept.")
    elif code == 3:
        raise StepStalled("the judge's API calls looked stuck and were stopped. Run this cell again: it resumes where it left off.")
    else:
        raise RuntimeError("the judge step did not finish (see the message above). Fix that and run the cell again: it resumes.")


def generation_stats(cfg: Config, step: str = "e2", drive_root: str | None = None) -> None:
    """Why is (or was) a generation step slow? Reads the output of a finished OR still running step from Drive and prints
    the GPU, the batch size, how long the poems are, how many hit the token limit, and the measured speed.

    Pass `drive_root` (for example the other model's folder, ".../shaer-repro-allam") to look at another run, so two runs can be compared.
    """
    root = pathlib.Path(drive_root or cfg.drive_root) / "runs" / step
    gen, meta_file = root / "generations.jsonl", root / "generations.meta.json"
    if not gen.exists():
        print(f"no generations yet in {root}")
        return
    rows = []
    for line in gen.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass                                             # a half-written last line
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    args = meta.get("args", {})
    max_new = int(args.get("max_new_tokens", 768))
    tokens = [int(r.get("gen_tokens", 0)) for r in rows]
    ended = sum(1 for r in rows if r.get("finished_with_eos"))
    capped = sum(1 for r in rows if not r.get("finished_with_eos") and int(r.get("gen_tokens", 0)) >= max_new)
    n = max(len(rows), 1)
    print(f"{step.upper()} in {root}")
    print(f"  model: {args.get('base_model')} | adapter: {meta.get('adapter')} | dtype {args.get('dtype')}, quantize {args.get('quantize')}")
    print(f"  GPU: {meta.get('gpu')} | batch size used: {meta.get('batch_size_used')} "
          f"({'auto' if str(args.get('batch_size', 'auto')).lower() == 'auto' else 'fixed'}) | samples per prompt: {args.get('num_samples')}")
    print(f"  poems so far: {len(rows):,}")
    if rows:
        print(f"  tokens per poem: mean {sum(tokens) / n:.0f}, longest {max(tokens)} | ended with EOS: {100 * ended / n:.0f}% | "
              f"cut off at the {max_new}-token limit: {100 * capped / n:.0f}%")
    seconds = sum(r.get("seconds", 0) for r in meta.get("runs", []))
    if not seconds and meta_file.exists():                   # a run in progress: meta was written when it started
        seconds = max(gen.stat().st_mtime - meta_file.stat().st_mtime, 0)
    if seconds and rows:
        print(f"  measured: {len(rows) / (seconds / 60):.1f} poems/min, {sum(tokens) / seconds:.0f} generated tokens/s over {_fmt(seconds)}")
    bs, gpu = meta.get("batch_size_used"), str(meta.get("gpu"))
    _warn_if_gpu_shared("this step")
    if isinstance(bs, int) and bs < 16:
        print(f"  -> small batch ({bs}): decoding is memory-bound, so poems/min is proportional to the batch size. Stop other GPU work "
              "(training, other notebooks) or restart the runtime so 'auto' can pick a larger batch, then run this step again; "
              "poems already written are kept.")
    if rows and capped / n > 0.5:
        print(f"  -> most poems run to the {max_new}-token limit (the model rarely emits end-of-sequence): every batch then takes the full {max_new} steps.")
    if any(x in gpu for x in ("T4", "L4")):
        print(f"  -> {gpu} has a fraction of an A100's memory bandwidth; expect several times slower generation.")


# ------------------------------------------------------------------------------------------------ E3: training
def train_smoke_test(cfg: Config) -> None:
    """Train 6 steps with the real 4-bit configuration and report the measured peak GPU memory."""
    _warn_if_gpu_shared("the smoke test")
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


def start_training(cfg: Config, gradient_checkpointing: bool = True, save_steps: int = 200,
                   save_minutes: float | str | None = "auto", background: bool = False, auto_restart: bool = True,
                   max_restarts: int = 3, eval_batch_size: int | str = "auto", allow_fresh_start: bool = False) -> None:
    """Run the full training in this cell, printing its progress as it happens; resumes from the newest complete checkpoint.

    The output (step bar with ETA, loss lines, evaluation) streams into the cell and is also appended to the log on Drive. While the
    run prints nothing (model loading, data packing, a checkpoint being written) a heartbeat line every 20 s shows that it is alive
    (elapsed, silence, GPU and CPU use). Stopping the cell stops training cleanly; checkpoints are kept and running this again
    resumes from the newest complete one. If the run is silent AND idle (GPU and CPU) for `cfg.stall_minutes`, it is stopped and,
    with `auto_restart`, restarted from the last checkpoint (up to `max_restarts` times).
    background=True: the old behaviour, a detached process followed with sc.watch_training(cfg).
    allow_fresh_start: this run already recorded training steps but has no complete checkpoint to resume from, it refuses to start
    from step 0 (and prints sc.checkpoint_report) unless this is True.
    eval_batch_size: rows per evaluation batch; "auto" (default) = 32 on a GPU with 70 GB or more (A100/H100 80 GB), 16 with 35 GB
    or more, else 8 (the paper's configuration uses 1). Evaluation over the 3,481 eval rows
    runs every 200 steps and took ~31 min on an L4 at batch size 1. Training is not affected; the reported eval loss shifts slightly,
    because a batch's loss is averaged over all its tokens instead of per row. After an out-of-memory error during evaluation the
    run restarts with half the batch size.

    save_steps: write a resume checkpoint every this many optimiser steps. The default, 200, is the paper's schedule. A smaller value
    (for example 50) ADDS extra checkpoints in between, so an interruption loses fewer steps; evaluation, the probe and the choice of
    the best checkpoint stay on the 200-step schedule, so the results are not affected. Each checkpoint is about 1 GB written to
    Drive and costs some time to write. Only two are kept on Drive: the latest (to resume from) and the best (lowest eval loss so
    far); when the latest is also the best, just one. The previous latest is deleted only after the new one is completely written.
    save_minutes: also write a resume checkpoint once this many minutes of training have passed since the last checkpoint.
    "auto" (default): 20 on an A100 / H100 (fast steps; fewer ~1 GB checkpoints for Drive to upload, and the regular 200-step
    checkpoint comes every ~35 min there anyway), 10 on other GPUs (an L4 at ~27 s/step reaches 200 steps only every ~90 min).
    Like save_steps it only adds resume checkpoints and does not change the results. A number sets it; None or 0 turns it off.
    When the foreground run stops (stop button, finished, or failed), Drive is flushed (sc.flush_drive) so the newest checkpoint is
    uploaded before the session can end, then mounted again.
    """
    if not 1 <= int(save_steps) <= 200:
        raise ValueError("save_steps must be between 1 and 200: the regular 200-step checkpoints (the paper's evaluation schedule) "
                         "are always written, and Hugging Face requires the regular save interval to be a multiple of the evaluation interval")
    if save_minutes == "auto":
        gpu = _gpu_name()
        save_minutes = 20 if any(k in gpu.upper() for k in ("A100", "H100", "H200")) else 10
        print(f"  GPU {gpu or 'unknown'}: resume checkpoint every {save_minutes} minutes (save_minutes='auto')")
    if save_minutes is not None and float(save_minutes) < 0:
        raise ValueError("save_minutes must be positive (or None / 0 to turn the time-based checkpoints off)")
    if _training_pid(cfg):
        print("training is already running (pid %s), e.g. a background run started earlier. Follow it with sc.watch_training(cfg), "
              "or stop it with sc.stop_training(cfg) and run this cell again to train here in the foreground." % _training_pid(cfg))
        return
    if (cfg.run_dir / "run_summary.json").exists():
        print(f"run '{cfg.run_name}' already finished (run_summary.json exists): go to the generation step. "
              "Use a new RUN_NAME to train again.")
        return
    _require_drive(cfg.drive_root)              # unmounted, the run would see no checkpoint and start at step 0 on local disk
    complete = [s for s, _, p in _checkpoints(cfg) if not p]
    logged = _last_logged_step(cfg)
    if not complete and logged > 0 and not allow_fresh_start:
        print(f"✖ NOT STARTING: run '{cfg.run_name}' already trained to step {logged} (metrics.jsonl), but there is no complete checkpoint "
              "to resume from, so training would start again at step 0.")
        checkpoint_report(cfg)
        print("  If the previous session was ended before Drive finished uploading, the checkpoint may be lost. Check the folder above in\n"
              "  drive.google.com (and its Trash). To start over anyway: sc.start_training(cfg, ..., allow_fresh_start=True), or use a new RUN_NAME.")
        return
    if complete:
        print(f"  resuming from checkpoint-{max(complete)} (last step recorded in metrics.jsonl: {logged})")
        if logged - max(complete) > 100:
            print(f"  ⚠ {logged - max(complete)} recorded steps are newer than that checkpoint and will be trained again "
                  "(a newer checkpoint did not reach Drive); see sc.checkpoint_report(cfg)")
    _warn_if_gpu_shared("training")
    env = cfg.env(SFT_RESUME_MODE="auto")        # resumes from the newest complete checkpoint of this RUN_NAME on Drive
    # keep only the latest and the best checkpoint: with load_best_model_at_end, Hugging Face treats save_total_limit=1 as
    # "latest + best" (it never deletes the best one, nor the latest, which is needed to resume)
    env["SFT_SAVE_TOTAL_LIMIT"] = "1"
    if not gradient_checkpointing:
        env["SFT_GRADIENT_CHECKPOINTING"] = "0"  # faster (no recomputed forward pass), uses more memory; same maths
        print("  gradient checkpointing off: faster steps, more GPU memory"
              + ("" if background else " (falls back to on by itself if the GPU runs out of memory)"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation, so the larger activations fit
    if eval_batch_size == "auto":
        stats = _gpu_stats()
        total_gb = stats[2] / 1024 if stats else 0
        eval_batch_size = 32 if total_gb >= 70 else 16 if total_gb >= 35 else 8
    if int(eval_batch_size) < 1:
        raise ValueError("eval_batch_size must be at least 1")
    env["SFT_PER_DEVICE_EVAL_BATCH_SIZE"] = str(int(eval_batch_size))
    if int(eval_batch_size) != 1:
        print(f"  evaluation in batches of {int(eval_batch_size)} rows (the paper's configuration: 1); training is unchanged")
    if int(save_steps) < 200:
        env["SFT_RESCUE_SAVE_STEPS"] = str(int(save_steps))
        print(f"  extra resume checkpoint every {int(save_steps)} steps (plus the regular one every 200)")
    if save_minutes and float(save_minutes) > 0:
        env["SFT_RESCUE_SAVE_MINUTES"] = str(float(save_minutes))
        print(f"  extra resume checkpoint every {float(save_minutes):g} minutes of training (plus the regular one every 200 steps)")
    log = cfg.logs / f"train_{cfg.run_name}.log"
    cmd = [cfg.py, "-u", "train_sft.py", "--mode", "train", "--run_name", cfg.run_name]
    if not background:
        _train_foreground(cfg, cmd, env, log, auto_restart, max_restarts)
        return
    with open(log, "ab") as handle:
        proc = subprocess.Popen(cmd, cwd=pathlib.Path(cfg.repo_dir, "sft"), env=env, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)      # survives the notebook cell, and a closed browser tab
    _pid_file(cfg).write_text(str(proc.pid))
    resumed = sorted(p.name for p in cfg.run_dir.glob("checkpoint-*")) if cfg.run_dir.exists() else []
    print(f"▶ training started (pid {proc.pid})" + (f", resuming from the newest complete of {resumed}" if resumed else ""))
    print(f"  log: {log}\n  follow it live with: sc.watch_training(cfg)")


def _log_has_oom(log: pathlib.Path, tail_bytes: int = 20000, during_eval: bool = False) -> bool:
    """Whether the end of the training log reports a CUDA out-of-memory error (with during_eval: raised inside evaluation)."""
    try:
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            text = f.read().decode("utf-8", "replace").lower()
    except OSError:
        return False
    oom = "out of memory" in text and ("cuda" in text or "torch" in text)
    if not during_eval:
        return oom
    last = text[text.rfind("traceback"):] if "traceback" in text else text
    return oom and any(k in last for k in ("evaluation_loop", "prediction_step", "def evaluate", "in evaluate", "meteraux"))


def _train_foreground(cfg: Config, cmd: list, env: dict, log: pathlib.Path, auto_restart: bool, max_restarts: int) -> None:
    """start_training's default: run train_sft.py in this cell, stream its output, restart it when it hangs."""
    t0 = _start(f"training run '{cfg.run_name}' in this cell",
                "startup (model load, tokenising, packing) takes several minutes, then the step bar appears")
    version = code_version(cfg)
    print("  " + version, flush=True)
    with open(log, "a", encoding="utf-8") as handle:     # so the log on Drive also says which code produced each start
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC | start_training | {version}\n")
    restarts = 0
    try:
        while True:
            resumed = sorted(p.name for p in cfg.run_dir.glob("checkpoint-*")) if cfg.run_dir.exists() else []
            print("  resuming from the newest complete of " + str(resumed) if resumed else "  starting from step 0 (no checkpoint yet)",
                  flush=True)
            try:
                run(cmd, label="training", cwd=pathlib.Path(cfg.repo_dir, "sft"), env=env, stall_after=cfg.stall_seconds,
                    log_path=log, on_start=lambda proc: _pid_file(cfg).write_text(str(proc.pid)), stop_on_interrupt=True)
                break
            except StepStalled as exc:
                if not auto_restart or restarts >= max_restarts:
                    raise
                restarts += 1
                print(f"⚠ {exc}\n  restarting from the last checkpoint (restart {restarts}/{max_restarts}) ...", flush=True)
            except RuntimeError:
                if not _log_has_oom(log):
                    raise
                eval_bs = int(env.get("SFT_PER_DEVICE_EVAL_BATCH_SIZE", "1"))
                if eval_bs > 1 and _log_has_oom(log, during_eval=True):
                    env["SFT_PER_DEVICE_EVAL_BATCH_SIZE"] = str(eval_bs // 2)    # not part of the resume fingerprint
                    print(f"⚠ the GPU ran out of memory during evaluation: restarting from the last checkpoint with evaluation "
                          f"batches of {eval_bs // 2}", flush=True)
                elif env.get("SFT_GRADIENT_CHECKPOINTING") == "0":
                    env["SFT_GRADIENT_CHECKPOINTING"] = "1"  # does not change the maths, and is not part of the resume fingerprint
                    print("⚠ the GPU ran out of memory without gradient checkpointing: restarting with it ON from the last "
                          "checkpoint (slower steps, much less memory)", flush=True)
                elif eval_bs > 1:
                    env["SFT_PER_DEVICE_EVAL_BATCH_SIZE"] = str(eval_bs // 2)
                    print(f"⚠ the GPU ran out of memory: restarting from the last checkpoint with evaluation batches of "
                          f"{eval_bs // 2}", flush=True)
                else:
                    raise
    except KeyboardInterrupt:
        print("■ training stopped. Checkpoints are kept: run this cell again to resume from the newest complete one.", flush=True)
        stopped = True
    else:
        stopped = False
    finally:
        _pid_file(cfg).unlink(missing_ok=True)
        flush_drive()                            # stopped, finished or failed: get the newest checkpoint onto Drive now
    if stopped:
        return
    print("✔ training finished: run the generation step (7c)" if (cfg.run_dir / "run_summary.json").exists()
          else "training process exited", flush=True)
    _finish(t0, "training")


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


def _render_training(info: dict, alive: bool, finished: bool, gpu: tuple | None, cpu_share: float | None = None) -> str:
    state = "FINISHED" if finished else ("RUNNING" if alive else "NOT running")
    out = [f"training: {state} | {_gpu_text(gpu)}" + ("" if cpu_share is None else f" | CPU {100 * cpu_share:.0f}%")]
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


_CKPT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt")


def _checkpoint_problem(path: pathlib.Path) -> str | None:
    """None when the checkpoint can be resumed from (same test as train_sft.latest_complete_checkpoint_step), else why not."""
    missing = [n for n in _CKPT_FILES if not (path / n).exists()]
    if not ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()):
        missing.append("adapter_model.safetensors")
    if missing:
        return "missing " + ", ".join(missing)
    try:
        json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "trainer_state.json is truncated"
    return None


def _checkpoints(cfg: Config) -> list[tuple[int, pathlib.Path, str | None]]:
    if not cfg.run_dir.is_dir():
        return []
    found = []
    for child in cfg.run_dir.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if m and child.is_dir():
            found.append((int(m.group(1)), child, _checkpoint_problem(child)))
    return sorted(found)


def _last_logged_step(cfg: Config) -> int:
    """Highest training step recorded in metrics.jsonl (0 if none): evidence that this run already made progress."""
    best = 0
    path = cfg.run_dir / "metrics.jsonl"
    if path.exists():
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                if '"global_step"' not in raw:
                    continue
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if row.get("mode") == "train":
                    best = max(best, int(row.get("global_step") or 0))
    return best


def checkpoint_report(cfg: Config) -> None:
    """What a restart would resume from: every checkpoint folder on Drive with its size and whether it is complete, the last step
    recorded in metrics.jsonl, the last resume decision, and the free space on Drive."""
    _require_drive(cfg.drive_root)
    print(f"run folder: {cfg.run_dir}  ({'exists' if cfg.run_dir.is_dir() else 'DOES NOT EXIST: check RUN_NAME, BASE_MODEL and the Drive account'})")
    if cfg.run_dir.parent.is_dir():
        others = sorted(p.name for p in cfg.run_dir.parent.iterdir() if p.is_dir() and p.name != cfg.run_name)
        if others:
            print("  other runs in the same folder:", others)
    ckpts = _checkpoints(cfg)
    if not ckpts:
        print("  no checkpoint-* folders")
    for step, path, problem in ckpts:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        mtime = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(path.stat().st_mtime))
        print(f"  {path.name:<18} {size / 2**30:5.2f} GB  modified {mtime}  " + ("✔ complete" if not problem else f"✖ {problem}"))
    complete = [s for s, _, p in ckpts if not p]
    print(f"  a restart resumes from: {'checkpoint-%d' % max(complete) if complete else 'NOTHING (step 0)'}")
    print(f"  last training step in metrics.jsonl: {_last_logged_step(cfg)}")
    decision = cfg.run_dir / "resume_decision.json"
    if decision.exists():
        print("  last resume decision:", decision.read_text().strip()[:400])
    try:
        usage = shutil.disk_usage(cfg.drive_root)
        print(f"  Drive: {usage.free / 2**30:.1f} GB free of {usage.total / 2**30:.1f} GB")
    except OSError:
        pass


def flush_drive(remount: bool = True) -> bool:
    """Upload everything still waiting in Colab's Drive cache (checkpoints are ~1 GB each), then mount Drive again (remount=True).

    Run before you end or switch a Colab session; otherwise the newest checkpoint may never reach Drive and the next session cannot
    resume from it. start_training does this by itself when training stops, finishes or fails. Returns True when the flush finished.
    """
    try:
        from google.colab import drive
    except ImportError:
        print("  (not on Colab: nothing to flush)")
        return False
    t0 = _start("uploading pending files to Google Drive (do not end the session until this finishes)", "seconds to several minutes")
    try:
        drive.flush_and_unmount()
    except KeyboardInterrupt:
        print("⚠ Drive flush interrupted: the newest checkpoint may not be on Drive yet. Run sc.flush_drive() before ending the session.")
        return False
    _finish(t0, "Drive flushed: everything is uploaded, it is safe to end this session")
    if remount:
        try:
            drive.mount("/content/drive")
        except Exception as exc:                 # e.g. the authorisation prompt was not answered
            print(f"  Drive is unmounted ({exc.__class__.__name__}); run section 1 again before using Drive in this session")
    return True


def training_status(cfg: Config) -> None:
    """One-shot status: running or not, progress bar, loss, ETA, GPU, last log line, checkpoints."""
    info, alive = _training_progress(cfg), _training_pid(cfg) is not None
    print(_render_training(info, alive, (cfg.run_dir / "run_summary.json").exists(), _gpu_stats()))
    print("  checkpoints:", sorted(p.name for p in cfg.run_dir.glob("checkpoint-*")) if cfg.run_dir.exists() else "none yet")


def watch_training(cfg: Config, every: int = 20, auto_restart: bool = False, max_restarts: int = 3) -> None:
    """Live view of the background run; returns when training finishes, dies, or looks stuck.

    Stopping this cell (the stop button) is safe: training keeps running in the background. Stuck means: the log has not changed
    for `cfg.stall_minutes`, the GPU is idle AND the training process uses almost no CPU (so the long, silent, CPU-only data
    preparation before the first step, e.g. packing the training set, is never mistaken for a hang). With `auto_restart=True` a stuck run is stopped and restarted from its last
    checkpoint (up to `max_restarts` times), which is what you want when Colab hangs while you are away.
    """
    try:
        from IPython.display import clear_output
    except Exception:
        clear_output = None
    began, restarts = time.time(), 0
    cpu_pid, last_cpu, last_tick = None, None, None
    try:
        while True:
            pid = _training_pid(cfg)
            info, alive = _training_progress(cfg), pid is not None
            finished, gpu = (cfg.run_dir / "run_summary.json").exists(), _gpu_stats()
            now, cpu = time.time(), (_cpu_seconds(pid) if pid else None)
            cpu_share = None
            if pid is not None and pid == cpu_pid and cpu is not None and last_cpu is not None:
                cpu_share = (cpu - last_cpu) / max(now - last_tick, 1e-9)     # 1.0 = one core fully busy
            cpu_pid, last_cpu, last_tick = pid, cpu, now
            if clear_output:
                clear_output(wait=True)
            print(_render_training(info, alive, finished, gpu, cpu_share))
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
            cpu_busy = cpu_share is None or cpu_share >= 0.2    # unknown (first look) counts as busy: never restart blind
            if age >= cfg.stall_seconds and not (gpu and gpu[0] >= 10) and not cpu_busy:
                print(f"⚠ no log output for {_fmt(age)}, " + ("the GPU is idle" if gpu else "no GPU activity could be read")
                      + f" and the CPU is idle ({100 * cpu_share:.0f}%): the run looks stuck (Colab sometimes hangs).")
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
    judge_lines = []
    for key, name in steps:
        f = cfg.runs / "e4" / key / "judge_scores.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        mark = "" if s.get("judge_model") == PAPER_JUDGE_MODEL else f" [judge {s.get('judge_model')}: not comparable]"
        judge_lines.append(f"| {name}{mark} (n={s['rows']:,}) | " + " | ".join(f"{s['mean'][m]:.2f}" for m in JUDGE_METRICS) + " |")
    if judge_lines:
        header = ["| System (LLM judge, 1-5) | Description adherence | Meaning | Fluency | Coherence | Poeticness |", "|---|---:|---:|---:|---:|---:|"]
        paper = [f"| {n} (paper{suffix}) | " + " | ".join(f"{v:.2f}" for v in vals) + " |" for n, vals in PAPER_JUDGE_SCORES.items()]
        table += "\n\n" + "\n".join(header + paper + judge_lines)
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
