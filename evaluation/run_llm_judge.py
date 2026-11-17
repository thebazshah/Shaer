#!/usr/bin/env python3
"""E4: judge a generations file with the repository's own LLM-judge machinery, in parallel and safe to re-run.

It orchestrates the pieces that produced the paper's judge numbers, unchanged: the prompts (evaluation/judge_prompts_v2_strict.yaml),
the client (judge_llm.JudgeClient: temperature 0, forced `submit_score` tool call, OpenRouter), the resumable worker
(full_judge_worker.py: all five metrics per poem) and the merge / summary (merge_full_judge_results.py). The paper's judge model is
`qwen/qwen3-235b-a22b-2507`.

What this script adds around them:
  * input   : turns a generations.jsonl (from generate_shaer_test.py) into the rows the worker expects (`judge_poem` = generated_text,
              `judge_description` = enhanced_description). By default only sample 0 of each prompt is judged (--all-samples for all).
              Only the poem and the description are ever sent to the judge, never the reference poem.
  * preflight: one real call first, so a wrong key / unknown model / no credits stops the run immediately with the API's message.
  * safe re-runs: results are kept; rows whose judging failed (API error, no parsable score) are dropped and judged again, and cached
              API errors are cleared (the repository's client would otherwise replay a cached error forever); rows are re-sharded when the
              number of workers or the row limit changes (the merge rejects duplicates); a poem that changed since it was judged is re-judged.
  * progress: a bar with rows, calls, rows/min, ETA and the API-reported cost; --max-cost-usd stops the run at a budget; a run in which
              no row completes for --stall-minutes is stopped (re-run to resume).

Environment: OPENROUTER_API_KEY (required except with --dry-run); optional JUDGE_BASE_URL, JUDGE_REQUEST_SLEEP_SECONDS, ...

Exit codes: 0 done | 1 some workers failed | 2 setup / preflight problem | 3 stalled | 4 budget reached
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from judge_llm import DEFAULT_METRICS, JudgeClient, extract_judge_json, load_prompt_config, render_user_prompt, safe_int_score
except ModuleNotFoundError as exc:                      # judge_llm needs curl_cffi, pyyaml, python-dotenv
    raise SystemExit(f"missing package for the LLM judge: {exc.name}. Install it with:  pip install curl_cffi pyyaml python-dotenv")

PAPER_JUDGE_MODEL = "qwen/qwen3-235b-a22b-2507"
PROMPT_FILE = HERE / "judge_prompts_v2_strict.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", required=True, help="generations.jsonl written by generate_shaer_test.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--key", default="", help="label for the rows (default: the name of --out-dir)")
    ap.add_argument("--judge-model", default=PAPER_JUDGE_MODEL)
    ap.add_argument("--prompt-file", default=str(PROMPT_FILE))
    ap.add_argument("--metrics", nargs="*", default=None, help=f"default: {' '.join(DEFAULT_METRICS)}")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-rows", type=int, default=0, help="judge only the first N poems (a trial); a later full run reuses them")
    ap.add_argument("--all-samples", action="store_true", help="judge every sample of every prompt, not only sample 0")
    ap.add_argument("--max-cost-usd", type=float, default=0.0, help="stop when the API-reported cost of this run exceeds this (0 = no limit)")
    ap.add_argument("--stall-minutes", type=float, default=15.0)
    ap.add_argument("--poll-seconds", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true", help="prepare the input, print the plan and one rendered prompt; make no API call")
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass                                    # a half-written line
    return rows


def build_input(generations: list[dict], key: str, all_samples: bool, limit: int) -> tuple[list[dict], int]:
    """Rows in the format of full_judge_worker.py (see materialize_judge_dataset.normalize_row), in a deterministic order."""
    rows = [r for r in generations if all_samples or int(r.get("sample_idx", 0)) == 0]
    rows.sort(key=lambda r: (int(r.get("sample_idx", 0)), str(r.get("id"))))
    out, empty = [], 0
    for r in rows:
        poem = str(r.get("generated_text") or "").strip()
        if not poem:
            empty += 1
            continue
        source_id = str(r["id"])
        rec = dict(r)
        rec.update(judge_dataset_key=key, judge_row_index=len(out), judge_source_id=source_id, judge_poem=poem,
                   judge_description=str(r.get("enhanced_description") or r.get("description") or "").strip(),
                   judge_row_id=f"{key}::{source_id}" + (f"::s{int(r.get('sample_idx', 0))}" if all_samples else ""))
        out.append(rec)
    return (out[:limit] if limit else out), empty


def reshard(workers_dir: Path, rows: list[dict], workers: int) -> tuple[int, int, int]:
    """Keep every completed, error-free, still-current result and redistribute it for the current number of workers.

    Returns (kept, dropped_errors, dropped_stale). Rows with errors are dropped so they are judged again.
    """
    position = {r["judge_row_id"]: i for i, r in enumerate(rows)}
    poem_of = {r["judge_row_id"]: r["judge_poem"] for r in rows}
    kept: dict[str, dict] = {}
    errors = stale = 0
    for f in sorted(workers_dir.glob("worker_*.jsonl")):
        for rec in read_jsonl(f):
            rid = str(rec.get("judge_row_id"))
            if rid not in position:
                continue                                        # not part of this run (for example after a smaller --limit-rows)
            if rec.get("judge_metric_errors"):
                errors += 1
            elif rec.get("judge_poem") != poem_of[rid]:
                stale += 1                                      # the poem was regenerated since it was judged
            else:
                kept[rid] = rec
    for f in workers_dir.glob("worker_*.jsonl"):
        f.unlink()
    for w in range(workers):
        mine = [rec for rid, rec in kept.items() if position[rid] % workers == w]
        tmp = workers_dir / f"worker_{w}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in sorted(mine, key=lambda r: position[r["judge_row_id"]])),
                       encoding="utf-8")
        tmp.replace(workers_dir / f"worker_{w}.jsonl")
    return len(kept), errors, stale


def clear_cached_errors(cache_dir: Path) -> int:
    """JudgeClient caches failed API calls as well, which would make a retry return the same error. Remove those entries."""
    removed = 0
    for f in cache_dir.glob("*/*.json"):
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            f.unlink()
            removed += 1
            continue
        if entry.get("error") or not entry.get("response"):
            f.unlink()
            removed += 1
    return removed


def count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


class Tally:
    """Cost and failed rows of THIS run, read incrementally from the workers' result files (one row granularity)."""

    def __init__(self, files: list[Path]) -> None:
        self.files = files
        self.offset = {f: (f.stat().st_size if f.exists() else 0) for f in files}     # rows kept from earlier runs are not counted
        self.cost = 0.0
        self.failed = 0

    def update(self) -> None:
        for f in self.files:
            try:
                with open(f, "rb") as handle:
                    handle.seek(self.offset[f])
                    chunk = handle.read()
            except OSError:
                continue
            complete = chunk[: chunk.rfind(b"\n") + 1]                                 # only whole lines
            self.offset[f] += len(complete)
            for line in complete.splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                self.cost += float(rec.get("judge_total_cost_usd") or 0.0)
                self.failed += 1 if rec.get("judge_metric_errors") else 0


def bar(fraction: float, width: int = 20) -> str:
    n = int(round(min(max(fraction, 0.0), 1.0) * width))
    return "█" * n + "░" * (width - n)


def fmt(seconds: float) -> str:
    m, s = divmod(int(max(seconds, 0)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    workers_dir, cache_dir, logs_dir = out_dir / "workers", out_dir / "cache", out_dir / "logs"
    for d in (workers_dir, cache_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    key = args.key or out_dir.name
    metrics = list(args.metrics or DEFAULT_METRICS)
    prompt_config = load_prompt_config(Path(args.prompt_file))

    generations = read_jsonl(Path(args.generations))
    rows, empty = build_input(generations, key, args.all_samples, args.limit_rows)
    if not rows:
        print(f"nothing to judge in {args.generations} ({len(generations)} rows, {empty} with an empty poem)")
        return 2
    input_path = out_dir / "judge_input.jsonl"
    input_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    total, calls = len(rows), len(rows) * len(metrics)
    print(f"plan: judge {total:,} poems x {len(metrics)} metrics = {calls:,} API calls with {args.judge_model} "
          f"(prompts {prompt_config.get('version')}, {args.workers} parallel workers)"
          + (f"; {empty} empty poems skipped" if empty else "") + ("" if args.all_samples else "; sample 0 of each prompt"), flush=True)
    print("only the poem and its description are sent to the judge (never the reference poem)", flush=True)

    if args.dry_run:
        m = metrics[0]
        print(f"\n--- dry run: the {m} prompt for the first poem (system prompt, then user prompt) ---")
        print(prompt_config["shared_system_prompt"][:600] + (" ..." if len(prompt_config["shared_system_prompt"]) > 600 else ""))
        print("---")
        print(render_user_prompt(prompt_config["metrics"][m], poem=rows[0]["judge_poem"], description=rows[0]["judge_description"])[:1200])
        print(f"\nno API call was made. input written to {input_path}")
        return 0

    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        print("OPENROUTER_API_KEY is not set. Add it as a Colab secret named OPENROUTER_API_KEY (key icon in the left sidebar) "
              "and run the configuration cell again.")
        return 2

    # Results of a different judge model, prompt version or metric set must never be mixed with this run's: archive them, do not reuse.
    prompt_version = str(prompt_config.get("version") or "")
    previous = next((rec for f in sorted(workers_dir.glob("worker_*.jsonl")) for rec in read_jsonl(f)[:1]), None)
    if previous and (previous.get("judge_model") != args.judge_model or str(previous.get("judge_prompt_version")) != prompt_version
                     or list(previous.get("judge_metrics") or []) != metrics):
        archive = out_dir / f"previous-{time.strftime('%Y%m%d-%H%M%S')}"
        archive.mkdir()
        workers_dir.rename(archive / "workers")
        for name in ("judged.jsonl", "judge_summary.json", "judge_scores.json"):
            if (out_dir / name).exists():
                (out_dir / name).rename(archive / name)
        workers_dir.mkdir()
        print(f"found results from another judge setup ({previous.get('judge_model')}, prompts {previous.get('judge_prompt_version')}, "
              f"{len(previous.get('judge_metrics') or [])} metrics): moved them to {archive.name}; judging from scratch with "
              f"{args.judge_model}, prompts {prompt_version}", flush=True)

    kept, dropped_errors, dropped_stale = reshard(workers_dir, rows, args.workers)
    cleared = clear_cached_errors(cache_dir)
    if kept or dropped_errors or dropped_stale:
        print(f"resuming: {kept:,} poems already judged and kept; {dropped_errors} with errors will be judged again"
              + (f"; {dropped_stale} changed since they were judged" if dropped_stale else "")
              + (f"; {cleared} cached API errors cleared" if cleared else ""), flush=True)
    if kept == total:
        print("all poems are already judged")
    else:
        # preflight: one real call, so a bad key, unknown model or exhausted credits is reported before any worker starts
        client = JudgeClient(cache_dir=str(cache_dir), cache_namespace="preflight", model=args.judge_model, json_mode=True, max_tokens=128)
        first = next(r for r in rows)
        user = render_user_prompt(prompt_config["metrics"][metrics[0]], poem=first["judge_poem"], description=first["judge_description"])
        response, _ = client.judge(prompt_config["shared_system_prompt"], user)
        parsed = extract_judge_json(response, score_key="score")
        if response.get("error") or safe_int_score(parsed.get("score")) <= 0:
            (cache_dir / "preflight" / f"{response.get('cache_key')}.json").unlink(missing_ok=True)
            print(f"preflight call failed: {response.get('error') or 'no score in the answer'}. Nothing was started. "
                  "Check the key, the model name and your OpenRouter credits.")
            return 2
        print(f"preflight OK (score {safe_int_score(parsed.get('score'))} for the first poem, {metrics[0]})", flush=True)

        procs = []
        for w in range(args.workers):
            cmd = [sys.executable, "-u", str(HERE / "full_judge_worker.py"), "--input-jsonl", str(input_path),
                   "--output-jsonl", str(workers_dir / f"worker_{w}.jsonl"), "--progress-json", str(workers_dir / f"worker_{w}.progress.json"),
                   "--worker-index", str(w), "--worker-count", str(args.workers), "--judge-model", args.judge_model,
                   "--cache-dir", str(cache_dir), "--prompt-file", str(args.prompt_file), "--metrics", *metrics]
            log = open(logs_dir / f"worker_{w}.log", "w", encoding="utf-8")
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))

        tally = Tally([workers_dir / f"worker_{w}.jsonl" for w in range(args.workers)])
        start, done0 = time.time(), kept
        last_done, last_change, status = kept, time.time(), 0
        try:
            while True:
                time.sleep(args.poll_seconds)
                done = sum(count_lines(workers_dir / f"worker_{w}.jsonl") for w in range(args.workers))
                tally.update()
                cost, fail = tally.cost, tally.failed
                rate = (done - done0) / max((time.time() - start) / 60, 1e-9)
                eta = (total - done) / rate if rate > 0 else float("nan")
                print(f"[{bar(done / total)}] {100 * done / total:5.1f}% {done:,}/{total:,} poems | {done * len(metrics):,}/{calls:,} calls | "
                      f"{rate:.1f} poems/min | elapsed {fmt(time.time() - start)} | ETA ~{eta:.0f} min | cost so far ${cost:.4f} (API-reported) "
                      f"| row failures {int(fail)}", flush=True)
                if done != last_done:
                    last_done, last_change = done, time.time()
                alive = [p for p, _ in procs if p.poll() is None]
                if not alive:
                    break
                if args.max_cost_usd and cost >= args.max_cost_usd:
                    print(f"cost limit reached (${cost:.4f} >= ${args.max_cost_usd}): stopping. Run again with a higher --max-cost-usd to continue.", flush=True)
                    status = 4
                    break
                if time.time() - last_change >= args.stall_minutes * 60:
                    print(f"no poem finished for {fmt(time.time() - last_change)}: the API calls look stuck. Stopping the workers; "
                          "run again to resume where it left off.", flush=True)
                    status = 3
                    break
        finally:
            for p, log in procs:
                if p.poll() is None:
                    p.terminate()
            for p, log in procs:
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                log.close()
        if status:
            return status
        if any(p.returncode not in (0, None) for p, _ in procs):
            print(f"some workers failed; see {logs_dir}. Run again to resume.")
            return 1

    merge = subprocess.run([sys.executable, str(HERE / "merge_full_judge_results.py"), "--workers-dir", str(workers_dir),
                            "--expected-rows", str(total), "--output-jsonl", str(out_dir / "judged.jsonl"),
                            "--summary-json", str(out_dir / "judge_summary.json")], capture_output=True, text=True)
    if merge.returncode != 0:
        print("could not merge the results yet:", (merge.stderr.strip().splitlines() or ["?"])[-1], "\nRun again to finish the remaining poems.")
        return 1
    summary = json.loads((out_dir / "judge_summary.json").read_text(encoding="utf-8"))
    stats = summary["metric_stats"]
    (out_dir / "judge_scores.json").write_text(json.dumps({"judge_model": summary["judge_model"], "prompts": summary["judge_prompt_version"],
                                                          "rows": total, "mean": {m: stats[m]["mean"] for m in metrics},
                                                          "scored_rows": {m: stats[m]["scored_rows"] for m in metrics},
                                                          "error_rows": {m: stats[m]["missing_or_error_rows"] for m in metrics},
                                                          "total_cost_usd": summary["total_cost_usd"]}, indent=2), encoding="utf-8")
    print(f"\nLLM judge scores (1-5), {total:,} poems, judge {summary['judge_model']}, prompts {summary['judge_prompt_version']}:")
    for m in metrics:
        s = stats[m]
        print(f"  {m:22s} mean {s['mean']:.2f}   scored {s['scored_rows']:,}   missing/error {s['missing_or_error_rows']}")
    print(f"  total API-reported cost: ${summary['total_cost_usd']:.4f}")
    if any(stats[m]["missing_or_error_rows"] for m in metrics):
        print("some poems have missing or errored scores: run again to judge them again.")
    print(f"saved: {out_dir / 'judge_scores.json'} (means), {out_dir / 'judged.jsonl'} (per poem), {out_dir / 'judge_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
