# Colab Notebooks

This folder contains lightweight public notebooks for the released Shaer artifacts.

## Notebook

- [Use intended prompt for best performance](https://colab.research.google.com/github/AhmaddAbbass/Shaer/blob/main/colab/shaer_use_intended_prompt_for_best_performance.ipynb)

## Scope

This notebook is designed for:

- quick model loading
- intended prompt-format inspection
- loading real examples from the released split dataset `test` split
- sample generation with the public adapter

It is not a full end-to-end reproduction notebook for training.

## Reproducing the paper on a Colab A100 (branch `dev` of the fork)

- [reproduce_shaer_a100.ipynb](./reproduce_shaer_a100.ipynb) runs experiments E0-E3 and saves everything to Google Drive:
  E0 re-scores the released generations, E1 uses the released adapter, E2 the untuned base model, E3 re-trains.
- The notebook is plain Python: each cell calls a function of [`shaer_colab.py`](./shaer_colab.py), which runs the repository's scripts
  (only the package installation uses `!` commands). It clones this repository, installs an isolated Python 3.12 environment
  from [`requirements-colab.txt`](./requirements-colab.txt) (transformers 4.43.0, peft 0.12.0, bitsandbytes 0.43.3, torch 2.3.1)
  next to Colab's own packages, and starts the scripts with that environment's interpreter (`env_check.py` and
  `hf_access_check.py` are the two small checks that run there).
- **No blind waiting.** Every long step shows progress bars (downloads, model loading, tokenising, generation, scoring, training),
  prints a heartbeat every ~20 s (time running, time silent, GPU use), and stops a step that is both silent and idle
  (GPU below 10 % and almost no CPU) for `STALL_MINUTES`, telling you to re-run the cell, which resumes. Training runs in the
  cell itself (`sc.start_training(cfg)`): its step bar, loss and evaluations stream into the cell and the log on Drive, a hung run
  is restarted from its last checkpoint (up to 3 times), stopping the cell stops training cleanly, and a resume skips the step-0
  evaluation and probe already recorded. `sc.start_training(cfg, background=True)` keeps the old detached run, followed with
  `sc.watch_training(cfg)`.
- **E4, the LLM judge** (`sc.run_judge(cfg, "e3")`, optional and paid): scores the poems on the paper's five judge metrics with the repository's own
  judge (`qwen/qwen3-235b-a22b-2507`, strict v2 prompts, temperature 0, through OpenRouter; needs the Colab secret `OPENROUTER_API_KEY`).
  `evaluation/run_llm_judge.py` runs the repository's parallel workers and merge, and adds a preflight call, safe re-runs (failed poems
  are judged again, finished ones kept), stall detection, a cost limit, and a progress bar with the API-reported cost. One poem is five calls.
- `sc.start_training(cfg, save_steps=50)` adds a resume checkpoint every 50 optimiser steps (default 200, the paper's schedule) so an
  interruption loses fewer steps. Evaluation, the probe and best-checkpoint selection stay on the 200-step schedule (Hugging Face does not
  allow a save interval below the evaluation interval with `load_best_model_at_end`, so this is a separate resume-only mechanism,
  `SFT_RESCUE_SAVE_STEPS` in `sft/train_sft.py`, off by default).
- `sc.start_training(cfg)` also writes a resume checkpoint every 10 minutes of training by default (`save_minutes=10`; about every
  32 steps on an L4, about every 60 on an A100), so a slow GPU does not run for hours without one. Same resume-only mechanism
  (`SFT_RESCUE_SAVE_MINUTES`); `save_minutes=None` turns it off.
- `sc.start_training(cfg)` keeps only two checkpoints on Drive: the latest (to resume from) and the best (lowest eval loss), or one
  when they are the same. It sets `SFT_SAVE_TOTAL_LIMIT=1`, which Hugging Face treats as "latest + best" with `load_best_model_at_end`
  (the yaml's `save_total_limit: 3` stays the default for other runs).
- `evaluation/generate_shaer_test.py` generates the 3,481 test poems (the repository did not contain this step);
  `evaluation/summarize_structural.py` computes the paper's structural metrics and has been checked so that the released
  generations reproduce the paper's tables; `sft/train_sft.py` accepts `SFT_PUSH_TO_HUB=0` for runs that must not touch the Hugging Face Hub.
- Base model, chosen in the first cell (`BASE_MODEL`), selects one of two flows:
  - `Navid-AI/Yehia-7B-preview` (gated): the **paper flow**. E0, E1 (released adapter), E2 and E3 run as in the paper and
    are compared with the paper's numbers. Outputs go to `MyDrive/shaer-repro-yehia`.
  - `ALLaM-AI/ALLaM-7B-Instruct-preview` (not gated; the model Yehia was fine-tuned from, byte-identical tokenizer): the
    **substitute flow**. E1 is skipped because the released adapter was trained on Yehia's weights
    (`evaluation/generate_shaer_test.py` refuses that combination unless `--allow-adapter-mismatch` is given), E2/E3 measure
    ALLaM, and the results table labels the paper's rows as Yehia-based. Outputs go to `MyDrive/shaer-repro-allam`.
  Any other value raises an error.
- GPU: an A100 (40 or 80 GB) is what the notebook is written for. 40 GB is enough: generation (bf16 model + adaptive KV-cache batch)
  and 4-bit QLoRA training (about 12-14 GB estimated; the smoke test in the notebook prints the measured peak). The notebook checks the GPU and says which steps it supports.
