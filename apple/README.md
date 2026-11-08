# Shaer on Apple Silicon (MLX port)

Original repository: https://github.com/AhmaddAbbass/Shaer (commit `bba73cb`)  
Fork with this port: https://github.com/thebazshah/Shaer (folder `apple/`, branch `main`)

This folder runs the Shaer SFT recipe and evaluation on an Apple Silicon Mac (target: Mac mini M4,
24 GB unified memory) instead of an NVIDIA GPU. Nothing outside this folder is modified: the data,
prompts, hyperparameters and the official evaluator (`evaluation/score_meter_count.py`,
`evaluation/judge_llm.py`) are used as they are.

## Why a port is needed

The official training and inference path depends on CUDA-only components:

| Component | Official (CUDA) | This port (Apple Silicon) |
|---|---|---|
| 4-bit base model | bitsandbytes NF4, double quantisation | MLX affine 4-bit, group size 64 (`mlx_lm.convert -q`) |
| Optimiser | `paged_adamw_8bit` (bitsandbytes) | `mlx.optimizers.AdamW` (32-bit states) |
| Training loop | HF `Trainer` + PEFT LoRA (`sft/train_sft.py`) | `train_mlx_qlora.py`: same LoRA config, loss mask, packing, weighted sampler, schedule |
| Generation | HF `generate` / vLLM 0.5.3 | `generate_mlx.py` (`mlx_lm.generate`), same decoding settings |
| Released adapter | PEFT adapter on an NF4 base | merged into the bf16 base on the CPU (`fuse_official_adapter.py`), then converted to MLX 4-bit |
| Meter / count scorer | `evaluation/score_meter_count.py` | **unchanged**, runs on the CPU |
| LLM judge | `evaluation/judge_llm.py` via OpenRouter | **unchanged** |

Kept identical to the official run: LoRA r 64, alpha 128, dropout 0.05, rsLoRA (scale alpha/sqrt(r)),
all seven linear projections; lr 8e-5, 3% warmup then cosine, weight decay 0, gradient clip 1.0;
batch 1 x 16 gradient accumulation; 1,024-token packing within sampler groups; loss on poem tokens
only; sampler weights (1/count)^0.6; 2 epochs; eval every 200 steps; seed 42. Decoding:
temperature 0.55, top-p 0.90, repetition penalty 1.08, at most 768 new tokens.

## Files

| File | Purpose |
|---|---|
| `setup_apple.sh` | venv, `requirements-apple.txt`, Hugging Face login, convert Yehia-7B to MLX 4-bit |
| `common.py` | dataset/model ids, tokenisation, packing and sampler weights ported from `sft/` |
| `prepare_data.py` | tokenise and pack the official splits into `data/` |
| `train_mlx_qlora.py` | MLX training loop (resumable; logs to `train_log.jsonl`) |
| `generate_mlx.py` | generate the 3,481 test poems (resumable) |
| `fuse_official_adapter.py` | merge the released adapter for E1 |
| `export_released_generations.py` | export the authors' released generations for E0 |
| `score.sh` | run the official scorer |
| `judge_summary.py` | average the LLM-judge scores (E4) |
| `memorisation.py` | copying analysis against the training set |
| `collect_results.py` | compute the paper's four structural metrics and fill LaTeX macros |
| `plot_training_curves.py` | training/eval loss and learning-rate figure |
| `run_all.sh` | runs the experiments below |

## Running

```bash
cd apple
./setup_apple.sh                      # once; accept the Navid-AI/Yehia-7B-preview licence on Hugging Face first
. .venv-apple/bin/activate
./run_all.sh e0                       # re-score the authors' released generations (CPU only)
./run_all.sh e2                       # untuned Yehia-7B
./run_all.sh e1                       # released Shaer adapter
./run_all.sh e3                       # full re-training on the Mac, then generation and scoring
OPENROUTER_API_KEY=... ./run_all.sh e4   # optional LLM judge (paid API; never commit the key)
./run_all.sh collect                  # metrics -> runs/summary.json and runs/values.tex
```

| Experiment | What it measures |
|---|---|
| E0 | The evaluator: released generations re-scored on this machine |
| E1 | Inference with the released adapter on an MLX 4-bit base |
| E2 | The untuned base model, as the paper's Yehia-7B row |
| E3 | Full re-training on Apple Silicon |
| E4 | LLM-judge scores (strict v2 prompts, `qwen/qwen3-235b-a22b-2507`) of the E3 poems |

Every step can be stopped and re-run: training resumes from `runs/e3_train/latest`, generation skips
ids already written. Logs are appended to `logs/<step>.log`.

Outputs:

- `runs/e3_train/train_log.jsonl`: step, train loss, lr, grad norm, tokens/s, peak memory, eval loss
- `runs/e3_train/best/`: best adapter by eval loss (on a fixed 500-prompt eval subset)
- `runs/e3_train/test_loss.json`: full eval and test loss of the best adapter
- `runs/<exp>/generations.jsonl` and `scored.jsonl`
- `runs/summary.json`

E1 needs about 16 GB of free memory while merging, because the base is loaded in bf16 on the CPU.

## Known differences from the official run

- **Quantisation.** MLX affine 4-bit is not NF4, so the frozen base weights differ slightly. In E1
  the adapter trained against NF4 is applied to a merged and re-quantised model.
- **Optimiser precision.** The optimiser keeps 32-bit AdamW states instead of paged 8-bit ones.
- **Eval subset.** Eval loss during training uses a fixed random 500-prompt subset (seed 42) of the
  eval split, to keep evaluation affordable. The final `test_loss.json` uses the full splits.
- **Sampling.** Different hardware and samplers give different poems at temperature 0.55. Seeds are
  fixed per prompt, so a run on the same machine is repeatable.

## Test status

Tested in a Linux container with MLX's CPU backend, using a 135M stand-in model (SmolLM2-135M) and
a few rows of the real dataset:

- data preparation;
- training: logging, eval, checkpoints, best/latest and resume;
- generation with an adapter, including resume;
- the official scorer and `collect_results.py`. The re-scored `meter` and `count_adherence` match
  the authors' released values exactly on the rows checked.

**Not tested yet:**

- the Metal backend (it needs a Mac);
- the full 7B model;
- `fuse_official_adapter.py`;
- the LLM judge.
