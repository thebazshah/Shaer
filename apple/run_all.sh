#!/usr/bin/env bash
# Run the reproduction experiments on Apple Silicon. Each step is resumable; pick steps by name:
#   ./run_all.sh e0          re-score the authors' released generations with the official evaluator
#   ./run_all.sh e1          released adapter (merged on CPU) -> MLX 4-bit -> generate -> score
#   ./run_all.sh e2          untuned Yehia-7B (MLX 4-bit) -> generate -> score
#   ./run_all.sh e3          prepare data -> QLoRA-style training on MLX -> generate -> score -> memorisation
#   ./run_all.sh e4          LLM judge (strict v2 prompts) on E3 generations; needs OPENROUTER_API_KEY
#   ./run_all.sh collect     fill the LaTeX result macros (VALUES_TEX) and plot the training curves
#   ./run_all.sh all         e0 e2 e1 e3 collect   (e4 is opt-in because it calls a paid API)
# Logs go to logs/<step>.log. Run setup_apple.sh first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SHAER_ROOT="${SHAER_ROOT:-$(cd "$HERE/.." && pwd)}"
export SHAER_ROOT
BASE_MLX="${BASE_MLX:-models/yehia-7b-mlx-4bit}"
VALUES_TEX="${VALUES_TEX:-runs/values.tex}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen/qwen3-235b-a22b-2507}"
mkdir -p logs runs

e0() {
  [ -f runs/e0/generations.jsonl ] || python export_released_generations.py --out runs/e0/generations.jsonl
  ./score.sh runs/e0/generations.jsonl runs/e0/scored.jsonl
}

e1() {
  [ -d models/shaer-official-merged-bf16 ] || python fuse_official_adapter.py --out models/shaer-official-merged-bf16
  [ -d models/shaer-official-mlx-4bit ] || mlx_lm.convert --hf-path models/shaer-official-merged-bf16 \
      -q --q-bits 4 --q-group-size 64 --mlx-path models/shaer-official-mlx-4bit
  [ -f data/test_prompts.jsonl ] || python prepare_data.py --tokenizer "$BASE_MLX" --out data
  python generate_mlx.py --model models/shaer-official-mlx-4bit --prompts data/test_prompts.jsonl --out runs/e1/generations.jsonl
  ./score.sh runs/e1/generations.jsonl runs/e1/scored.jsonl
}

e2() {
  [ -f data/test_prompts.jsonl ] || python prepare_data.py --tokenizer "$BASE_MLX" --out data
  python generate_mlx.py --model "$BASE_MLX" --prompts data/test_prompts.jsonl --out runs/e2/generations.jsonl
  ./score.sh runs/e2/generations.jsonl runs/e2/scored.jsonl
}

e3() {
  [ -f data/train_packed.npz ] || python prepare_data.py --tokenizer "$BASE_MLX" --out data
  python train_mlx_qlora.py --model "$BASE_MLX" --data data --out runs/e3_train
  python generate_mlx.py --model "$BASE_MLX" --adapter runs/e3_train/best --prompts data/test_prompts.jsonl \
      --out runs/e3/generations.jsonl
  ./score.sh runs/e3/generations.jsonl runs/e3/scored.jsonl
  python memorisation.py --generations runs/e3/generations.jsonl --out runs/e3/memorisation.json
}

e4() {
  : "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (never commit it)}"
  mkdir -p runs/e4
  for m in description_adherence meaning fluency coherence poeticness; do
    (cd "$SHAER_ROOT" && python evaluation/judge_llm.py --input-jsonl "$HERE/runs/e3/generations.jsonl" \
        --output-jsonl "$HERE/runs/e4/judge_$m.jsonl" --metric "$m" \
        --prompt-file evaluation/judge_prompts_v2_strict.yaml --judge-model "$JUDGE_MODEL" \
        --cache-dir "$HERE/runs/e4/cache")
  done
  python judge_summary.py --judge-dir runs/e4 --out runs/e4/judge_summary.json
}

collect() {
  [ -f "$VALUES_TEX" ] || cp values_template.tex "$VALUES_TEX"
  python collect_results.py --runs-dir runs --out "$VALUES_TEX"
  [ -f runs/e3_train/train_log.jsonl ] && python plot_training_curves.py --log runs/e3_train/train_log.jsonl \
      --out runs/training_curves.pdf || true
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && { sed -n '2,10p' "$0"; exit 1; }
[ "${steps[0]}" = "all" ] && steps=(e0 e2 e1 e3 collect)
for s in "${steps[@]}"; do
  echo "=== $s ($(date)) ===" | tee -a "logs/$s.log"
  "$s" 2>&1 | tee -a "logs/$s.log"
done
