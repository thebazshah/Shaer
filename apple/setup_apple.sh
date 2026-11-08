#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac (tested target: Mac mini M4, 24 GB unified memory).
#  1. creates a virtual environment and installs requirements-apple.txt
#  2. logs in to Hugging Face (the base model Navid-AI/Yehia-7B-preview is gated: accept its licence first)
#  3. converts the base model to MLX affine 4-bit (group size 64), which replaces bitsandbytes NF4
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || echo "warning: MLX's Metal backend needs macOS on Apple Silicon" >&2
python3 -m venv .venv-apple
. .venv-apple/bin/activate
pip install --upgrade pip
pip install -r requirements-apple.txt

hf auth whoami >/dev/null 2>&1 || hf auth login

mkdir -p models
[ -d models/yehia-7b-mlx-4bit ] || mlx_lm.convert --hf-path Navid-AI/Yehia-7B-preview \
    -q --q-bits 4 --q-group-size 64 --mlx-path models/yehia-7b-mlx-4bit

# Optional: let the GPU wire more of the 24 GB (default limit is about 2/3 of RAM). Resets on reboot.
#   sudo sysctl iogpu.wired_limit_mb=20480
echo "setup done; activate with: . $(pwd)/.venv-apple/bin/activate"
