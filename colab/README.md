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
- It clones this repository, builds an isolated Python 3.12 environment with the pinned versions of `requirements.txt`
  (transformers 4.43.0, peft 0.12.0, bitsandbytes 0.43.3, torch 2.3.1), and works on top of Colab's own packages.
- `evaluation/generate_shaer_test.py` generates the 3,481 test poems (the repository did not contain this step);
  `evaluation/summarize_structural.py` computes the paper's structural metrics and has been checked so that the released
  generations reproduce the paper's tables; `sft/train_sft.py` accepts `SFT_PUSH_TO_HUB=0` for runs that must not touch the Hugging Face Hub.
- GPU: an A100 (40/80 GB) is what the notebook is written for. The notebook checks the GPU and says which steps it supports.
