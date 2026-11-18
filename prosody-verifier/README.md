# prosody-verifier

Research line on top of the reproduced Shaer baseline: a formal Arabic prosody verifier (meter, qafiya, radeef in both
senses, poem-level consistency), used to audit the baseline and to guide generation (idea 1: verifier-guided decoding).

All code for this line lives in this folder. Everything runs from one notebook, [`prosody_main.ipynb`](prosody_main.ipynb),
on Colab (A100 preferred, L4 works; M0 needs only a CPU runtime). Outputs go to `MyDrive/shaer-prosody/`; the baseline's
outputs in `MyDrive/shaer-repro-<model>/` are only read.

| File | What |
|---|---|
| `prosody_main.ipynb` | the main notebook (plain Python cells; only the install cell uses `!pip` / `!uv`) |
| `pv_colab.py` | notebook helpers; reuses `colab/shaer_colab.py` (unchanged) for the environment, progress and stall detection |
| `m0_audit.py` | M0 headroom audit: count, meter (per-bayt BiLSTM predictions), heuristic rhyme, diacritics; report |
| `pv_text.py` | Arabic text helpers: scorer-identical line split, rhyme-ending heuristics, diacritic coverage |
| `meter_score_batched.py` | the repository's meter/count scorer with the classifier batched (same results, checked every run) |
| `tests/` | `python -m pytest prosody-verifier/tests -q` |

## Milestones

| | Work | Status |
|---|---|---|
| M0 | Headroom audit of baseline and human reference poems | in the notebook |
| M1-M2 | Rule specification, verifier, gold set | next |
| M3 | Strict re-scoring of the baselines with the verifier | |
| M4 | Verifier-guided decoding vs best-of-N at equal compute | |

## Standing rules

* Branch `dev`, this folder only.
* Long steps show progress and a heartbeat, and a silent idle step is stopped with resume instructions.
* Generation and scoring runs write each finished row to Drive and resume from it.
* Training runs (none yet; idea 2 was deferred) save a checkpoint about every 10 minutes of wall-clock time and keep the two
  most recent plus the best one. Saving is time-based rather than a fixed step count, so it adapts to the GPU. Reference
  speeds for this recipe: 10.65 s/step in the authors' run (4,236 steps in 12.5 h, about 56 steps per 10 minutes);
  18.8 s/step on our Colab A100 while the GPU was shared with a generation job (about 32 steps per 10 minutes).
