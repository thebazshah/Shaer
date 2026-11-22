# prosody-verifier

Research line on top of the reproduced Shaer baseline: a formal Arabic prosody verifier (meter, qāfiya, radīf in both senses,
poem-level consistency), used to audit the baseline and to guide generation (idea 1: verifier-guided decoding).

All code for this line lives in this folder. Everything runs from one notebook, [`prosody_main.ipynb`](prosody_main.ipynb),
on Colab (A100 preferred, L4 works; M0 and M1 need only a CPU runtime). Outputs go to `MyDrive/shaer-prosody/`; the
baseline's outputs in `MyDrive/shaer-repro-<model>/` are only read.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thebazshah/Shaer/blob/dev/prosody-verifier/prosody_main.ipynb)
Open the latest notebook in Colab: https://colab.research.google.com/github/thebazshah/Shaer/blob/dev/prosody-verifier/prosody_main.ipynb
(it opens the `dev` branch's copy; if you save it from Colab, save it under another name so a stale copy never replaces it).

## The verifier (`prosody/`, standard library only)

| Module | What |
|---|---|
| `feet.py` | the 8 base feet as units (sabab / watad); every zihāf (by letter ordinal, only on the second letter of a sabab) and ʿilla as an operation |
| `meters.py` | the 16 meters with their forms (tām, majzūʾ, mukhallaʿ, aḥadhdh), ʿarūḍ / ḍarb tables and pairings, rare zihāfāt, muʿāqaba, murāqaba, kharm; Shaer's 21 labels |
| `arudi.py` | prosodic writing: text with full, partial or no diacritics → the mutaḥarrik/sākin readings it allows (waṣl, article and sun letters, shadda, tanwīn, long vowels, hidden alifs, pronoun lengthening, word boundaries, the corpus's spelling conventions) |
| `lexicon.py` | the pronunciation prior: how each bare word is read in the train split, plus a letter model for unseen words |
| `scan.py` | scansion of a hemistich against every meter pattern (trie + dynamic programme); nearest pattern and broken foot on failure |
| `qafiya.py` | rhyme: rawī, majrā (iqwāʾ / iṣrāf), ridf, taʾsīs, waṣl, khurūj, muṭlaqa / muqayyada, īṭāʾ; Urdu-style radīf (refrain) |
| `verify.py` | poem verdicts: per hemistich and bayt (allowed pairs, taṣrīʿ, tadwīr), ḍarb / ʿarūḍ consistency, rhyme, length; `ignore_diacritics` mode |

`data/lexicon_train.json.gz` is the shipped pronunciation lexicon (built by `build_lexicon.py` from the train split only).

## Tools

| File | What |
|---|---|
| `prosody_main.ipynb` | the main notebook (plain Python cells; only the install cell uses `!pip` / `!uv`) |
| `pv_colab.py` | notebook helpers; reuses `colab/shaer_colab.py` (unchanged) for the environment, progress and stall detection |
| `m0_audit.py`, `pv_text.py`, `meter_score_batched.py` | M0 headroom audit |
| `prepare_splits.py` | the Shaer train / eval / test splits as JSONL |
| `build_lexicon.py` | builds the pronunciation lexicon from the train split |
| `validate_verifier.py` | validation on human poems (positives) and broken copies (negatives), across thresholds |
| `compare_bilstm.py` | bayt-by-bayt agreement with the Shaer evaluation's BiLSTM classifier |
| `m3_rescore.py` | M3: strict re-scoring of every baseline (score / control / summarize), resumable and version-pinned |
| `m4_search.py` | M4: the hemistich search (candidate checks against the poem so far, the search loop, best-of-N selection); no GPU needed |
| `m4_decode.py` | M4 on the GPU: whole poems (I0/I1) or the hemistich search (I2) with the released adapter; `selfcheck` mode |
| `m4_report.py` | M4 summary: best-of-N against the search at equal compute, independent views, quality, integrity checks |
| `m4_replay.py` | M4 validation without a GPU: the search on real lines, scored by the full M3 pipeline |
| `tests/` | `python -m pytest prosody-verifier/tests -q` |
| `validation/` | written reports: the rule-table cross-check, the M1, M3 and M4 validations |

## Milestones

| | Work | Status |
|---|---|---|
| M0 | Headroom audit of baseline and human reference poems | done |
| M1 | The verifier and its validation | done (see `validation/M1_report.md`) |
| M2 | Expert-checked gold set; sign-off of the rule tables | needs a prosody expert |
| M3 | Strict re-scoring of the baselines with the verifier | done for the reference and E0 (`validation/M3_report.md`); E1–E3 are scored on Colab with the same code |
| M4 | Verifier-guided decoding vs best-of-N at equal compute | code and local validation in `validation/M4_report.md`; the runs need a GPU on Colab |

## Where things are on Drive

| folder | what | written by |
|---|---|---|
| `MyDrive/shaer-repro-yehia/` | the baseline: the E3 adapter in `sft_outputs/train/<RUN_NAME>/` (`paper_repro_a100` by default; `checkpoint-*`, `run_summary.json` naming the best one) and the generations in `runs/e0` … `runs/e3/` | the baseline notebook only; this notebook only reads it |
| `MyDrive/shaer-repro-yehia-prosody/` | the same layout, beside the baseline's: anything this research trains or generates with the baseline's tools (`sc.start_training`, `sc.run_generation`), under run names `prosody_*` | this notebook (`pv.training_config(pvc, "prosody_<name>")`) |
| `MyDrive/shaer-prosody/` | M0–M4 outputs (audits, verifier records, M4 runs, reports) | this notebook |

`pv.make_config` replaces the baseline's configuration by the sibling one and points the repository's `sft/outputs` and
`runs/` links at the sibling. A run name that does not start with `prosody_`, or that the baseline already uses, is
refused.

## Standing rules

* Branch `dev`, this folder only.
* Never write into the baseline's Drive folder; this notebook's training runs go to the sibling folder above.
* Long steps show progress and a heartbeat, and a silent idle step is stopped with resume instructions.
* Generation and scoring runs write each finished row to Drive and resume from it.
* Training runs (none yet; idea 2 was deferred) save a checkpoint about every 10 minutes of wall-clock time and keep the two
  most recent plus the best one. Saving is time-based rather than a fixed step count, so it adapts to the GPU. Reference
  speeds for this recipe: 10.65 s/step in the authors' run (4,236 steps in 12.5 h, about 56 steps per 10 minutes);
  18.8 s/step on our Colab A100 while the GPU was shared with a generation job (about 32 steps per 10 minutes).
