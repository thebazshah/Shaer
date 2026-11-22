# M5 — self-training on verifier-approved poems (idea 2): validation

## 1. Expectations, written before the GPU runs

This section is committed before any M5 training or generation. It is not edited afterwards; results will be read
against it.

### 1a. What is trained and compared

All variants use the baseline's recipe and runner, unchanged: QLoRA on Yehia-7B-preview, `sft/train_sft.py` (2 epochs,
the paper's schedule), run with `sc.start_training`, which handles resume and the verified upload of checkpoints to
Drive. Only the data changes.

| variant | data | starts from |
|---|---|---|
| **2a** | the Shaer SFT corpus; every prompt gets the line `حرف الروي: <letter>` (the verifier's rawī of its poem) before the topic | a fresh LoRA, like E3 |
| **2b** | 2a, train split limited to poems the verifier accepts (meter in the labelled form with no margin, and rhyme); eval and test as 2a | a fresh LoRA |
| **2c, round n** | poems the previous model wrote with M4's hemistich search for 4,000 seeded train prompts, kept by the quality gate, mixed 1:1 with original 2a/2b train rows; one epoch | the previous variant's best checkpoint |

**The quality gate of 2c.** A poem is kept only if all of these hold:
* the search finished without a forced hemistich;
* it has exactly the requested length;
* it is strictly valid for the evaluation verifier (letters mode, margin 1.5);
* it uses the requested rawī;
* no hemistich is repeated within the poem;
* no hemistich is copied from any training poem;
* every hemistich has at least 3 words.

**Evaluation.**
* Each variant writes **one ordinary sample per test prompt** (3,481 prompts, the paper's decoding, no search, no
  selection). At test time it is asked for the rawī of the human poem of the prompt.
* It is compared, paired on the same prompts, with **E3**: the baseline notebook's re-training, the same recipe without
  the rhyme field, and the like-for-like baseline. If E3 is missing, **E1** (the released adapter) is used.
* The authors' best GRPO adapter (`Shaer-AI/Shaer-adapters-grpo-short1k-no-trio-v2`, `best-checkpoint`) is optional.
* Scoring is M3's: strict validity in letters mode and binding mode, the BiLSTM metric, quality and copying, and "rawī as
  requested".

**Decision rule (fixed now).** Go on from 2a to 2b and 2c only if both hold:
* 2a's strict validity or its rhyme pass rate beats the baseline with a 95% interval above 0;
* its quality is not worse: distinct-2 within 3 points, repeated hemistichs at most 1 point higher, copied lines at most
  twice the baseline's.

Otherwise M5 is reported as a negative result for 2a.

### 1b. Exact checks (any failure is a bug)

| check | expected |
|---|---|
| a 2a prompt is the original prompt plus exactly one line `حرف الروي: …` before `الموضوع:`; the training text is the new prompt plus the original completion text | every row |
| 2a has as many rows as the corpus in every split; 2b's train split is a subset of 2a's; eval and test are identical across 2a, 2b and 2c | exact |
| a training set is rebuilt into a new folder whenever its inputs change (labels, accepted poems, code), so the training script's data cache (keyed by the folder's path) is never stale | by construction (hash in the folder name) |
| every M5 training run writes only under `MyDrive/shaer-repro-yehia-prosody/` | never into `shaer-repro-yehia/` (tested) |
| M5 summary: missing verifier records, missing scored rows | 0 |
| M5 test prompts without a rawī | at most the number of human test poems with no rawī label (3 of 3,481 in M3) |

### 1c. Predictions (letters mode, margin 1.5; ranges)

**Baselines.**
* E3 should be close to E0: about 9% strictly valid, 38% rhyme passes.
* It should hit the human poem's rawī without being asked for it on about 8–20% of prompts. The most common rawī, ر,
  covers 16% of the human test poems.

| quantity | 2a − E3 | 2b − 2a | 2c (round 1) − 2a |
|---|---|---|---|
| rawī as requested | +45 to +80 pp (2a: 60–90%) | ±5 pp | +0 to +10 pp |
| rhyme check passes | +5 to +25 pp | ±5 pp | +3 to +15 pp |
| strictly valid | +1 to +8 pp | 0 to +5 pp | +3 to +15 pp |
| bayts on meter | ±3 pp (the meter is not targeted) | 0 to +5 pp | +2 to +10 pp |
| BiLSTM bayts | ±1.5 pp | ±1.5 pp | 0 to +2 pp |
| distinct-2 (over all poems) | ±3 points | −5 to +1 | −5 to +1 |
| hemistichs copied from the training poems | ≤ 2× E3 | ≤ 2× 2a | ≤ 2× 2a (the gate refuses copies) |

Notes:
* **Rhyme.** Naming the rawī should raise the rhyme pass rate. The rawī is only one of the rhyme checks (the vowel,
  ridf and taʾsīs still apply), so the gain in the rhyme pass rate is smaller than the gain in hitting the rawī.
* **2b.** It trains on about 70% of the corpus: the human test poems pass at about that rate with no margin.
* **2c.** It is the only variant that trains on the model's own verified outputs. The most likely failure is reduced
  diversity, so its quality columns matter as much as its validity.
* **The GRPO adapter.** It was rewarded by the BiLSTM. Expect a BiLSTM rate at or above E3's, but no higher strict
  validity than 2c.

## 2. Validation of the code (local, no GPU)

### 2a. Tests

Ten M5/M6 tests are in `tests/test_m5_m6.py`; the full suite is 237 tests (231 + 6 skipped without torch). They cover:
* the prompt insertion and the rebuilt training text;
* the labels, and resume;
* 2a and 2b: row counts, the rawī in every prompt, the 2b filter, folder reuse, loading with `datasets`;
* every reason of the 2c gate;
* the 2c build (completion replaced, mixing, eval and test from the base);
* the M5 summary: paired differences, McNemar, "rawī as requested", integrity checks;
* the version check, and the training configuration (its extra variables reach the training environment, and its run
  folder is in the sibling).

### 2b. The unchanged training script on an M5 folder

A 2a folder built from a sample of real eval rows was loaded by `sft/train_sft.py`'s own functions, with
`SFT_DATASET_ID=<folder>`:
* `load_or_build_dataset_splits` reads it as published train / eval / test splits;
* `annotate_dataset` and `filter_dataset_for_run` accept it;
* `preprocess_dataset`, run with a tokenizer byte-identical to Yehia's, masks the whole prompt including the new line
  and supervises exactly the poem: the last masked token is `[/INST]`, and supervision ends with `</s>`.

The rawī line reads `عدد الأشطر المطلوب: 4\nحرف الروي: ن\nالموضوع: …`.

### 2c. The notebook helpers

`pv.m5_relabel`, `m5_build` (2a, 2b), `m5_import_baselines`, `m5_score` and `m5_report` were run end to end on a fake
Drive with a small corpus sample and stand-in poems:
* the labels, the two data folders and `variants.json` were written; the folders have the same content-hash names as a
  separate manual build, so the builds are deterministic;
* the baseline's E1 poems were copied from its folder, which was only read: its files were unchanged afterwards;
* both runs were scored by the repository's scorer and the verifier, and the report was written with its integrity
  checks at 0.

Training and generation need the GPU. They use the baseline's runner (`sc.start_training`) and M4's generator
(`m4_decode.py sample`), both validated before.

## 3. Cost

| step | runtime | time |
|---|---|---|
| relabelling the corpus (116k poems) | CPU | about 1–2 h with 12 CPUs; several hours on a 2-CPU runtime |
| training 2a or 2b | A100 | about 12–20 h (as E3) |
| one 2c round | A100 | the search on 4,000 train prompts (1–2 h) and one epoch on about 2× the accepted poems (2–3 h) |
| each test generation (3,481 prompts, one sample) | A100 | about 1.5 h |
| scoring | CPU | as for M4 |

With the decision rule, the minimum is relabel + 2a + its test generation: roughly 15–25 A100 hours.

## 4. Known limitations

* **Checkpoints.** The baseline's runner keeps two checkpoints on Drive (the latest and the best) and writes a resume
  checkpoint about every 10 minutes (`save_minutes=10`). Keeping the two latest plus the best would need a change to
  `colab/shaer_colab.py`, the baseline's module, which this work does not modify.
* **Only the rawī is named.** The rhyme field names the rawī alone, not its vowel or ridf. A model can follow the letter
  and still fail iqwāʾ or sinād.
* **The verifier trains and scores.** 2b and 2c use the verifier to choose training data, and the same verifier scores
  them. The BiLSTM metric, binding mode, M6's judge and the human study are the independent checks.
