# M6 — quality of the final systems: validation

## 1. Expectations, written before the judge and the human study run

This section is committed before any paid judge call or human rating. It is not edited afterwards.

### 1a. What is measured

**Systems.** One poem per test prompt from each of:
* the M4 test conditions: I0, the matched best-of-N and the chosen search run;
* the M5 systems: E3 or E1, 2a, 2b, the 2c rounds, and optionally the GRPO adapter;
* the human poems of the same prompts, as the ceiling.

**Measures.**
* **The paper's LLM judge** (`evaluation/run_llm_judge.py`, unchanged: the judge model, its strict v2 prompts and
  temperature 0). It scores five metrics, 1–5: description adherence, meaning, fluency, coherence and poeticness.
  * Every system is judged on the same prompts: the shared prompts, sorted by id, up to `limit_rows`.
  * The judge sees only the poem and its description, never the system's name.
* **Copying and diversity:**
  * hemistichs copied from the training poems and from the reference poem;
  * repeated hemistichs;
  * distinct-1/2 over all poems and within a poem;
  * distinct rhyme words.
* **Strict validity** (the verifier, letters mode) on the judged prompts, so quality and validity are read on the same
  poems.
* **A blinded human study.**
  * A CSV sheet lists, for a seeded sample of prompts, every chosen system's poem in a random order under item codes.
  * The key that maps codes to systems is a separate file.
  * Several raters can each fill a copy; a poem's score is the mean of its raters', a system's the mean over prompts.

All differences are paired, against a baseline (E3 by default), with bootstrap intervals over prompts.

### 1b. Exact checks

| check | expected |
|---|---|
| every system is judged on the same prompts | the judge input of every system lists the same ids in the same order |
| the human sheet names no system | no system name in the sheet (tested) |
| unblinding | every item code maps to exactly one (system, prompt) |

### 1c. Predictions

**The judge.**
* The paper reports for Shaer: description adherence 4.75, meaning 3.72, fluency 4.31, coherence 3.69, poeticness 3.82.
  I0 and E3 should land within ±0.15 of these on each metric.
* The human poems should score at or above every system on meaning and poeticness.
* The M4 search keeps the model's own first acceptable line, so it should stay within ±0.15 of I0 on every metric. A
  drop larger than 0.2 on meaning or coherence would mean its line-by-line selection hurts the whole poem.
* Best-of-N should stay within ±0.1 of I0; it only chooses among whole samples.
* 2a should stay within ±0.2 of E3 on every metric. 2c is where a drop is most likely: poeticness or description
  adherence down more than 0.2 would argue against further rounds.

**Copying.** Hemistichs copied from the training poems should stay at or below 0.5% for every system: E0 is at 0.13%,
the human test poems at 4.1% (the corpus repeats lines across poems).

**The human study.** No prediction of levels. The ranking of systems on "overall" should agree with the judge's on
poeticness for the extremes (the human poems first, the weakest system last).

## 2. Validation of the code (local)

**Tests** (`tests/test_m5_m6.py`):
* collect from M5 runs, with the human poems added;
* the human sheet: blinded, the key covering every item, two raters scored and averaged;
* the summary: judge means, paired judge differences, strict validity and human ratings, read from the same files
  `run_llm_judge.py` writes (`judged.jsonl`).

**The notebook helpers** (`pv.m6_collect`, `m6_human_sheet`, `m6_report`) call `m6_quality.py`, whose functions the
tests above cover. The judge helper only calls the baseline's `run_llm_judge.py`; it was not run here because it is a
paid API. Start with `dry_run=True`,
which prints the plan and one exact prompt without any call.

## 3. Cost and practical notes

* **The judge** makes 5 calls per poem. 300 prompts × 6 systems = 9,000 calls.
  * The 10-row trial shows the real cost per poem; set `max_cost_usd` per system.
  * Finished poems are kept, so a larger `limit_rows` later only adds poems.
* **The human study.** 40 prompts × 3 systems = 120 poems per rater. Raters should be native readers of classical Arabic
  poetry, and should not be told which systems are in the sheet.
* **Runtime.** Everything but the judge's API calls runs on a CPU runtime.
