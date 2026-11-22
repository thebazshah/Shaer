# M4 — verifier-guided decoding against best-of-N: validation

## 1. Expectations, written before the GPU runs

This section is committed before any run of the real model. It is not edited afterwards; the results will be read
against it and deviations discussed in later sections.

### 1a. What is compared

All runs use the released Shaer adapter on Yehia-7B-preview, bf16, and the paper's decoding (temperature 0.55, top-p
0.90, top-k off, repetition penalty 1.08). Prompts are the official `sft_prompt` strings.

| condition | what it does |
|---|---|
| **I0** | one sample per prompt (the paper's setting) |
| **I1 best-of-n** | n samples; the first strictly valid one (letters mode, margin 1.5), else the one with the fewest defects |
| **I2 k, R** | hemistich search: per hemistich, k candidates per round and up to R rounds; the first candidate that adds no new defect to the poem so far, else the one adding the fewest; exactly the requested number of hemistichs |

* **Budget.** The budget is the number of generated tokens per prompt, rejected candidates included. Encoded context
  tokens and GPU seconds are reported next to it. The search re-encodes the context once per hemistich; a production
  implementation with prefix caching would not.
* **Matched compute.** Each search run is compared with the best-of-n with the smallest n whose budget was at least as
  large, so best-of-N always gets at least the search's budget. The comparison uses paired bootstrap intervals over
  prompts and an exact McNemar test.
* **Development sweep.** 400 seeded eval prompts (`--subset 400 --subset-seed 2026`). It has one run of 16 samples per
  prompt (giving I0 and best-of-2/4/8/16) and six search runs: (k, R) = (1, 4), (1, 8), (4, 1), (4, 2), (8, 1), and
  (4, 2) checked in binding mode.
* **Test runs.** These are fixed by a rule set now. Among the letters-mode search runs whose development budget is at
  most that of best-of-4, take the highest strict validity (ties: the smaller budget). Best-of-n for the test uses the
  smallest n whose development budget was at least that run's. Both run on all 3,481 test prompts. `m4_report` applies
  the rule (`choice` in `m4_summary.json`).
* **A negative result.** If the chosen search run does not beat the matched best-of-n on test strict validity with a
  95% interval above 0, M4's gate is not met, and that will be reported as such.

### 1b. Exact checks (any failure is a bug)

| check | expected |
|---|---|
| self-check on the real model (`m4_decode.py selfcheck`): next-token argmax from the shared cache vs a full forward pass | 100% agreement; max logit difference small (bf16 rounding; < 0.5) |
| self-check: greedy decoding one hemistich at a time (accept everything) vs one-shot greedy decoding | identical for at least 75% of prompts (identical in float32, see section 2; bf16 can break a near-tie differently) |
| every search poem has the requested number of hemistichs | 0 exceptions, except a hemistich where every candidate was empty (counted separately; expected 0) |
| a search poem with no forced hemistich is strictly valid in the mode it was checked in | 0 exceptions |
| every selected poem has scorer rows and verifier records in both modes; line counts agree | 0 missing, 0 mismatches |
| budget of best-of-n | the sum of the n samples' tokens |
| sample mode vs the baseline's E1 (`m4_check_sampling`, 300 test prompts) | every 95% interval of the differences contains 0 (meter score, exact count, count adherence) |

### 1c. Predictions (ranges, not equalities)

They are based on M3's released Shaer poems (E0, test split, the verifier as fixed in this milestone: see section 3) and
on E0's own next-hemistich pass rates. E0 is the same model and decoding as I0, so the eval prompts should behave alike.

**Sources of the numbers:**

* **E0 strict validity (letters).** 8.99%, by length 1–3 / 4–6 / 7–10 / 11–20 bayts as in M3.
* **E0 length.** 163 generated tokens per poem, 14.8 per requested hemistich.
* **Best-of-n.** Within each cell of requested length × meter, prompts are given E0's pass rate. That is an upper bound,
  since prompts in a cell would then be equally easy. A lower bound treats each prompt's pass rate as drawn from a Beta
  distribution with that cell mean and concentration 1 (strong prompt-to-prompt variation).
* **Search.** Given a valid poem so far, how often E0's own next hemistich passes the search check (1,500 E0 poems):
  * sadr 76.4%;
  * first ʿajuz 70.6% (meter only, as the first bayt fixes the rawī);
  * later ʿajuz 51.4% (the rhyme now binds);
  * per meter, ʿajuz after the first bayt ranges from 31% (al-munsariḥ) to 62% (al-ṭawīl).

  The poem passes if every hemistich finds a passing candidate among its k·R tries. The upper bound treats candidates as
  independent. The lower bound counts half as many independent tries, since candidates for one prompt are correlated.
* **A bias to keep in mind.** The pass rates were measured on prefixes where E0 happened to be doing well, so the search
  predictions lean optimistic. The ranges below are widened downward for that.

| quantity (letters, margin 1.5) | predicted |
|---|---|
| I0 strictly valid on the 400 eval prompts | 5–14% (E0 test: 8.99%; ±3 pp of sampling noise) |
| best-of-2 / 4 / 8 / 16 | 11–17% / 15–28% / 18–40% / 20–51% |
| I2 with 4 tries per hemistich (k=1 R=4; k=4 R=1) | 25–75% |
| I2 with 8 tries (k=1 R=8; k=4 R=2; k=8 R=1) | 50–96% |
| budget relative to I0 (generated tokens) | k=1 R=4: 1.4–2.5×; k=1 R=8: 1.5–3×; k=4 R=1: about 4×; k=4 R=2: 4–5×; k=8 R=1: about 8× |
| first candidate passes (sequential runs) | 55–75% of hemistichs |
| the pre-registered choice | k=1 R=8 or k=1 R=4, tested against best-of-2 or best-of-4 |
| matched compute (dev and test): I2 − I1 strictly valid | positive, the interval excluding 0; for the choice, +25 to +70 pp |
| I2 strictly valid by length | still decreasing with length, much less steeply than I0 (I0 on 11–20 bayts: about 0%) |
| valid ignoring length (meter, rhyme, even lines) | I2's gain over I1 smaller than its strict gain (the search fixes length by construction) |
| BiLSTM bayts on target (the paper's metric) | I2 ≥ I0; I2 at 96–99% vs I0 at about 95% |
| strictly valid in binding mode, letters-mode search runs | far below their letters rate (E0: 40.8% of bayts in binding vs 54.6% in letters), 10–40% |
| the binding-mode search run | binding strict validity well above the letters-mode runs' binding rate; letters rate close to theirs |
| quality: distinct-2 over all poems (E0: 81.9; human: 89.9) | within ±4 points of I0 |
| hemistichs repeated within a poem (E0: 1.26%) | at most 1 point above I0 |
| hemistichs copied from the training poems (E0: 0.13%; human test poems: 4.09%) | at most 0.5% (lines that scan well could be memorised ones) |
| early end of text (the model wanted to stop before the requested length) | a few per 100 poems; it is where I0 loses most of its length errors |

## 2. Validation of the code (local, no GPU)

### 2a. Tests

There are 216 tests. 22 are M4 tests, in `tests/test_m4.py`, run in the notebook's stack (torch 2.3.1, transformers
4.43.0, peft 0.12.0).

**The candidate checker:**
* the true lines of a human poem pass, in order;
* a line in another meter, a foreign rawī, an empty line, and a rhyme word repeated in the second bayt (read as a
  refrain) are rejected;
* the search verifier agrees with the evaluation verifier wherever tadwīr is not involved;
* memoised bayts are not shared between positions;
* binding mode works.

**The search loop:**
* it takes the first passing candidate in sample order;
* the checks stop there, but every generated candidate's tokens are counted;
* the context grows by exactly the accepted ids;
* rounds work, and the fallback takes the fewest new defects;
* an empty candidate never beats a written line;
* the length is exact, and an early end of text is counted.

**Best-of-N:**
* it takes the first valid sample, and its budget is the sum over the samples;
* the McNemar test is exact;
* the summary fixture checks selection, budgets, matched comparisons, the pre-registered choice and the integrity checks.

**The generator, with a tiny random Llama, a character tokenizer, and with and without a LoRA adapter:**
* **Shared cache.** The context encoding (no logits) and its cache repeated k times give the same next-token logits as a
  full forward pass on a left-padded batch (≤ 1e-5).
* **Cached candidates.** Greedy candidates generated from the shared cache equal plain `generate` on the full context.
* **Encoding reuse.** A later round of the same step reuses the encoding and is not counted again.
* **Step-by-step equals one-shot.** With k = 1 and a checker that accepts everything, the search reproduces one-shot
  greedy decoding token for token over 4 hemistichs, with the repetition penalty and line splitting included.
* **Candidate parsing.** The line text, the appended ids, the end-of-text replacement and the token limit are handled
  correctly.

### 2b. The search on real lines (`m4_replay.py`)

This runs the search with a generator that proposes, for each hemistich, the released Shaer line and then the human line
of the same prompt. Each poem is then scored by the full M3 pipeline (letters, margin 1.5, tadwīr allowed).

| 600 test prompts, 6,316 hemistichs | result |
|---|---:|
| poems not of the requested length | 0 |
| poems with no forced hemistich | 265 (44.2%) |
| … of them not strictly valid for the evaluation | **0** |
| poems strictly valid | 274 (45.7%; the extra 9 pass the evaluation through its lenient points: tadwīr, īṭāʾ) |
| hemistichs taken from E0 / from the human poem / forced | 3,554 / 1,779 / 983 |
| accounting errors | 0 |

So, on several thousand real hemistichs with partial diacritics, tadwīr, refrains and mismatched lines, the search never
accepted a hemistich the evaluation rejects.

### 2c. The whole pipeline

I ran it with a tiny random Llama saved with a tokenizer byte-identical to Yehia's and a random LoRA
adapter.

**Command lines on CPU:** `m4_decode.py sample` and `search`, then the repository's scorer (`meter_score_batched.py`, the
BiLSTM), then `m3_rescore.py score` in both modes, then `m4_report.py summarize`. Checked:
* the output schema;
* the ids (`shaer_eval_…`, and `shaer_test_…` matching the released rows);
* exact search lengths;
* the accounting (k·R candidates of 16 tokens each when every candidate is cut);
* the integrity table.

**The notebook helpers on a fake Drive:** `pv.m4_dev` run twice (the second run resumes and does nothing), `pv.m4_score`
run twice (the second skips), then `pv.m4_report`, which applies the pre-registered rule.

**Found and fixed: a search poem one hemistich short.** When every candidate of a hemistich was empty, the fallback
accepted an empty line, and empty lines vanish when a poem is split into hemistichs. Empty candidates now rank below
every written line. A hemistich whose candidates are all empty is counted (`empty_lines`), and the integrity check
excludes only those.

### 2d. Tokenizer facts used by the generator

* The only token containing a line break is `<0x0A>` (id 33). The training completions use it between hemistichs, and
  the prompt is a token prefix of the training text.
* E0's hemistichs are 14.0 tokens on average (90th percentile 19, maximum 38), so the 64-token limit per candidate never
  cuts a real line.
* Prompts are 139–205 tokens.

### 2e. Speed

* A candidate check costs about 11 ms, because scans and bayt verdicts are cached per poem.
* Checks stop at the first pass and run in worker processes, each poem staying with one worker so its cache stays warm.
* The scan of a hemistich costs 16 ms. The tadwīr search, which is off during the search, costs about 52 ms per failing
  bayt.

## 3. A verifier error found by M4: tadwīr skipped the margin

**How it was found.** While writing the checker tests, a kāmil sadr followed by a ṭawīl ʿajuz passed as one kāmil bayt.
In M3's wrong-meter control, the tadwīr path accounted for 87 of the 182 bayts accepted in the wrong meter (19,206
bayts in all), and for all 3 control poems that passed.

**The cause.** In letters mode, the connected reading of a bayt (tadwīr: the line read across the hemistich break) was
accepted at any cost up to 6, without the margin rule that every ordinary hemistich must meet. For example, the kāmil
bayt كَم قُلتَ لَستُ بِعائِدٍ في سَوأَةٍ / وَنَذَرتَ فيها ثُمَّ صِرتَ تَعودُ passed as ṭawīl through a connected reading of cost 5.3,
against 1.0 for its ordinary reading.

**The fix** (`prosody/verify.py`). The connected reading may cost at most the margin more than the two hemistichs read
as ordinary lines, each in its likeliest meter. Consequences:
* `m3_rescore.py` records the tadwīr cost and its rival, so verdicts at other margins stay exact;
* the record version is 2, so records on Drive made with the old code are set aside and scored again;
* tests are added for the regression case, the margin recomputation and a genuine mudawwar khafīf bayt that stays
  accepted.

**Measured effect** (bayts accepted through tadwīr, letters):

| | before | after |
|---|---:|---:|
| human reference poems | 309 | 229 |
| released Shaer poems (E0) | 403 | 229 |
| wrong-meter control | 87 | 11 |

**A stricter rule I tried and rejected.** It allowed the written words to be glued into one word only when the result is
a real word. It removed 4 more control bayts but also 45 metrical human bayts. One residual false accept is known: the
kāmil sadr + ṭawīl ʿajuz above passes through a glued non-word read cheaply (cost 0.71). The search itself never accepts
tadwīr.

The full M3 numbers with the fixed verifier are in `M3_report.md` section 11.

## 4. What the GPU runs must show, and their cost

The order is: `m4_selfcheck` (must pass) → `m4_dev` → `m4_score` / `m4_report` on eval → `m4_test_from_choice` →
`m4_score` / `m4_report` on test. All runs resume from Drive.

**Rough A100 times** (from E1's measured 38 poems per minute and the per-step costs):

| step | time |
|---|---|
| self-check | 10 min |
| development sample run (6,400 poems) | 2–3 h |
| each development search run | 10–30 min |
| test search run | about 1.5 h |
| test best-of-2 | about 3 h (about 6 h for best-of-4) |

* Each run also loads the model (a few minutes).
* An L4 decodes about 3–5 times more slowly; per hour it costs fewer compute units, so the total is similar.
* Scoring is CPU work: a CPU runtime is cheapest. On a GPU runtime's CPUs it is about 30 minutes for the sample pool.

## 5. Risks and limitations, known now

* **The selector is the scorer.** Both best-of-N and the search optimise the same letters-mode verifier that scores
  them. The report therefore puts the BiLSTM metric, binding mode and no margin beside it. The binding-mode search run
  shows whether the model can also write correctly vowelled metrical lines.
* **Rejection sampling, not beam search.** The search keeps the first acceptable candidate, so the model's distribution
  decides between valid lines. It cannot plan ahead: a sadr that admits no rhyming ʿajuz is only discovered too late.
  There is no beam (B partial poems); a beam is a follow-up if forced hemistichs concentrate there.
* **Stricter than the evaluation in two places.** The search refuses tadwīr (a mudawwar bayt cannot be written) and
  treats a repeated rhyme word (īṭāʾ) as a defect. Both only reject candidates the evaluation might accept.
* **Refrains.** Two bayts that end in the same word read as a refrain (radīf) poem, as in the evaluation; with īṭāʾ a
  defect, such a candidate is rejected unless the words before it rhyme too.
* **Budget accounting.** Tokens favour neither method. Wall-clock time favours best-of-N slightly here, because the
  search re-encodes its context every hemistich. The search's batches are homogeneous, which favours it, while
  best-of-N's batches wait for their longest poem.
* **The predictions rest on E0.** E0 is the same model and decoding on the test split. The search predictions use pass
  rates measured on prefixes where E0 was doing well, so they lean optimistic.
* **Quality** is measured by distinct-n, repetition and copying. The paper's LLM judge (E4) can be run on the selected
  poems later (paid).
