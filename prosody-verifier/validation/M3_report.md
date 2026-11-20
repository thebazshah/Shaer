# M3 — strict re-scoring of the baselines: validation

## 1. Expectations, written before the full runs

This section was committed before any full M3 run, so that the results below can be read against it. It is not edited
afterwards; deviations are discussed in the results.

Systems run locally (CPU): the human reference poems of the 3,481 test prompts (`reference`), the released Shaer poems
(`e0_released_shaer`), and two controls built from the reference poems. The other baselines (E1–E3) live on the user's
Drive and are scored by the notebook with the same code. Margin 1.5 unless stated; *letters* = written vowels ignored,
*binding* = written vowels count.

### 1a. Exact equalities (any difference is a bug)

| check | expected |
|---|---|
| hemistich counts vs the scorer's `parsed_num_lines`, every row | 0 differences |
| BiLSTM bayt texts vs the verifier's pairing of lines into bayts | 0 misaligned |
| meter verdict recomputed from the record vs the verifier's own verdict at 1.5, every poem | 0 mismatches |
| M3 exact-length rate − the repository's `exact_count_accuracy_pct` on the same poems | 0.0 pp |
| E0 official structural metrics (`summarize_structural.summarise`, unchanged) | 0.9064 / 95.17% / 83.40% / 0.9792 (the paper's numbers, reproduced in M0) |
| reference poems: exact length | 100% |
| BiLSTM bayts on target, E0 / reference (bayts the BiLSTM scored) | 95.37% / 97.30% (M0, M1) |
| M1 cross-check on the BiLSTM-scored bayts, E0 letters / binding | verifier (any form) 55.74% / 41.73%; target form 54.64% / 40.81%; no reading 25.82% / 50.12% |
| M1 cross-check, reference letters / binding | any form 86.20% / 84.97%; target form 82.48% / 81.20%; no reading 5.15% / 9.75% |
| bayts on meter at margins 1.0 ≤ 1.5 ≤ 3.0 ≤ none, every system | monotone (by construction) |
| two summaries of the same records | identical (seeded bootstrap) |

The M1 runs scanned without the scan cache added in M3, so the M1 cross-check also shows that the cache changes no
verdict at full scale.

### 1b. Predictions (from M0 and M1; ranges, not equalities)

| quantity | reference | E0 released Shaer | basis |
|---|---|---|---|
| poems: meter, every bayt (letters) | 53–62% | 12–25% | M1 test split poem level 53.4% (binding, ≥ 4 lines); short poems pass more often; E0 bayt rate 55.7% compounds over ~6 bayts with correlated errors |
| poems: meter (binding) | 50–58% | 5–15% | E0's written vowels contradict the meter (no reading 50% of bayts) |
| poems: rhyme check passes | 84–90% | 40–55% | M1 85.6% on ≥ 4 lines; M0 written-form heuristic: one rawī in 43.8% of E0 poems |
| poems: exact length | 100% | 83.40% (exact) | M0 |
| poems: strictly valid (letters) | 45–55% | 5–15% | meter × rhyme, positively correlated |
| E0 − reference, strictly valid (letters), paired | — | −35 to −50 pp, interval excludes 0 | |
| E0 − reference, BiLSTM bayts | — | about −2 pp | M0 |
| E0 strict validity by requested length | — | decreasing from 1–3 to 11–20 bayts | M0: count and rhyme consistency degrade with length |
| margin effect, E0 letters, strictly valid from margin 1.0 to none | small (≤ 6 pp) | small (≤ 6 pp) | E0 failures are mostly "no reading at all", which no margin fixes |

### 1c. Controls (built from the reference poems)

| control | what is changed | expected |
|---|---|---|
| `control_wrong_meter` | each poem judged against a different meter (form tām) | poems: meter ≤ 3%; bayts on meter ≤ 15% (pairs that overlap by rule, e.g. al-kāmil/al-rajaz, can pass); rhyme and length unchanged from the reference |
| `control_mixed_rhyme` | poems of ≥ 4 bayts: first half joined to the second half of a same-meter, same-form poem with another rawī (written-form heuristic) | rhyme check passes ≤ 10% (heuristic errors can pick a partner with the same rawī); bayts on meter within 2 pp of the reference's long poems; poem-level meter lower than the reference because the two halves may use different ḍarb forms |

## 2. Runs and code

All numbers below come from one full local run of the code at commit `0e8c3d2` (branch `dev`) with the shipped lexicon
`data/lexicon_train.json.gz`, 4 worker processes, on the 3,481 test prompts:

```
python m3_rescore.py score     --scored <m0>/reference/scored.jsonl --system reference          --out-dir <m3>
python m3_rescore.py score     --scored <m0>/e0/scored.jsonl        --system e0_released_shaer  --out-dir <m3>
python m3_rescore.py control   --scored <m0>/reference/scored.jsonl --kind wrong_meter|mixed_rhyme --out <m3>/controls/<kind>.jsonl
python m3_rescore.py score     --scored <m3>/controls/<kind>.jsonl  --system control_<kind> --modes letters --out-dir <m3>
python m3_rescore.py summarize --dir <m3> --reps 1000          # run twice; the two m3_summary.json files are compared
```

The `scored.jsonl` inputs are M0's: the repository's own scorer (`summarize_structural.py` with the BiLSTM, unchanged)
applied to the human reference poems and to the released Shaer poems. The generated report is kept as
[`M3_results_local.md`](M3_results_local.md); the tables below quote it.

M3 also found two errors in the rhyme check (section 6). Both were fixed, and the full run was repeated after each fix. The
meter code did not change, so every meter number is identical in the three runs (0 meter verdicts changed of
3,481 × 2 systems × 2 modes).

Run time: reference ≈ 8 min per mode, E0 ≈ 20 min per mode (4 workers). The scan of a line with *tadwīr* (a word split
across the hemistichs) is about 70% of the time; see section 9.

## 3. Exact equalities (section 1a): all hold

| check | expected | observed |
|---|---|---|
| hemistich counts vs the scorer's `parsed_num_lines` (all 6 system/mode runs and both controls) | 0 | 0 |
| BiLSTM bayt texts vs the verifier's pairing | 0 misaligned | 0 |
| meter verdict recomputed from the record at 1.5 vs the verifier's verdict | 0 mismatches | 0 |
| exact length − `exact_count_accuracy_pct`, unrounded | 0.0 pp | 0.0 pp (all four system/mode runs) |
| E0 official structural metrics | 0.9064 / 95.17% / 83.40% / 0.9792 | 0.9064 / 95.1738% / 83.3956% / 0.9792 |
| reference official metrics (not predicted) | — | 0.9482 / 96.9262% / 100% / 1.0 |
| reference exact length | 100% | 100% |
| BiLSTM bayts on target, E0 / reference | 95.37% / 97.30% | 95.37% / 97.30% |
| M1 cross-check, E0 letters: target form / any form / no reading | 54.64 / 55.74 / 25.82 | 54.64 / 55.74 / 25.82 |
| M1 cross-check, E0 binding | 40.81 / 41.73 / 50.12 | 40.81 / 41.73 / 50.12 |
| M1 cross-check, reference letters | 82.48 / 86.20 / 5.15 | 82.48 / 86.20 / 5.15 |
| M1 cross-check, reference binding | 81.20 / 84.97 / 9.75 | 81.20 / 84.97 / 9.75 |
| bayts on meter monotone over margins 1.0 ≤ 1.5 ≤ 3.0 ≤ none | every system | every system and mode (e.g. reference letters 80.5 ≤ 82.5 ≤ 87.1 ≤ 88.9) |
| two summaries of the same records | identical | byte-identical `m3_summary.json` |
| control inputs rebuilt after the rhyme fixes | (not predicted) | byte-identical to the earlier run: the controls do not depend on the verifier |

The M1 cross-check matches to the hundredth on 19,927 (E0) and 19,198 (reference) bayts. So the scan cache added in M3,
and the reworked pipeline, change no meter verdict at full scale. Every margin-independent field (rhyme, length, no
reading) is the same at all four margins.

## 4. Predictions (section 1b) against the results

Letters mode unless stated; 95% bootstrap intervals over prompts.

| quantity | predicted | observed | |
|---|---|---|---|
| reference: meter, every bayt (letters) | 53–62% | 57.5 [56.0, 59.1] | ✓ |
| E0: meter, every bayt (letters) | 12–25% | 15.9 [14.7, 17.2] | ✓ |
| reference: meter (binding) | 50–58% | 55.1 [53.4, 56.7] | ✓ |
| E0: meter (binding) | 5–15% | 7.8 [6.9, 8.8] | ✓ |
| reference: rhyme check passes | 84–90% | 93.1 [92.3, 93.9] | ✗ above (section 6) |
| E0: rhyme check passes | 40–55% | 38.1 [36.6, 39.6] | ✗ below |
| exact length, reference / E0 | 100% / 83.40% | 100.0 / 83.4 | ✓ |
| reference: strictly valid (letters) | 45–55% | 55.3 [53.8, 56.9] | ✗ by 0.3 pp |
| E0: strictly valid (letters) | 5–15% | 9.2 [8.3, 10.3] | ✓ |
| E0 − reference, strictly valid, paired | −35 to −50 pp, excluding 0 | −46.1 [−47.8, −44.3] | ✓ |
| E0 − reference, BiLSTM bayts, paired | about −2 pp | −1.9 [−2.6, −1.2] | ✓ |
| E0 strictly valid by length (1–3 / 4–6 / 7–10 / 11–20 bayts) | decreasing | 17.9 / 2.5 / 0.6 / 0.0 | ✓ |
| margin effect 1.0 → none, strictly valid, E0 | ≤ 6 pp | 8.6 → 13.5 (+4.9) | ✓ |
| margin effect 1.0 → none, strictly valid, reference | ≤ 6 pp | 51.4 → 69.7 (+18.3) | ✗ |

Eleven of fourteen predictions hold. The four misses:

* **Reference rhyme above the range.** The range came from M1's 85.63%, which was produced by the rhyme code before the
  fixes in section 6. Run with that code, M3 gave 85.38%, inside the range. The range was right for the code it was
  made for, but that code wrongly flagged many human poems. The fixed check passes 93.1%.
* **E0 rhyme below the range.** The basis was M0's written-form heuristic: one rawī in 43.8% of E0 poems. The full check
  also fails poems with a single rawī for iqwāʾ, iṣrāf, sinād and mixed muṭlaqa/muqayyada, so it should sit below that
  number. The lower end of the range did not allow for this. Before the second fix the observed value was 41.3%, inside
  the range, but only because of false passes (section 6).
* **Reference strictly valid 0.3 pp above the range.** This follows from the higher rhyme pass rate. Meter, the main
  term, is in range.
* **The reference margin effect is three times the prediction.** The reasoning ("failures are mostly bayts with no
  reading at all, which no margin fixes") holds for E0: 25.8% of its bayts have no reading, and the margin moves its
  strict rate by only 4.9 pp. It does not hold for the human poems. Only 5.2% of their bayts have no reading. Most
  rejected human bayts *do* scan in the requested meter, but a more common reading of a word scans better in another
  meter, by more than the margin. Bayts on meter go from 80.5% to 88.9% between margin 1.0 and no margin. Over the
  roughly 6 bayts of a human poem, that compounds to +18 pp of poems. This is the weakness M1 identified (the margin
  rule rejecting metrical bare-text lines), now measured at the poem level. It has two consequences:
  * the strict rate of human poems (55%) is a lower bound on their true validity;
  * comparisons between systems should also be read at margin none, where the rate is an upper bound (69.7% for humans
    against 13.5% for E0; the gap is large under both).

## 5. Controls (section 1c)

| control | expectation | observed | |
|---|---|---|---|
| wrong meter: poems on meter | ≤ 3% | 0.1 [0.0, 0.2] | ✓ |
| wrong meter: bayts on meter | ≤ 15% | 0.9 [0.7, 1.3] (6.1 with no margin) | ✓ |
| wrong meter: length | unchanged | 100%, paired 0.0 | ✓ |
| wrong meter: rhyme | unchanged | 93.13% = reference 93.13%; paired 0.0 [−0.1, 0.1] | ✓ (2 poems swap, see below) |
| mixed rhyme: rhyme passes | ≤ 10% | 0.27% (4 of 1,487) [0.1, 0.5] | ✓ |
| mixed rhyme: bayts on meter vs the same prompts' reference | within 2 pp | −0.3 [−1.5, 0.8] | ✓ |
| mixed rhyme: poem meter lower | lower | 39.3% vs paired reference: −7.3 [−9.7, −4.8] | ✓ |
| mixed rhyme: strictly valid | (not predicted) | 0.0% | |

* **Wrong meter.** A human poem judged against another meter almost never passes: 3 poems of 3,481, all of them
  1–3 bayts long. Bayts pass at 0.9%, mainly from meters that share feet: al-kāmil passes 5.1% of bayts, and
  al-ramal/al-mutadārik pass about 2%. 83.6% of the bayts have no reading at all in the wrong meter.
  * The rhyme rate is the same, but two poems swap verdicts (one fails, one passes). The rhyme analysis reads the last
    word with the scansion chosen in the requested meter, so judging against another meter can change how that word
    is read. This is expected, and it affects 0.06% of poems.
* **Mixed rhyme.** The controls are built with M0's written-form heuristic, not with the verifier, so their inputs are
  identical across the three runs. Rhyme passes were 3.7% after the first fix and are 0.27% after the second; the four
  remaining passes are listed in section 6. The meter is unaffected (−0.3 pp on bayts). The poem-level meter drops by
  7 pp because the two halves can use different ḍarb forms, which the consistency check rightly rejects.

## 6. The rhyme check: two errors found by M3, fixed and audited

M3 was the first time the rhyme check ran on many generated poems and on the full human test split with all checks
side by side. Two errors showed up, in opposite directions.

**Round 1: false flags on human poems (commit `1c707ba`).**
* What was wrong: the rhyme word's ridf and taʾsīs were read from the scanner's chosen reading. In letters mode that
  reading comes from bare letters, and the end of a line rarely constrains it, so a written alif could disappear or a
  long vowel could become a consonant.
* How it showed: sinād flags on human poems that the text did not support.
* The fix: the scanner's reading now only locates the rawī, and the letters before it are read from the written word.
  Further details:
  * a wāw or yāʾ counts as a consonant only on written evidence (a vowel, a shadda, first letter, or after a sukūn or a
    sākin alif); an unmarked one is open, so it can vote but is never flagged;
  * a taʾsīs alif must be pronounced;
  * an alif written as آ, and the unwritten alifs of الله and ذلك, count as ridf.
* Audit after the fix:
  * E0: 20 of 20 sampled sinād flags are real faults;
  * human poems: about 22 of 30 remaining flags are real. The false ones mostly come from wrong written vowels in the
    corpus (المَنْونِ, سُليَمى) and from ambiguous words (باسمه).

**Round 2: false passes, found through the mixed-rhyme control (commit `0e8c3d2`).**
* What was wrong: 3.7% of the mixed poems still passed. When the bayts split between two rhyme consonants, every
  bayt's *second* candidate could win the vote and become the shared rawī: the alif of iṭlāq (القمرا … هنا), or a
  pronoun hāʾ (بعقدِهِ … بسلمِهِ, كريمه … قتيله).
* This is also a reward-hacking hole for M4: ending every line in ـا would pass.
* The fix: such readings are now *weak*. A weak reading counts only when at least three bayts in four take the same
  letter as a *strong* reading:
  * a maqṣūra alif (ى, or ا after a stem of at most two letters, as in دعا, الرضا, هنا);
  * a hāʾ after a long vowel or a sukūn (تراهُ, منهُ);
  * a root hāʾ (الله, a two-letter stem);
  * ـية, or an unmarked ـيه.
* The rawī is the letter that most bayts take as a strong reading. This also fixed a genuine maqṣūra poem
  (الهوى / النوى / الجوى / المنى) that had been flagged because و outvoted the alif. A voweled tāʾ marbūṭa is a tāʾ.
* Tests: 12 new rhyme cases (6 fail on the earlier code); 193 tests pass.

**Effect of each round** (full runs, letters; meter identical throughout):

| | before the fixes | after round 1 | after round 2 (final) |
|---|---:|---:|---:|
| reference: rhyme passes | 85.38% | 93.36% | 93.13% |
| reference: strictly valid | 52.28% | 55.47% | 55.33% |
| E0: rhyme passes | 37.49% | 41.28% | 38.12% |
| E0: strictly valid | 9.42% | 9.94% | 9.25% |
| E0 − reference, strictly valid (pp) | −42.9 | −45.5 [−47.3, −43.7] | −46.1 [−47.8, −44.3] |
| mixed-rhyme control: rhyme passes | (not run) | 3.70% | 0.27% |
| mixed-rhyme control: strictly valid | (not run) | 1.34% | 0.00% |
| M1 test split, poems with ≥ 4 hemistichs: rhyme passes | 85.63% | 92.0% | 91.79% |
| M1 planted rhyme breaks flagged | 100.0% | 100.0% | 100.0% |

The M1 validation (`validate_verifier.py`, test split) was re-run with the final code. All of its meter numbers are
unchanged (87.1% of hemistichs accepted, 86.09% identification, 71.89% of broken copies rejected, 53.36% of poems with
every bayt on meter).

**Audit of round 2.** Every poem whose verdict changed between rounds 1 and 2 was listed.
* E0: 113 poems changed from pass to fail and 3 from fail to pass (letters).
  * Two independent random samples of 30 pass→fail changes: 60 of 60 are real mixed rawīs, typically an iṭlāq alif
    after different consonants.
  * The 3 fail→pass changes:
    * two are correct: a refrain detected with the rhyme on إلى / يشتهى, and استوى / فاشتهى;
    * one, يشتهيه / سويّه, is accepted under the latitude kept for unmarked ـيه.
* Reference: 8 poems changed from pass to fail and none the other way.
  * 5 are correct: صدغها / دمعها; بها / بعدها / انتهى / لها; النُهى / اِنتَهى / وَزِنتَها / لَها; بطشه / بكشبه; فيْ / كَيْ.
  * 1 is debatable: وذا / حيثما.
  * 2 are false flags:
    * المنبّه / مشوّه: a root hāʾ after a voweled letter is read as waṣl;
    * حبّيها / فيها: a reading artefact.
  * Net cost on human poems: about 2 false flags in 3,481 poems.

**The remaining false passes** in the mixed control (4 of 1,487, all audited):
* `000839`, وقفا / السلفا with الجنى / الأذى: the stem rule takes the root و of وقفا for the conjunction, so وقفا
  looks like a two-letter stem and its alif like a maqṣūra.
* `002031`, ـما (مغرما, الدرهما, but also الظما, الدمى, اللمى) with ـى after various letters: five of the first half's
  rhyme words are themselves maqṣūra, which meets the three-in-four threshold. This is a borderline case: some
  prosodists accept a maqṣūra alif as waṣl in a poem rhymed on م.
* `000522`, معانيهِ / لمناغيهِ with البقّه / دقه: with the written vowels ignored, دقه can be read as daqh (a hāʾ after
  a sukūn, which is a rawī).
* `002131`: the first poem's ʿajuz lines are literally "..." in the dataset (4 reference poems have such lines). Bayts
  with no rhyme word are skipped by the rhyme check; the meter check rejects them, so the poem is not strictly valid.

**Known recall losses, accepted for precision.**
* An unmarked final و/ي is never flagged for sinād.
* A root hāʾ after a voweled letter (المنبّه) is read as waṣl.
* An unmarked ـيه is ambiguous between -iyyah and -īh, and both are accepted.

The two corrections move the reference and E0 rhyme rates in opposite directions, by 8 pp and 3 pp. This is why
section 1b's rhyme ranges, anchored on the earlier code, missed.

## 7. Main results

The full tables are in [`M3_results_local.md`](M3_results_local.md). In letters mode, margin 1.5:

| | reference (human) | E0 released Shaer | paired difference |
|---|---:|---:|---:|
| Shaer's metric: meter conformity (BiLSTM) | 0.9482 | 0.9064 | |
| Shaer's metric: bayts on target (BiLSTM) | 97.3% | 95.4% | −1.9 [−2.6, −1.2] |
| bayts on meter (verifier) | 82.5% | 54.6% | −27.8 [−29.2, −26.4] |
| bayts with no reading at all in the meter | 5.2% | 25.8% | +20.6 [19.6, 21.7] |
| poems: every bayt on meter, form kept | 57.5% | 15.9% | −41.6 [−43.5, −39.7] |
| poems: rhyme check passes | 93.1% | 38.1% | −55.0 [−56.8, −53.3] |
| poems: exact length | 100% | 83.4% | −16.6 [−17.8, −15.4] |
| **poems: strictly valid** | **55.3%** | **9.2%** | **−46.1 [−47.8, −44.3]** |
| strictly valid, no margin (upper bound) | 69.7% | 13.5% | |
| strictly valid, binding (written vowels count) | 52.9% | 5.1% | −47.8 [−49.4, −45.9] |

**Interpretation.**
* **Shaer's metric hides most failures.** By the BiLSTM metric, the released Shaer poems are 1.9 pp behind human
  poems. By the verifier, the gap is 28 pp on bayts and 46 pp on whole poems. The BiLSTM accepts 95% of E0's bayts, but
  a quarter of E0's bayts have no reading at all in the requested meter. M1's manual check supports these rejections.
* **Rhyme is E0's largest single failure.** Only 38% of E0 poems pass, against 93% of human poems. The most frequent
  E0 issues are a mixed rawī (52% of poems), sinād al-ridf (30%), sinād al-taʾsīs (13%), iṣrāf (9%) and īṭāʾ (8%).
  For human poems the most frequent are īṭāʾ (4.2%, a warning only), rawī (3.5%) and sinād al-ridf (3.3%).
* **Length.** Rhyme and count fail more often as requested length grows, so E0's strict rate goes from 17.9% (1–3
  bayts) to 0% (11–20 bayts). Over the same lengths, human poems go from 70.7% to 25.8%. Every-bayt validity
  compounds, so long human poems are also often rejected; part of this is the margin effect of section 4.
* **No drift by position.** E0's bayt rate is flat across positions (52.5–57.5% from bayt 1 to 10+). Errors do not pile
  up later in the poem. They are spread over the poem, and 45% of failing E0 bayts already break at the first foot.
* **By meter.** E0's best meter is al-ṭawīl (72% of bayts, 15.1% of poems strictly valid). Al-ramal, al-rajaz,
  al-hazaj, al-munsariḥ and al-sarīʿ are the weakest (30–36% of bayts). Human poems in al-rajaz and al-ramal also score lowest
  (60% and 64% of bayts), which suggests remaining verifier strictness on those meters as well.
* **Binding mode.** Counting E0's written vowels makes it much worse: 50% of its bayts have no reading. The vowels
  E0 writes often contradict the meter its letters could follow. Human poems lose only 1–2 pp.

## 8. Limitations

* **No expert gold set yet (M2).** Precision figures come from my own audits of sampled verdicts, not from a
  prosodist's. The rule tables and the rhyme decisions (weak readings, the three-in-four threshold, ـيه) need the
  expert's sign-off.
* **The margin rule.** With bare text, the margin is the largest source of rejections of human poems (section 4).
  Strict rates of human poems are lower bounds. Margin none gives the upper bound and is reported next to it.
* **Corpus noise.** The human reference includes typing errors, wrong diacritics, truncated lines ("...") and a few
  mislabelled meters. The reference rate is not a ceiling of 100%.
* **One sample per prompt.** The released E0 poems have one sample per prompt, so pass@k is not reported for them.
  `summarize` computes pass@k when a run has several.
* **E1–E3 are not scored here.** Their outputs live on the user's Drive; the notebook scores them with the same code
  (section 10).

## 9. Performance

On 4 local workers, scoring runs at 1.7–7.7 poems/s depending on the system and mode (the wrong-meter control is the slowest, because no reading is found quickly). A 2-core Colab CPU runtime should
need roughly 30–60 minutes per system and mode, as the notebook says.

About 70% of the time is the tadwīr search (a word split across the two hemistichs). For M4, where the verifier runs
inside decoding, two things matter:
* cache that search per hemistich prefix, or skip it when the hemistich boundary falls between words and both halves
  already scan;
* do not prune it away, because tadwīr is legitimate in several meters.

## 10. Running M3 on Colab for E1–E3

1. Use a CPU runtime (no GPU needed). Run sections 1–3 of `prosody_main.ipynb` (install, Drive, setup), then section 4
   (M0) if it has not been run.
2. Section 6: `pv.m3_rescore(pvc)`. It finds the reference, E0, and every complete baseline run on Drive (E1–E3), and
   writes each scored poem at once to `MyDrive/shaer-prosody/m3/<system>/<mode>.jsonl`.
   * Re-running the cell resumes.
   * Records written by another version of the verifier or lexicon are renamed `.stale-<time>.jsonl` and re-scored,
     never mixed in.
   * `pv.m3_rescore(pvc, limit=200)` gives a quick look.
3. `pv.m3_report(pvc)` writes `m3_report.md` and `m3_summary.json` in the same folder. Check that the integrity table is
   all 0, and that the reference and E0 rows match [`M3_results_local.md`](M3_results_local.md).

## 11. Later change: the margin now applies to tadwīr (found in M4)

M4's validation found a verifier error. In letters mode, the connected reading of a bayt (tadwīr) was accepted without
the margin rule, so a bayt in one meter could pass as another. The fix and its analysis are in `M4_report.md`
section 3.

**What changed.**
* `prosody/verify.py` (commit `e26e236`): the connected reading may cost at most the margin more than the two hemistichs
  read as ordinary lines, each in its likeliest meter.
* M3 records now carry the tadwīr cost and its rival (record version 2), so margins can still be recomputed exactly.
  Records on Drive made with the earlier code are set aside and scored again by `pv.m3_rescore`.

**The re-run.** The full local run was repeated with this code. Integrity checks are all 0, the two summaries are
byte-identical, and the control inputs are byte-identical to the earlier runs. The tables in
[`M3_results_local.md`](M3_results_local.md) are now those of this run. Sections 2–10 above keep the numbers of the
earlier code; this is what changed (letters, margin 1.5 unless stated):

| | before | after |
|---|---:|---:|
| reference: strictly valid | 55.33% | 54.93% |
| reference: meter, every bayt | 57.51% | 57.08% |
| reference: bayts on meter | 82.45% | 82.09% |
| E0: strictly valid | 9.25% | 8.99% |
| E0: meter, every bayt | 15.94% | 15.51% |
| E0: bayts on meter | 54.64% | 53.78% |
| E0: rhyme passes | 38.12% | 38.18% |
| E0 − reference, strictly valid (pp) | −46.1 [−47.8, −44.3] | −45.9 [−47.7, −44.2] |
| wrong-meter control: poems on meter | 0.09% (3 poems) | 0.00% |
| wrong-meter control: bayts on meter | 0.95% | 0.57% |
| mixed-rhyme control: bayts on meter (paired vs reference: −0.3 pp before) | 82.07% | 81.64% |
| binding mode, E0 / reference strictly valid | 5.08% / 52.86% | 5.06% / 52.48% |
| any margin = none (every number) | | unchanged |

Notes on the table:
* **Margin none.** These numbers are unchanged, as they must be: with no margin every tadwīr reading is still
  accepted.
* **Rhyme.** It moves only where a bayt's last word is now read through its ordinary scansion instead of the
  connected one (2 E0 poems, both now passing).
* **The M1 cross-check (section 3).** It now reads E0 letters 53.79 / 54.81 / 25.82 and reference letters
  82.13 / 85.83 / 5.15, because M1's numbers were made with the earlier code. The no-reading column is unchanged.
* **The M1 test-split validation** was re-run as well: 52.93% of poems with every bayt on meter (was 53.36%). The
  rhyme rate (91.79%), the planted breaks (100.0%) and every hemistich-level number are unchanged.

None of this changes a conclusion of sections 4–7. The predictions keep their verdicts, except that the human strict
validity (54.93%) is now inside its predicted range of 45–55%.
