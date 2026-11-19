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
