# M3 — strict re-scoring with the prosody verifier

Margin 1.5 unless stated; 95% bootstrap intervals over prompts (1000 resamples) in brackets. First sample per prompt. *letters*: written vowels ignored (the conservative meter verdict); *binding*: written vowels count.

**Strictly valid** = every bayt in the requested meter and form (ḍarb/ʿarūḍ form kept), rhyme check passes, exact number of hemistichs, no odd hemistich. *No reading*: share of bayts with a hemistich that cannot be read in the requested meter at all (any form, any cost; independent of the margin).

## Poems, letters

| system | poems | strictly valid | meter (form) | meter (any form) | rhyme | exact length |
|---|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 1,487 | 0.0 [0.0, 0.0] | 39.3 [36.9, 41.7] | 39.6 [37.0, 42.0] | 0.3 [0.1, 0.5] | 100.0 [100.0, 100.0] |
| control_wrong_meter | 3,481 | 0.1 [0.0, 0.2] | 0.1 [0.0, 0.2] | 0.1 [0.0, 0.3] | 93.1 [92.3, 93.9] | 100.0 [100.0, 100.0] |
| e0_released_shaer | 3,481 | 9.2 [8.3, 10.3] | 15.9 [14.7, 17.2] | 16.3 [15.0, 17.5] | 38.1 [36.6, 39.6] | 83.4 [82.2, 84.6] |
| reference | 3,481 | 55.3 [53.8, 56.9] | 57.5 [56.0, 59.1] | 59.9 [58.3, 61.5] | 93.1 [92.3, 93.9] | 100.0 [100.0, 100.0] |

## Bayts, letters

| system | bayts on meter (verifier) | any form | no reading in the meter | BiLSTM (Shaer metric) |
|---|---:|---:|---:|---:|
| control_mixed_rhyme | 82.1 [81.0, 83.1] | 86.1 [85.2, 87.0] | 5.4 [4.8, 6.0] | – |
| control_wrong_meter | 0.9 [0.7, 1.3] | 1.2 [0.9, 1.5] | 83.6 [82.5, 84.6] | – |
| e0_released_shaer | 54.6 [53.5, 55.7] | 55.7 [54.6, 56.7] | 25.8 [25.0, 26.6] | 95.4 [95.0, 95.8] |
| reference | 82.5 [81.3, 83.6] | 86.2 [85.3, 87.0] | 5.2 [4.6, 5.7] | 97.3 [96.7, 97.8] |

**Margin sensitivity (letters)**: strictly valid poems / bayts on meter

| system | margin 1.0 | margin 1.5 | margin 3.0 | margin none |
|---|---:|---:|---:|---:|
| control_mixed_rhyme | 0.0 / 80.4 | 0.0 / 82.1 | 0.1 / 86.8 | 0.1 / 88.5 |
| control_wrong_meter | 0.1 / 0.9 | 0.1 / 0.9 | 0.6 / 2.2 | 1.7 / 6.1 |
| e0_released_shaer | 8.6 / 52.6 | 9.2 / 54.6 | 12.0 / 61.7 | 13.5 / 65.5 |
| reference | 51.4 / 80.5 | 55.3 / 82.5 | 66.3 / 87.1 | 69.7 / 88.9 |

**Paired difference to the human reference poems (letters)**, percentage points:

| system | prompts | strictly valid | meter | rhyme | exact length | bayts on meter | no reading | BiLSTM bayts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 1,487 | -44.0 [-46.2, -41.3] | -7.3 [-9.7, -4.8] | -90.5 [-91.9, -89.0] | 0.0 [0.0, 0.0] | -0.3 [-1.5, 0.8] | -0.1 [-0.8, 0.5] | – |
| control_wrong_meter | 3,481 | -55.2 [-56.8, -53.7] | -57.4 [-59.0, -55.9] | 0.0 [-0.1, 0.1] | 0.0 [0.0, 0.0] | -81.5 [-82.6, -80.3] | 78.4 [77.2, 79.5] | – |
| e0_released_shaer | 3,481 | -46.1 [-47.8, -44.3] | -41.6 [-43.5, -39.7] | -55.0 [-56.8, -53.3] | -16.6 [-17.8, -15.4] | -27.8 [-29.2, -26.4] | 20.6 [19.6, 21.7] | -1.9 [-2.6, -1.2] |

## Poems, binding

| system | poems | strictly valid | meter (form) | meter (any form) | rhyme | exact length |
|---|---:|---:|---:|---:|---:|---:|
| e0_released_shaer | 3,481 | 5.1 [4.4, 5.9] | 7.8 [6.9, 8.8] | 8.0 [7.1, 9.0] | 33.8 [32.4, 35.3] | 83.4 [82.2, 84.6] |
| reference | 3,481 | 52.9 [51.1, 54.5] | 55.1 [53.4, 56.7] | 57.7 [56.0, 59.3] | 92.3 [91.4, 93.2] | 100.0 [100.0, 100.0] |

## Bayts, binding

| system | bayts on meter (verifier) | any form | no reading in the meter | BiLSTM (Shaer metric) |
|---|---:|---:|---:|---:|
| e0_released_shaer | 40.8 [39.8, 41.8] | 41.7 [40.8, 42.7] | 50.1 [49.1, 51.2] | 95.4 [95.0, 95.8] |
| reference | 81.2 [80.0, 82.3] | 84.9 [84.1, 85.8] | 9.8 [9.1, 10.4] | 97.3 [96.7, 97.8] |

**Margin sensitivity (binding)**: strictly valid poems / bayts on meter

| system | margin 1.0 | margin 1.5 | margin 3.0 | margin none |
|---|---:|---:|---:|---:|
| e0_released_shaer | 4.8 / 40.2 | 5.1 / 40.8 | 6.0 / 43.4 | 6.5 / 44.9 |
| reference | 51.3 / 80.2 | 52.9 / 81.2 | 57.2 / 83.6 | 58.6 / 84.5 |

**Paired difference to the human reference poems (binding)**, percentage points:

| system | prompts | strictly valid | meter | rhyme | exact length | bayts on meter | no reading | BiLSTM bayts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e0_released_shaer | 3,481 | -47.8 [-49.4, -45.9] | -47.2 [-48.9, -45.5] | -58.5 [-60.2, -56.7] | -16.6 [-17.8, -15.4] | -40.4 [-41.9, -39.0] | 40.4 [39.2, 41.7] | -1.9 [-2.6, -1.2] |

## Shaer's own metrics (repository definitions) next to the strict ones (letters)

| system | meter conformity | base-meter accuracy | exact count | count adherence | strictly valid | meter (form) |
|---|---:|---:|---:|---:|---:|---:|
| e0_released_shaer | 0.9064 | 95.17% | 83.40% | 0.9792 | 9.2% | 15.9% |
| reference | 0.9482 | 96.93% | 100.00% | 1.0 | 55.3% | 57.5% |

## By requested length (letters)

| system | bayts | poems | strictly valid | meter | rhyme | exact length | bayts on meter | BiLSTM bayts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 4-6 | 805 | 0.0 | 47.45 | 0.37 | 100.0 | 81.19 | – |
| control_mixed_rhyme | 7-10 | 419 | 0.0 | 36.04 | 0.0 | 100.0 | 82.51 | – |
| control_mixed_rhyme | 11-20 | 263 | 0.0 | 19.77 | 0.38 | 100.0 | 82.56 | – |
| control_wrong_meter | 1-3 | 1,668 | 0.18 | 0.18 | 96.52 | 100.0 | 0.84 | – |
| control_wrong_meter | 4-6 | 805 | 0.0 | 0.0 | 92.42 | 100.0 | 0.63 | – |
| control_wrong_meter | 7-10 | 473 | 0.0 | 0.0 | 91.97 | 100.0 | 0.89 | – |
| control_wrong_meter | 11-20 | 535 | 0.0 | 0.0 | 84.67 | 100.0 | 1.18 | – |
| e0_released_shaer | 1-3 | 1,668 | 17.93 | 27.88 | 57.31 | 98.44 | 49.96 | 97.01 |
| e0_released_shaer | 4-6 | 805 | 2.48 | 8.7 | 31.06 | 93.17 | 51.58 | 94.99 |
| e0_released_shaer | 7-10 | 473 | 0.63 | 3.38 | 16.7 | 75.26 | 55.43 | 94.29 |
| e0_released_shaer | 11-20 | 535 | 0.0 | 0.75 | 7.85 | 28.97 | 57.62 | 95.36 |
| reference | 1-3 | 1,668 | 70.74 | 72.54 | 96.58 | 100.0 | 82.0 | 97.24 |
| reference | 4-6 | 805 | 50.93 | 53.54 | 92.42 | 100.0 | 80.72 | 96.44 |
| reference | 7-10 | 473 | 41.86 | 43.97 | 91.75 | 100.0 | 82.44 | 97.17 |
| reference | 11-20 | 535 | 25.79 | 28.6 | 84.67 | 100.0 | 83.49 | 97.8 |

## Bayts on meter by position in the poem (letters)

| system | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 82.65 | 82.04 | 82.52 | 83.39 | 82.14 | 82.54 | 82.99 | 80.42 | 82.11 | 79.4 |
| control_wrong_meter | 0.89 | 0.83 | 1.11 | 0.83 | 0.77 | 0.68 | 1.19 | 1.06 | 0.54 | 1.26 |
| e0_released_shaer | 53.78 | 52.73 | 52.48 | 54.95 | 55.38 | 55.6 | 55.39 | 57.49 | 55.71 | 56.53 |
| reference | 82.28 | 82.35 | 82.54 | 82.29 | 83.34 | 82.28 | 83.04 | 82.86 | 83.51 | 81.88 |

## Rhyme and form issues (letters, share of poems)

| system | arud_changes | count | darb_changes | iqwa | israf | ita | mutlaqa_muqayyada | odd_tail | radif_missing | rawi | sinad_ridf | sinad_tasis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 5.78 | 0.0 | 10.56 | 1.41 | 7.87 | 3.7 | 0.2 | 0.0 | 0.0 | 99.66 | 48.22 | 19.84 |
| control_wrong_meter | 1.18 | 0.0 | 1.21 | 0.03 | 0.57 | 4.22 | 0.09 | 0.0 | 0.14 | 3.48 | 3.33 | 2.59 |
| e0_released_shaer | 1.75 | 16.6 | 8.07 | 2.15 | 8.91 | 7.7 | 0.29 | 2.36 | 0.78 | 52.14 | 29.91 | 12.96 |
| reference | 1.75 | 0.0 | 1.41 | 0.06 | 0.66 | 4.22 | 0.09 | 0.0 | 0.14 | 3.48 | 3.27 | 2.61 |

## By meter (letters; strictly valid / bayts on meter / BiLSTM bayts)

| meter | control_mixed_rhyme | control_wrong_meter | e0_released_shaer | reference |
|---|---|---|---|---|
| tawil | 0.0 / 89.57 / – (352) | 0.0 / 1.36 / – (191) | 15.1 / 72.04 / 99.96 (841) | 66.23 / 89.87 / 98.2 (841) |
| kamil | 0.0 / 80.75 / – (284) | 0.0 / 5.12 / – (199) | 7.4 / 53.61 / 95.04 (649) | 60.25 / 82.97 / 97.74 (649) |
| basit | 0.0 / 85.17 / – (230) | 0.0 / 1.28 / – (173) | 9.45 / 57.8 / 98.29 (529) | 61.44 / 84.57 / 97.75 (529) |
| wafir | 0.0 / 86.27 / – (149) | 0.0 / 0.0 / – (209) | 9.41 / 52.9 / 96.54 (340) | 60.88 / 87.06 / 94.93 (340) |
| khafif | 0.0 / 84.31 / – (141) | 0.0 / 0.47 / – (188) | 8.14 / 51.11 / 98.22 (295) | 51.19 / 83.94 / 97.61 (295) |
| mujtath | 0.0 / 80.0 / – (24) | 0.0 / 0.14 / – (255) | 4.92 / 54.18 / 86.96 (61) | 57.38 / 81.97 / 95.92 (61) |
| mudari | – | 0.0 / 0.0 / – (245) | – | – |
| rajaz | 0.0 / 62.24 / – (53) | 0.0 / 1.39 / – (244) | 6.57 / 33.29 / 84.24 (137) | 27.01 / 60.05 / 94.44 (137) |
| madid | 0.0 / 70.77 / – (9) | 0.0 / 0.51 / – (241) | 0.0 / 40.0 / 65.26 (17) | 41.18 / 75.27 / 94.62 (17) |
| muqtadab | – | 0.0 / 0.0 / – (230) | – | – |
| mutaqarib | 0.0 / 70.18 / – (58) | 0.0 / 0.68 / – (229) | 8.33 / 45.22 / 96.15 (156) | 33.97 / 71.68 / 97.39 (156) |
| munsarih | 0.0 / 80.08 / – (37) | 0.0 / 0.63 / – (219) | 1.19 / 36.45 / 82.0 (84) | 40.48 / 78.77 / 97.17 (84) |
| hazaj | 0.0 / 73.58 / – (9) | 0.0 / 0.17 / – (218) | 5.56 / 33.65 / 80.77 (18) | 44.44 / 68.93 / 92.23 (18) |
| sari | 0.0 / 77.43 / – (78) | 0.0 / 0.54 / – (211) | 5.09 / 36.48 / 87.71 (216) | 39.81 / 75.32 / 96.88 (216) |
| mutadarik | – | 0.93 / 1.59 / – (215) | – | – |
| ramal | 0.0 / 58.09 / – (63) | 0.47 / 1.94 / – (214) | 2.17 / 29.59 / 86.84 (138) | 25.36 / 64.16 / 97.71 (138) |

## Integrity checks (all must be 0)

| system/mode | line counts differing from Shaer's | BiLSTM bayts misaligned | margin recomputation mismatches | exact length vs official (pp) | errors |
|---|---:|---:|---:|---:|---:|
| control_mixed_rhyme/letters | 0 | 0 | 0 | – | 0 |
| control_wrong_meter/letters | 0 | 0 | 0 | – | 0 |
| e0_released_shaer/letters | 0 | 0 | 0 | 0.0 | 0 |
| e0_released_shaer/binding | 0 | 0 | 0 | 0.0 | 0 |
| reference/letters | 0 | 0 | 0 | 0.0 | 0 |
| reference/binding | 0 | 0 | 0 | 0.0 | 0 |
