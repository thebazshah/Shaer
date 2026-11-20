# M3 — strict re-scoring with the prosody verifier

Margin 1.5 unless stated; 95% bootstrap intervals over prompts (1000 resamples) in brackets. First sample per prompt. *letters*: written vowels ignored (the conservative meter verdict); *binding*: written vowels count.

**Strictly valid** = every bayt in the requested meter and form (ḍarb/ʿarūḍ form kept), rhyme check passes, exact number of hemistichs, no odd hemistich. *No reading*: share of bayts with a hemistich that cannot be read in the requested meter at all (any form, any cost; independent of the margin).

## Poems, letters

| system | poems | strictly valid | meter (form) | meter (any form) | rhyme | exact length |
|---|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 1,487 | 0.0 [0.0, 0.0] | 38.8 [36.3, 41.1] | 39.1 [36.5, 41.4] | 0.3 [0.1, 0.5] | 100.0 [100.0, 100.0] |
| control_wrong_meter | 3,481 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] | 93.1 [92.3, 93.9] | 100.0 [100.0, 100.0] |
| e0_released_shaer | 3,481 | 9.0 [8.0, 10.0] | 15.5 [14.3, 16.8] | 15.8 [14.5, 17.0] | 38.2 [36.6, 39.6] | 83.4 [82.2, 84.6] |
| reference | 3,481 | 54.9 [53.4, 56.5] | 57.1 [55.6, 58.7] | 59.4 [57.9, 61.1] | 93.1 [92.3, 93.9] | 100.0 [100.0, 100.0] |

## Bayts, letters

| system | bayts on meter (verifier) | any form | no reading in the meter | BiLSTM (Shaer metric) |
|---|---:|---:|---:|---:|
| control_mixed_rhyme | 81.6 [80.5, 82.7] | 85.7 [84.8, 86.6] | 5.4 [4.8, 6.0] | – |
| control_wrong_meter | 0.6 [0.4, 0.8] | 0.7 [0.5, 0.9] | 83.6 [82.5, 84.6] | – |
| e0_released_shaer | 53.8 [52.6, 54.8] | 54.8 [53.7, 55.8] | 25.8 [25.0, 26.6] | 95.4 [95.0, 95.8] |
| reference | 82.1 [81.0, 83.2] | 85.8 [85.0, 86.7] | 5.2 [4.6, 5.7] | 97.3 [96.7, 97.8] |

**Margin sensitivity (letters)**: strictly valid poems / bayts on meter

| system | margin 1.0 | margin 1.5 | margin 3.0 | margin none |
|---|---:|---:|---:|---:|
| control_mixed_rhyme | 0.0 / 79.9 | 0.0 / 81.6 | 0.1 / 86.6 | 0.1 / 88.5 |
| control_wrong_meter | 0.0 / 0.5 | 0.0 / 0.6 | 0.5 / 2.0 | 1.7 / 6.1 |
| e0_released_shaer | 8.4 / 51.7 | 9.0 / 53.8 | 11.9 / 61.3 | 13.6 / 65.5 |
| reference | 51.0 / 80.1 | 54.9 / 82.1 | 66.0 / 87.0 | 69.7 / 88.9 |

**Paired difference to the human reference poems (letters)**, percentage points:

| system | prompts | strictly valid | meter | rhyme | exact length | bayts on meter | no reading | BiLSTM bayts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 1,487 | -43.4 [-45.7, -40.8] | -7.2 [-9.6, -4.8] | -90.5 [-91.9, -89.0] | 0.0 [0.0, 0.0] | -0.4 [-1.5, 0.7] | -0.1 [-0.8, 0.5] | – |
| control_wrong_meter | 3,481 | -54.9 [-56.5, -53.4] | -57.1 [-58.7, -55.6] | 0.0 [-0.1, 0.1] | 0.0 [0.0, 0.0] | -81.5 [-82.7, -80.3] | 78.4 [77.2, 79.5] | – |
| e0_released_shaer | 3,481 | -45.9 [-47.7, -44.2] | -41.6 [-43.3, -39.7] | -55.0 [-56.7, -53.2] | -16.6 [-17.8, -15.4] | -28.3 [-29.7, -26.9] | 20.6 [19.6, 21.7] | -1.9 [-2.6, -1.2] |

## Poems, binding

| system | poems | strictly valid | meter (form) | meter (any form) | rhyme | exact length |
|---|---:|---:|---:|---:|---:|---:|
| e0_released_shaer | 3,481 | 5.1 [4.3, 5.8] | 7.7 [6.8, 8.6] | 7.9 [7.0, 8.8] | 33.8 [32.4, 35.3] | 83.4 [82.2, 84.6] |
| reference | 3,481 | 52.5 [50.8, 54.1] | 54.7 [53.0, 56.3] | 57.2 [55.6, 58.9] | 92.3 [91.4, 93.2] | 100.0 [100.0, 100.0] |

## Bayts, binding

| system | bayts on meter (verifier) | any form | no reading in the meter | BiLSTM (Shaer metric) |
|---|---:|---:|---:|---:|
| e0_released_shaer | 40.4 [39.4, 41.4] | 41.3 [40.3, 42.3] | 50.1 [49.1, 51.2] | 95.4 [95.0, 95.8] |
| reference | 80.8 [79.7, 82.0] | 84.6 [83.8, 85.5] | 9.8 [9.1, 10.4] | 97.3 [96.7, 97.8] |

**Margin sensitivity (binding)**: strictly valid poems / bayts on meter

| system | margin 1.0 | margin 1.5 | margin 3.0 | margin none |
|---|---:|---:|---:|---:|
| e0_released_shaer | 4.8 / 39.8 | 5.1 / 40.4 | 6.0 / 43.2 | 6.5 / 44.9 |
| reference | 50.9 / 79.8 | 52.5 / 80.8 | 56.9 / 83.4 | 58.6 / 84.5 |

**Paired difference to the human reference poems (binding)**, percentage points:

| system | prompts | strictly valid | meter | rhyme | exact length | bayts on meter | no reading | BiLSTM bayts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e0_released_shaer | 3,481 | -47.4 [-49.1, -45.6] | -47.0 [-48.7, -45.2] | -58.5 [-60.2, -56.7] | -16.6 [-17.8, -15.4] | -40.4 [-42.0, -39.0] | 40.4 [39.2, 41.7] | -1.9 [-2.6, -1.2] |

## Shaer's own metrics (repository definitions) next to the strict ones (letters)

| system | meter conformity | base-meter accuracy | exact count | count adherence | strictly valid | meter (form) |
|---|---:|---:|---:|---:|---:|---:|
| e0_released_shaer | 0.9064 | 95.17% | 83.40% | 0.9792 | 9.0% | 15.5% |
| reference | 0.9482 | 96.93% | 100.00% | 1.0 | 54.9% | 57.1% |

## By requested length (letters)

| system | bayts | poems | strictly valid | meter | rhyme | exact length | bayts on meter | BiLSTM bayts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 4-6 | 805 | 0.0 | 47.2 | 0.37 | 100.0 | 80.78 | – |
| control_mixed_rhyme | 7-10 | 419 | 0.0 | 34.84 | 0.0 | 100.0 | 82.01 | – |
| control_mixed_rhyme | 11-20 | 263 | 0.0 | 19.39 | 0.38 | 100.0 | 82.18 | – |
| control_wrong_meter | 1-3 | 1,668 | 0.0 | 0.0 | 96.52 | 100.0 | 0.51 | – |
| control_wrong_meter | 4-6 | 805 | 0.0 | 0.0 | 92.42 | 100.0 | 0.34 | – |
| control_wrong_meter | 7-10 | 473 | 0.0 | 0.0 | 91.97 | 100.0 | 0.51 | – |
| control_wrong_meter | 11-20 | 535 | 0.0 | 0.0 | 84.67 | 100.0 | 0.73 | – |
| e0_released_shaer | 1-3 | 1,668 | 17.51 | 27.22 | 57.37 | 98.44 | 49.06 | 97.01 |
| e0_released_shaer | 4-6 | 805 | 2.24 | 8.2 | 31.06 | 93.17 | 50.65 | 94.99 |
| e0_released_shaer | 7-10 | 473 | 0.63 | 3.38 | 16.91 | 75.26 | 54.46 | 94.29 |
| e0_released_shaer | 11-20 | 535 | 0.0 | 0.75 | 7.85 | 28.97 | 56.88 | 95.36 |
| reference | 1-3 | 1,668 | 70.5 | 72.3 | 96.58 | 100.0 | 81.77 | 97.24 |
| reference | 4-6 | 805 | 50.19 | 52.67 | 92.42 | 100.0 | 80.3 | 96.44 |
| reference | 7-10 | 473 | 41.44 | 43.55 | 91.75 | 100.0 | 82.13 | 97.17 |
| reference | 11-20 | 535 | 25.42 | 28.22 | 84.67 | 100.0 | 83.08 | 97.8 |

## Bayts on meter by position in the poem (letters)

| system | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 82.45 | 81.44 | 82.18 | 83.05 | 81.69 | 81.84 | 82.26 | 80.23 | 81.86 | 78.9 |
| control_wrong_meter | 0.6 | 0.49 | 0.62 | 0.44 | 0.35 | 0.51 | 0.79 | 0.7 | 0.27 | 0.72 |
| e0_released_shaer | 53.17 | 51.87 | 51.5 | 54.18 | 54.55 | 54.77 | 54.3 | 56.68 | 55.43 | 55.42 |
| reference | 82.1 | 81.92 | 82.27 | 81.91 | 83.0 | 81.86 | 82.24 | 82.51 | 83.38 | 81.48 |

## Rhyme and form issues (letters, share of poems)

| system | arud_changes | count | darb_changes | iqwa | israf | ita | mutlaqa_muqayyada | odd_tail | radif_missing | rawi | sinad_ridf | sinad_tasis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_mixed_rhyme | 5.78 | 0.0 | 10.56 | 1.41 | 7.87 | 3.7 | 0.13 | 0.0 | 0.0 | 99.66 | 48.22 | 19.84 |
| control_wrong_meter | 1.18 | 0.0 | 1.21 | 0.03 | 0.57 | 4.22 | 0.09 | 0.0 | 0.14 | 3.48 | 3.33 | 2.59 |
| e0_released_shaer | 1.75 | 16.6 | 8.07 | 2.04 | 8.68 | 7.7 | 0.23 | 2.36 | 0.78 | 52.14 | 29.91 | 12.96 |
| reference | 1.75 | 0.0 | 1.41 | 0.03 | 0.66 | 4.22 | 0.09 | 0.0 | 0.14 | 3.48 | 3.27 | 2.61 |

## By meter (letters; strictly valid / bayts on meter / BiLSTM bayts)

| meter | control_mixed_rhyme | control_wrong_meter | e0_released_shaer | reference |
|---|---|---|---|---|
| tawil | 0.0 / 89.19 / – (352) | 0.0 / 0.68 / – (191) | 14.63 / 71.43 / 99.96 (841) | 65.64 / 89.65 / 98.2 (841) |
| kamil | 0.0 / 80.37 / – (284) | 0.0 / 4.33 / – (199) | 7.4 / 53.12 / 95.04 (649) | 59.48 / 82.7 / 97.74 (649) |
| basit | 0.0 / 84.93 / – (230) | 0.0 / 1.17 / – (173) | 9.07 / 56.78 / 98.29 (529) | 61.25 / 84.43 / 97.75 (529) |
| wafir | 0.0 / 86.09 / – (149) | 0.0 / 0.0 / – (209) | 9.41 / 52.31 / 96.54 (340) | 60.88 / 86.95 / 94.93 (340) |
| khafif | 0.0 / 84.04 / – (141) | 0.0 / 0.19 / – (188) | 7.8 / 50.28 / 98.22 (295) | 50.85 / 83.43 / 97.61 (295) |
| mujtath | 0.0 / 80.0 / – (24) | 0.0 / 0.14 / – (255) | 4.92 / 52.51 / 86.96 (61) | 57.38 / 81.97 / 95.92 (61) |
| mudari | – | 0.0 / 0.0 / – (245) | – | – |
| rajaz | 0.0 / 59.73 / – (53) | 0.0 / 1.06 / – (244) | 5.84 / 31.9 / 84.24 (137) | 27.01 / 57.95 / 94.44 (137) |
| madid | 0.0 / 70.77 / – (9) | 0.0 / 0.22 / – (241) | 0.0 / 37.89 / 65.26 (17) | 41.18 / 75.27 / 94.62 (17) |
| muqtadab | – | 0.0 / 0.0 / – (230) | – | – |
| mutaqarib | 0.0 / 69.92 / – (58) | 0.0 / 0.34 / – (229) | 7.69 / 44.41 / 96.15 (156) | 33.97 / 71.45 / 97.39 (156) |
| munsarih | 0.0 / 80.08 / – (37) | 0.0 / 0.47 / – (219) | 1.19 / 34.85 / 82.0 (84) | 39.29 / 78.54 / 97.17 (84) |
| hazaj | 0.0 / 73.58 / – (9) | 0.0 / 0.0 / – (218) | 5.56 / 31.73 / 80.77 (18) | 44.44 / 68.93 / 92.23 (18) |
| sari | 0.0 / 76.46 / – (78) | 0.0 / 0.27 / – (211) | 5.09 / 34.86 / 87.71 (216) | 39.35 / 74.54 / 96.88 (216) |
| mutadarik | – | 0.0 / 0.35 / – (215) | – | – |
| ramal | 0.0 / 57.45 / – (63) | 0.0 / 0.34 / – (214) | 2.17 / 28.1 / 86.84 (138) | 25.36 / 63.47 / 97.71 (138) |

## Integrity checks (all must be 0)

| system/mode | line counts differing from Shaer's | BiLSTM bayts misaligned | margin recomputation mismatches | exact length vs official (pp) | errors |
|---|---:|---:|---:|---:|---:|
| control_mixed_rhyme/letters | 0 | 0 | 0 | – | 0 |
| control_wrong_meter/letters | 0 | 0 | 0 | – | 0 |
| e0_released_shaer/letters | 0 | 0 | 0 | 0.0 | 0 |
| e0_released_shaer/binding | 0 | 0 | 0 | 0.0 | 0 |
| reference/letters | 0 | 0 | 0 | 0.0 | 0 |
| reference/binding | 0 | 0 | 0 | 0.0 | 0 |
