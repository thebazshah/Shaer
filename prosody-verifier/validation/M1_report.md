# M1 — the prosody verifier: design and validation

Numbers in this report were produced by the code at commit `e17d0f4` on branch `dev` (shipped lexicon `data/lexicon_train.json.gz`, margin 1.5, strict mode). They are reproduced by section 5 of `prosody_main.ipynb` (`limit=0` for whole splits).

## Summary

* On **human poems of the held-out test split** (3,481 poems, 38,412 hemistichs), **87.1%** of hemistichs are accepted in their labelled meter and form (91.12% in their labelled meter, any form), and the verifier's most plausible analysis is the labelled meter for **86.09%**. Corpus labels and texts are noisy, so these are lower bounds.
* Systematically **broken copies** of accepted hemistichs are rejected **71.89%** of the time (a word added 99.59%, a word swapped for one of another length 80.51%, a word dropped 75.13%, one letter dropped 24.37%; many one-letter drops are still metrical, see below).
* **Rhyme**: 85.63% of human test poems pass the full qāfiya check, and **100.0%** of poems with one planted rhyme break are flagged.
  *Erratum (M3):* 85.63% was produced by the rhyme code at `e17d0f4`. M3 found two errors in it, both fixed (`1c707ba`, `0e8c3d2`; see `M3_report.md` section 6). With the fixed code, **91.79%** of these poems pass the rhyme check, planted breaks are still flagged at 100.0%, and every meter number in this report is unchanged. *Erratum (M4):* the margin now also applies to tadwīr (`M4_report.md` section 3; `M3_report.md` section 11). With that change, the poems with every bayt accepted are 52.93% (was 53.36%), and the verifier accepts 54.81% / 85.83% of the bayts in the BiLSTM cross-check (letters, any form, E0 / human; was 55.74% / 86.2%); hemistich-level numbers are unchanged.
* **Against the BiLSTM** used by the Shaer evaluation: the BiLSTM accepts 95.37% of the released Shaer bayts and 97.3% of the human bayts on the same prompts. Judging the letters only (written vowels ignored), the verifier accepts 55.74% and 86.2%. The margin-independent measure is sharper: among the bayts the BiLSTM accepts, **24.85%** of Shaer's have a hemistich with **no reading at all** in the requested meter (any form, any cost), against **4.07%** of the human ones. The BiLSTM metric hides a large share of unmetrical lines, and the manual check below supports these rejections.
* The main weakness is in the other direction: with bare text, the **margin rule** sometimes rejects a metrical line because a more common reading of one of its words scans in another meter. This accounts for most rejections of human poems in the manual check. A prosody expert still has to sign off the rule tables and a gold set (M2).

## What the verifier decides

For a poem and a requested meter and form, the verifier decides for every hemistich, every bayt and the whole poem:

* **meter**: the hemistich scans in the requested meter form; the bayt's ʿarūḍ and ḍarb are a permitted pair (taṣrīʿ is allowed in the opening bayt; a word split across the hemistichs, tadwīr, is handled in both written layouts); the ḍarb form, and outside the opening bayt the ʿarūḍ form, stays the same in every bayt (an ʿilla is binding; zihāfāt and tashʿīth may vary);
* **rhyme**: one rawī; its vowel (iqwāʾ, iṣrāf); ridf and taʾsīs all-or-none (sinād); waṣl and khurūj; muṭlaqa vs muqayyada; īṭāʾ (the same rhyme word within 7 bayts, a warning); and an Urdu-style refrain (radīf) when the poem has one or one is required;
* **length**: the number of hemistichs against the request, and an odd final hemistich.

A failing hemistich comes with the nearest valid pattern of the requested meter, the number of mutaḥarrik/sākin edits to reach it, and the foot where the first edit falls.

## How it works

1. **Rule tables** (`prosody/feet.py`, `prosody/meters.py`). Feet are built from sabab/watad units. Zihāfāt are defined by the classical letter ordinals and can only touch the second letter of a sabab; ʿilal act on the end of the last foot. Every pattern is derived by these operations, never typed: 16 meters, 26 forms, 1,867 hemistich realisations.
2. **Prosodic writing** (`prosody/arudi.py`). Each word becomes the set of mutaḥarrik/sākin readings its spelling allows (the rules are listed in the module docstring), constrained by Arabic phonotactics. Written diacritics are binding, or ignored in the `ignore_diacritics` mode.
3. **Pronunciation prior** (`prosody/lexicon.py`, `build_lexicon.py`). Bare text allows many readings, enough to force a line into several meters. The prior counts, per bare word form, how the **train** split's poems are read when scanned in their labelled meter (two rounds); a letter model covers unseen words. Reading costs are −log relative frequencies.
4. **Scansion** (`prosody/scan.py`): a dynamic programme over the words and a trie of every hemistich pattern of every meter, resolving word boundaries on the way. A hemistich is accepted in the requested meter when an analysis exists, it costs at most 6.0, and it is at most `margin` (1.5) costlier than the most plausible analysis **in any meter for the same position** (a sadr outside the opening bayt can only take ʿarūḍ forms).
5. **Tadwīr**: a bayt that fails as two hemistichs is scanned as one connected line against every permitted ʿarūḍ + ḍarb concatenation, and the split must fall strictly inside a word.
6. **Rhyme** (`prosody/qafiya.py`) on the chosen reading of the last word. Where the text leaves the rawī position open (a final ا/و/ي/ى, ه or ة), the poem-level choice that makes most bayts agree is taken.

## Validation design

* **Unit tests** (`tests/`, run by the notebook): 52 textbook equalities (each zihāf/ʿilla gives the named foot), structural invariants of the tables, word-reading and boundary rules, al-Ṣafī al-Ḥillī's mnemonic verse for each of the 16 meters, the foot names read as text, famous verses, rhyme cases (maqṣūra, hāʾ waṣl + khurūj, iqwāʾ, sinād al-ridf / al-taʾsīs, refrain), poem verdicts, and a regression test for every bug found by the corpus runs below.
* **Independent cross-check of the tables** against ARBML Bohour and pyarud (`validation/spec_crosscheck.md`; their sources were read, not run).
* **Human poems as positives**: the eval split (tuning) and the test split (report). The lexicon was built from the train split only. Corpus labels are scraped and the texts have typing errors, so acceptance on human poems is a lower bound.
* **Broken copies as negatives**: for every accepted human hemistich, one letter dropped, one word dropped, a word inserted, a word swapped for one of another length. Dropping a letter often removes a long vowel, which is exactly what khabn or qabḍ do, so many such copies are still metrical: that row overstates false acceptance.
* **Planted rhyme breaks**: the last word of one ʿajuz replaced by a word with another rawī.
* **Agreement with the BiLSTM classifier** of the Shaer evaluation, bayt by bayt, on the released Shaer poems and on the human poems of the same 3,481 test prompts, with diacritics binding and ignored.
* **Author's manual check** of random disagreement samples (not an expert's check; see M2).

## Choosing the margin (eval split)

The margin trades acceptance of real poems against rejection of broken ones and against how many *other* meters also accept a hemistich. It was chosen on the eval split; the test split was not used for any choice.

| margin | hemistichs accepted | other meters also accepting (mean) | broken copies rejected |
|---:|---:|---:|---:|
| 0.5 | 84.16% | 0.253 | 74.78% |
| 1.0 | 84.89% | 0.318 | 72.84% |
| 1.5 | 85.62% | 0.365 | 71.89% |
| 2.0 | 86.01% | 0.423 | 71.02% |
| 3.0 | 87.17% | 0.838 | 67.79% |
| none | 89.94% | 2.496 | 59.53% |

Without the prior and the margin (last row), bare text lets a hemistich be read in about 2.496 other meters on average; at 1.5 this falls to 0.365 for about 4.3 points of acceptance. The curve has no sharp knee: 1.5 is a compromise set on this split, and M3 reports its headline numbers at more than one margin.

## Results on the test split

3,481 poems, 38,412 hemistichs, margin 1.5.

| measure | value |
|---|---:|
| hemistichs accepted in their labelled meter and form | 87.1% |
| ... in their labelled meter, any form | 91.12% |
| ... with any margin (an analysis in the labelled meter and form exists) | 91.05% |
| labelled meter = most plausible analysis | 86.09% |
| other meters also accepting a hemistich (mean) | 0.329 |
| broken copies rejected | 71.89% |
| ... add word | 99.59% |
| ... swap word | 80.51% |
| ... drop word | 75.13% |
| ... drop letter | 24.37% |
| poems with at least 4 hemistichs: every bayt accepted | 53.36% of 3,263 |
| ... rhyme check passes | 85.63% (91.79% after the M3 fixes, see the erratum above) |
| planted rhyme breaks flagged | 100.0% |

Every-bayt acceptance compounds the per-hemistich rate over 10 to 20 hemistichs, so it is far lower than the hemistich rate; with label and typing noise in the corpus it is a pessimistic number.

**By meter** (hemistichs; acceptance at margin 1.5 and with any margin; labelled meter = most plausible analysis):

| meter | hemistichs | accepted @1.5 | analysis exists | top-1 |
|---|---:|---:|---:|---:|
| al-ṭawīl | 9,202 | 93.38% | 94.6% | 91.98% |
| al-kāmil | 7,446 | 85.75% | 92.6% | 86.74% |
| al-basīṭ | 5,780 | 86.68% | 89.9% | 92.46% |
| al-wāfir | 3,632 | 89.84% | 92.48% | 88.79% |
| al-khafīf | 3,512 | 86.73% | 89.64% | 84.4% |
| al-sarīʿ | 2,050 | 88.0% | 93.46% | 73.07% |
| al-ramal | 1,752 | 71.63% | 79.91% | 77.97% |
| al-mutaqārib | 1,688 | 91.05% | 93.78% | 87.86% |
| al-rajaz | 1,522 | 63.93% | 71.81% | 63.86% |
| al-munsariḥ | 848 | 88.92% | 93.16% | 82.67% |
| al-mujtathth | 588 | 85.37% | 92.86% | 47.28% |
| al-hazaj | 206 | 79.13% | 83.5% | 76.21% |
| al-madīd | 186 | 91.4% | 92.47% | 84.41% |

**By diacritic coverage** of the hemistich (share of letters carrying a mark):

| coverage | hemistichs | accepted @1.5 | other meters also accepting |
|---|---:|---:|---:|
| 0-20% | 17,160 | 86.98% | 0.419 |
| 20-40% | 6,407 | 85.72% | 0.303 |
| 40-60% | 8,281 | 88.13% | 0.248 |
| 60-80% | 5,195 | 88.64% | 0.215 |
| 80-100% | 1,369 | 82.83% | 0.249 |

More written vowels cut the number of other meters that also accept a hemistich by almost half. Acceptance changes little with coverage and is lowest in the most vowelled band; written vowels are binding, so a wrong vowel in the corpus counts against the line (compare the binding and letters-only rows of the next section).

### Which other meters also accept a hemistich

Share of the hemistichs of each meter (test split) that another meter also accepts at margin 1.5, the four largest:

| meter | also accepted as |
|---|---|
| al-ṭawīl | al-kāmil 3.4%, al-rajaz 1.1%, al-basīṭ 0.9%, al-khafīf 0.8% |
| al-kāmil | al-rajaz 15.9%, al-ṭawīl 12.2%, al-basīṭ 4.6%, al-munsariḥ 2.4% |
| al-basīṭ | al-kāmil 1.5%, al-ṭawīl 1.5%, al-wāfir 1.1%, al-munsariḥ 0.8% |
| al-wāfir | al-ṭawīl 3.9%, al-mutaqārib 2.8%, al-khafīf 2.7%, al-hazaj 2.1% |
| al-khafīf | al-kāmil 5.1%, al-mutadārik 3.9%, al-madīd 3.5%, al-ṭawīl 2.9% |
| al-sarīʿ | al-kāmil 31.6%, al-rajaz 17.3%, al-munsariḥ 7.7%, al-mutadārik 7.3% |
| al-ramal | al-mutadārik 12.9%, al-madīd 9.1%, al-kāmil 7.3%, al-khafīf 6.7% |
| al-mutaqārib | al-mutadārik 8.1%, al-sarīʿ 4.3%, al-khafīf 4.1%, al-ṭawīl 3.1% |
| al-rajaz | al-kāmil 72.7%, al-mujtathth 6.4%, al-basīṭ 6.0%, al-wāfir 5.7% |
| al-munsariḥ | al-basīṭ 12.9%, al-kāmil 7.0%, al-rajaz 3.3%, al-khafīf 2.1% |
| al-mujtathth | al-muḍāriʿ 29.3%, al-kāmil 27.2%, al-mutadārik 9.0%, al-rajaz 5.1% |
| al-hazaj | al-wāfir 86.4%, al-mutadārik 8.7%, al-rajaz 7.8%, al-kāmil 4.9% |
| al-madīd | al-mutadārik 18.3%, al-khafīf 10.8%, al-ramal 7.0%, al-sarīʿ 4.8% |

The largest overlaps are prosodic facts rather than verifier errors. Al-kāmil with iḍmār in every foot is identical to al-rajaz (mutafāʿilun → mustafʿilun), and the majzūʾ al-wāfir with ʿaṣb in every foot is identical to al-hazaj (mufāʿalatun → mafāʿīlun). Classical prosody assigns such lines by the rest of the poem, as the poem-level consistency check does. Al-sarīʿ shares its first two feet with al-rajaz and al-kāmil (mustafʿilun mustafʿilun). Al-mujtathth and al-muḍāriʿ are rotations of the same circle and become hard to separate in bare text. The overlaps involving al-mutadārik come from tashʿīth, which makes many short-syllable sequences fit it.

## Agreement with the Shaer BiLSTM metric

Bayt by bayt, on the 3,481 test prompts: the released Shaer poems (E0, first sample) and the human reference poems. "Verifier" accepts a bayt in any form of the requested meter (the BiLSTM knows only base meters). The last column is the margin-independent measure: the share of the bayts the BiLSTM accepts that have a hemistich with no reading at all in the requested meter.

| poems | written vowels | bayts | BiLSTM accepts | verifier accepts | Cohen's κ | BiLSTM-accepted, no reading in the meter |
|---|---|---:|---:|---:|---:|---:|
| Shaer (released) | binding | 19,927 | 95.37% | 41.73% | 0.054 | 48.8% |
| Shaer (released) | ignored | 19,927 | 95.37% | 55.74% | 0.082 | 24.85% |
| human | binding | 19,198 | 97.3% | 84.97% | 0.193 | 8.35% |
| human | ignored | 19,198 | 97.3% | 86.2% | 0.208 | 4.07% |

With its own written vowels treated as binding, nearly half of the Shaer bayts that the BiLSTM accepts cannot be read in the requested meter at all: the model writes vowels that contradict the meter. Judged by the letters alone, about a quarter still cannot. For the human poems the same figures are small, and part of them is corpus noise.

κ is low because the BiLSTM accepts nearly everything, so almost all disagreement is in one cell: bayts the BiLSTM accepts and the verifier rejects. The verifier accepts very few bayts the BiLSTM rejects (147 Shaer and 133 human bayts, letters only).

**Why the verifier rejects bayts the BiLSTM accepts** (share of those bayts; a bayt can have two reasons):

| reason | Shaer, binding | Shaer, letters only | human, binding | human, letters only |
|---|---:|---:|---:|---:|
| sadr: no reading in any meter | 33.0% | 15.0% | 21.8% | 7.7% |
| ʿajuz: no reading in any meter | 38.5% | 14.5% | 17.7% | 5.5% |
| sadr: scans only in another meter or form | 13.9% | 16.5% | 15.6% | 14.2% |
| ʿajuz: scans only in another meter or form | 19.7% | 22.6% | 15.9% | 12.9% |
| sadr: in the meter, but another meter is far more natural | 4.9% | 14.5% | 12.0% | 25.2% |
| ʿajuz: in the meter, but another meter is far more natural | 4.7% | 16.5% | 12.5% | 29.1% |
| sadr: only through implausible readings | 2.7% | 7.7% | 6.1% | 7.8% |
| ʿajuz: only through implausible readings | 3.3% | 8.7% | 7.3% | 7.7% |
| ʿarūḍ and ḍarb not a permitted pair | 1.2% | 1.1% | 2.9% | 2.3% |
| bayts | 10,762 | 8,044 | 2,495 | 2,264 |

For Shaer, the first four rows dominate: a hemistich that cannot be read in the requested meter at all. For human poems with the letters only, the margin kind dominates: the line can be read in its meter, but a more common reading of some word fits another meter better.

**By meter** (bayts accepted, letters only):

| meter | Shaer bayts | BiLSTM | verifier | human bayts | BiLSTM | verifier |
|---|---:|---:|---:|---:|---:|---:|
| al-ṭawīl | 4,781 | 100.0% | 72.0% | 4,601 | 98.2% | 89.9% |
| al-kāmil | 3,852 | 95.0% | 55.7% | 3,723 | 97.7% | 90.2% |
| al-basīṭ | 3,040 | 98.3% | 58.4% | 2,890 | 97.8% | 90.3% |
| al-wāfir | 1,881 | 96.5% | 53.2% | 1,816 | 94.9% | 89.8% |
| al-khafīf | 1,800 | 98.2% | 51.1% | 1,755 | 97.6% | 84.5% |
| al-sarīʿ | 1,050 | 87.7% | 36.5% | 1,025 | 96.9% | 75.3% |
| al-ramal | 935 | 86.8% | 38.7% | 875 | 97.7% | 77.8% |
| al-mutaqārib | 858 | 96.2% | 45.2% | 843 | 97.4% | 71.9% |
| al-rajaz | 793 | 84.2% | 36.8% | 756 | 94.4% | 73.3% |
| al-munsariḥ | 439 | 82.0% | 36.4% | 424 | 97.2% | 78.8% |
| al-mujtathth | 299 | 87.0% | 54.2% | 294 | 95.9% | 82.0% |
| al-hazaj | 104 | 80.8% | 33.7% | 103 | 92.2% | 68.9% |
| al-madīd | 95 | 65.3% | 40.0% | 93 | 94.6% | 75.3% |

## Manual check (author, not an expert)

Random samples of bayts that the BiLSTM accepts and the verifier rejects, letters only, were scanned by hand, 12 from the
released Shaer poems and 12 from the human poems (`disagreement_sample` in the comparison outputs; this check used the run
before the tadwīr fix described at the end of this section; the disagreements are of the same kind).

**Shaer (12 bayts).** 9 have a hemistich that is genuinely unmetrical: most are one syllable short or long (for example
وَمَنعةٍ من أذى قومٍ إذا ما leaves al-basīṭ's ḍarb without its last foot; فإِنَّ قلبيَ ظَمآنٌ بِدَنِّهِ is one mutaḥarrik short), one
ends the ḍarb of al-kāmil al-tāmm on two sākins; the nearest valid pattern was 1 to 4 mutaḥarrik/sākin edits away.
In 1 more the verdict is right (the ʿajuz is broken) but the sadr was also flagged, wrongly. 2 are false rejections of
the margin kind: a bare word has a second reading that the prior prefers and that fits another meter (وجدي read as wajdī
rather than jaddī; هجراني read as hajarānī rather than hijrāniya). 10 of 12 bayt verdicts are right.

**Human (12 bayts).** 9 are false rejections caused by reading costs, all but one by the margin: the hemistich has a
reading in its meter (an exact one, in most cases) but a more common reading of a bare word fits another meter, for
example النقد read as naqd/naqad rather than nuqda, وارتشفها as a past tense rather than an imperative. In the other, اصيب
written without its hamza made the needed reading cost more than the cap. 1 is labelled tām but is majzūʾ al-rajaz (the
verifier finds it in that form), 1 is a corpus line consisting of "...", and 1 is a mudawwar bayt written with the whole
split word in the sadr (ربما تجزع النفوس من الامر / له فرجة كحل العقال), which the verifier then missed and now handles
(tadwīr in both layouts, with a regression test).

So on Shaer's output the verifier's rejections are mostly real errors that the BiLSTM metric does not see, while on human
poems most rejections in letters-only mode are caused by the margin rule. The margin protects against the opposite error
(bare text forced into a wrong meter); the remedies are diacritics (generated text can be asked to carry them, and they
are binding) and a better pronunciation prior. The margin-independent column above is the conservative measure to quote
for unmetrical output.

Earlier corpus runs found and fixed further reading and rule bugs, each now covered by a test: a hamza-alif taken as the
article, implicit shadda, pronoun hāʾ and mīm al-jamʿ lengthening, tanwīn written on the alif or on ى, tashʿīth treated
as binding, and a rival reading that only a ḍarb can take being used to reject a sadr. The table invariants caught zihāf
entries that would have ended a hemistich on a mutaḥarrik.

## Limitations and what M2 must sign off

* **Bare-text ambiguity.** With few diacritics a hemistich can often be read in more than one meter; the prior and the
  margin reduce this to about one other meter for every three hemistichs, and cause the false rejections above. Partial
  vowelling reduces the number of other meters that accept a line (table by coverage).
* **One-letter perturbations.** A dropped letter is often a dropped long vowel, which zihāfāt license, so the 25% rejection
  of that kind understates the verifier's sensitivity; a better negative would change a syllable's weight in a position
  where no zihāf applies.
* **Corpus noise.** Scraped meter and form labels and typing errors depress every acceptance figure on human poems; the
  test-split numbers are lower bounds, not error rates.
* **Al-rajaz.** It has the lowest acceptance, probably in part because urjūza poems (every hemistich rhymes and the sadr
  takes ḍarb forms) are not modelled as a separate form; this has not been measured. It also overlaps al-kāmil by rule.
* **Ridf and taʾsīs** depend on the chosen reading of the last word; with bare text a sinād flag can come from a misread
  long vowel (sinād is the most frequent rhyme issue on human poems).
* **No expert review yet.** The rule tables were cross-checked against two open-source implementations and the textbook
  equalities, not by a prosodist. M2 needs a prosody expert to (1) sign off the zihāf/ʿilla tables per meter and form,
  including the rare zihāfāt allowed at a cost and the decisions taken on the differences listed in `spec_crosscheck.md`; (2) label a gold set
  of hemistichs and bayts from Shaer output and human poems (metrical or not, meter and form, broken foot) to measure the
  verifier's precision and recall directly; (3) rule on the rhyme decisions (rawī with final hāʾ, ridf with و/ي after
  a fatḥa, īṭāʾ window).
