# Paper: Verifying the Verse

IEEE conference-format paper (`IEEEtran`) on the reproduction of Shaer and the prosody-verifier study (M0–M6).

| file | what |
|---|---|
| `main.tex` | preamble, title, abstract; inputs the sections |
| `sections/*.tex` | one file per section (introduction … appendix) |
| `references.bib` | BibTeX, IEEE numbered style; entries checked against Crossref (DOI) or the arXiv API |
| `scripts/analysis.py` | writes `generated/`: `numbers.tex` (one macro per number), `tab_logit.tex`, `tab_nested.tex`, `results.json` and the figures (PNG, 400 DPI) |
| `generated/` | output of `analysis.py`; do not edit by hand |
| `main.pdf` | the compiled paper |

## Build

LuaLaTeX is required (the Arabic text uses babel with `bidi=basic` and the Amiri font; the main font is TeX Gyre Termes):

```
latexmk -lualatex main.tex
```

## Regenerate the numbers and figures

```
python scripts/analysis.py --m3 <m3 dir> --repro <repro dir> --m0 <m0_summary.json> --m1 <M1 validation json> --out generated
```

* `--m3`: the M3 output folder (`m3_rescore.py score` for every system, then `m3_rescore.py summarize --reps 5000`).
* `--repro`: one folder per run `e0`…`e3` with the baseline's `summary_drive.json`, plus the training log `metrics.jsonl`.
* `PAPER_REPS` (environment) sets the bootstrap resamples; the paper uses 5,000 (the default).

## Placeholders

Results that are not available yet print in red:

* `\TBD` / `\pending{...}`: written by hand in the text (M4–M6 results, the human-study reliability, author details).
* `\val{name}`: a number from `generated/numbers.tex`; if the analysis has not produced it, it prints `[TBD]`.

To fill a pending result, either add it to `analysis.py` (preferred, so the number is traceable) or replace the
`\TBD` in the section file. Search for `TBD` and `pending` to find every placeholder.
