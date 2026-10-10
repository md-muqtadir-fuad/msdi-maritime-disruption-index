# Figure readability and dry edits

Redraws the manuscript figures at text width (6.5 in) so every label prints at 8 pt or more (axis labels and titles at 9 pt). It reads only saved results: no forecasting, VAR, DML, PCMCI or LLM step is re-run. The original plotting functions run with disk writes blocked, and each run checks that nothing under `outputs/`, `data/` or `validation/` changed.

## Run

From `scripts/`, with the repo interpreter:

```
..\..\..\.venv\Scripts\python.exe run_all.py                      # redraw all 19 into output_figs/
..\..\..\.venv\Scripts\python.exe run_all.py pcmci_links          # redraw one
..\..\..\.venv\Scripts\python.exe run_all.py --install            # redraw, back up latex/ PDFs, copy new ones in
..\..\..\.venv\Scripts\python.exe measure_latex_fig_fonts.py ..\..\..\latex\pcmci_links.pdf   # printed font sizes
..\..\..\.venv\Scripts\python.exe compare.py dml_predictive_effects   # text diff: latex/ vs output_figs/
```

`--install` copies the current `latex/` PDFs to `original_latex_pdfs/<timestamp>/` before overwriting them. `original_latex_pdfs/20261011_before_readability_fix/` holds the PDFs as they were before the first install.

## Where to change things

| Change | File |
|---|---|
| Font sizes (8 / 9 / 9 pt), canvas width | `common.py`: `TICK`, `LABEL`, `TITLE`, `WIDTH_IN` |
| Figure height, legend position, tick spacing | `fig24_tune.py` (script-24 figures), `fig_mid.py` (47, 48, 46), `eda_tune.py` (31, 39, 40, 43, 44), `portmap.py` (49) |
| PCMCI graph geometry and labels | `pcmci_draw.py` |
| Which figures exist and how each is built | `run_all.py`: `FIGURES` |

`eda_tune.py` works on figures captured by `fig_eda.py` and stored in `output_figs/_cache/`. After the first `run_all.py`, `eda_tune.py <name>` re-lays-out a figure without re-reading the data.

## Notes

- `frame-work.pdf` is a drawn diagram with no source in the repo, so it is not regenerated. Its text prints at about 5 pt.
- `pairwise_pdp_grid` refits script 46's seeded random forest, because its surfaces were never saved. The script asserts that the refit reproduces all 15 saved H values.
- Script 31's figures (`eda_correlation`, `eda_distributions`, `empirical_relationships`) are drawn from the current monthly panel. That panel is newer than the PDFs these replaced.
- Mathtext subscripts ($H_{jk}$, $S_i$) print at 0.7 of the base size (about 6.4 pt). This is standard subscript sizing.
- `audit_float_placement.py main.tex` lists figures and tables that sit directly under a heading or appear before their first citation.
