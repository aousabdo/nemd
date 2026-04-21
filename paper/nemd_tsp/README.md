# N-EMD — IEEE TSP Manuscript

Publication-ready draft targeting *IEEE Transactions on Signal Processing*.
Author: Aous A. Abdo, Analytica Data Science Solutions.

## Status

First draft complete. 12 pages, ~7,300 words, compiles cleanly with
TeX Live 2025 (`IEEEtran` class).

| Section                                    | Status         |
|--------------------------------------------|----------------|
| Abstract                                   | Drafted        |
| §1 Introduction                            | Drafted        |
| §2 Background and Related Work             | Drafted        |
| §3 Method                                  | Drafted        |
| §4 Case for Architectural Inductive Bias   | Drafted        |
| §5 Experiments                             | Drafted        |
| §6 Discussion and Limitations              | Drafted        |
| §7 Conclusion                              | Drafted        |
| Appendix A: Proofs (FFT conventions)       | Drafted        |
| Appendix B: Architectural details          | Drafted        |
| Appendix C: Training hyperparameters       | Drafted        |
| Appendix D: Additional ablation notes      | Drafted        |
| Figures (architecture, filters, ...)       | Complete       |

## Files

- `main.tex`          — manuscript source (IEEEtran `journal` class)
- `refs.bib`          — bibliography
- `figs/`             — figures referenced by `main.tex`
- `main.pdf`          — compiled PDF (after build)

## Build

```
cd paper/nemd_tsp
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Requires `IEEEtran.cls` (ships with TeX Live 2020+), `amsmath`, `amsthm`,
`graphicx`, `booktabs`, `cite`, `tikz`, `hyperref`.

## Provenance of figures

| Figure in `figs/`                   | Source                                   |
|-------------------------------------|------------------------------------------|
| `filters_canonical.png`             | `paper/figures/phase25b_v3_filters.png`  |
| `decomposition_compare.png`         | `paper/figures/phase25b_v3_compare.png`  |
| `decomposition_spectrograms.png`    | `paper/figures/phase25b_v3_spectrograms.png` |
| `filter_responses_taskaware.png`    | `paper/figures/phase3_exp3/filter_responses.png` |
| `confusion_matrices_snr3.png`       | `paper/figures/phase3_exp3/confusion_matrices.png` |
| `training_curves_snr3.png`          | `paper/figures/phase3_exp3/training_curves.png` |
| `snr_sweep_accuracy.png`            | `make_figs.py` — grouped bar chart over SNR 3/10/20 dB |
| `generalization_metrics.png`        | `make_figs.py` — mode-mix / ortho / energy across test sets A–E |
| `inference_time.png`                | `make_figs.py` — per-signal CPU time (in-dist set) |

Rebuild derived figures after changing the experiment JSON:

```
python paper/nemd_tsp/make_figs.py
```
