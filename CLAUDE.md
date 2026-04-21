# CLAUDE.md — N-EMD research project

## Project

Neural Empirical Mode Decomposition (N-EMD): a differentiable, physics-constrained
signal decomposition that replaces EMD's hand-crafted sifting with a learned
neural operator. Target venue: IEEE Transactions on Signal Processing.

Owner: Aous A. Abdo, Analytica Data Science Solutions. Sole author.

## Current state (2026-04-21)

Phases 1–4 are complete. The final architecture (Phase 2.5b v3) is the
`NEMD` class (`nemd/model.py`) — a softmax partition-of-unity filter bank
operating in the frequency domain. The legacy `NEMDSifting` class is kept
for the negative-result ablation in §4 of the paper.

Main deliverable in progress: a publication-ready IEEE TSP paper at
`paper/nemd_tsp/` (new directory). The earlier draft at `paper/main.tex`
is a starting point and is not being edited further.

## Code layout

```
nemd/
  model.py       — NEMD (primary), NEMDSifting (legacy), SignalAnalyzer,
                   FrequencyPartition, ResBlock1d, SiftNet (legacy U-Net)
  losses.py      — NEMDLoss (4 terms), NEMDSiftingLoss (7 terms, legacy),
                   CentroidSeparationLoss, FilterBalanceLoss,
                   FilterSharpnessLoss, and others
  layers.py      — differentiable Hilbert, IF, envelopes, spectral bandwidth
  classical.py   — ClassicalEMD, EEMD, VMD baselines
  utils.py       — synthetic signal generation, metrics, helpers
  train.py       — training loop
  applications/  — downstream tasks (phase 3, phase 4)
  benchmarks/    — comparison scripts
  synthetic/     — data generators
  data/          — dataset loaders incl. OpenNeuro ds003838
experiments/     — experiment drivers (applications, benchmarks, synthetic)
tests/           — pytest suite, 125+ tests passing
```

Trained checkpoints:
- `checkpoints_p25b_v3/` — primary; K=3, hidden=64, 3 ResBlocks, kernel=5
- Older: `checkpoints_p25*`, `checkpoints_fast`, `checkpoints_pupil`, etc.

## Figures (experimental artifacts)

All under `paper/figures/`:
- `phase25b_v3_{compare,filters,spectrograms}.png` — final decomposition
- `phase3_exp1/` — nonstationary comparisons (5 signal types, time + IF)
- `phase3_exp2/` — 5-set generalization with `summary.json`
- `phase3_exp3/`, `phase3_exp3_snr10/`, `phase3_exp3_snr20/` — task-aware
  classification at SNR 3/10/20 dB, with `summary.json`, confusion matrices,
  filter responses, training curves
- `phase4/` — OpenNeuro ds003838 pupillometry, decomposition + 3/6-class
  classification (negative classification result, positive decomposition
  result)

## Paper writing conventions

Formal IEEE style. Do **not** use these words: delve, crucial, landscape,
leverage, facilitate, comprehensive, multifaceted, streamline, moreover,
furthermore, additionally. Selective contractions only. Numbered refs
(IEEEtran style). No hedging. No skeletonized content.

## Compilation

```
cd paper/nemd_tsp
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Git policy

No `Co-Authored-By`, no `Generated with Claude Code`, no tool attribution
in commits or PRs. Commits are solely authored by the human git user.
