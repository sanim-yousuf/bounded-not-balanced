# Bounded, Not Balanced: experiment design notes

Code for the study *What quantum classifier heads do to rare-attack detection*: bounded-logit heads
(a 10-qubit variational head and its controls) on TON_IoT and CICIoT2023, the reachability theory for
post-hoc logit adjustment, and inference on IBM `ibm_kingston`.

**To run it, follow [KAGGLE_INSTRUCTIONS.md](KAGGLE_INSTRUCTIONS.md).**

```
bbd/            the package (see bbd/__init__.py for the module map)
notebooks/      00 setup+tests · 01 data · 02 grid (GPU, resumable) · 03 IBM hardware · 04 analysis + figures
                05 revision runs · 06 hardware session 2 · 07 revision analysis
tests/          pytest suite (runs in notebook 00; also `python -m pytest tests`)
tools/          build_notebooks.py regenerates the notebooks
```

## Design in one paragraph

Every model is `x → shared encoder → h ∈ (−1,1)^10 → head → logits`. Heads: **Q** (s·⟨Z_k⟩ from a
10-qubit, 3-layer RY·RZ + CZ-path circuit, simulated exactly in PyTorch with Qiskit conventions), **Q-noent**
(no CZ), **C-bounded** (s·tanh), **C-cosine**, **C-host** (unbounded), plus MLP and LightGBM references. Losses:
cross-entropy, class-balanced, logit-adjusted; post-hoc adjustment is evaluated on every cross-entropy run.
Imbalance ρ ∈ {10, 100, 1000} (exponential profile) and the natural profile; 10 seeds for the core grid; a fixed
balanced test set. Every run saves its ledger record, logits and checkpoint (packed into a few zip files per
session, `bbd/store.py`), so all metrics and figures can be recomputed without retraining.

## Grid size

| Tier | Runs | Purpose |
| --- | --- | --- |
| tuning | 384 trials | 8 (lr, wd) trials per dataset × head family × loss |
| hw | 4 | the two models sent to ibm_kingston (+ matched C-bounded) |
| core | 1,560 | Q, C-bounded at s ∈ {1, 2, 4, learned}; Q-noent, C-cosine at s ∈ {2, 4}; C-host; CE and LA; 3 ρ; 2 datasets; 10 seeds |
| sweep | 356 | Q, C-bounded at s ∈ {1.5, 3, 6}; 5 seeds |
| ablation | 450 | class-balanced loss, learned bias, re-uploading, double-width head, natural profile, all-class TON_IoT (with MITM and ransomware; ρ ∈ {100, 1000}) |
| ref | 180 | MLP (CE, LA), LightGBM |

Total: 2,550 grid runs, 1,090 of them quantum (`python -m bbd.runner status` prints progress per tier).

| revision | 1,690 | peer-review controls (notebook 05): Q-noent at s ∈ {1, 1.5, learned}; classical readout controls C-sin-diag, C-diag, C-sin, C-bounded-init; learning-rate sweep; capped validation; C-host without output bias; two models for hardware session 2 |

Revision analyses live in `bbd/revision.py` (notebook 07); hardware session 2 is notebook 06.
The data figures of the paper (Figs. 2, 4 and 5) are drawn by `bbd/paper_figures.py` from released outputs only, in
the last cells of notebook 07 or with `python -m bbd.paper_figures LEDGER REV_TABLES HW1_JSON HW2_JSON OUT`.
`bbd/paper_numbers.py` recomputes every manuscript number that no other cell prints and lists it beside the
paper's value (notebook 07, cell "Paper numbers not printed above"; output `paper_figures/paper_numbers.csv`).

## Methodology notes (for the paper's setup section)

- **Cleaning before splitting.** Rows whose features appear with more than one label are removed, then exact
  duplicates are removed, before the pool/validation/test split. No feature vector is shared between splits.
- **TON_IoT classes.** The main set (`toniot8`) excludes MITM (unreliable labels) and ransomware: after removing
  timestamps, IPs and ports, only 248 of its 20,000 rows are unique and label-consistent, too few for 300/100
  test/validation rows and for the ρ = 10 training profile. Both classes are kept in `toniot10`, used as a
  sensitivity check at ρ = 100 and 1000.
- **Fixed balanced evaluation sets.** Validation and test sets are balanced per class and fixed by `DATA_SEED`;
  only the training subset changes with ρ and seed.
- **Model selection.** Early stopping and hyperparameter tuning both use macro-F1 on the balanced validation set.
  Tuning is done once per dataset × head family × loss at s = 2, ρ = 100, seed 0; the chosen (lr, wd) is reused for
  every s, ρ and seed of that family. Class-balanced loss reuses the cross-entropy settings.
- **Realised imbalance.** If a class's pool is smaller than the target count, the class is capped and a warning
  is printed; the ledger records the realised ratio as `actual_rho`.
- **Pairing in the statistics.** Differences against a shared baseline (C-host, unbounded) are averaged over the
  bounded heads and scales paired with the same baseline run before pooling, so a shared run is counted once.
  The pooled estimate uses a random intercept per design cell (dataset × ρ × loss, plus s for matched-s
  comparisons) with t inference on cells − 1 degrees of freedom, and reports the between-cell SD and a 95%
  prediction interval; TOST uses a ±1 macro-F1-point margin; H2a is a Wilcoxon signed-rank test paired within
  baseline runs; Holm correction is applied over H1, H2a, H2b and H3. Other analyses are exploratory.
- **Readout correction on hardware.** The code uses tensored (per-qubit) readout inversion from the two
  calibration circuits, not M3; the affine correction of Lemma 3 is fitted on the held-out calibration inputs.
