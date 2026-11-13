"""Bounded, Not Balanced: quantum classifier heads and rare-attack detection.

Package layout
--------------
config    paths, constants, the experiment grid
data      TON_IoT / CICIoT2023 loading, cleaning, splits, imbalance sampling
qsim      batched PyTorch state-vector simulator (Qiskit conventions)
models    shared encoder and the head ladder (Q, Q-noent, C-bounded, C-host, ...)
losses    cross-entropy, class-balanced, logit-adjusted, post-hoc adjustment
train     early-stopped training
metrics   per-run evaluation (macro-F1, tail metrics, saturation, effective range, ECE)
theory    reachability (Lemma 2) and the clipped-target minimiser (Lemma 1)
runner    resumable, sharded grid execution with ledgers and checkpoints
stats     paired pooled tests (t inference, prediction intervals), TOST equivalence, Holm correction
figures   publication figures (LNCS widths, vector PDF + 300 dpi PNG)
tables    LaTeX tables
hardware  IBM Quantum execution (SamplerV2), readout calibration, H4/H5 analysis, session-2 helpers
revision  analyses added for peer review (pre-test reachability, controls, lr sweep, leakage, ...)
store     run artifacts packed into zip files
"""
__version__ = "1.1.0"
