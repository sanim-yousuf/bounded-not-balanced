"""Paths, constants and the experiment grid.

Every run is a frozen RunSpec. Its run_id is a deterministic string, so ledgers from
different Kaggle sessions merge without duplicates and a finished run is never repeated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

# ----------------------------------------------------------------------------- paths
ON_KAGGLE = Path("/kaggle").exists()
# BBD_INPUT / BBD_WORK override the defaults (the test suite uses this to stay in a temp folder)
INPUT_ROOT = Path(os.environ.get("BBD_INPUT", "/kaggle/input" if ON_KAGGLE else "./input"))
WORK_ROOT = Path(os.environ.get("BBD_WORK", "/kaggle/working" if ON_KAGGLE else "./work"))
PREPARED_DIR = WORK_ROOT / "prepared"
RESULTS_DIR = WORK_ROOT / "results"
FIG_DIR = WORK_ROOT / "figures"
TABLE_DIR = WORK_ROOT / "tables"
HW_DIR = WORK_ROOT / "hardware"

# ----------------------------------------------------------------------------- constants
N_QUBITS = 10            # one Pauli-Z per class on qubits 0..K-1
N_LAYERS = 3             # hardware-efficient layers (RY.RZ per qubit + CZ on a path)
TRAIN_SIZE = int(os.environ.get("BBD_TRAIN_SIZE", 20_000))   # rows per controlled-imbalance training subset
N_TEST = int(os.environ.get("BBD_N_TEST", 300))             # balanced test rows per class (fixed)
N_VAL = int(os.environ.get("BBD_N_VAL", 100))               # balanced validation rows per class (fixed)
DATA_SEED = 2027         # seed for the fixed test/validation split
DELTA = 0.01             # equivalence margin: 1 macro-F1 point
ENC_HIDDEN = 64          # shared encoder width
BATCH_SIZE = 512
MAX_EPOCHS = int(os.environ.get("BBD_MAX_EPOCHS", 150))
PATIENCE = int(os.environ.get("BBD_PATIENCE", 15))
SHOTS_EVAL = 1024        # shot-noise evaluation of quantum heads in simulation
POSTHOC_TAUS = (0.5, 1.0)

DATASETS = {
    # name: (source, drop_classes)
    # main analysis: MITM labels are unreliable, and only ~1% of ransomware rows survive de-duplication
    # (too few for the rho = 10 profile and for 300/100 test/validation rows)
    "toniot8": ("toniot", ("mitm", "ransomware")),
    "toniot10": ("toniot", ()),         # sensitivity: all ten classes (rho = 100 and 1000 only, see build_grid)
    "ciciot8": ("ciciot", ()),          # 7 attack categories + benign
}

QUANTUM_MODELS = ("Q", "Q-noent")
REVISION_CONTROLS = ("C-sin-diag", "C-diag", "C-sin", "C-bounded-init")
BOUNDED_MODELS = ("Q", "Q-noent", "C-bounded", "C-bounded-wide", "C-cosine") + REVISION_CONTROLS
ALL_MODELS = ("Q", "Q-noent", "C-bounded", "C-bounded-wide", "C-cosine", "C-host", "MLP", "LightGBM",
              "C-host-nobias") + REVISION_CONTROLS

Scale = Optional[Union[float, str]]   # float, "learned", or None for unbounded heads


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    rho: Union[int, str]          # 10 / 100 / 1000 or "nat"
    model: str
    s: Scale
    loss: str                     # "CE", "CB", "LA"
    seed: int
    tags: tuple = ()              # "bias", "reupload"; revision: "valcap", "lr<value>"
    tier: str = field(default="core", compare=False)
    priority: int = field(default=5, compare=False)

    @property
    def run_id(self) -> str:
        s = "none" if self.s is None else (self.s if isinstance(self.s, str) else f"{self.s:g}")
        tags = ("-" + "+".join(sorted(self.tags))) if self.tags else ""
        return f"{self.dataset}_r{self.rho}_{self.model}_s{s}_{self.loss}{tags}_seed{self.seed}"

    @property
    def family(self) -> str:
        """Hyper-parameters are tuned once per (dataset, family, loss)."""
        tag = ("+" + "+".join(sorted(self.tags))) if "reupload" in self.tags else ""
        return f"{self.model}{tag}"

    @property
    def is_quantum(self) -> bool:
        return self.model in QUANTUM_MODELS

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        d["run_id"] = self.run_id
        d["family"] = self.family
        return d


# ----------------------------------------------------------------------------- grid
RHOS = (10, 100, 1000)
CORE_S = (1.0, 2.0, 4.0, "learned")
SWEEP_S = (1.5, 3.0, 6.0)
SEEDS10 = tuple(range(10))
SEEDS5 = tuple(range(5))
MAIN_DATASETS = ("toniot8", "ciciot8")
HW_SPEC = dict(dataset="toniot8", rho=100, s=3.0)   # the model sent to ibm_kingston
# Revision, hardware session 2: a post-hoc model whose range sits just above the rho = 100 threshold
# (2 s = 4.70 > log 100 = 4.61), so a measured hardware contraction alpha < 0.98 makes the head class
# unreachable (Lemma 3), and an entanglement-free model to separate CZ noise from readout noise.
HW2_S_POSTHOC = 2.35
LR_SWEEP = (1e-3, 3e-3, 1e-2, 3e-2)


def lr_tag(lr: float) -> str:
    return f"lr{lr:g}"


def tag_lr(tags) -> float | None:
    for t in tags:
        if t.startswith("lr"):
            return float(t[2:])
    return None


def _add(out: dict, spec: RunSpec):
    """Keep one copy per run_id, at its most urgent priority."""
    old = out.get(spec.run_id)
    if old is None or spec.priority < old.priority:
        out[spec.run_id] = spec


def build_grid(tiers=("hw", "core", "sweep", "ablation", "ref")) -> list[RunSpec]:
    out: dict[str, RunSpec] = {}
    for tier in tiers:
        if tier == "hw":
            for loss in ("CE", "LA"):
                for model in ("Q", "C-bounded"):
                    _add(out, RunSpec(HW_SPEC["dataset"], HW_SPEC["rho"], model, HW_SPEC["s"], loss, 0,
                                      tier="hw", priority=0))
        elif tier == "core":
            for ds in MAIN_DATASETS:
                for rho in RHOS:
                    for seed in SEEDS10:
                        pr = 1 if seed < 5 else 2         # first five seeds first: early analysis possible
                        for loss in ("CE", "LA"):
                            _add(out, RunSpec(ds, rho, "C-host", None, loss, seed, tier="core", priority=pr))
                            for s in CORE_S:
                                for model in ("Q", "C-bounded"):
                                    _add(out, RunSpec(ds, rho, model, s, loss, seed, tier="core", priority=pr))
                            for s in (2.0, 4.0):
                                _add(out, RunSpec(ds, rho, "Q-noent", s, loss, seed, tier="core", priority=pr))
                                _add(out, RunSpec(ds, rho, "C-cosine", s, loss, seed, tier="core", priority=pr))
        elif tier == "sweep":
            for ds in MAIN_DATASETS:
                for rho in RHOS:
                    for seed in SEEDS5:
                        for loss in ("CE", "LA"):
                            for s in SWEEP_S:
                                for model in ("Q", "C-bounded"):
                                    _add(out, RunSpec(ds, rho, model, s, loss, seed, tier="sweep", priority=3))
        elif tier == "ablation":
            for ds in MAIN_DATASETS:
                for rho in RHOS:
                    for seed in SEEDS5:
                        for model in ("Q", "C-bounded", "C-host"):
                            _add(out, RunSpec(ds, rho, model, None if model == "C-host" else 2.0, "CB", seed,
                                              tier="ablation", priority=4))
                        for model in ("Q", "C-bounded"):
                            for loss in ("CE", "LA"):
                                _add(out, RunSpec(ds, rho, model, 2.0, loss, seed, ("bias",), tier="ablation", priority=4))
                        for loss in ("CE", "LA"):
                            _add(out, RunSpec(ds, rho, "Q", 2.0, loss, seed, ("reupload",), tier="ablation", priority=4))
                            _add(out, RunSpec(ds, rho, "C-bounded-wide", 2.0, loss, seed, tier="ablation", priority=4))
                # natural imbalance profile
                for seed in SEEDS5:
                    for loss in ("CE", "LA"):
                        _add(out, RunSpec(ds, "nat", "C-host", None, loss, seed, tier="ablation", priority=4))
                        for model in ("Q", "C-bounded"):
                            _add(out, RunSpec(ds, "nat", model, 2.0, loss, seed, tier="ablation", priority=4))
            # sensitivity to the excluded classes (MITM, ransomware); rho = 10 is infeasible here because the
            # ransomware pool (~150 rows) is below its rho = 10 target
            for rho in (100, 1000):
                for seed in SEEDS5:
                    for loss in ("CE", "LA"):
                        _add(out, RunSpec("toniot10", rho, "C-host", None, loss, seed, tier="ablation", priority=4))
                        for model in ("Q", "C-bounded"):
                            _add(out, RunSpec("toniot10", rho, model, 2.0, loss, seed, tier="ablation", priority=4))
        elif tier == "revision":
            ds0 = HW_SPEC["dataset"]
            # R0: models for hardware session 2
            _add(out, RunSpec(ds0, 100, "Q", HW2_S_POSTHOC, "CE", 0, tier="revision", priority=0))
            _add(out, RunSpec(ds0, 100, "Q-noent", HW_SPEC["s"], "LA", 0, tier="revision", priority=0))
            for ds in MAIN_DATASETS:
                for rho in RHOS:
                    for loss in ("CE", "LA"):
                        # R1: the entanglement-free circuit where the Q - C-bounded gap is largest
                        for s in (1.0, 1.5, "learned"):
                            for seed in SEEDS10:
                                _add(out, RunSpec(ds, rho, "Q-noent", s, loss, seed, tier="revision", priority=1))
                        # R2: classical readout controls
                        for model in REVISION_CONTROLS:
                            for s in (1.0, 2.0, 4.0):
                                for seed in SEEDS5:
                                    _add(out, RunSpec(ds, rho, model, s, loss, seed, tier="revision", priority=2))
                        # R5: unbounded head without output bias (H1 confound)
                        for seed in SEEDS10:
                            _add(out, RunSpec(ds, rho, "C-host-nobias", None, loss, seed, tier="revision",
                                              priority=2))
                # R3: learning-rate robustness, evaluation seeds 1-3 (tuning used seed 0)
                for model in ("Q", "Q-noent", "C-bounded"):
                    for s in (1.0, 4.0):
                        for loss in ("CE", "LA"):
                            for lr in LR_SWEEP:
                                for seed in (1, 2, 3):
                                    _add(out, RunSpec(ds, 100, model, s, loss, seed, (lr_tag(lr),),
                                                      tier="revision", priority=3))
                # R4: early stopping on a validation set capped at each class's training count
                for rho in (100, 1000):
                    for loss in ("CE", "LA"):
                        for seed in SEEDS5:
                            _add(out, RunSpec(ds, rho, "C-host", None, loss, seed, ("valcap",), tier="revision",
                                              priority=3))
                            for model in ("Q", "C-bounded"):
                                for s in (2.0, 4.0):
                                    _add(out, RunSpec(ds, rho, model, s, loss, seed, ("valcap",),
                                                      tier="revision", priority=3))
        elif tier == "ref":
            for ds in MAIN_DATASETS:
                for rho in RHOS:
                    for seed in SEEDS10:
                        for loss in ("CE", "LA"):
                            _add(out, RunSpec(ds, rho, "MLP", None, loss, seed, tier="ref", priority=2))
                        _add(out, RunSpec(ds, rho, "LightGBM", None, "CB", seed, tier="ref", priority=2))
        else:
            raise ValueError(f"unknown tier {tier!r}")
    return sorted(out.values(), key=lambda r: (r.priority, r.seed, r.dataset, str(r.rho), r.model, str(r.s), r.loss, r.tags))


# ----------------------------------------------------------------------------- tuning
TUNE_RHO = 100
TUNE_SEED = 0
TUNE_GRID = [dict(lr=lr, wd=wd) for lr in (1e-3, 3e-3, 1e-2, 3e-2) for wd in (0.0, 1e-4)]   # 8 trials
TUNE_S = 2.0
TUNE_FAMILIES = [("Q", ()), ("Q-noent", ()), ("Q", ("reupload",)), ("C-bounded", ()), ("C-bounded-wide", ()),
                 ("C-cosine", ()), ("C-host", ()), ("MLP", ()),
                 # revision controls
                 ("C-sin-diag", ()), ("C-diag", ()), ("C-sin", ()), ("C-bounded-init", ()), ("C-host-nobias", ())]


def build_tuning() -> list[tuple[RunSpec, dict, int]]:
    """(spec, hyper-parameters, trial index) for every tuning trial."""
    out = []
    for ds in DATASETS:
        for model, tags in TUNE_FAMILIES:
            for loss in ("CE", "LA"):
                s = None if model in ("C-host", "C-host-nobias", "MLP") else TUNE_S
                spec = RunSpec(ds, TUNE_RHO, model, s, loss, TUNE_SEED, tags, tier="tune", priority=0)
                for i, hp in enumerate(TUNE_GRID):
                    out.append((spec, hp, i))
    return out


def tune_key(dataset: str, family: str, loss: str) -> str:
    # class-balanced runs reuse the cross-entropy hyper-parameters
    return f"{dataset}|{family}|{'CE' if loss == 'CB' else loss}"
