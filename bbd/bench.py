"""Measure training-step time on the current device and estimate the grid's GPU hours."""
from __future__ import annotations

import math
import time

import torch

from .config import BATCH_SIZE, TRAIN_SIZE, RunSpec, build_grid, build_tuning
from .losses import Objective
from .models import build_model


def step_ms(model_name: str, s=2.0, device=None, batch: int = BATCH_SIZE, d: int = 40, k: int = 9, reps: int = 30,
            tags=()) -> float:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    spec = RunSpec("toniot8", 100, model_name, None if model_name in ("C-host", "MLP") else s, "CE", 0, tags)
    model = build_model(spec, d, k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    obj = Objective("CE", [100] * k, device=device)
    x = torch.randn(batch, d, device=device)
    y = torch.randint(0, k, (batch,), device=device)

    def step():
        loss = obj(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(5):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / reps


def autotune_group(device=None, choices=(2, 3, 4, 5, 10), reps: int = 12, verbose: bool = True) -> int:
    """Pick the fused-block size of the simulator that trains fastest on this device (sets qsim.GROUP)."""
    import os
    from . import qsim
    if os.environ.get("BBD_QGROUP"):
        qsim.GROUP = int(os.environ["BBD_QGROUP"])
        return qsim.GROUP
    times = {}
    for g in choices:
        qsim.GROUP = g
        times[g] = step_ms("Q", device=device, reps=reps)
    qsim.GROUP = min(times, key=times.get)
    if verbose:
        print("simulator block size:", qsim.GROUP, {g: round(t, 2) for g, t in times.items()}, "ms/step", flush=True)
    return qsim.GROUP


def estimate_hours(ms: dict, n_workers: int = 1, mean_epochs: int = 60, tiers=("hw", "core", "sweep", "ablation", "ref")):
    """ms: {"Q": ..., "C": ...} milliseconds per step. Returns estimated wall-clock hours (grid, tuning)."""
    steps = mean_epochs * math.ceil(TRAIN_SIZE / BATCH_SIZE)
    overhead = 4.0                                                   # seconds per run: data, eval, saving
    sec = lambda spec: overhead + steps * (ms["Q"] if spec.is_quantum else ms["C"]) / 1000
    grid = sum(sec(s) for s in build_grid(tiers) if s.model != "LightGBM")
    tune = sum(sec(s) for s, _, _ in build_tuning())
    return dict(grid_hours=grid / 3600 / n_workers, tuning_hours=tune / 3600 / n_workers,
                n_grid=len(build_grid(tiers)), n_quantum=sum(s.is_quantum for s in build_grid(tiers)))


def report(device=None, n_workers=1):
    autotune_group(device)
    ms = {"Q": step_ms("Q", device=device), "Q-noent": step_ms("Q-noent", device=device),
          "C": step_ms("C-bounded", device=device), "C-host": step_ms("C-host", device=device)}
    est = estimate_hours({"Q": ms["Q"], "C": ms["C"]}, n_workers)
    return ms, est
