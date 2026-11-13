"""Resumable, sharded execution of the tuning stage and the experiment grid.

Layout of a results directory
  ledger/<session>_w<i>.jsonl   one JSON record per (run, evaluation variant); append-only
  packs/<session>_w<i>_<n>.zip  run artifacts, a few hundred per zip (bbd.store):
                                  logits/<run_id>.npz  test and validation logits, counts, curves
                                  ckpt/<run_id>.pt     model weights (hardware and analysis notebooks)
  logs/                         per-worker logs

A run is done when its "raw" record exists in any ledger, so a new Kaggle session that
carries the previous ledgers forward (carry_forward) only runs what is left. Workers
claim runs through lock files, so two GPUs share one queue without overlap.

CLI
  python -m bbd.runner tune   --worker 0 --n-workers 2 --budget-hours 11
  python -m bbd.runner run    --worker 0 --n-workers 2 --budget-hours 11 --tiers hw,core,sweep,ablation,ref
  python -m bbd.runner status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import qsim
from .config import (tag_lr, INPUT_ROOT, POSTHOC_TAUS, QUANTUM_MODELS, RESULTS_DIR, SHOTS_EVAL, RunSpec,
                     build_grid, build_tuning, tune_key)
from .data import load_prepared, sample_train
from .losses import Objective, posthoc
from .metrics import evaluate
from .models import build_model, count_params
from .train import fit, predict_logits
from . import store

SESSION = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha1(socket.gethostname().encode()).hexdigest()[:4]


# ----------------------------------------------------------------------------- ledgers
def ledger_dir(results: Path) -> Path:
    d = Path(results) / "ledger"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_records(results: Path = RESULTS_DIR) -> list[dict]:
    recs, seen = [], set()
    for f in sorted(ledger_dir(results).glob("*.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:              # a line cut off by a killed session
                continue
            key = (r.get("stage"), r.get("run_id"), r.get("variant"), r.get("trial"))
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)
    return recs


def done_keys(results: Path) -> set:
    out = set()
    for r in read_records(results):
        if r.get("stage") == "grid" and r.get("variant") == "raw":
            out.add(r["run_id"])
        elif r.get("stage") == "tune":
            out.add(f"{r['run_id']}#t{r['trial']}")
    return out


def append(results: Path, worker: int, records: list[dict]):
    """One O_APPEND write per batch, so a reader never sees half a batch from a healthy worker."""
    payload = "".join(json.dumps(r, default=_jsonable) + "\n" for r in records).encode()
    fd = os.open(ledger_dir(results) / f"{SESSION}_w{worker}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def claim(key: str, claims: Path) -> bool:
    claims.mkdir(parents=True, exist_ok=True)
    path = claims / (hashlib.sha1(key.encode()).hexdigest() + ".lock")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, key.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def carry_forward(results: Path = RESULTS_DIR, root: Path = INPUT_ROOT, verbose: bool = True,
                  kinds=("ledger", "packs", "logits", "ckpt")) -> int:
    """Copy ledgers (and, by default, artifact packs) from attached outputs of earlier sessions.

    Notebook 02 copies everything so its latest output is complete; notebooks 03 and 04 copy only the
    ledgers and read artifacts directly from the attached inputs (see bbd.store)."""
    results = Path(results)
    n = 0
    for led in sorted(Path(root).glob("**/results/ledger")):
        src = led.parent
        if src.resolve() == results.resolve():
            continue
        for sub in kinds:
            if not (src / sub).exists():
                continue
            (results / sub).mkdir(parents=True, exist_ok=True)
            for f in (src / sub).iterdir():
                dst = results / sub / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    n += 1
        if verbose:
            print(f"carried forward from {src}")
    store.refresh()
    return n


# ----------------------------------------------------------------------------- hyper-parameters
def best_hyperparameters(results: Path) -> dict:
    best = {}
    for r in read_records(results):
        if r.get("stage") != "tune" or not np.isfinite(r.get("val_macro_f1", np.nan)):
            continue
        key = r["tune_key"]
        cand = (r["val_macro_f1"], -r["hp"]["lr"], -r["hp"]["wd"])
        if key not in best or cand > best[key][0]:
            best[key] = (cand, r["hp"])
    return {k: v[1] for k, v in best.items()}


DEFAULT_HP = {"lr": 3e-3, "wd": 0.0}


# ----------------------------------------------------------------------------- execution
class DataCache:
    def __init__(self, device):
        self.device = device
        self.store = {}

    def get(self, name):
        if name not in self.store:
            p = load_prepared(name)
            X = torch.as_tensor(p.X, device=self.device)
            y = torch.as_tensor(p.y, device=self.device)
            val = torch.as_tensor(p.split == 1, device=self.device)
            test = torch.as_tensor(p.split == 2, device=self.device)
            self.store[name] = dict(p=p, X=X, y=y, Xval=X[val], yval=y[val], Xte=X[test], yte=y[test])
        return self.store[name]


def capped_val_index(yval: np.ndarray, counts, seed: int) -> np.ndarray:
    """Validation rows for early stopping, at most the training count of each class (at least 1)."""
    rng = np.random.default_rng(1_000_003 + seed)
    out = []
    for c, n in enumerate(np.asarray(counts)):
        rows = np.flatnonzero(yval == c)
        out.append(rng.choice(rows, size=int(min(len(rows), max(1, n))), replace=False))
    return np.sort(np.concatenate(out))


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lightgbm(spec, d, idx, counts):
    import lightgbm as lgb
    p = d["p"]
    Xtr, ytr = p.X[idx], p.y[idx]
    clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=spec.seed,
                             verbose=-1, n_jobs=int(os.environ.get("BBD_THREADS", "2")))
    Xv, yv = p.part("val")
    clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    Xt, _ = p.part("test")
    zt = clf.predict(Xt, raw_score=True)
    zv = clf.predict(Xv, raw_score=True)
    hist = {"best_epoch": int(clf.best_iteration_ or 0), "epochs_run": int(clf.best_iteration_ or 0)}
    return np.asarray(zt, np.float32), np.asarray(zv, np.float32), hist, {"params_total": None, "params_head": None}


_WRITERS: dict = {}


def writer_for(results: Path, worker: int = 0) -> store.PackWriter:
    key = (str(results), worker)
    if key not in _WRITERS:
        _WRITERS[key] = store.PackWriter(results, SESSION, worker)
    return _WRITERS[key]


def execute(spec: RunSpec, hp: dict, cache: DataCache, device, results: Path | None, trial: int | None = None,
            shots: bool = True, worker: int = 0) -> list[dict]:
    """Train and evaluate one run. Returns ledger records; saves logits + checkpoint unless tuning."""
    d = cache.get(spec.dataset)
    p = d["p"]
    k = p.k
    idx, counts = sample_train(p, spec.rho, spec.seed)
    lr_fixed = tag_lr(spec.tags)
    if lr_fixed is not None:                         # learning-rate sweep: the tag overrides tuning
        hp = {"lr": lr_fixed, "wd": 0.0}
    Xval_fit, yval_fit = d["Xval"], d["yval"]
    if "valcap" in spec.tags:                        # early stopping on as many rows per class as training has
        vidx = capped_val_index(d["yval"].cpu().numpy(), counts, spec.seed)
        vt = torch.as_tensor(vidx, device=d["Xval"].device)
        Xval_fit, yval_fit = d["Xval"][vt], d["yval"][vt]
    log_pi = np.log(counts / counts.sum())
    _seed_all(spec.seed)
    t0 = time.time()
    probs_te = None
    if spec.model == "LightGBM":
        zt, zv, hist, npar = _lightgbm(spec, d, idx, counts)
        bound, s_final = None, None
    else:
        model = build_model(spec, p.d, k).to(device)
        obj = Objective(spec.loss, counts, device=device)
        idx_t = torch.as_tensor(idx, device=device)
        _, hist = fit(model, d["X"][idx_t], d["y"][idx_t], Xval_fit, yval_fit, obj, k,
                      lr=hp["lr"], wd=hp["wd"], seed=spec.seed)
        zt = predict_logits(model, d["Xte"]).cpu().numpy()
        zv = predict_logits(model, d["Xval"]).cpu().numpy()
        npar = count_params(model)
        bound = model.bound
        s_final = bound
        if spec.is_quantum and shots and trial is None:
            with torch.no_grad():
                h = model.encoder(d["Xte"])
                probs_te = qsim.probabilities(model.head.angles(h), model.head.phi, model.head.entangle,
                                              model.head.reupload)
    yte = p.part("test")[1]
    yval = p.part("val")[1]
    base = dict(spec.to_dict(), stage="tune" if trial is not None else "grid", trial=trial, hp=hp,
                tune_key=tune_key(spec.dataset, spec.family, spec.loss), k=k, class_names=p.class_names,
                n_train=int(counts.sum()), n_val_fit=int(len(yval_fit)),
                train_counts=counts.tolist(), actual_rho=float(counts.max() / counts.min()), s_final=s_final,
                seconds=time.time() - t0, device=str(device), session=SESSION,
                **npar, **{kk: hist[kk] for kk in ("best_epoch", "epochs_run") if kk in hist})
    if trial is not None:
        mv = evaluate(zv, yval, k, bound)
        return [dict(base, variant="val", val_macro_f1=mv["macro_f1"], val_bacc=mv["bacc"])]

    recs = [dict(base, variant="raw", **evaluate(zt, yte, k, bound), val_macro_f1=hist.get("best_val_macro_f1"))]
    if spec.loss in ("CE", "CB"):
        for tau in POSTHOC_TAUS:
            recs.append(dict(base, variant=f"posthoc{tau:g}", tau=tau, **evaluate(posthoc(zt, log_pi, tau), yte, k, None)))
    extra = {}
    if probs_te is not None:
        gen = torch.Generator(device=probs_te.device).manual_seed(spec.seed)
        ez = qsim.sampled_expectations(probs_te, k, SHOTS_EVAL, gen).cpu().numpy()
        zs = bound * ez
        if getattr(model, "class_bias", None) is not None:
            zs = zs + model.class_bias.detach().cpu().numpy()
        zs = zs.astype(np.float32)
        extra["z_test_shots"] = zs
        recs.append(dict(base, variant=f"shots{SHOTS_EVAL}", **evaluate(zs, yte, k, bound)))
        if spec.loss in ("CE", "CB"):
            recs.append(dict(base, variant=f"shots{SHOTS_EVAL}+posthoc1", tau=1.0,
                             **evaluate(posthoc(zs, log_pi, 1.0), yte, k, None)))
    if results is not None:                          # artifacts first, ledger record after (see run_worker)
        w = writer_for(results, worker)
        w.put(f"logits/{spec.run_id}.npz", store.npz_bytes(
            z_test=zt.astype(np.float32), z_val=zv.astype(np.float32), y_test=yte, y_val=yval, counts=counts,
            log_pi=log_pi, train_loss=np.asarray(hist.get("train_loss", [])),
            val_macro_f1=np.asarray(hist.get("val_macro_f1", [])), **extra))
        if spec.model != "LightGBM":
            w.put(f"ckpt/{spec.run_id}.pt", store.torch_bytes(
                {"state_dict": {kk: v.cpu() for kk, v in model.state_dict().items()}, "spec": spec.to_dict(),
                 "hp": hp, "d": p.d, "k": k, "counts": counts.tolist()}))
    return recs


def spec_from_dict(d: dict) -> RunSpec:
    s = d["s"]
    return RunSpec(d["dataset"], d["rho"] if d["rho"] == "nat" else int(d["rho"]), d["model"], s, d["loss"],
                   int(d["seed"]), tuple(d.get("tags", ())), d.get("tier", "core"), d.get("priority", 5))


def load_model(results: Path, run_id: str, device="cpu"):
    ck = store.load_ckpt(results, run_id, device)
    spec = spec_from_dict(ck["spec"])
    model = build_model(spec, ck["d"], ck["k"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, spec, ck


# ----------------------------------------------------------------------------- workers
def _log(worker, msg):
    print(f"[{time.strftime('%H:%M:%S')}] w{worker} {msg}", flush=True)


def run_worker(stage: str, worker: int = 0, n_workers: int = 1, budget_hours: float = 11.0,
               tiers=("hw", "core", "sweep", "ablation", "ref"), results: Path = RESULTS_DIR,
               device: str | None = None, shots: bool = True, deadline: float | None = None,
               launch_id: str | None = None):
    results = Path(results)
    torch.set_num_threads(int(os.environ.get("BBD_THREADS", "2")))     # two workers share Kaggle's 4 vCPUs
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    deadline = deadline or (time.time() + budget_hours * 3600)
    from .bench import autotune_group
    autotune_group(device, verbose=False)
    _log(worker, f"simulator block size {qsim.GROUP}")
    # workers of one launch share a claims folder; a relaunch gets a fresh one
    launch_id = launch_id or os.environ.get("BBD_LAUNCH", "default")
    claims = Path(os.environ.get("BBD_CLAIMS", "/tmp/bbd_claims")) / f"{stage}_{launch_id}"   # not in the output
    cache = DataCache(device)
    done = done_keys(results)
    if stage == "tune":
        jobs = [(spec, hp, i, f"{spec.run_id}#t{i}") for spec, hp, i in build_tuning()]
    else:
        hps = best_hyperparameters(results)
        jobs = []
        for spec in build_grid(tiers):
            key = tune_key(spec.dataset, spec.family, spec.loss)
            jobs.append((spec, hps.get(key, DEFAULT_HP), None, spec.run_id))
            if key not in hps and spec.model != "LightGBM":
                _log(worker, f"no tuning result for {key}; using defaults {DEFAULT_HP}")
    jobs = [j for j in jobs if j[3] not in done]
    _log(worker, f"{stage}: {len(jobs)} jobs pending on {device}")
    est: dict[str, float] = {}
    n_ok = 0
    for spec, hp, trial, key in jobs:
        fam = f"{spec.dataset}|{spec.family}"
        guess = est.get(fam, 60.0 if spec.is_quantum else 20.0)
        if deadline - time.time() < 1.5 * guess + 60:
            _log(worker, "time budget reached; stopping cleanly")
            break
        if not claim(key, claims):
            continue
        try:
            recs = execute(spec, hp, cache, device, None if trial is not None else results, trial, shots, worker)
            append(results, worker, recs)
            sec = recs[0]["seconds"]
            est[fam] = sec if fam not in est else 0.7 * est[fam] + 0.3 * sec
            n_ok += 1
            r0 = recs[0]
            score = r0.get("val_macro_f1") if trial is not None else r0.get("macro_f1")
            _log(worker, f"{key}  {sec:5.0f}s  macro-F1 {score:.3f}")
        except Exception as e:
            append(results, worker, [dict(spec.to_dict(), stage="error", trial=trial, error=repr(e),
                                          traceback=traceback.format_exc()[-2000:], session=SESSION)])
            _log(worker, f"ERROR {key}: {e!r}")
            if "out of memory" in repr(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
    _log(worker, f"{stage}: finished {n_ok} jobs")
    return n_ok


def status(results: Path = RESULTS_DIR, tiers=("hw", "core", "sweep", "ablation", "ref")) -> dict:
    done = done_keys(results)
    grid = build_grid(tiers)
    tune = build_tuning()
    by_tier = {}
    for s in grid:
        t = by_tier.setdefault(s.tier, [0, 0])
        t[1] += 1
        t[0] += s.run_id in done
    n_tune_done = sum(f"{s.run_id}#t{i}" in done for s, _, i in tune)
    errors = [r for r in read_records(results) if r.get("stage") == "error"]
    out = {"tuning": (n_tune_done, len(tune)), **{f"grid:{k}": tuple(v) for k, v in by_tier.items()},
           "errors": len(errors)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "run", "status"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--budget-hours", type=float, default=11.0)
    ap.add_argument("--deadline", type=float, default=None, help="absolute unix time to stop launching runs")
    ap.add_argument("--tiers", default="hw,core,sweep,ablation,ref")
    ap.add_argument("--results", default=str(RESULTS_DIR))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-shots", action="store_true")
    ap.add_argument("--launch-id", default=None)
    a = ap.parse_args(argv)
    tiers = tuple(t for t in a.tiers.split(",") if t)
    if a.stage == "status":
        print(json.dumps(status(Path(a.results), tiers), indent=1))
        return
    run_worker("tune" if a.stage == "tune" else "grid", a.worker, a.n_workers, a.budget_hours, tiers,
               Path(a.results), a.device, not a.no_shots, a.deadline, a.launch_id)


if __name__ == "__main__":
    main()
