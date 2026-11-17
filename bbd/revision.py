"""Analyses added in response to peer review. Every function reads the run ledgers (records_frame) and, where
it needs per-input predictions, the saved logits (bbd.store); none of them retrains anything.

  reachability_pretest   Lemma 2 predicted from information available before the test set is seen
                         (nominal s, or the effective range on the validation logits), not from test logits
  h1_decomposition       where C-bounded loses decisions to C-host under CE: saturated inputs or not
  pooled_grid            pooled paired differences A - B by s and loss, with t intervals (controls, Q-noent, ...)
  lr_sweep_table         the Q - C-bounded gap at every learning rate and at the validation-selected one
  valcap_table           does early stopping on a capped (imbalanced) validation set change the comparisons?
  leakage_audit          near-duplicate test rows (nearest training row in feature space) and comparisons
                         recomputed on the test rows without a near duplicate
  natural_prior_table    benign false-alarm rate and attack precision at the natural class prior
  hyperparameter_table   the learning rate / weight decay selected for every (dataset, family, loss)
  figures                rev_fig_scale_gap, rev_fig_lr_sweep
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import store
from . import style as S
from .config import DELTA, REVISION_CONTROLS
from .stats import FIXED_S, KEYS, paired, pooled
from .theory import unreachable_count

BOUNDED_BIASFREE = ("Q", "Q-noent", "C-bounded", "C-bounded-wide", "C-cosine") + REVISION_CONTROLS
MAIN = ("toniot8", "ciciot8")


def _main(df):
    return df[(df["dataset"].isin(MAIN)) & (df["rho_str"] != "nat")]


def _concat(parts) -> pd.DataFrame:
    parts = [p for p in parts if isinstance(p, pd.DataFrame) and len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _half_spread(z: np.ndarray) -> float:
    return float(0.5 * (z.max(1) - z.min(1)).max())


# ----------------------------------------------------------------------------- 1. reachability before testing
def reachability_pretest(df: pd.DataFrame, results) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agreement between predicted unreachable classes and classes never predicted on the test set, with the
    prediction made from (a) the test logits (as in the first draft), (b) the validation logits, (c) the nominal
    s (fixed-s runs; the learned s at its final value otherwise). Only (b) and (c) are genuine predictions."""
    ph = df[df["variant"].isin(["posthoc0.5", "posthoc1"]) & df["model"].isin(BOUNDED_BIASFREE)
            & ~df["tags_str"].str.contains("bias") & ~df["tags_str"].str.contains("lr")].copy()
    raw = df[df["variant"] == "raw"].set_index("run_id")
    rows = []
    for r in ph.itertuples():
        tau = 0.5 if r.variant == "posthoc0.5" else 1.0
        pi = np.asarray(r.train_counts, float)
        pi = pi / pi.sum()
        z = store.load_logits(results, r.run_id)
        if z is None:
            continue
        r_test = float(raw.at[r.run_id, "r_eff"]) if r.run_id in raw.index else _half_spread(z["z_test"])
        r_val = _half_spread(z["z_val"])
        s_nom = float(r.s) if r.s_label not in ("learned", "none") else float(r.s_final)
        rows.append(dict(run_id=r.run_id, model=r.model, dataset=r.dataset, tau=tau, learned=r.s_label == "learned",
                         observed=int(r.never_predicted),
                         pred_test=unreachable_count(pi, r_test, tau), pred_val=unreachable_count(pi, r_val, tau),
                         pred_nominal=unreachable_count(pi, s_nom, tau)))
    d = pd.DataFrame(rows)
    if not len(d):
        return d, d
    summ = []
    for col, label in (("pred_test", "effective range on test logits (draft)"),
                       ("pred_val", "effective range on validation logits"),
                       ("pred_nominal", "nominal s, before training")):
        for sub, name in ((d, "all"), (d[~d["learned"]], "fixed s only")):
            summ.append(dict(predictor=label, subset=name, n=len(sub),
                             exact=float((sub["observed"] == sub[col]).mean()),
                             observed_ge=float((sub["observed"] >= sub[col]).mean()),
                             observed_lt=int((sub["observed"] < sub[col]).sum()),
                             mae=float(np.abs(sub["observed"] - sub[col]).mean())))
    per_model = d.groupby("model").apply(lambda g: pd.Series(dict(
        n=len(g), exact_val=(g["observed"] == g["pred_val"]).mean(),
        exact_nominal=(g["observed"] == g["pred_nominal"]).mean()))).reset_index()
    return pd.DataFrame(summ), per_model


# ----------------------------------------------------------------------------- 2. H1 decomposition
def h1_decomposition(df: pd.DataFrame, results, host: str = "C-host", loss: str = "CE") -> pd.DataFrame:
    """For every (dataset, rho, seed), compare C-bounded with the unbounded head input by input. A decision is
    'lost' if the host is right and C-bounded wrong. Corollary 1 allows departures only through ties at +s
    (saturated inputs: top-two logits >= 0.95 s) or limited capacity; the split says which it is."""
    b = _main(df)
    b = b[(b["variant"] == "raw") & (b["tags_str"] == "") & (b["loss"] == loss)]
    cb = b[(b["model"] == "C-bounded") & b["s_label"].isin(FIXED_S)]
    hs = b[b["model"] == host].set_index(["dataset", "rho_str", "seed"])
    rows = []
    for r in cb.itertuples():
        key = (r.dataset, r.rho_str, r.seed)
        if key not in hs.index:
            continue
        zb, zh = store.load_logits(results, r.run_id), store.load_logits(results, hs.loc[key, "run_id"])
        if zb is None or zh is None:
            continue
        y = zb["y_test"]
        pb, ph_ = zb["z_test"].argmax(1), zh["z_test"].argmax(1)
        top2 = np.sort(zb["z_test"], 1)[:, -2:]
        sat = np.all(top2 >= 0.95 * float(r.s), 1)
        lost = (ph_ == y) & (pb != y)
        gained = (ph_ != y) & (pb == y)
        n = len(y)
        rows.append(dict(dataset=r.dataset, rho=r.rho_str, seed=r.seed, s=r.s_label, n=n,
                         saturated=sat.mean(), lost=lost.sum() / n, gained=gained.sum() / n,
                         net_loss=(lost.sum() - gained.sum()) / n,
                         net_loss_saturated=((lost & sat).sum() - (gained & sat).sum()) / n,
                         net_loss_unsaturated=((lost & ~sat).sum() - (gained & ~sat).sum()) / n,
                         lost_on_saturated=(lost & sat).sum() / max(lost.sum(), 1),
                         disagree=(pb != ph_).mean()))
    d = pd.DataFrame(rows)
    if not len(d):
        return d
    out = d.groupby("s").agg(n_pairs=("n", "size"), saturated=("saturated", "mean"), lost=("lost", "mean"),
                             gained=("gained", "mean"), net_loss=("net_loss", "mean"),
                             net_loss_saturated=("net_loss_saturated", "mean"),
                             net_loss_unsaturated=("net_loss_unsaturated", "mean"),
                             share_of_lost_on_saturated=("lost_on_saturated", "mean")).reset_index()
    out["host"] = host
    return out.sort_values("s", key=lambda c: c.map(lambda v: float(v)))


# ----------------------------------------------------------------------------- 3. pooled comparisons by s, loss
def pooled_grid(df: pd.DataFrame, a: str, b: str, s_values=FIXED_S, losses=("CE", "LA"), variant: str = "raw",
                match_s: bool = True, metric: str = "macro_f1", tags: str = "") -> pd.DataFrame:
    """Pooled paired difference a - b for each (s, loss), plus all s pooled per loss. Pairs share dataset,
    rho, loss, seed (and s when match_s); only seeds present in both are used."""
    base = _main(df)
    base = base[base["tags_str"] == tags]
    rows = []
    for loss in losses:
        for s in list(s_values) + ["all"]:
            sl = list(s_values) if s == "all" else [s]
            fa = {"model": a, "variant": variant, "loss": loss, "s_label": sl}
            fb = {"model": b, "variant": variant, "loss": loss}
            if match_s:
                fb["s_label"] = sl
            d = paired(base, fa, fb, metric=metric, match_s=match_s)
            if not len(d):
                continue
            r = pooled(d)
            rows.append(dict(a=a, b=b, loss=loss, s=s, n=r["n"], n_cells=r["n_cells"], est=100 * r["est"],
                             lo95=100 * r["ci95"][0], hi95=100 * r["ci95"][1], tau=100 * r["tau"],
                             p=r["p_two_sided"], p_tost=r["p_tost"]))
    return pd.DataFrame(rows)


def controls_table(df: pd.DataFrame) -> pd.DataFrame:
    """Which classical head reproduces Q? Every control and Q / Q-noent against C-bounded and against Q."""
    parts = []
    for m in ("Q", "Q-noent") + REVISION_CONTROLS:
        parts.append(pooled_grid(df, m, "C-bounded", s_values=("1", "2", "4")))
        if m != "Q":
            parts.append(pooled_grid(df, m, "Q", s_values=("1", "2", "4")))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------------------------------------------------------- 4. learning-rate sweep
def lr_sweep_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _main(df)
    d = d[(d["variant"] == "raw") & d["tags_str"].str.startswith("lr")].copy()
    if not len(d):
        return d, d
    d["lr"] = d["tags_str"].str[2:].astype(float)
    by_lr = (d.groupby(["dataset", "model", "s_label", "loss", "lr"])
             .agg(macro_f1=("macro_f1", "mean"), val_macro_f1=("val_macro_f1", "mean"), n=("seed", "size"))
             .reset_index())
    # gap Q - C-bounded (and Q-noent - C-bounded) at each lr, paired on dataset / seed
    rows = []
    key = ["dataset", "s_label", "loss", "seed"]
    cb = d[d["model"] == "C-bounded"]
    for m in ("Q", "Q-noent"):
        q = d[d["model"] == m]
        for lr in sorted(d["lr"].unique()):
            mm = q[q["lr"] == lr].merge(cb[cb["lr"] == lr], on=key, suffixes=("", "_b"))
            for (s, loss), g in mm.groupby(["s_label", "loss"]):
                diff = 100 * (g["macro_f1"] - g["macro_f1_b"])
                rows.append(dict(model=m, s=s, loss=loss, lr=f"{lr:g}", gap=diff.mean(),
                                 se=diff.std(ddof=1) / np.sqrt(len(diff)), n=len(diff)))
        # each family at its own validation-selected lr (per dataset, s, loss)
        sel = (by_lr.sort_values("val_macro_f1").groupby(["dataset", "model", "s_label", "loss"]).tail(1)
               [["dataset", "model", "s_label", "loss", "lr"]])
        qs = q.merge(sel[sel["model"] == m], on=["dataset", "model", "s_label", "loss", "lr"])
        bs = cb.merge(sel[sel["model"] == "C-bounded"], on=["dataset", "model", "s_label", "loss", "lr"])
        mm = qs.merge(bs, on=key, suffixes=("", "_b"))
        for (s, loss), g in mm.groupby(["s_label", "loss"]):
            diff = 100 * (g["macro_f1"] - g["macro_f1_b"])
            rows.append(dict(model=m, s=s, loss=loss, lr="selected", gap=diff.mean(),
                             se=diff.std(ddof=1) / np.sqrt(len(diff)), n=len(diff)))
    return by_lr, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 5. capped validation
def valcap_table(df: pd.DataFrame) -> pd.DataFrame:
    d = _main(df)
    d = d[d["variant"].isin(["raw", "posthoc1"])]
    std = d[(d["tags_str"] == "") & (d["seed"] < 5)]
    cap = d[d["tags_str"] == "valcap"]
    if not len(cap):
        return pd.DataFrame()
    key = ["dataset", "rho_str", "model", "s_label", "loss", "variant", "seed"]
    m = cap.merge(std, on=key, suffixes=("_cap", "_std"))
    m["delta"] = 100 * (m["macro_f1_cap"] - m["macro_f1_std"])
    out = (m.groupby(["model", "s_label", "loss", "variant", "rho_str"])
           .agg(std=("macro_f1_std", "mean"), cap=("macro_f1_cap", "mean"), delta=("delta", "mean"),
                n=("seed", "size")).reset_index())
    out[["std", "cap"]] *= 100
    # do the head comparisons survive? Q - C-bounded and C-bounded - C-host under both protocols
    comp = []
    for tag, frame in (("standard", std), ("capped validation", cap)):
        f = frame.assign(tags_str="")
        for a, b, ms in (("Q", "C-bounded", True), ("C-bounded", "C-host", False)):
            for loss in ("CE", "LA"):
                dd = paired(f, {"model": a, "variant": "raw", "loss": loss, "s_label": ["2", "4"]},
                            {"model": b, "variant": "raw", "loss": loss, **({"s_label": ["2", "4"]} if ms else {})},
                            match_s=ms)
                dd = dd[dd["rho_str"].isin(["100", "1000"])]
                if len(dd):
                    r = pooled(dd)
                    comp.append(dict(protocol=tag, comparison=f"{a} - {b}", loss=loss, est=100 * r["est"],
                                     lo95=100 * r["ci95"][0], hi95=100 * r["ci95"][1], n=r["n"]))
    return out, pd.DataFrame(comp)


# ----------------------------------------------------------------------------- 6. leakage audit
def leakage_audit(df: pd.DataFrame, results, thresholds=(1e-6, 0.1, 0.5), rho: int = 100,
                  seeds=range(10), clean_thresholds=(1e-6, 0.1)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distance from each test row to its nearest training row (standardised features), and the main paired
    comparisons recomputed on test rows whose nearest training row is farther than `clean_threshold`."""
    from sklearn.neighbors import NearestNeighbors
    from .data import load_prepared, sample_train
    from .metrics import evaluate
    rows, masks = [], {}
    for ds in MAIN:
        p = load_prepared(ds)
        Xt, yt = p.part("test")
        for seed in seeds:
            idx, _ = sample_train(p, rho, seed)
            nn_ = NearestNeighbors(n_neighbors=1).fit(p.X[idx])
            dist = nn_.kneighbors(Xt, return_distance=True)[0][:, 0]
            masks[(ds, seed)] = dist
            rows.append(dict(dataset=ds, rho=rho, seed=seed, **{f"share_within_{t:g}": float((dist <= t).mean())
                                                               for t in thresholds},
                             median_nn_distance=float(np.median(dist))))
    audit = pd.DataFrame(rows).groupby(["dataset", "rho"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    # recompute macro-F1 on the clean test rows and compare paired differences
    b = _main(df)
    b = b[(b["variant"] == "raw") & (b["tags_str"] == "") & (b["rho_str"] == str(rho))
          & b["model"].isin(["Q", "Q-noent", "C-bounded", "C-host"])]
    clean = []
    for r in b.itertuples():
        dist = masks.get((r.dataset, r.seed))
        z = store.load_logits(results, r.run_id)
        if dist is None or z is None:
            continue
        k = len(r.train_counts)
        bound = None if r.model == "C-host" else float(r.s_final or 1.0)
        row = dict(run_id=r.run_id)
        for th in clean_thresholds:
            m = dist > th
            row[f"macro_f1_clean{th:g}"] = evaluate(z["z_test"][m], z["y_test"][m], k, bound)["macro_f1"]
            row[f"clean_share{th:g}"] = float(m.mean())
        clean.append(row)
    c = b.merge(pd.DataFrame(clean), on="run_id")
    comp = []
    for metric in ["macro_f1"] + [f"macro_f1_clean{th:g}" for th in clean_thresholds]:
        for a, bb, ms, sl in (("Q", "C-bounded", True, FIXED_S), ("Q", "Q-noent", True, ["2", "4"]),
                              ("C-bounded", "C-host", False, FIXED_S)):
            for loss in ("CE", "LA"):
                dd = paired(c, {"model": a, "loss": loss, "s_label": sl},
                            {"model": bb, "loss": loss, **({"s_label": sl} if ms else {})}, metric=metric, match_s=ms)
                if len(dd) > 2:
                    r = pooled(dd)
                    comp.append(dict(test_rows="all" if metric == "macro_f1" else
                                     "no training row within " + metric.replace("macro_f1_clean", ""),
                                     comparison=f"{a} - {bb}", loss=loss, est=100 * r["est"],
                                     lo95=100 * r["ci95"][0], hi95=100 * r["ci95"][1], n=r["n"]))
    return audit, pd.DataFrame(comp)


# ----------------------------------------------------------------------------- 7. natural-prior metrics
def natural_prior_table(df: pd.DataFrame, results, rhos=("100", "1000")) -> pd.DataFrame:
    """Metrics an operator sees at the natural class mix: benign false-alarm rate (prior-free), attack
    detection rate, and macro precision / F1 with the confusion matrix re-weighted to the natural prior."""
    from .data import load_prepared
    from .losses import posthoc
    rows = []
    priors = {}
    for ds in MAIN:
        p = load_prepared(ds)
        priors[ds] = (np.asarray(p.natural_counts, float) / np.sum(p.natural_counts),
                      [i for i, c in enumerate(p.class_names) if c.lower() in ("normal", "benign")][0])
    b = _main(df)
    sel = b[(b["tags_str"] == "") & b["rho_str"].isin(rhos) & (b["variant"] == "raw")
            & (((b["model"].isin(["Q", "Q-noent", "C-bounded"])) & (b["s_label"] == "4"))
               | b["model"].isin(["C-host", "MLP", "LightGBM"]))]
    for r in sel.itertuples():
        z = store.load_logits(results, r.run_id)
        if z is None:
            continue
        pi_nat, benign = priors[r.dataset]
        conds = [(r.loss if r.model != "LightGBM" else "LightGBM", z["z_test"])]
        if r.loss in ("CE", "CB") and r.model != "LightGBM":
            # post-hoc adjustment of a CB-trained model corrects for the prior twice; label it apart
            cond_ph = "post-hoc" if r.loss == "CE" else f"{r.loss}+post-hoc"
            conds.append((cond_ph, posthoc(z["z_test"], z["log_pi"], 1.0)))
        y = z["y_test"]
        k = len(pi_nat)
        for cond, zz in conds:
            pred = zz.argmax(1)
            C = np.zeros((k, k))
            for i in range(k):
                C[i] = np.bincount(pred[y == i], minlength=k) / max((y == i).sum(), 1)
            rec = np.diag(C)
            col = (C * pi_nat[:, None]).sum(0)
            prec = np.where(col > 0, np.diag(C) * pi_nat / np.maximum(col, 1e-12), 0.0)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
            attack = [i for i in range(k) if i != benign]
            rows.append(dict(dataset=r.dataset, rho=r.rho_str, model=r.model, s=r.s_label, condition=cond,
                             seed=r.seed, benign_false_alarm=1 - C[benign, benign],
                             attack_detection=float(np.mean([1 - C[i, benign] for i in attack])),
                             macro_precision_natural=float(prec.mean()), macro_f1_natural=float(f1.mean())))
    d = pd.DataFrame(rows)
    if not len(d):
        return d
    return (d.groupby(["dataset", "rho", "model", "s", "condition"]).mean(numeric_only=True)
            .drop(columns="seed").reset_index())


# ----------------------------------------------------------------------------- 8. hyper-parameters
def hyperparameter_table(results) -> pd.DataFrame:
    from .runner import read_records
    best = {}
    for r in read_records(results):
        if r.get("stage") != "tune" or not np.isfinite(r.get("val_macro_f1", np.nan)):
            continue
        cand = (r["val_macro_f1"], -r["hp"]["lr"], -r["hp"]["wd"])
        if r["tune_key"] not in best or cand > best[r["tune_key"]][0]:
            best[r["tune_key"]] = (cand, r["hp"])
    rows = []
    for key, (cand, hp) in sorted(best.items()):
        ds, fam, loss = key.split("|")
        rows.append(dict(dataset=ds, family=fam, loss=loss, lr=hp["lr"], wd=hp["wd"], val_macro_f1=cand[0]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def rev_fig_scale_gap(df: pd.DataFrame, out):
    """Paired difference against C-bounded by logit bound, for Q, Q-noent and the classical readout controls."""
    import matplotlib.pyplot as plt
    S.apply()
    models = [m for m in ("Q", "Q-noent") + REVISION_CONTROLS if m in set(df["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 2.2), sharey=True)
    svals = [s for s in FIXED_S]
    xpos = {s: i for i, s in enumerate(svals)}
    any_ = False
    for ax, loss, letter in zip(axes, ("CE", "LA"), "ab"):
        ax.axhspan(-100 * DELTA, 100 * DELTA, color=S.LIGHT, alpha=0.6, lw=0)
        ax.axhline(0, color=S.GREY, lw=0.5)
        for j, m in enumerate(models):
            g = pooled_grid(df, m, "C-bounded", s_values=svals, losses=(loss,))
            if not len(g):
                continue
            g = g[g["s"] != "all"]
            if not len(g):
                continue
            any_ = True
            x = np.array([xpos[s] for s in g["s"]]) + (j - len(models) / 2) * 0.07
            ax.errorbar(x, g["est"], yerr=[g["est"] - g["lo95"], g["hi95"] - g["est"]], color=S.MODEL_COLORS[m],
                        marker=S.MODEL_MARKERS[m], ms=3, lw=1.0, elinewidth=0.6, label=S.MODEL_LABELS[m])
        ax.set_xticks(range(len(svals)))
        ax.set_xticklabels(svals)
        ax.set_xlabel("logit bound $s$")
        ax.set_title(S.COND_LABELS[loss])
        S.panel(ax, letter)
    if not any_:
        plt.close(fig)
        return []
    axes[0].set_ylabel("macro-F1 minus C-bounded (points)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=min(6, len(l)), fontsize=6.2)
    return S.save(fig, "rev_fig_scale_gap", out)


def rev_fig_lr_sweep(df: pd.DataFrame, out):
    import matplotlib.pyplot as plt
    S.apply()
    by_lr, _ = lr_sweep_table(df)
    if not len(by_lr):
        return []
    g = by_lr.groupby(["model", "s_label", "loss", "lr"])["macro_f1"].mean().reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(S.FULL, 1.9), sharey=True)
    for ax, (s, loss), letter in zip(axes, [("1", "CE"), ("1", "LA"), ("4", "CE"), ("4", "LA")], "abcd"):
        for m in ("Q", "Q-noent", "C-bounded"):
            h = g[(g["model"] == m) & (g["s_label"] == s) & (g["loss"] == loss)].sort_values("lr")
            if len(h):
                ax.plot(h["lr"], 100 * h["macro_f1"], color=S.MODEL_COLORS[m], marker=S.MODEL_MARKERS[m], ms=3,
                        label=S.MODEL_LABELS[m])
        ax.set_xscale("log")
        ax.set_title(f"$s$={s}, {loss}")
        ax.set_xlabel("learning rate")
        S.panel(ax, letter)
    axes[0].set_ylabel("macro-F1 (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=6.4)
    return S.save(fig, "rev_fig_lr_sweep", out)


# ----------------------------------------------------------------------------- run everything
def run_all(df: pd.DataFrame, results, tables: Path, figures: Path, verbose: bool = True) -> dict:
    """Compute every revision analysis, write CSVs to `tables` and figures to `figures`."""
    tables, figures = Path(tables), Path(figures)
    tables.mkdir(parents=True, exist_ok=True)
    out = {}

    def job(name, fn):
        try:
            res = fn()
            out[name] = res
            for i, frame in enumerate(res if isinstance(res, tuple) else (res,)):
                if isinstance(frame, pd.DataFrame) and len(frame):
                    frame.to_csv(tables / (f"rev_{name}" + (f"_{i}" if isinstance(res, tuple) else "") + ".csv"),
                                 index=False)
            if verbose:
                print(f"ok   {name}")
        except Exception as e:                       # one failing analysis must not stop the others
            out[name] = None
            print(f"FAIL {name}: {e!r}")

    job("reachability_pretest", lambda: reachability_pretest(df, results))
    job("h1_decomposition", lambda: _concat([h1_decomposition(df, results, "C-host"),
                                             h1_decomposition(df, results, "C-host-nobias")]))
    job("h1_nobias", lambda: _concat([pooled_grid(df, "C-bounded", "C-host", match_s=False),
                                      pooled_grid(df, "C-bounded", "C-host-nobias", match_s=False)]))
    job("q_vs_qnoent", lambda: pooled_grid(df, "Q", "Q-noent"))
    job("controls", lambda: controls_table(df))
    job("lr_sweep", lambda: lr_sweep_table(df))
    job("valcap", lambda: valcap_table(df))
    job("leakage", lambda: leakage_audit(df, results))
    job("natural_prior", lambda: natural_prior_table(df, results))
    job("hyperparameters", lambda: hyperparameter_table(results))
    for name, fn in (("rev_fig_scale_gap", rev_fig_scale_gap), ("rev_fig_lr_sweep", rev_fig_lr_sweep)):
        try:
            paths = fn(df, figures)
            print(f"{'ok  ' if paths else 'skip'} {name}")
        except Exception as e:
            print(f"FAIL {name}: {e!r}")
    return out
