"""Publication figures. Every function takes the records frame (stats.records_frame) and an output
folder, draws at final LNCS size, saves PDF + PNG, and returns the saved paths. Functions skip
gracefully (returning []) when the results they need are not there yet, so the analysis notebook
can be run at any point of the grid.

  fig1_reachability      phase diagram of unreachable classes + theorem check on every run
  fig2_main              macro-F1 vs imbalance ratio, all heads, three training conditions
  fig3_collapse          post-hoc vs LA-loss against the normalised range 2 r_eff / (tau log rho)
  fig4_forest            paired differences with TOST band (Q vs C-bounded, Q vs Q-noent, C-bounded vs C-host)
  fig5_tail_tradeoff     tail recall against tail false-positive rate
  fig6_saturation        tail-recall gain of bounded heads vs saturation rate (H1 secondary)
  fig7_perclass          per-class recall heat map at rho = 100
  fig8_calibration       ECE vs logit bound, and reliability diagrams
  fig9_range_shots       effective range vs nominal bound, and shot-noise decision flips
  fig10_optimisation     training budget check: best epoch and validation curves per head
  hardware figures       see fig_hw_* (inputs: hardware.analyse() output)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from . import store
from . import style as S
from .metrics import reliability, softmax
from .stats import FIXED_S, KEYS, paired, pooled, seed_ci
from .theory import unreachable_count

S.apply()


def _sel(df, **f):
    m = np.ones(len(df), bool)
    for k, v in f.items():
        m &= df[k].isin(v).values if isinstance(v, (list, tuple, set)) else (df[k] == v).values
    return df[m]


def _agg(df, by, metric):
    rows = []
    for key, g in df.groupby(by, dropna=False):
        m, lo, hi = seed_ci(g[metric].values)
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), mean=m, lo=lo, hi=hi, n=len(g)))
    return pd.DataFrame(rows)


def _clean(df):
    return df[(df["tags_str"] == "") & (df["rho_str"] != "nat")]


def _legend_models(models, extra=()):
    h = [Line2D([], [], color=S.MODEL_COLORS[m], marker=S.MODEL_MARKERS[m], lw=1.2, ms=3.6, label=S.MODEL_LABELS[m])
         for m in models]
    return h + list(extra)


# ============================================================================ Fig 1
def unreachable_grid(k, rho, s, tau=1.0):
    t = 2 * s * (k - 1) / (tau * np.log(rho))
    return np.clip(k - np.ceil(t - 1e-12), 0, k - 1)


def fig1_reachability(df, out, k=None, tau=1.0):
    if k is None:
        kk = df[df["dataset"].isin(["toniot8", "ciciot8"])]["k"] if len(df) and "k" in df else []
        k = int(np.max(kk)) if len(kk) else 8
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 2.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    rho = np.logspace(np.log10(3), np.log10(3000), 400)
    s = np.linspace(0.2, 6.5, 400)
    R, Sg = np.meshgrid(rho, s)
    Z = unreachable_grid(k, R, Sg, tau)
    base = plt.get_cmap("Greys")
    cmap = ListedColormap([base(x) for x in np.linspace(0.0, 0.85, k)])
    norm = BoundaryNorm(np.arange(-0.5, k, 1), k)
    pm = a.pcolormesh(R, Sg, Z, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    a.plot(rho, tau * np.log(rho) / 2, color="#000000", lw=1.1)
    a.plot(rho, 0.5 * tau * np.log(rho) / 2, color="#000000", lw=0.9, ls=(0, (4, 1.6)))
    a.axhline(1.0, color=S.ACCENT_POSTHOC, lw=0.9, ls=":")
    a.annotate("raw $\\langle Z\\rangle$ head ($s=1$)", xy=(5.0, 1.0), xytext=(3.6, 2.35), color=S.ACCENT_POSTHOC,
               fontsize=6.5, arrowprops=dict(arrowstyle="-", color=S.ACCENT_POSTHOC, lw=0.6))
    a.text(1100, 3.85, "$s=\\frac{\\tau}{2}\\log\\rho$\n($\\tau$=1)", fontsize=6.5, ha="center", va="bottom")
    a.text(1500, 1.45, "$\\tau$=0.5", fontsize=6.5, ha="center", va="bottom")
    for r0 in (10, 100, 1000):
        a.axvline(r0, color="white", lw=0.5, alpha=0.6)
    a.set_xscale("log")
    a.set_xticks([10, 100, 1000])
    a.set_xticklabels(["10", "100", "1000"])
    a.set_xlim(rho[0], rho[-1])
    a.set_ylim(s[0], s[-1])
    a.set_xlabel("imbalance ratio $\\rho$")
    a.set_ylabel("logit half-range $s$")
    a.grid(False)
    cb = fig.colorbar(pm, ax=a, ticks=range(0, k, 2), fraction=0.05, pad=0.02)
    cb.set_label(f"classes never predicted (K = {k})", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    S.panel(a, "a")

    ok = False
    if len(df):
        raw = df[df["variant"] == "raw"][["run_id", "r_eff"]].rename(columns={"r_eff": "r_raw"})
        bounded = ["Q", "Q-noent", "C-bounded", "C-bounded-wide", "C-cosine"]     # Prop. 2 is about bounded, bias-free heads
        ph = df[df["variant"].str.startswith("posthoc") & df["model"].isin(bounded)
                & ~df["tags_str"].str.contains("bias")].merge(raw, on="run_id")
        if len(ph):
            ok = True
            pred = [unreachable_count(np.asarray(c) / np.sum(c), r, t) for c, r, t in
                    zip(ph["train_counts"], ph["r_raw"], ph["tau"].astype(float))]
            ph = ph.assign(pred=pred)
            cnt = ph.groupby(["pred", "never_predicted"]).size().reset_index(name="n")
            mx = int(max(cnt["pred"].max(), cnt["never_predicted"].max())) + 1
            b.plot([-0.5, mx], [-0.5, mx], color=S.GREY, lw=0.8, ls=(0, (4, 1.6)), zorder=1)
            b.fill_between([-0.5, mx], [-0.5, mx], [mx, mx], color=S.ACCENT_LA, alpha=0.06, lw=0, zorder=0)
            b.scatter(cnt["pred"], cnt["never_predicted"], s=6 + 60 * np.sqrt(cnt["n"] / cnt["n"].max()),
                      color=S.ACCENT_POSTHOC, alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
            frac = float((ph["never_predicted"] >= ph["pred"]).mean())
            eq = float((ph["never_predicted"] == ph["pred"]).mean())
            b.text(0.97, 0.04, f"observed $\\geq$ predicted: {100 * frac:.0f}%\nobserved = predicted: {100 * eq:.0f}%\n"
                   f"{len(ph):,} post-hoc evaluations", transform=b.transAxes, va="bottom", ha="right", fontsize=6.3,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
            b.set_xlim(-0.5, mx)
            b.set_ylim(-0.5, mx)
            b.set_xticks(range(0, mx + 1, max(1, mx // 5)))
            b.set_yticks(range(0, mx + 1, max(1, mx // 5)))
    if not ok:
        b.text(0.5, 0.5, "no post-hoc runs yet", transform=b.transAxes, ha="center", color=S.GREY)
    b.set_xlabel("unreachable classes predicted by Prop. 2")
    b.set_ylabel("classes never predicted (test)")
    S.panel(b, "b")
    return S.save(fig, "fig1_reachability", out)


# ============================================================================ Fig 2
MAIN_MODELS = ["C-host", "C-bounded", "Q", "Q-noent", "C-cosine"]
CONDITIONS = [("CE", "raw", "CE"), ("post-hoc", "posthoc1", "CE"), ("LA", "raw", "LA")]


def fig2_main(df, out, s_main="2", datasets=("toniot8", "ciciot8")):
    d = _clean(df)
    datasets = [x for x in datasets if x in set(d["dataset"])]
    if not datasets:
        return []
    fig, axes = plt.subplots(len(datasets), 3, figsize=(S.FULL, 1.55 * len(datasets) + 0.45), sharex=True,
                             sharey="row", squeeze=False)
    dodge = dict(zip(MAIN_MODELS + ["MLP"], np.linspace(-0.06, 0.06, len(MAIN_MODELS) + 1)))
    for i, ds in enumerate(datasets):
        for j, (cond, variant, loss) in enumerate(CONDITIONS):
            ax = axes[i, j]
            for m in MAIN_MODELS + ["MLP"]:
                g = _sel(d, dataset=ds, model=m, variant=variant, loss=loss)
                if m not in ("C-host", "MLP"):
                    g = g[g["s_label"] == s_main]
                if not len(g):
                    continue
                a = _agg(g, ["rho_num"], "macro_f1").sort_values("rho_num")
                x = a["rho_num"] * 10 ** dodge[m]
                style = dict(color=S.MODEL_COLORS[m], marker=S.MODEL_MARKERS[m], lw=1.1 if m != "MLP" else 0.8,
                             ms=3.2, mfc="white" if m == "Q-noent" else S.MODEL_COLORS[m], mew=0.8,
                             alpha=1.0 if m != "MLP" else 0.8, zorder=3 if m == "Q" else 2)
                lo, hi = a["lo"].clip(0, 1), a["hi"].clip(0, 1)
                ax.errorbar(x, a["mean"], yerr=[a["mean"] - lo, hi - a["mean"]], elinewidth=0.7, **style)
            lg = _sel(d, dataset=ds, model="LightGBM", variant="raw")
            if len(lg):
                a = _agg(lg, ["rho_num"], "macro_f1").sort_values("rho_num")
                ax.plot(a["rho_num"], a["mean"], color=S.MODEL_COLORS["LightGBM"], lw=0.8, ls=(0, (1, 1.2)),
                        marker=S.MODEL_MARKERS["LightGBM"], ms=3, zorder=1)
            S.rho_axis(ax)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(S.COND_LABELS[cond])
            if i < len(datasets) - 1:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(f"{S.DATASET_LABELS[ds]}\nmacro-F1 (balanced test)")
            S.panel(ax, "abcdef"[i * 3 + j])
    handles = _legend_models(MAIN_MODELS + ["MLP", "LightGBM"])
    for h in handles:
        if h.get_label() == S.MODEL_LABELS["Q-noent"]:
            h.set_markerfacecolor("white")
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=6.5, handlelength=1.6)
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02)
    return S.save(fig, "fig2_main", out)


# ============================================================================ Fig 3
def collapse_frame(df):
    d = _clean(df)
    raw = d[d["variant"] == "raw"][["run_id", "r_eff"]].rename(columns={"r_eff": "r_raw"})
    rows = []
    for variant, loss, cond in (("posthoc1", "CE", "post-hoc"), ("raw", "LA", "LA")):
        b = _sel(d, variant=variant, loss=loss, model=["Q", "C-bounded", "Q-noent", "C-cosine"])
        b = b[b["s_label"].isin(FIXED_S + ["learned"])].merge(raw, on="run_id")
        h = _sel(d, variant=variant, loss=loss, model="C-host")[KEYS + ["macro_f1"]]
        m = b.merge(h, on=KEYS, suffixes=("", "_host"))
        m["gap"] = m["macro_f1"] - m["macro_f1_host"]
        m["x"] = 2 * m["r_raw"] / np.log(m["actual_rho"])
        m["cond"] = cond
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fig3_collapse(df, out, datasets=("toniot8", "ciciot8")):
    m = collapse_frame(df)
    if not len(m):
        return []
    datasets = [x for x in datasets if x in set(m["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(S.FULL, 2.05), sharey=True, squeeze=False)
    bins = np.logspace(np.log10(0.08), np.log10(12), 15)
    for ax, ds, letter in zip(axes[0], datasets, "ab"):
        ax.axhspan(-100 * 0.01, 100 * 0.01, color=S.LIGHT, alpha=0.5, lw=0)
        ax.axvline(1.0, color="#000000", lw=0.8)
        ax.axhline(0, color=S.GREY, lw=0.5)
        for cond, col in (("post-hoc", S.ACCENT_POSTHOC), ("LA", S.ACCENT_LA)):
            g = m[(m["dataset"] == ds) & (m["cond"] == cond)]
            if not len(g):
                continue
            for mod, mk in (("C-bounded", "s"), ("Q", "o"), ("Q-noent", "o"), ("C-cosine", "D")):
                gg = g[g["model"] == mod]
                ax.scatter(gg["x"], 100 * gg["gap"], s=5, marker=mk, color=col, alpha=0.22, lw=0,
                           facecolors=col if mod != "Q-noent" else "none")
            idx = np.digitize(g["x"], bins)
            med = g.groupby(idx)["gap"].agg(["median", lambda v: v.quantile(0.25), lambda v: v.quantile(0.75), "size"])
            med.columns = ["med", "q1", "q3", "n"]
            med = med[(med["n"] >= 3) & (med.index > 0) & (med.index < len(bins))]
            xc = np.sqrt(bins[med.index - 1] * bins[med.index])
            ax.fill_between(xc, 100 * med["q1"], 100 * med["q3"], color=col, alpha=0.18, lw=0)
            ax.plot(xc, 100 * med["med"], color=col, lw=1.4, ls=S.COND_STYLES[cond], label=S.COND_LABELS[cond])
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
        ax.set_xlim(0.08, 12)
        ax.set_xlabel("normalised range  $2r_{\\mathrm{eff}} / (\\tau\\log\\rho)$")
        ax.set_title(S.DATASET_LABELS[ds])
        ax.text(1.06, 0.98, "every class reachable →", transform=ax.get_xaxis_transform(), fontsize=6.2,
                va="top", color="#333333")
        S.panel(ax, letter)
    axes[0, 0].set_ylabel("macro-F1 minus C-host\n(points, same seed)")
    h, _ = axes[0, 0].get_legend_handles_labels()
    h += [Line2D([], [], color="#555555", marker=mk, ls="none", ms=3.5, mfc=fc, label=lab)
          for mk, fc, lab in (("s", "#555555", "C-bounded"), ("o", "#555555", "Q"), ("o", "white", "Q-noent"),
                              ("D", "#555555", "C-cosine"))]
    fig.legend(handles=h, loc="outside lower center", fontsize=6.3, ncol=6)
    return S.save(fig, "fig3_collapse", out)


# ============================================================================ Fig 4
COMPARISONS = [
    ("Q − C-bounded", {"model": "Q", "variant": "raw"}, {"model": "C-bounded", "variant": "raw"}, True),
    ("Q − Q-noent", {"model": "Q", "variant": "raw"}, {"model": "Q-noent", "variant": "raw"}, True),
    ("C-bounded − C-host", {"model": "C-bounded", "variant": "raw"}, {"model": "C-host", "variant": "raw"}, False),
]


def fig4_forest(df, out, delta=0.01, row_s="2"):
    d = _clean(df)
    d = d[d["loss"].isin(["CE", "LA"]) & d["dataset"].isin(["toniot8", "ciciot8"])]
    if not len(d):
        return []
    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 2.9), sharey=True)
    cells = [(ds, r, l) for ds in ("toniot8", "ciciot8") for r in ("10", "100", "1000") for l in ("CE", "LA")]
    labels = [f"{'TON' if ds.startswith('ton') else 'CIC'}  ρ={r:>4}  {l}" for ds, r, l in cells]
    any_ = False
    for ax, (name, A, B, match), letter in zip(axes, COMPARISONS, "abc"):
        ax.axvspan(-100 * delta, 100 * delta, color=S.LIGHT, alpha=0.6, lw=0)
        ax.axvline(0, color=S.GREY, lw=0.5)
        dif = paired(d, dict(A, s_label=FIXED_S), B, match_s=match)
        if not len(dif):
            ax.set_title(name)
            continue
        any_ = True
        for yi, (ds, r, l) in enumerate(cells):
            g = dif[(dif["dataset"] == ds) & (dif["rho_str"] == r) & (dif["loss"] == l) & (dif["s_label"] == row_s)]
            if len(g) < 2:
                continue
            mu, lo, hi = seed_ci(100 * g["diff"].values, level=0.90)
            col = S.ACCENT_LA if l == "LA" else "#333333"
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.0)
            ax.plot(mu, yi, marker="o", color=col, ms=3)
        pr = pooled(dif, delta)
        y0 = len(cells) + 0.8
        if np.isfinite(pr.get("est", np.nan)):
            e, (lo, hi) = 100 * pr["est"], [100 * v for v in pr["ci90"]]
            ax.fill([lo, e, hi, e], [y0, y0 - 0.38, y0, y0 + 0.38], color=S.ACCENT_POSTHOC, lw=0)
            verdict = "equivalent" if pr["p_tost"] < 0.05 else ("different" if pr["p_two_sided"] < 0.05 else "inconclusive")
            pt = "< 0.001" if pr["p_tost"] < 0.001 else f"= {pr['p_tost']:.3f}"
            ax.set_xlabel(f"pooled {e:+.2f} pts, TOST p {pt}\n({verdict})", fontsize=6.5)
        ax.set_title(name)
        S.panel(ax, letter)
    if not any_:
        plt.close(fig)
        return []
    axes[0].set_yticks(list(range(len(cells))) + [len(cells) + 0.8])
    axes[0].set_yticklabels(labels + ["pooled (all s)"], fontsize=6.3, family="monospace")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", visible=False)
        lim = max(abs(np.array(ax.get_xlim()))) * 1.05
        ax.set_xlim(-lim, lim)
    fig.supxlabel(f"Difference in macro-F1 (points). Rows: s = {row_s}, mean and 90% CI over seeds. Diamond: pooled over all s"
                  f" (90% CI). Band: ±{100 * delta:.0f}-point margin.", fontsize=6.3)
    return S.save(fig, "fig4_forest", out)


# ============================================================================ Fig 5
def fig5_tail_tradeoff(df, out, rho="100", s_main="2", datasets=("toniot8", "ciciot8")):
    d = _clean(df)
    d = d[d["rho_str"] == rho]
    datasets = [x for x in datasets if x in set(d["dataset"])]
    if not datasets:
        return []
    fig, axes = plt.subplots(1, len(datasets), figsize=(S.FULL, 2.2), squeeze=False)
    markers = {"CE": "o", "post-hoc": "s", "LA": "^"}
    for ax, ds, letter in zip(axes[0], datasets, "ab"):
        for m in ["C-host", "C-bounded", "Q", "Q-noent", "C-cosine", "MLP"]:
            for cond, variant, loss in CONDITIONS:
                g = _sel(d, dataset=ds, model=m, variant=variant, loss=loss)
                if m not in ("C-host", "MLP"):
                    g = g[g["s_label"] == s_main]
                if not len(g):
                    continue
                x, y = g["tail_fpr"].values, g["tail_recall"].values
                ax.errorbar(100 * x.mean(), 100 * y.mean(), xerr=100 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0,
                            yerr=100 * y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0, fmt=markers[cond],
                            color=S.MODEL_COLORS[m], ms=4, mew=0.7, elinewidth=0.6,
                            mfc=S.MODEL_COLORS[m] if cond != "post-hoc" else "white")
        ax.set_xlabel("tail false-positive rate (%)")
        ax.set_title(f"{S.DATASET_LABELS[ds]},  ρ = {rho}")
        S.panel(ax, letter)
    axes[0, 0].set_ylabel("tail recall (%)")
    h = _legend_models(["C-host", "C-bounded", "Q", "Q-noent", "C-cosine", "MLP"])
    for hh in h:
        hh.set_linestyle("none")
    h += [Line2D([], [], color="#333333", marker=markers[c], ls="none", ms=4, mfc="white" if c == "post-hoc" else "#333333",
                 label=S.COND_LABELS[c]) for c, _, _ in CONDITIONS]
    fig.legend(handles=h, loc="outside right center", fontsize=6.3)
    return S.save(fig, "fig5_tail_tradeoff", out)


# ============================================================================ Fig 6
def fig6_saturation(df, out):
    d = _clean(df)
    dif = paired(d, {"model": ["C-bounded", "Q"], "variant": "raw", "loss": "CE", "s_label": FIXED_S},
                 {"model": "C-host", "variant": "raw", "loss": "CE"}, metric="tail_recall", match_s=False)
    sat = d[d["variant"] == "raw"][KEYS + ["s_label", "model", "saturation"]].rename(columns={"model": "model_a"})
    dif = dif.merge(sat, on=KEYS + ["s_label", "model_a"], how="left").dropna(subset=["saturation"])
    if len(dif) < 5:
        return []
    fig, ax = plt.subplots(figsize=(S.FULL * 0.72, 2.3))
    svals = sorted(dif["s_label"].unique(), key=float)
    cmap = plt.get_cmap("Blues")
    cols = {s: cmap(0.4 + 0.55 * i / max(1, len(svals) - 1)) for i, s in enumerate(svals)}
    for (mod, s), g in dif.groupby(["model_a", "s_label"]):
        ax.scatter(100 * g["saturation"], 100 * g["diff"], s=7, marker="o" if mod == "Q" else "s", color=cols[s],
                   alpha=0.55, lw=0)
    x, y = 100 * dif["saturation"].values, 100 * dif["diff"].values
    if np.ptp(x) > 0:
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, coef[0] + coef[1] * xs, color="#000000", lw=1.0)
        ax.text(0.97, 0.95, f"slope {coef[1]:+.2f} pts per % saturated", transform=ax.transAxes, ha="right",
                va="top", fontsize=6.3)
    ax.axhline(0, color=S.GREY, lw=0.5)
    ax.set_xlim(left=-0.02 * max(1.0, x.max()), right=max(1.0, 1.05 * x.max()))
    ax.set_xlabel("saturation rate (% of test inputs)")
    ax.set_ylabel("tail recall minus C-host (points)")
    h = [Line2D([], [], color=cols[s], marker="o", ls="none", ms=4, label=f"s = {s}") for s in svals]
    h += [Line2D([], [], color="#333333", marker="o", ls="none", ms=4, label="Q"),
          Line2D([], [], color="#333333", marker="s", ls="none", ms=4, label="C-bounded")]
    fig.legend(handles=h, fontsize=6, loc="outside lower center", ncol=len(h))
    return S.save(fig, "fig6_saturation", out)


# ============================================================================ Fig 7
def fig7_perclass(df, out, dataset="toniot8", rho="100"):
    d = _clean(df)
    d = d[(d["dataset"] == dataset) & (d["rho_str"] == rho)]
    cols = [("C-host", "none", "CE", "raw", "C-host\nCE"), ("C-host", "none", "CE", "posthoc1", "C-host\npost-hoc"),
            ("C-host", "none", "LA", "raw", "C-host\nLA"), ("C-bounded", "1", "CE", "posthoc1", "C-bnd s=1\npost-hoc"),
            ("C-bounded", "1", "LA", "raw", "C-bnd s=1\nLA"), ("Q", "1", "CE", "posthoc1", "Q s=1\npost-hoc"),
            ("Q", "2", "CE", "raw", "Q s=2\nCE"), ("Q", "2", "LA", "raw", "Q s=2\nLA"),
            ("Q", "4", "LA", "raw", "Q s=4\nLA")]
    mats, names, labels = [], None, []
    for model, s, loss, variant, lab in cols:
        g = _sel(d, model=model, s_label=s, loss=loss, variant=variant)
        if not len(g):
            continue
        mats.append(np.mean(np.vstack(g["recall"].values), axis=0))
        labels.append(lab)
        if names is None and "class_names" in g:
            names = g["class_names"].iloc[0]
            counts = np.mean(np.vstack(g["train_counts"].values), axis=0)
    if len(mats) < 2:
        return []
    M = np.vstack(mats).T
    k = M.shape[0]
    fig, ax = plt.subplots(figsize=(S.FULL, 0.22 * k + 0.9))
    im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    for i in range(k):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0") if M[i, j] < 1 else "1", ha="center", va="center",
                    fontsize=5.8, color="white" if M[i, j] < 0.55 else "black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ylab = [f"{n}  ({int(c):,})" for n, c in zip(names, counts)] if names else [str(i) for i in range(k)]
    ax.set_yticks(range(k))
    ax.set_yticklabels(ylab, fontsize=6.3)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("recall", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.outline.set_linewidth(0.4)
    ax.set_ylabel("class (training rows), most frequent first", fontsize=7)
    return S.save(fig, f"fig7_perclass_{dataset}", out)


# ============================================================================ Fig 8
def fig8_calibration(df, out, results=None, dataset="toniot8", rho="100"):
    d = _clean(df)
    d = d[(d["dataset"] == dataset) & (d["rho_str"] == rho) & (d["variant"] == "raw")]
    if not len(d):
        return []
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 2.1), gridspec_kw=dict(width_ratios=[1.2, 1]))
    order = FIXED_S
    for m in ("C-bounded", "Q"):
        for loss in ("CE", "LA"):
            g = d[(d["model"] == m) & (d["loss"] == loss)]
            a_ = _agg(g[g["s_label"].isin(order)], ["s_label"], "ece")
            if not len(a_):
                continue
            a_["sv"] = a_["s_label"].astype(float)
            a_ = a_.sort_values("sv")
            a.errorbar(a_["sv"], a_["mean"], yerr=[a_["mean"] - a_["lo"].clip(lower=0), a_["hi"] - a_["mean"]],
                       color=S.MODEL_COLORS[m], marker=S.MODEL_MARKERS[m], ls=S.COND_STYLES[loss], ms=3, elinewidth=0.6,
                       label=f"{m}, {loss}")
            gl = g[g["s_label"] == "learned"]
            if len(gl):
                a.errorbar(gl["s_final"].mean(), gl["ece"].mean(), xerr=gl["s_final"].std(), fmt="*",
                           color=S.MODEL_COLORS[m], ms=6, mew=0, elinewidth=0.6)
    for loss in ("CE", "LA"):
        h = d[(d["model"] == "C-host") & (d["loss"] == loss)]
        if len(h):
            a.axhline(h["ece"].mean(), color=S.MODEL_COLORS["C-host"], ls=S.COND_STYLES[loss], lw=0.8,
                      label=f"C-host, {loss}")
    a.set_xlabel("logit bound $s$   (★ learned $s$, at its final value)")
    a.set_ylabel("expected calibration error")
    a.set_ylim(bottom=0)
    ha, la = a.get_legend_handles_labels()
    S.panel(a, "a")
    b.plot([0, 1], [0, 1], color=S.GREY, lw=0.7, ls=(0, (4, 1.6)))
    drawn = False
    if results is not None:
        for m, s in (("C-host", "none"), ("C-bounded", "2"), ("Q", "2"), ("Q", "1")):
            g = d[(d["model"] == m) & (d["s_label"] == s) & (d["loss"] == "CE")]
            Z, Y = [], []
            for rid in g["run_id"]:
                z = store.load_logits(results, rid)
                if z is not None:
                    Z.append(z["z_test"])
                    Y.append(z["y_test"])
            if not Z:
                continue
            rel = reliability(softmax(np.vstack(Z).astype(np.float64)), np.concatenate(Y))
            b.plot(rel[:, 1], rel[:, 2], color=S.MODEL_COLORS[m], marker=S.MODEL_MARKERS[m], ms=2.8,
                   ls="-" if s != "1" else ":", label=f"{m}" + ("" if s == "none" else f" s={s}"))
            drawn = True
    b.set_xlim(0, 1)
    b.set_ylim(0, 1)
    b.set_xlabel("confidence")
    b.set_ylabel("accuracy")
    b.set_title(f"reliability, CE, ρ = {rho}")
    if drawn:
        b.legend(fontsize=6, loc="lower right")
    S.panel(b, "b")
    if ha:
        fig.legend(ha, la, loc="outside lower center", fontsize=6.2, ncol=6)
    return S.save(fig, "fig8_calibration", out)


# ============================================================================ Fig 9
def fig9_range_shots(df, out, shot_df: pd.DataFrame | None = None):
    """(a) share of the nominal bound a head actually uses, r_eff / s; (b) shot-noise decision flips."""
    d = _clean(df)
    d = d[(d["variant"] == "raw") & d["model"].isin(["Q", "Q-noent", "C-bounded", "C-cosine"])].copy()
    if not len(d):
        return []
    d["used"] = d["r_eff95"] / d["s_final"]          # typical inputs; the max over inputs saturates
    two = shot_df is not None and len(shot_df)
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(S.FULL if two else S.HALF + 0.5, 2.15), squeeze=False)
    a = axes[0, 0]
    jit = {"Q": -0.045, "Q-noent": -0.015, "C-bounded": 0.015, "C-cosine": 0.045}
    a.axhline(1.0, color=S.GREY, lw=0.7, ls=(0, (4, 1.6)))
    for m in jit:
        g = d[d["model"] == m]
        fixed = g[g["s_label"].isin(FIXED_S)]
        a_ = _agg(fixed, ["s_label"], "used")
        if len(a_):
            a_["sv"] = a_["s_label"].astype(float)
            a_ = a_.sort_values("sv")
            a.errorbar(a_["sv"] * np.exp(jit[m]), a_["mean"], yerr=[a_["mean"] - a_["lo"].clip(lower=0),
                                                                    a_["hi"].clip(upper=1) - a_["mean"]],
                       color=S.MODEL_COLORS[m], marker=S.MODEL_MARKERS[m], ms=3, lw=1.0, elinewidth=0.6,
                       mfc="white" if m == "Q-noent" else S.MODEL_COLORS[m], label=S.MODEL_LABELS[m])
        gl = g[g["s_label"] == "learned"]
        if len(gl):
            a.scatter(gl["s_final"], gl["used"], marker="*", s=14, color=S.MODEL_COLORS[m], lw=0, alpha=0.6)
    a.set_xscale("log")
    a.set_xticks([1, 1.5, 2, 3, 4, 6])
    a.set_xticklabels(["1", "1.5", "2", "3", "4", "6"])
    a.minorticks_off()
    a.set_ylim(0, 1.08)
    a.set_xlabel("logit bound $s$  (★ learned, at its final value)")
    a.set_ylabel("logit half-spread / $s$\n(95th percentile of test inputs)")
    a.legend(fontsize=6, loc="lower left")
    S.panel(a, "a")
    if two:
        b = axes[0, 1]
        svals = sorted(shot_df["s"].unique())
        for i, (s_, g) in enumerate(shot_df.groupby("s")):
            g = g.sort_values("shots")
            col = plt.get_cmap("Blues")(0.45 + 0.5 * i / max(1, len(svals) - 1))
            b.plot(g["shots"], 100 * g["flip_rate"], marker="o", ms=2.8, color=col, label=f"Q, s = {s_:g}")
        for x0, lab in ((512, "512"), (1024, "1024")):
            b.axvline(x0, color=S.GREY, lw=0.6, ls=":")
        b.set_xscale("log", base=2)
        b.set_xticks([64, 256, 1024, 4096])
        b.set_xticklabels(["64", "256", "1024", "4096"])
        b.set_ylim(bottom=0)
        b.set_xlabel("shots per input")
        b.set_ylabel("decisions changed by shot noise (%)")
        b.legend(fontsize=6)
        S.panel(b, "b")
    return S.save(fig, "fig9_range_shots", out)


def shot_curve(results, df, dataset="toniot8", rho="100", loss="LA", s_values=("1", "2", "4"), seeds=range(5),
               shots=(64, 128, 256, 512, 1024, 2048, 4096), device="cpu"):
    """Decision flip rate of Q heads when exact <Z> is replaced by a finite-shot estimate."""
    import torch
    from . import qsim
    from .data import load_prepared
    from .runner import load_model
    p = load_prepared(dataset)
    Xt = torch.as_tensor(p.part("test")[0], device=device)
    rows = []
    for s_ in s_values:
        for seed in seeds:
            rid = f"{dataset}_r{rho}_Q_s{s_}_{loss}_seed{seed}"
            if not store.has(results, "ckpt", rid):
                continue
            model, spec, _ = load_model(results, rid, device)
            with torch.no_grad():
                h = model.encoder(Xt)
                probs = qsim.probabilities(model.head.angles(h), model.head.phi, model.head.entangle, model.head.reupload)
                exact = (probs @ qsim.z_matrix(model.head.n, model.head.k, device)).argmax(1)
                for n_sh in shots:
                    g = torch.Generator(device=device).manual_seed(seed * 1000 + n_sh)
                    ez = qsim.sampled_expectations(probs, model.head.k, n_sh, g)
                    rows.append(dict(s=float(s_), seed=seed, shots=n_sh,
                                     flip_rate=float((ez.argmax(1) != exact).float().mean())))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["s", "shots"], as_index=False)["flip_rate"].mean()


# ============================================================================ Fig 10
def fig10_optimisation(df, out, results=None, dataset="toniot8", rho="100", s_main="2"):
    d = _clean(df)
    d = d[(d["variant"] == "raw") & d["model"].isin(["Q", "Q-noent", "C-bounded", "C-host", "C-cosine"])]
    if not len(d):
        return []
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 2.0))
    models = ["C-host", "C-bounded", "C-cosine", "Q-noent", "Q"]
    data = [d[d["model"] == m]["best_epoch"].dropna().values for m in models]
    pos = np.arange(len(models))
    for i, (m, v) in enumerate(zip(models, data)):
        if not len(v):
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        a.plot([i, i], [np.percentile(v, 5), np.percentile(v, 95)], color=S.MODEL_COLORS[m], lw=0.8)
        a.add_patch(plt.Rectangle((i - 0.22, q1), 0.44, q3 - q1, facecolor=S.MODEL_COLORS[m], alpha=0.25,
                                  edgecolor=S.MODEL_COLORS[m], lw=0.8))
        a.plot([i - 0.22, i + 0.22], [med, med], color=S.MODEL_COLORS[m], lw=1.4)
    from .config import MAX_EPOCHS
    a.axhline(MAX_EPOCHS, color=S.GREY, lw=0.6, ls=":")
    a.text(len(models) - 0.5, MAX_EPOCHS, "epoch cap", fontsize=6, ha="right", va="bottom", color=S.GREY)
    a.set_xticks(pos)
    a.set_xticklabels([S.MODEL_LABELS[m] for m in models], fontsize=6.5)
    a.set_ylabel("best epoch (5–95%, IQR, median)")
    a.grid(axis="x", visible=False)
    S.panel(a, "a")
    drawn = False
    if results is not None:
        for m in models:
            g = d[(d["model"] == m) & (d["dataset"] == dataset) & (d["rho_str"] == rho) & (d["loss"] == "CE")]
            if m != "C-host":
                g = g[g["s_label"] == s_main]
            curves = []
            for rid in g["run_id"]:
                z = store.load_logits(results, rid)
                if z is not None:
                    c = z["val_macro_f1"]
                    if len(c):
                        curves.append(c)
            if not curves:
                continue
            L = max(len(c) for c in curves)
            M = np.full((len(curves), L), np.nan)
            for i, c in enumerate(curves):
                M[i, :len(c)] = c
                M[i, len(c):] = np.nan
            mu = np.nanmean(M, 0)
            b.plot(np.arange(1, L + 1), mu, color=S.MODEL_COLORS[m], lw=1.1, label=S.MODEL_LABELS[m])
            drawn = True
    b.set_xlabel("epoch")
    b.set_ylabel("validation macro-F1")
    b.set_title(f"CE, s = {s_main}, ρ = {rho}")
    if drawn:
        b.legend(fontsize=6, loc="best")
    S.panel(b, "b")
    return S.save(fig, "fig10_optimisation", out)


# ============================================================================ hardware
COND_ORDER = ["sim_exact", "sim_shots", "raw", "global", "affine", "readout"]
COND_NAMES = {"sim_exact": "sim.\nexact", "sim_shots": "sim.\nshots", "raw": "hardware\nraw",
              "global": "global\nrescale", "affine": "per-qubit\naffine", "readout": "readout\ninversion"}


def _series_style(name):
    if "post-hoc" in name:
        return dict(color=S.ACCENT_POSTHOC, alpha=0.9, hatch=None)
    if "layout B" in name:
        return dict(color=S.ACCENT_LA, alpha=0.45, hatch=None)
    return dict(color=S.ACCENT_LA, alpha=0.95, hatch=None)


def fig_hw_conditions(hw: dict, out):
    if not hw or "conditions" not in hw:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 2.25))
    names = list(hw["conditions"].keys())
    width = 0.8 / max(1, len(names))
    for ax, metric, lab, letter in ((axes[0], "macro_f1", "macro-F1", "a"),
                                    (axes[1], "tail_fpr", "tail false-positive rate", "b")):
        ax.axvspan(-0.5, 1.5, color=S.LIGHT, alpha=0.35, lw=0)
        for i, name in enumerate(names):
            conds = [c for c in COND_ORDER if c in hw["conditions"][name]]
            vals = np.array([hw["conditions"][name][c][metric] for c in conds])
            ci = np.array([hw["conditions"][name][c].get(metric + "_ci", (v, v)) for c, v in zip(conds, vals)])
            x = np.array([COND_ORDER.index(c) for c in conds]) + (i - (len(names) - 1) / 2) * width
            st = _series_style(name)
            ax.bar(x, vals, width=width * 0.9, color=st["color"], alpha=st["alpha"], lw=0, label=name)
            ax.errorbar(x, vals, yerr=[vals - ci[:, 0], ci[:, 1] - vals], fmt="none", ecolor="#222222", elinewidth=0.6)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels([COND_NAMES[c] for c in COND_ORDER], fontsize=6.3)
        ax.set_xlim(-0.5, len(COND_ORDER) - 0.5)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
        S.panel(ax, letter)
    axes[0].text(0.5, 0.99, "simulation", transform=axes[0].get_xaxis_transform(), ha="center", va="top",
                 fontsize=6, color="#555555")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=len(names), fontsize=6.3)
    return S.save(fig, "fig_hw_conditions", out)


def fig_hw_bias(hw: dict, out):
    """Over-prediction per class on hardware against the readout offset beta_k (pre-specified H4) and against the
    contrast alpha_k of the qubit that reads the class (exploratory), for both layouts."""
    if not hw or "bias" not in hw:
        return []
    from scipy.stats import spearmanr
    b = pd.DataFrame(hw["bias"])
    b = b[b["model"] == b["model"].iloc[0]]
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 2.05), sharey=True)
    specs = (("beta", "readout offset $\\beta_k$ of the reading qubit (%)", 100, "a"),
             ("alpha_fit", "fitted contrast $\\alpha_k$ of the reading qubit", 1, "b"))
    for ax, (col, xlabel, mult, letter) in zip(axes, specs):
        for _, g in b.groupby("class_index"):
            if len(g) == 2:
                g = g.sort_values("layout")
                ax.plot(mult * g[col], 100 * g["over_prediction"], color=S.LIGHT, lw=0.8, zorder=1)
        for lay, mk, colr in (("A", "o", S.ACCENT_LA), ("B", "s", S.ACCENT_POSTHOC)):
            g = b[b["layout"] == lay]
            ax.scatter(mult * g[col], 100 * g["over_prediction"], marker=mk, color=colr, s=15, lw=0, zorder=3,
                       label=f"layout {lay}" + (" (reversed path)" if lay == "B" else ""))
        for _, r in b[np.abs(b["over_prediction"]) >= 0.03].iterrows():
            ax.annotate(r["class_name"], (mult * r[col], 100 * r["over_prediction"]), xytext=(3, 2),
                        textcoords="offset points", fontsize=5.6, color="#333333")
        ax.axhline(0, color=S.GREY, lw=0.5)
        rs = spearmanr(b[col], b["over_prediction"])
        ax.set_title(f"Spearman $r$ = {rs.statistic:+.2f}, p = {rs.pvalue:.2f} (n = {len(b)})", fontsize=6.3)
        ax.set_xlabel(xlabel)
        S.panel(ax, letter)
    axes[0].set_ylabel("over-prediction (pp)")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=6.2, loc="outside lower center", ncol=2)
    return S.save(fig, "fig_hw_bias", out)


def fig_hw_confusion(hw: dict, out, model_key=None):
    if not hw or "confusion" not in hw:
        return []
    model_key = model_key or list(hw["confusion"].keys())[0]
    conf = hw["confusion"][model_key]
    conds = [c for c in ("sim_exact", "raw", "affine") if c in conf]
    fig, axes = plt.subplots(1, len(conds), figsize=(S.FULL, 1.85), squeeze=False)
    for ax, c, letter in zip(axes[0], conds, "abc"):
        M = np.asarray(conf[c], float)
        M = M / np.maximum(M.sum(1, keepdims=True), 1)
        im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1)
        ax.set_title(COND_NAMES[c].replace("\n", " "))
        ax.set_xlabel("predicted class")
        ax.grid(False)
        ax.set_xticks(range(M.shape[0]))
        ax.set_yticks(range(M.shape[0]))
        ax.tick_params(labelsize=5.5, length=0)
        S.panel(ax, letter)
    axes[0, 0].set_ylabel("true class")
    cb = fig.colorbar(im, ax=axes[0, -1], fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    return S.save(fig, "fig_hw_confusion", out)


# ============================================================================ data profile
def fig_datasets(prepared: list, out, total=None):
    """Natural class sizes after cleaning (bars, log scale) and the controlled training profiles (lines)."""
    from .config import TRAIN_SIZE
    from .data import train_counts
    total = total or TRAIN_SIZE
    prepared = [p for p in prepared if p is not None]
    if not prepared:
        return []
    fig, axes = plt.subplots(1, len(prepared), figsize=(S.FULL, 2.2), squeeze=False)
    shades = {10: 0.35, 100: 0.6, 1000: 0.9}
    for ax, p, letter in zip(axes[0], prepared, "abc"):
        x = np.arange(p.k)
        ax.bar(x, p.natural_counts, color=S.LIGHT, edgecolor=S.GREY, lw=0.5, label="full data, cleaned")
        for rho, sh in shades.items():
            ax.plot(x, train_counts(p, rho, total), color=plt.get_cmap("Blues")(sh), lw=1.1, marker="o",
                    ms=2.4, label=f"training subset, ρ = {rho}")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(p.class_names, rotation=60, ha="right", fontsize=6)
        ax.set_title(S.DATASET_LABELS.get(p.name, p.name))
        ax.grid(axis="x", visible=False)
        S.panel(ax, letter)
    axes[0, 0].set_ylabel("rows per class")
    axes[0, 0].legend(fontsize=5.8, loc="lower left")
    return S.save(fig, "fig_datasets", out)


# ============================================================================ all
def make_all(df, out, results=None, hw=None, shot_df=None, verbose=True):
    jobs = [("fig1_reachability", lambda: fig1_reachability(df, out)),
            ("fig2_main", lambda: fig2_main(df, out)),
            ("fig3_collapse", lambda: fig3_collapse(df, out)),
            ("fig4_forest", lambda: fig4_forest(df, out)),
            ("fig5_tail_tradeoff", lambda: fig5_tail_tradeoff(df, out)),
            ("fig6_saturation", lambda: fig6_saturation(df, out)),
            ("fig7_perclass_toniot8", lambda: fig7_perclass(df, out, "toniot8")),
            ("fig7_perclass_ciciot8", lambda: fig7_perclass(df, out, "ciciot8")),
            ("fig8_calibration", lambda: fig8_calibration(df, out, results)),
            ("fig9_range_shots", lambda: fig9_range_shots(df, out, shot_df)),
            ("fig10_optimisation", lambda: fig10_optimisation(df, out, results)),
            ("fig_hw_conditions", lambda: fig_hw_conditions(hw, out)),
            ("fig_hw_bias", lambda: fig_hw_bias(hw, out)),
            ("fig_hw_confusion", lambda: fig_hw_confusion(hw, out))]
    made = {}
    for name, fn in jobs:
        try:
            paths = fn()
            made[name] = paths
            if verbose:
                print(f"{'ok  ' if paths else 'skip'} {name}")
        except Exception as e:                      # one broken figure must not stop the rest
            made[name] = []
            print(f"FAIL {name}: {e!r}")
    return made
