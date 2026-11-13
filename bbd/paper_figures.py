"""The data figures of the paper, drawn from released outputs only (no training, no hardware access).

    Fig. 2  fig1_reachability.pdf   main-grid run ledger (notebook 02)
    Fig. 3  fig2_main.pdf           drawn by figures.fig2_main (notebook 07)
    Fig. 4  fig_h1.pdf              rev_h1_decomposition.csv (notebook 07)
    Fig. 5  fig_hw.pdf              hardware/hw_results.json (notebook 03) and
                                    hardware/session2/hw_results.json (notebook 06)

Fig. 1 of the paper is a TikZ schematic inside main.tex and holds no data.

Usage (from a notebook, or the command line):

    from bbd import paper_figures
    paper_figures.make_all(df, rev_dir, hw1_json, hw2_json, out)

    python -m bbd.paper_figures LEDGER_DIR REV_TABLE_DIR HW1_JSON HW2_JSON OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from . import style as S
from .figures import unreachable_grid
from .theory import unreachable_count

# Tiers of the main grid (notebook 02); the second tier added after a first analysis is "revision".
MAIN_GRID_TIERS = ("hw", "core", "sweep", "ablation", "ref")
# Lemma 2 concerns bounded, bias-free heads; these five are the bounded heads of the main grid.
BOUNDED_MAIN = ["Q", "Q-noent", "C-bounded", "C-bounded-wide", "C-cosine"]
# TON_IoT class order used by the hardware records (benign is stored as "normal").
TON_CLASSES = ["normal", "injection", "password", "ddos", "xss", "dos", "scanning", "backdoor"]


def _panel(ax, letter, size=8):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, fontsize=size, fontweight="bold", va="bottom",
            ha="right")


def _save(fig, name, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


# ============================================================================ Fig. 2: reachability
def reachability_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed-s post-hoc evaluations of the bias-free bounded heads in the main grid, with the Lemma 2 count
    computed from the nominal s (known before training) in column `pred`."""
    d = df[df["tier"].isin(MAIN_GRID_TIERS) & (df["s_label"] != "learned")]
    ph = d[d["variant"].str.startswith("posthoc") & d["model"].isin(BOUNDED_MAIN)
           & ~d["tags_str"].str.contains("bias")].copy()
    pred = [unreachable_count(np.asarray(c) / np.sum(c), float(s), float(t))
            for c, s, t in zip(ph["train_counts"], ph["s"], ph["tau"])]
    return ph.assign(pred=pred)


def fig_reachability(df: pd.DataFrame, out, k: int = 8, tau: float = 1.0):
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 2.35), gridspec_kw=dict(width_ratios=[1.3, 1]))
    # (a) theory: classes never predicted under post-hoc adjustment, exponential profile
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
               fontsize=7.5, arrowprops=dict(arrowstyle="-", color=S.ACCENT_POSTHOC, lw=0.6))
    a.text(1100, 3.85, "$s=\\frac{\\tau}{2}\\log\\rho$\n($\\tau$=1)", fontsize=7.5, ha="center", va="bottom")
    a.text(1500, 1.45, "$\\tau$=0.5", fontsize=7.5, ha="center", va="bottom")
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
    cb.set_label(f"classes never predicted (K = {k})", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    _panel(a, "a")

    # (b) predicted from the nominal s against observed on the test set
    ph = reachability_frame(df)
    stats = {}
    if len(ph):
        cnt = ph.groupby(["pred", "never_predicted"]).size().reset_index(name="n")
        mx = int(max(cnt["pred"].max(), cnt["never_predicted"].max())) + 1
        b.plot([-0.5, mx], [-0.5, mx], color=S.GREY, lw=0.8, ls=(0, (4, 1.6)), zorder=1)
        b.fill_between([-0.5, mx], [-0.5, mx], [mx, mx], color=S.ACCENT_LA, alpha=0.06, lw=0, zorder=0)
        b.scatter(cnt["pred"], cnt["never_predicted"], s=6 + 60 * np.sqrt(cnt["n"] / cnt["n"].max()),
                  color=S.ACCENT_POSTHOC, alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        pos = ph[ph["pred"] > 0]
        stats = dict(n=len(ph), exact=float((ph["never_predicted"] == ph["pred"]).mean()),
                     never_below=float((ph["never_predicted"] >= ph["pred"]).mean()),
                     guess0=float((ph["never_predicted"] == 0).mean()),
                     exact_if_pred=float((pos["never_predicted"] == pos["pred"]).mean()) if len(pos) else np.nan)
        b.text(0.97, 0.04, f"exact {100 * stats['exact']:.0f}% (guess 0: {100 * stats['guess0']:.0f}%)\n"
               f"if $\\geq$1 predicted: {100 * stats['exact_if_pred']:.0f}%", transform=b.transAxes,
               va="bottom", ha="right", fontsize=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        b.set_xlim(-0.5, mx)
        b.set_ylim(-0.5, mx)
        b.set_xticks(range(0, mx + 1, max(1, mx // 5)))
        b.set_yticks(range(0, mx + 1, max(1, mx // 5)))
    else:
        b.text(0.5, 0.5, "no post-hoc runs", transform=b.transAxes, ha="center", color=S.GREY)
    b.set_xlabel("predicted by Lemma 2")
    b.set_ylabel("classes never predicted (test)")
    _panel(b, "b")
    return _save(fig, "fig1_reachability", out), stats


# ============================================================================ Fig. 4: H1 decomposition
def fig_h1(rev_dir, out):
    dec = pd.read_csv(Path(rev_dir) / "rev_h1_decomposition.csv")
    dec = dec[dec["host"] == "C-host"]
    fig, ax = plt.subplots(figsize=(S.FULL * 0.6, 1.85))
    x = np.arange(len(dec))
    sat = 100 * dec["lost"] * dec["share_of_lost_on_saturated"]
    ax.bar(x, sat, 0.6, color="#D55E00", label="lost, saturated")
    ax.bar(x, 100 * dec["lost"] - sat, 0.6, bottom=sat, color="#F2B38F", label="lost, unsaturated")
    ax.bar(x, -100 * dec["gained"], 0.6, color="#0072B2", label="gained")
    ax.plot(x, 100 * dec["saturated"], color="#333333", marker="o", ms=3, lw=0.9, label="saturated inputs")
    ax.axhline(0, color="#333333", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in dec["s"]])
    ax.set_xlabel("logit bound $s$")
    ax.set_ylabel("% of test inputs")
    ax.legend(fontsize=6.5, loc="upper right", handlelength=1.1, frameon=False)
    return _save(fig, "fig_h1", out)


# ============================================================================ Fig. 5: hardware
def fig_hw(hw1: dict, hw2: dict, out, classes=TON_CLASSES):
    fig = plt.figure(figsize=(S.FULL, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
    # (a) per-class recall, raw hardware minus exact simulation, LA model (s = 3), both sessions and layouts
    ax = fig.add_subplot(gs[0])
    series = [("S1, layout A", hw1, "LA / layout A", "#0072B2", ""), ("S1, layout B", hw1, "LA / layout B", "#56B4E9", ""),
              ("S2, layout A", hw2, "LA / layout A", "#0072B2", "////"),
              ("S2, layout B", hw2, "LA / layout B", "#56B4E9", "////")]
    k, w = len(classes), 0.2
    for i, (lab, d, key, c, hatch) in enumerate(series):
        rec = np.array(d["conditions"][key]["raw"]["recall"]) - np.array(d["conditions"][key]["sim_exact"]["recall"])
        ax.bar(np.arange(k) + (i - 1.5) * w, 100 * rec, w, color=c if not hatch else "white", edgecolor=c,
               hatch=hatch, lw=0.6, label=lab)
    ax.axhline(0, color="#333333", lw=0.5)
    ax.axvspan(k - 3.5, k - 0.5, ymax=0.62, color=S.LIGHT, alpha=0.45, lw=0, zorder=0)   # three rarest classes
    ax.set_ylim(-36, 42)
    ax.set_xticks(range(k))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_ylabel("recall, hardware minus\nexact simulation (points)")
    ax.set_title("LA model, $s=3$")
    _panel(ax, "a")
    ax.legend(ncol=2, loc="upper left", fontsize=6.5, handlelength=1.3, columnspacing=0.8)
    # (b) where the benign test inputs go under the post-hoc model at s = 2.35 (session 2)
    ax = fig.add_subplot(gs[1])
    conf = hw2["confusion"]["CE235 + post-hoc / layout A"]
    conds = [("simulation", "sim_exact"), ("raw", "raw"), ("global\nrescale", "global"), ("readout\ninversion", "readout")]
    y = np.arange(len(conds))[::-1]
    n_benign = int(np.array(conf["sim_exact"])[0].sum())
    for yi, (lab, key) in zip(y, conds):
        row = np.array(conf[key])[0]
        n, left = row.sum(), 0
        for v, c, lab2 in [(row[0], "#000000", "benign (correct)"), (row[k - 1], "#D55E00", "backdoor (rarest)"),
                           (n - row[0] - row[k - 1], "#BBBBBB", "other")]:
            ax.barh(yi, v, left=left, color=c, height=0.62, label=lab2 if yi == y[0] else None)
            if v >= 5:
                ax.text(left + v / 2, yi, str(v), ha="center", va="center", fontsize=7, color="white")
            left += v
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in conds])
    ax.set_xlim(0, n_benign)
    ax.set_xlabel(f"benign test inputs (of {n_benign})")
    ax.set_title("H5: post-hoc, $s=2.35$")
    ax.grid(False, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2, fontsize=6.5, handlelength=1.0, frameon=False,
              columnspacing=0.8)
    _panel(ax, "b")
    return _save(fig, "fig_hw", out)


# ============================================================================ all
def make_all(df: pd.DataFrame, rev_dir, hw1, hw2, out, verbose=True):
    """df: stats.records_frame of the ledgers; rev_dir: folder with rev_h1_decomposition.csv;
    hw1, hw2: session-1 and session-2 hw_results.json (paths or loaded dicts)."""
    S.apply()
    hw1 = json.loads(Path(hw1).read_text()) if not isinstance(hw1, dict) else hw1
    hw2 = json.loads(Path(hw2).read_text()) if not isinstance(hw2, dict) else hw2
    made, stats = fig_reachability(df, out)
    made += fig_h1(rev_dir, out)
    made += fig_hw(hw1, hw2, out)
    if verbose:
        print("Fig. 2 statistics:", {key: round(v, 4) if isinstance(v, float) else v for key, v in stats.items()})
        for p in made:
            print("wrote", p)
    return made, stats


if __name__ == "__main__":
    from .stats import records_frame
    ledger, rev, h1, h2, dest = sys.argv[1:6]
    make_all(records_frame(ledger), rev, h1, h2, dest)
