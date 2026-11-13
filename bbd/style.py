"""One figure style for the whole paper.

* Sizes follow the Springer LNCS / CCIS text block (122 mm wide), which QC+AI uses.
  FULL = 4.80 in, HALF = 2.33 in. Figures are drawn at final size, so the 8 pt type is 8 pt in print.
* Okabe-Ito colours (colour-blind safe), one colour per model in every figure.
* Line style encodes the training condition: CE solid, post-hoc dashed, LA dash-dot.
* Vector PDF with embedded TrueType fonts (pdf.fonttype 42) plus a 300 dpi PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FULL, HALF = 4.80, 2.33

MODEL_COLORS = {
    "Q": "#0072B2", "Q-noent": "#56B4E9", "C-bounded": "#D55E00", "C-bounded-wide": "#E69F00",
    "C-cosine": "#CC79A7", "C-host": "#000000", "MLP": "#8C8C8C", "LightGBM": "#009E73",
    # revision controls
    "C-sin-diag": "#009E73", "C-diag": "#F0E442", "C-sin": "#882255", "C-bounded-init": "#AA4499",
    "C-host-nobias": "#555555",
}
MODEL_LABELS = {
    "Q": "Q", "Q-noent": "Q-noent", "C-bounded": "C-bounded", "C-bounded-wide": "C-bounded (2×)",
    "C-cosine": "C-cosine", "C-host": "C-host", "MLP": "MLP (ref.)", "LightGBM": "LightGBM (ref.)",
    "C-sin-diag": "C-sin-diag", "C-diag": "C-diag", "C-sin": "C-sin", "C-bounded-init": "C-bounded-init",
    "C-host-nobias": "C-host (no bias)",
}
MODEL_MARKERS = {"Q": "o", "Q-noent": "o", "C-bounded": "s", "C-bounded-wide": "s", "C-cosine": "D",
                 "C-host": "^", "MLP": "v", "LightGBM": "P",
                 "C-sin-diag": "X", "C-diag": "<", "C-sin": ">", "C-bounded-init": "p", "C-host-nobias": "^"}
COND_STYLES = {"CE": "-", "post-hoc": (0, (4, 1.6)), "LA": (0, (5, 1.5, 1.2, 1.5)), "CB": ":"}
COND_LABELS = {"CE": "cross-entropy", "post-hoc": "CE + post-hoc ($\\tau$=1)", "LA": "logit-adjusted loss",
               "CB": "class-balanced"}
DATASET_LABELS = {"toniot8": "TON_IoT (K = 8)", "toniot10": "TON_IoT, all classes (K = 10)",
                  "ciciot8": "CICIoT2023 (K = 8)"}
ACCENT_POSTHOC = "#D55E00"
ACCENT_LA = "#0072B2"
GREY = "#7F7F7F"
LIGHT = "#D9D9D9"


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.titlesize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.45, "axes.axisbelow": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.color": "#333333", "ytick.color": "#333333",
        "lines.linewidth": 1.2, "lines.markersize": 3.6, "errorbar.capsize": 0,
        "legend.frameon": False, "legend.handlelength": 2.2, "legend.borderaxespad": 0.2,
        "legend.columnspacing": 1.0, "legend.handletextpad": 0.4,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.constrained_layout.use": True,
        "image.cmap": "cividis",
    })


def save(fig, name: str, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def panel(ax, letter: str, x: float = -0.02, y: float = 1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def rho_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlim(7, 1400)
    ax.set_xlabel("imbalance ratio $\\rho$")
