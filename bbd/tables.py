"""LaTeX tables (booktabs) and CSV copies for the paper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import seed_ci
from .style import DATASET_LABELS, MODEL_LABELS

ROWS = [("C-host", None), ("C-bounded", "2"), ("C-cosine", "2"), ("Q-noent", "2"), ("Q", "2"), ("Q", "4"),
        ("MLP", None), ("LightGBM", None)]
CONDS = [("CE", "raw", "CE"), ("post-hoc", "posthoc1", "CE"), ("LA", "raw", "LA")]


def main_table(df: pd.DataFrame, dataset: str, metric: str = "macro_f1") -> tuple[str, pd.DataFrame]:
    d = df[(df["dataset"] == dataset) & (df["tags_str"] == "") & (df["rho_str"].isin(["10", "100", "1000"]))]
    cells, recs = {}, []
    for model, s in ROWS:
        for cond, variant, loss in CONDS:
            for rho in ("10", "100", "1000"):
                if model == "LightGBM":
                    g = d[(d["model"] == model) & (d["variant"] == "raw") & (d["rho_str"] == rho)]
                else:
                    g = d[(d["model"] == model) & (d["variant"] == variant) & (d["loss"] == loss) & (d["rho_str"] == rho)]
                if s is not None:
                    g = g[g["s_label"] == s]
                v = 100 * g[metric].values
                if len(v):
                    cells[(model, s, cond, rho)] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v))
                    recs.append(dict(model=model, s=s, condition=cond, rho=rho, mean=v.mean(),
                                     sd=v.std(ddof=1) if len(v) > 1 else 0.0, n=len(v)))
    best = {}
    for (model, s, cond, rho), (m, _, _) in cells.items():
        if model in ("MLP", "LightGBM"):
            continue
        best[(cond, rho)] = max(best.get((cond, rho), -1), m)
    lines = [r"\begin{tabular}{l" + "rrr" * len(CONDS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{c}}}" for c, _, _ in CONDS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(CONDS))),
             r"Head & " + " & ".join([r"$\rho$=10", "100", "1000"] * len(CONDS)) + r" \\", r"\midrule"]
    for model, s in ROWS:
        if model == "MLP":
            lines.append(r"\midrule")
        name = MODEL_LABELS[model] + (f" ($s$={s})" if s else "")
        row = [name]
        for cond, _, _ in CONDS:
            for rho in ("10", "100", "1000"):
                c = cells.get((model, s, cond, rho))
                if c is None:
                    row.append("--")
                    continue
                txt = f"{c[0]:.1f}\\,{{\\scriptsize$\\pm${c[1]:.1f}}}"
                if model not in ("MLP", "LightGBM") and abs(c[0] - best.get((cond, rho), -9)) < 1e-9:
                    txt = r"\textbf{" + f"{c[0]:.1f}" + "}" + f"\\,{{\\scriptsize$\\pm${c[1]:.1f}}}"
                row.append(txt)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"% Macro-F1 (\\%) on the balanced test set, {DATASET_LABELS.get(dataset, dataset)}; mean $\\pm$ s.d. over "
               f"seeds. Bold: best head per column (references excluded). LightGBM uses class weights in every column.")
    return caption + "\n" + "\n".join(lines), pd.DataFrame(recs)


def hypothesis_latex(h: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Hyp. & Claim & Estimate (pts) & 90\% CI & $p$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for _, r in h.iterrows():
        est = 100 * r["est"] if pd.notna(r.get("est")) else np.nan
        ci = r.get("ci90")
        ci_s = f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]" if isinstance(ci, (tuple, list)) else "--"
        fmt = lambda v: "--" if v is None or pd.isna(v) else f"{v:.3g}"
        est_s = "--" if pd.isna(est) else f"{est:+.2f}"
        if isinstance(ci, (tuple, list)) and any(pd.isna(c) for c in ci):
            ci_s = "--"
        lines.append(f"{r['hypothesis']} & {r['claim']} & {est_s} & {ci_s} & {fmt(r.get('p'))} & "
                     f"{fmt(r.get('p_holm'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_all(df: pd.DataFrame, hyp: pd.DataFrame | None, out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ds in sorted(set(df["dataset"])):
        tex, frame = main_table(df, ds)
        (out / f"table_main_{ds}.tex").write_text(tex)
        frame.to_csv(out / f"table_main_{ds}.csv", index=False)
    if hyp is not None and len(hyp):
        (out / "table_hypotheses.tex").write_text(hypothesis_latex(hyp))
        hyp.to_csv(out / "table_hypotheses.csv", index=False)
