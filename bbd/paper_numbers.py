"""Every number in the paper that no other notebook cell prints, recomputed from the released outputs.

`paper_numbers(...)` returns one row per claim: the section, the claim, the value the paper states, the value
recomputed here, whether they agree at the paper's rounding, and the source files. Nothing is trained; the only
circuit simulations are 10-qubit state vectors on CPU (effective gate counts, Lemma 4, shot noise).

    from bbd import paper_numbers
    table = paper_numbers.paper_numbers(df, RESULTS, rev_dir, hw1_json, hw2_json, hw2_tests_json, dataset_summary_csv)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

from . import qsim, stats
from .paper_figures import MAIN_GRID_TIERS, reachability_frame
from .revision import pooled_grid
from .stats import FIXED_S, KEYS, paired, pooled

BENIGN, BACKDOOR = 0, 7          # TON_IoT class indices (benign is stored as "normal"; backdoor is the rarest)
H5_RUN = "toniot8_r100_Q_s2.35_CE_seed0"


def _pct(x, d=1):
    return round(100 * float(x), d)


def _ci(r, d=1):
    return f"{100 * r['est']:+.{d}f} [{100 * r['ci95'][0]:.{d}f}, {100 * r['ci95'][1]:.{d}f}]"


def _json(x):
    return x if isinstance(x, dict) else json.loads(Path(x).read_text())


# ----------------------------------------------------------------------------- circuit facts (Sect. 4, Lemma 4)
def effective_counts(k: int, n: int = 10, layers: int = 3, seed: int = 0) -> tuple[int, int]:
    """Parameters and CZ gates of the Q circuit that can change <Z_0..Z_{k-1}> (the rest are diagonal at the end
    or outside the past light cone of every read qubit). Checked numerically at random angles."""
    import torch
    torch.manual_seed(seed)
    theta = (torch.rand(32, n) * np.pi).double()
    phi = torch.randn(layers, n, 2, dtype=torch.float64, requires_grad=True)
    z = qsim.expectations(theta, phi, k, cdtype=torch.complex128)
    g = sum(torch.autograd.grad(z[:, j].sum(), phi, retain_graph=True)[0].abs() for j in range(k))
    n_params = int((g > 1e-9).sum())

    gates = qsim.ry_rz(phi.detach()[..., 0], phi.detach()[..., 1]).to(torch.complex128)
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    edges = qsim.path_edges(n)

    def run(drop=None):
        psi = qsim.product_state(theta, torch.complex128)
        for layer in range(layers):
            for q in range(n):
                psi = qsim.apply_1q(psi, gates[layer, q], q, n)
            d = np.ones(2 ** n)
            for e in edges:
                if (layer, e) != drop:
                    d *= np.where(bits[:, e[0]] & bits[:, e[1]], -1, 1)
            psi = psi * torch.as_tensor(d)
        p = psi.abs() ** 2
        return p @ qsim.z_matrix(n, k, "cpu", p.dtype)

    z0 = run()
    n_cz = sum((run((layer, e)) - z0).abs().max().item() > 1e-9 for layer in range(layers) for e in edges)
    return n_params, n_cz


def sinusoid_identity_error(n: int = 10, k: int = 8, layers: int = 3, seed: int = 0) -> float:
    """Lemma 4: max |<Z_k> - A_k cos(theta_k - phi_k)| for the entanglement-free circuit at random inputs."""
    import torch
    torch.manual_seed(seed)
    phi = torch.empty(layers, n, 2, dtype=torch.float64).uniform_(-np.pi, np.pi)
    th0 = torch.zeros(1, n, dtype=torch.float64)
    th1 = torch.full((1, n), np.pi / 2, dtype=torch.float64)
    uz = qsim.expectations(th0, phi, k, False, False, cdtype=torch.complex128)[0]    # u_z: Bloch vector e_z
    ux = qsim.expectations(th1, phi, k, False, False, cdtype=torch.complex128)[0]    # u_x: Bloch vector e_x
    theta = torch.rand(256, n, dtype=torch.float64) * np.pi
    ez = qsim.expectations(theta, phi, k, False, False, cdtype=torch.complex128)
    pred = ux * torch.sin(theta[:, :k]) + uz * torch.cos(theta[:, :k])
    return float((ez - pred).abs().max())


# ----------------------------------------------------------------------------- shot noise of a logit difference
def shot_sd_logit_diff(results, run_id: str = H5_RUN, h: int = BENIGN, j: int = BACKDOOR, shots: int = 512,
                       cls: int | None = BENIGN, alpha=None, beta=None, device="cpu") -> dict | None:
    """Exact standard deviation of the shot estimate of z_h - z_j = s(<Z_h> - <Z_j>) over the test inputs of
    class `cls`: Var = (2 - 2<Z_h Z_j> - (<Z_h> - <Z_j>)^2) / shots for one shot draw per input.

    With per-qubit `alpha`, `beta` (Lemma 3, fitted on hardware) each measured bit is modelled as flipping
    independently, so <Z_k>' = a_k <Z_k> + b_k and <Z_h Z_j>' = a_h a_j <Z_h Z_j> + a_h b_j <Z_h> + b_h a_j <Z_j> + b_h b_j;
    this is the shot noise of the contracted values a hardware run would measure."""
    import torch
    from . import store
    from .data import load_prepared
    from .runner import load_model
    if not store.has(results, "ckpt", run_id):
        return None
    model, spec, _ = load_model(results, run_id, device)
    try:
        p = load_prepared(spec.dataset)          # built by notebook 01 from the public data
    except FileNotFoundError:
        return None
    X, y = p.part("test")
    if cls is not None:
        X = X[y == cls]
    with torch.no_grad():
        hrep = model.encoder(torch.as_tensor(X, device=device))
        probs = qsim.probabilities(model.head.angles(hrep), model.head.phi, model.head.entangle, model.head.reupload)
        n = model.head.n
        bits = torch.as_tensor((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1, device=device)
        zh, zj = 1 - 2 * bits[:, h].double(), 1 - 2 * bits[:, j].double()
        P = probs.double()
        e_h, e_j, e_hj = P @ zh, P @ zj, P @ (zh * zj)
        if alpha is not None:
            ah, aj, bh, bj = alpha[h], alpha[j], beta[h], beta[j]
            e_hj = ah * aj * e_hj + ah * bj * e_h + bh * aj * e_j + bh * bj
            e_h, e_j = ah * e_h + bh, aj * e_j + bj
        var = (2 - 2 * e_hj - (e_h - e_j) ** 2).clamp_min(0) / shots
        s = model.head.scale.value()
        sd = s * var.sqrt().cpu().numpy()
    return dict(n=len(sd), median=float(np.median(sd)), p10=float(np.percentile(sd, 10)),
                p90=float(np.percentile(sd, 90)), s=s)


# ----------------------------------------------------------------------------- the table
def paper_numbers(df: pd.DataFrame, results=None, rev_dir=None, hw1=None, hw2=None, hw2_tests=None,
                  dataset_summary=None, circuit_checks: bool = True) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    rows = []

    def add(section, claim, paper, value, source):
        rows.append(dict(section=section, claim=claim, paper=str(paper), value=str(value), source=source))

    main = df[df["tier"].isin(MAIN_GRID_TIERS)]
    base = main[(main.tags_str == "") & (main.rho_str != "nat") & main.dataset.isin(["toniot8", "ciciot8"])]
    led = "main-grid ledger (nb 02)"

    # --- run counts
    n_main = main.run_id.nunique()
    n_rest = df[~df["tier"].isin(MAIN_GRID_TIERS)].run_id.nunique()
    add("4", "runs: main grid / second tier / total", "2,550 / 1,690 / 4,240", f"{n_main:,} / {n_rest:,} / {n_main + n_rest:,}",
        "all ledgers (nb 02, 05, 06)")

    # --- reachability, main grid (Fig. 2b, abstract)
    ph = reachability_frame(df)
    pos = ph[ph.pred > 0]
    add("abstract, 5.1, Fig. 2", "main-grid post-hoc evaluations; exact; guess 0; exact if >=1 predicted",
        "1,880; 84%; 65%; 67%",
        f"{len(ph):,}; {_pct((ph.never_predicted == ph.pred).mean(), 0):.0f}%; "
        f"{_pct((ph.never_predicted == 0).mean(), 0):.0f}%; {_pct((pos.never_predicted == pos.pred).mean(), 0):.0f}%", led)
    if rev_dir is not None and (Path(rev_dir) / "rev_reachability_pretest_1.csv").exists():
        per = pd.read_csv(Path(rev_dir) / "rev_reachability_pretest_1.csv")
        add("5.1", "evaluations behind the per-head rates (incl. learned s)", "3,362", f"{int(per['n'].sum()):,}",
            "rev_reachability_pretest_1.csv (nb 07)")

    # --- H2a: gaps and difference in differences
    w = stats._with_reff(base)

    def gap(variant):
        a = w[(w.variant == variant) & w.model.isin(["C-bounded", "Q"]) & w.s_label.isin(FIXED_S)]
        host = w[(w.variant == variant) & (w.model == "C-host")][KEYS + ["macro_f1"]]
        return a.merge(host, on=KEYS, suffixes=("", "_host"))

    m = gap("posthoc1")
    m["drop"] = m.macro_f1 - m.macro_f1_host
    m["below"] = m.x_reach < 1
    add("5.1", "post-hoc loss vs C-host per evaluation: below / above threshold", "39.5 / 0.8",
        f"{-_pct(m[m.below]['drop'].mean())} / {-_pct(m[~m.below]['drop'].mean())}", led)
    ce = gap("raw")[KEYS + ["model", "s_label", "macro_f1", "macro_f1_host"]].rename(
        columns={"macro_f1": "ce", "macro_f1_host": "ce_host"})
    mm = m.merge(ce, on=KEYS + ["model", "s_label"])
    mm["did"] = (mm.macro_f1 - mm.ce) - (mm.macro_f1_host - mm.ce_host)
    pv = mm.groupby(KEYS + ["below"], as_index=False)["did"].mean().pivot_table(index=KEYS, columns="below", values="did")
    both = (pv[True] - pv[False]).dropna()
    hw_ = sps.t.ppf(0.975, len(both) - 1) * both.std(ddof=1) / np.sqrt(len(both))
    add("5.1", "difference in differences [95% CI], one-sided Wilcoxon p", "-21.7 [-24.7, -18.7], p = 8.6e-12",
        f"{_pct(both.mean())} [{_pct(both.mean() - hw_)}, {_pct(both.mean() + hw_)}], "
        f"p = {sps.wilcoxon(both, alternative='less').pvalue:.1e}", led)
    t = main[(main.dataset == "toniot8") & (main.rho_str == "1000") & (main.variant == "posthoc1")
             & main.s_label.isin(["1", "1.5", "2"]) & (main.model != "C-host") & (main.tags_str == "")]
    nb = int((t.recall.apply(lambda r: r[BENIGN]) > 0).sum())
    add("5.1", "TON_IoT, rho = 1000, s <= 2, post-hoc: runs predicting any benign flow", "0",
        f"{nb} of {len(t)}", led)

    # --- H2b: class-balanced loss and learned bias (s = 2)
    def cmp(a, b):
        A = base[(base.variant == "raw") & (base.s_label == "2") & (base.model == a[0]) & (base.loss == a[1])]
        on = ["dataset", "rho_str", "seed"] + ([] if b[0] == "C-host" else ["s_label"])
        B = base[(base.variant == "raw") & (base.model == b[0]) & (base.loss == b[1])]
        if b[0] != "C-host":
            B = B[B.s_label == "2"]
        d = A.merge(B[on + ["macro_f1"]], on=on, suffixes=("", "_b"))
        d["diff"] = d.macro_f1 - d.macro_f1_b
        d["cell"] = d.dataset + "|" + d.rho_str
        d["unit"] = d.cell + "|" + d.seed.astype(str)
        return pooled(d)

    for a, b, paper in [(("C-bounded", "CB"), ("C-bounded", "LA"), "+9.4 [1.8, 17.0]"),
                        (("Q", "CB"), ("Q", "LA"), "+1.3 [-2.0, 4.7]"),
                        (("C-bounded", "CB"), ("C-host", "CB"), "-5.0 [-9.0, -1.0]"),
                        (("Q", "CB"), ("C-host", "CB"), "-4.8 [-8.0, -1.5]")]:
        add("5.2", f"{a[0]} {a[1]} minus {b[0]} {b[1]}, s = 2, 5 seeds, all rho", paper, _ci(cmp(a, b)), led)
    tb = base[(base.variant == "raw") & (base.s_label == "2") & (base.model == "C-bounded")
              & (base.dataset == "toniot8") & (base.rho_str == "1000")]
    cb_seeds = tb[tb.loss == "CB"].seed
    add("5.2", "C-bounded TON_IoT rho = 1000: CB against LA, same seeds", "74.1 against 63.3",
        f"{_pct(tb[tb.loss == 'CB'].macro_f1.mean())} against {_pct(tb[(tb.loss == 'LA') & tb.seed.isin(cb_seeds)].macro_f1.mean())}",
        led)
    for mdl, paper in (("C-bounded", "28.4 to 70.0"), ("Q", "23.0 to 58.1")):
        sel = (main.dataset == "toniot8") & (main.rho_str == "1000") & (main.s_label == "2") & (main.model == mdl) \
              & (main.loss == "CE") & (main.variant == "posthoc1")
        bb = main[sel & (main.tags_str == "bias")]
        nb_ = main[sel & (main.tags_str == "") & main.seed.isin(bb.seed)]
        add("5.2", f"learned bias, post-hoc, TON_IoT rho = 1000, s = 2: {mdl}", paper,
            f"{_pct(nb_.macro_f1.mean())} to {_pct(bb.macro_f1.mean())}", led)

    # --- H3: per-scale gaps, seed-matched Q row, capacity, re-uploading
    g = pooled_grid(main, "Q", "C-bounded", s_values=FIXED_S)
    ce_s = g[(g.loss == "CE") & (g.s != "all")]
    la_s = g[(g.loss == "LA")].set_index("s")
    add("5.4", "Q minus C-bounded under LA at s = 1, 2", "+20.5, +8.6",
        f"{la_s.loc['1', 'est']:+.1f}, {la_s.loc['2', 'est']:+.1f}", led)
    add("5.4", "Q minus C-bounded under CE, range over the six scales", "-2.9 to +7.7",
        f"{ce_s.est.min():+.1f} to {ce_s.est.max():+.1f}", led)
    g5 = pooled_grid(main[main.seed < 5], "Q", "C-bounded", s_values=("1", "2", "4")).query("s == 'all'").set_index("loss")
    add("Table 2", "Q seeds 0-4 minus C-bounded, CE / LA", "+1.8 [-1.9, +5.5] / +11.3 [+5.6, +17.0]",
        " / ".join(f"{g5.loc[l, 'est']:+.1f} [{g5.loc[l, 'lo95']:+.1f}, {g5.loc[l, 'hi95']:+.1f}]" for l in ("CE", "LA")), led)
    cw = main[(main.variant == "raw") & main.model.isin(["C-bounded-wide", "C-bounded"]) & (main.tags_str == "")]
    piv = cw.groupby(["dataset", "rho_str", "loss", "s_label", "model"]).macro_f1.mean().unstack("model").dropna()
    add("5.4", "cells where C-bounded-wide is below C-bounded", "11 of 12",
        f"{int((piv['C-bounded-wide'] < piv['C-bounded']).sum())} of {len(piv)}", led)
    re_ = main[(main.tags_str == "reupload") & (main.variant == "raw")]
    for loss, paper in (("CE", "+4.3 [-1.5, 10.1]"), ("LA", "+1.3 [-1.7, 4.3]")):
        on = ["dataset", "rho_str", "seed", "s_label"]
        B = base[(base.model == "Q") & (base.variant == "raw") & (base.loss == loss)]
        d = re_[re_.loss == loss].merge(B[on + ["macro_f1"]], on=on, suffixes=("", "_b"))
        d["diff"] = d.macro_f1 - d.macro_f1_b
        d["cell"] = d.dataset + "|" + d.rho_str + "|" + d.s_label.astype(str)
        d["unit"] = d.cell + "|" + d.seed.astype(str)
        add("5.4", f"data re-uploading minus Q, {loss}", paper, _ci(pooled(d)), led)

    # --- data
    if dataset_summary is not None and Path(dataset_summary).exists():
        ds = pd.read_csv(dataset_summary)
        cic, ton = ds[ds.dataset == "ciciot8"], ds[ds.dataset == "toniot8"]
        add("4", "CICIoT2023 rows (sum of raw per-category rows)", "7,845,673", f"{int(cic.raw_rows.sum()):,}",
            "dataset_summary.csv (nb 01)")
        add("4", "natural rho: TON_IoT / CICIoT2023", "12 / 2,775",
            f"{ton.natural_est.max() / ton.natural_est.min():.0f} / {cic.natural_est.max() / cic.natural_est.min():,.0f}",
            "dataset_summary.csv (nb 01)")
        add("4", "TON_IoT rows after cleaning (K = 8)", "92,018", f"{int(ton.natural_est.sum()):,}", "dataset_summary.csv (nb 01)")
    r1000 = main[(main.rho_str == "1000") & (main.dataset == "toniot8")].train_counts.iloc[0]
    add("4, 7", "training rows of the two rarest classes at rho = 1000", "34 and 13", f"{r1000[-2]} and {r1000[-1]}", led)

    # --- hardware
    if hw1 is not None and hw2 is not None and hw2_tests is not None:
        h1, h2, t2 = _json(hw1), _json(hw2), _json(hw2_tests)
        a1 = h1["fits"]["CE + post-hoc / layout A"]["alpha_global"]
        add("6", "session-1 contraction alpha (CE + post-hoc model)", "0.955", f"{a1:.3f}", "hw_results.json (nb 03)")
        add("6", "2 alpha s at s = 3 and at s = 2.35", "5.7 and 4.49", f"{2 * a1 * 3:.1f} and {2 * a1 * 2.35:.2f}",
            "hw_results.json (nb 03)")
        f2 = h2["fits"]["CE235 + post-hoc / layout A"]
        r = t2["h5_prediction"]["r_eff_sim"]
        margin = r * (f2["alpha"][BENIGN] + f2["alpha"][BACKDOOR]) + 2.35 * (f2["beta"][BENIGN] - f2["beta"][BACKDOOR])
        add("6", "session-2 per-qubit margin benign vs backdoor; realised threshold", "4.57; 4.60",
            f"{margin:.2f}; {t2['h5_prediction']['threshold']:.2f}", "hw_results.json, hw2_tests.json (nb 06)")
    if results is not None:
        fits = _json(hw2)["fits"]["CE235 + post-hoc / layout A"] if hw2 is not None else None
        for label, kw in (("simulation, no noise", {}),
                          ("hardware, session-2 per-qubit fits", dict(alpha=fits["alpha"], beta=fits["beta"]) if fits else None)):
            if kw is None:
                continue
            sd = shot_sd_logit_diff(results, **kw)
            add("6", f"512-shot s.d. of z_benign - z_backdoor, benign test inputs, {label} (10th-90th pct.)",
                "0.035" if kw else "-",
                "needs the checkpoint (release assets) and the prepared data (notebook 01)" if sd is None else
                f"{sd['p10']:.3f} to {sd['p90']:.3f} (median {sd['median']:.3f}, n = {sd['n']})",
                f"checkpoint {H5_RUN} (nb 05/06)" + (", hw_results.json (nb 06)" if kw else ""))

    # --- circuit
    if circuit_checks:
        p8, c8 = effective_counts(8)
        p10, c10 = effective_counts(10)
        add("4", "effective parameters and CZ gates, K = 8; K = 10", "44 and 17; 50 and 18", f"{p8} and {c8}; {p10} and {c10}",
            "state-vector simulation")
        add("3", "Lemma 4 identity, max error", "< 1e-10", f"{sinusoid_identity_error():.1e}", "state-vector simulation")

    out = pd.DataFrame(rows)
    out["agrees"] = [_agrees(p, v) for p, v in zip(out.paper, out.value)]
    return out


def _agrees(paper: str, value: str) -> str:
    """'yes' when every number of the paper's statement equals the recomputed one at the paper's rounding."""
    import re
    if "not attached" in value:
        return "n/a"
    if paper == "-":
        return "info"
    if paper.startswith("<"):
        return "yes" if float(value) < float(paper[1:]) else "no"
    if paper.startswith("about"):
        return "check"
    num = r"[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?"
    p_nums = re.findall(num, paper)
    v_nums = re.findall(num, value)
    if len(v_nums) < len(p_nums):
        return "no"
    for p, v in zip(p_nums, v_nums):
        p_, v_ = float(p.replace(",", "")), float(v.replace(",", ""))
        d = len(p.split(".")[1]) if "." in p and "e" not in p else 0
        tol = (abs(p_) * 0.05) if "e" in p else 0.5 * 10 ** (-d) + 1e-9
        if abs(p_ - v_) > tol:
            return "no"
    return "yes"
