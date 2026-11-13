"""Statistics for the pre-registered hypotheses.

Unit of analysis: a paired difference (model A minus model B) for the same dataset, imbalance
ratio, loss, logit bound and seed, so both runs saw the same training subset.

Pooled estimate: a random-intercept model  diff ~ 1 + (1 | cell), cell = (dataset, rho, loss, s),
fitted with statsmodels MixedLM; if it fails to converge, a cluster-robust mean is used instead.
Equivalence (TOST) at margin DELTA: p = max of the two one-sided t tests (df = cells - 1).
Non-inferiority: one-sided. Every pooled estimate also reports the between-cell SD and a 95% prediction interval.
Holm's step-down correction is applied across the primary tests.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import DELTA, RESULTS_DIR


def s_label(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "none"
    if isinstance(s, str):
        return s
    return f"{float(s):g}"


def records_frame(results=RESULTS_DIR) -> pd.DataFrame:
    from .runner import read_records
    recs = [r for r in read_records(results) if r.get("stage") == "grid"]
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["s_label"] = df["s"].map(s_label)
    df["tags_str"] = df["tags"].map(lambda t: "+".join(sorted(t)) if isinstance(t, (list, tuple)) and t else "")
    df["rho_str"] = df["rho"].astype(str)
    df["rho_num"] = pd.to_numeric(df["rho"], errors="coerce")
    df["cond"] = np.where(df["variant"].str.startswith("posthoc"), "post-hoc", df["loss"])
    return df


# ----------------------------------------------------------------------------- pairing
KEYS = ["dataset", "rho_str", "loss", "seed", "tags_str"]


def paired(df: pd.DataFrame, a: dict, b: dict, metric: str = "macro_f1", match_s: bool = True) -> pd.DataFrame:
    """Paired differences A - B. `a`/`b` are column filters, e.g. {"model": "Q", "variant": "raw"}."""
    def sel(f):
        m = np.ones(len(df), bool)
        for k, v in f.items():
            m &= df[k].isin(v).values if isinstance(v, (list, tuple, set)) else (df[k] == v).values
        return df[m]

    A, B = sel(a), sel(b)
    on = KEYS + (["s_label"] if match_s else [])
    cols_a = list(dict.fromkeys(on + [metric, "model", "s_label"]))
    m = A[cols_a].rename(columns={"model": "model_a"}).merge(B[on + [metric]], on=on, suffixes=("_a", "_b"))
    m["diff"] = m[f"{metric}_a"] - m[f"{metric}_b"]
    # cell: design cell for the random intercept; unit: the B row each difference is paired with.
    # With match_s=False one B row (e.g. C-host) is shared by several A rows (several s and models);
    # pooled() averages those to one value per unit so a shared baseline is not counted as several pairs.
    m["cell"] = m["dataset"] + "|" + m["rho_str"] + "|" + m["loss"]
    if match_s:
        m["cell"] = m["cell"] + "|" + m["s_label"].astype(str)
    m["unit"] = m["cell"] + "|" + m["seed"].astype(str) + "|" + m["tags_str"].astype(str)
    return m


def pooled(diffs: pd.DataFrame, delta: float = DELTA) -> dict:
    """Pooled mean difference from a random-intercept model, with small-sample (t) inference.

    Differences that share one baseline run are first averaged per baseline (`unit`). The intercept of
    diff ~ 1 + (1 | cell) is the estimate; its tests and intervals use a t distribution with
    n_cells - 1 degrees of freedom (a conservative stand-in for Satterthwaite), because the cells, not the
    seeds, are the replication units of a pooled claim. With fewer than three cells the cluster mean is used.
    Also returned: the between-cell SD (tau) and a 95% prediction interval for a new cell, so that an
    'equivalent on average' result is not read as 'equivalent everywhere'.
    """
    d = diffs.dropna(subset=["diff"])
    if "unit" in d and len(d):                        # one value per independent pair (see paired)
        d = d.groupby("unit", as_index=False).agg(diff=("diff", "mean"), cell=("cell", "first"))
    n, n_cells = len(d), d["cell"].nunique() if len(d) else 0
    nan = float("nan")
    if n < 3:
        return dict(n=n, n_cells=n_cells, est=nan, se=nan, df=nan, tau=nan, method="too few pairs",
                    ci90=(nan, nan), ci95=(nan, nan), pi95=(nan, nan), p_two_sided=nan, p_tost=nan,
                    p_noninferior=nan, p_wilcoxon=nan)
    cm = d.groupby("cell")["diff"].mean()
    est, se, tau, method = None, None, None, "mixedlm"
    if n_cells >= 3:
        try:
            import statsmodels.formula.api as smf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm("diff ~ 1", d, groups=d["cell"]).fit(reml=True)
            est, se = float(fit.params["Intercept"]), float(fit.bse["Intercept"])
            tau = float(np.sqrt(max(float(np.asarray(fit.cov_re)[0, 0]), 0.0)))
            if not np.isfinite(se) or se <= 0:
                est = None
        except Exception:
            est = None
    if est is None:                                   # mean of cell means, cells as clusters
        method = "cluster-mean"
        est = float(cm.mean()) if n_cells > 1 else float(d["diff"].mean())
        se = float(cm.std(ddof=1) / np.sqrt(n_cells)) if n_cells > 1 else float(d["diff"].std(ddof=1) / np.sqrt(n))
        tau = float(cm.std(ddof=1)) if n_cells > 1 else 0.0
    df = max(n_cells - 1, 1) if n_cells > 1 else max(n - 1, 1)
    t = sps.t(df)
    q90, q95 = float(t.ppf(0.95)), float(t.ppf(0.975))
    p_two = float(2 * t.sf(abs(est) / se))
    p_lower = float(t.sf((est + delta) / se))         # H0: diff <= -delta
    p_upper = float(t.cdf((est - delta) / se))        # H0: diff >= +delta
    half_pi = q95 * float(np.sqrt(se ** 2 + tau ** 2))
    try:
        p_wil = float(sps.wilcoxon(d["diff"]).pvalue)
    except ValueError:
        p_wil = np.nan
    return dict(n=n, n_cells=n_cells, est=est, se=se, df=df, tau=tau, method=method,
                ci90=(est - q90 * se, est + q90 * se), ci95=(est - q95 * se, est + q95 * se),
                pi95=(est - half_pi, est + half_pi),
                p_two_sided=p_two, p_tost=max(p_lower, p_upper), p_noninferior=p_lower, p_wilcoxon=p_wil)


def cell_table(diffs: pd.DataFrame, delta: float = DELTA) -> pd.DataFrame:
    """Per-cell mean difference with a 90% t interval over seeds and whether it lies inside +-delta."""
    d = diffs.dropna(subset=["diff"])
    if "unit" in d and len(d):
        d = d.groupby("unit", as_index=False).agg(diff=("diff", "mean"), cell=("cell", "first"))
    rows = []
    for cell, g in d.groupby("cell"):
        v = g["diff"].to_numpy()
        m = float(v.mean())
        h = float(sps.t.ppf(0.95, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        rows.append(dict(cell=cell, n=len(v), mean=m, lo90=m - h, hi90=m + h,
                         inside_margin=bool(m - h > -delta and m + h < delta),
                         outside_margin=bool(m - h > delta or m + h < -delta)))
    return pd.DataFrame(rows)


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def seed_ci(values, level: float = 0.95):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m = v.mean()
    if len(v) == 1:
        return m, m, m
    h = sps.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


# ----------------------------------------------------------------------------- hypotheses
FIXED_S = ["1", "1.5", "2", "3", "4", "6"]


def _with_reff(df):
    """Attach each run's raw-logit effective range to its post-hoc rows."""
    raw = df[df["variant"] == "raw"][["run_id", "r_eff"]].rename(columns={"r_eff": "r_eff_raw"})
    out = df.merge(raw, on="run_id", how="left")
    out["x_reach"] = 2 * out["r_eff_raw"] / np.log(out["actual_rho"])
    return out


def hypothesis_table(df: pd.DataFrame, delta: float = DELTA) -> pd.DataFrame:
    base = df[(df["tags_str"] == "") & (df["rho_str"] != "nat") & df["dataset"].isin(["toniot8", "ciciot8"])]
    rows = []

    # H1: no implicit rebalancing -> C-bounded equivalent to C-host under plain CE
    d1 = paired(base, {"model": "C-bounded", "variant": "raw", "loss": "CE", "s_label": FIXED_S},
                {"model": "C-host", "variant": "raw", "loss": "CE"}, match_s=False)
    r = pooled(d1, delta)
    rows.append(dict(hypothesis="H1", claim="C-bounded = C-host under CE (equivalence)", test="TOST",
                     p=r.get("p_tost"), **r))

    # H2a: post-hoc collapse below the reachability threshold
    w = _with_reff(base)
    ph = w[(w["variant"] == "posthoc1") & w["model"].isin(["C-bounded", "Q"]) & w["s_label"].isin(FIXED_S)]
    host = w[(w["variant"] == "posthoc1") & (w["model"] == "C-host")][KEYS + ["macro_f1"]]
    m = ph.merge(host, on=KEYS, suffixes=("", "_host"))
    m["drop"] = m["macro_f1"] - m["macro_f1_host"]
    m["below"] = m["x_reach"] < 1
    m = m.groupby(KEYS + ["below"], as_index=False)["drop"].mean()   # one value per C-host row and side
    # paired within baseline run: (mean drop below threshold) - (mean drop above), Wilcoxon signed-rank
    pv = m.pivot_table(index=KEYS, columns="below", values="drop")
    if True in pv.columns and False in pv.columns:
        both = (pv[True] - pv[False]).dropna()
        if len(both) > 2:
            t = sps.wilcoxon(both, alternative="less")
            h = float(sps.t.ppf(0.975, len(both) - 1) * both.std(ddof=1) / np.sqrt(len(both)))
            rows.append(dict(hypothesis="H2a", claim="post-hoc loses more below 2r/(tau log rho) = 1",
                             test="Wilcoxon signed-rank, paired within baseline run", p=float(t.pvalue),
                             est=float(both.mean()), ci95=(float(both.mean()) - h, float(both.mean()) + h),
                             n=len(both), method="paired per baseline run"))
    # H2b: LA-trained bounded heads non-inferior to C-host LA at rho = 1000
    d2 = paired(base[base["rho_str"] == "1000"], {"model": ["C-bounded", "Q"], "variant": "raw", "loss": "LA",
                                                 "s_label": FIXED_S},
                {"model": "C-host", "variant": "raw", "loss": "LA"}, match_s=False)
    d2b = d2[d2["s_label"].astype(str).isin(["2", "3", "4", "6"])]
    r = pooled(d2b, delta)
    rows.append(dict(hypothesis="H2b", claim="bounded + LA-loss non-inferior to C-host + LA at rho=1000 (s>=2)",
                     test="non-inferiority", p=r.get("p_noninferior"), **r))

    # H3: Q equivalent to C-bounded at matched s
    d3 = paired(base, {"model": "Q", "variant": "raw", "loss": ["CE", "LA"], "s_label": FIXED_S},
                {"model": "C-bounded", "variant": "raw"}, match_s=True)
    r = pooled(d3, delta)
    rows.append(dict(hypothesis="H3", claim="Q = C-bounded at matched s (equivalence)", test="TOST",
                     p=r.get("p_tost"), **r))
    # H3 follow-up: entanglement, at the scales fixed in advance (s = 2, 4)
    d4 = paired(base, {"model": "Q", "variant": "raw", "loss": ["CE", "LA"], "s_label": ["2", "4"]},
                {"model": "Q-noent", "variant": "raw"}, match_s=True)
    r = pooled(d4, delta)
    rows.append(dict(hypothesis="H3-ent", claim="Q = Q-noent at s in {2, 4} (entanglement adds nothing)",
                     test="TOST", p=r.get("p_tost"), **r))
    # revision: the same test over every fixed scale where both were run (adds s = 1, 1.5)
    d5 = paired(base, {"model": "Q", "variant": "raw", "loss": ["CE", "LA"], "s_label": FIXED_S},
                {"model": "Q-noent", "variant": "raw"}, match_s=True)
    if d5["s_label"].nunique() > 2:
        r = pooled(d5, delta)
        rows.append(dict(hypothesis="H3-ent-all", claim="Q = Q-noent at every fixed s run (revision)",
                         test="TOST", p=r.get("p_tost"), **r))

    out = pd.DataFrame(rows)
    for col in ("n", "est", "ci90", "p", "p_holm", "method"):
        if col not in out:
            out[col] = np.nan
    primary = out["hypothesis"].isin(["H1", "H2a", "H2b", "H3"])
    out.loc[primary, "p_holm"] = holm(out.loc[primary, "p"].fillna(1.0).values)
    out["role"] = np.where(primary, "primary", np.where(out["hypothesis"] == "H3-ent", "pre-specified follow-up",
                                                        "exploratory (revision)"))
    return out


def saturation_regression(df: pd.DataFrame) -> dict:
    """H1 secondary: tail-recall gain of bounded heads over C-host vs saturation rate (cluster-robust OLS)."""
    base = df[(df["tags_str"] == "") & (df["rho_str"] != "nat")]
    d = paired(base, {"model": ["C-bounded", "Q"], "variant": "raw", "loss": "CE", "s_label": FIXED_S},
               {"model": "C-host", "variant": "raw", "loss": "CE"}, metric="tail_recall", match_s=False)
    sat = base[base["variant"] == "raw"][KEYS + ["s_label", "model", "saturation"]]
    sat = sat[sat["model"].isin(["C-bounded", "Q"])].rename(columns={"model": "model_a"})
    d = d.merge(sat, on=KEYS + ["s_label", "model_a"], how="left").dropna(subset=["saturation"])
    if len(d) < 5:
        return {}
    import statsmodels.api as sm
    X = sm.add_constant(d["saturation"].values)
    # cluster on the shared C-host row: its noise enters every difference paired with it
    fit = sm.OLS(d["diff"].values, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["unit"])[0]})
    return dict(slope=float(fit.params[1]), se=float(fit.bse[1]), p=float(fit.pvalues[1]), n=len(d))
