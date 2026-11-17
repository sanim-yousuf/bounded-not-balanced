"""Analyses added in the second revision: reachability baselines, learned scales, the planned H2a test,
Q minus C-bounded by scale and loss, an operating-point view and exact hardware intervals.

Used by notebook 07 (cell "Second-revision analyses") and tools/revision_round2.py. Inputs are released
notebook outputs only (run ledgers and artifact packs); no model is trained."""
import numpy as np
import pandas as pd
import scipy.stats as sps

from . import store
from .paper_figures import reachability_frame
from .revision import BOUNDED_BIASFREE
from .stats import FIXED_S, KEYS, _with_reff, paired
from .theory import unreachable_count

BENIGN = {"toniot8": "normal", "ciciot8": "Benign"}


def reachability_pools(df):
    """Exactness of the pre-training bound with its two baselines, for both pools used in the paper."""
    rows = []
    main = reachability_frame(df).rename(columns={"never_predicted": "obs"})
    full = df[df["variant"].isin(["posthoc0.5", "posthoc1"]) & df["model"].isin(BOUNDED_BIASFREE)
              & ~df["tags_str"].str.contains("bias") & ~df["tags_str"].str.contains("lr")
              & (df["s_label"] != "learned")].copy()
    full["pred"] = [unreachable_count(np.asarray(c) / np.sum(c), float(s), 0.5 if v == "posthoc0.5" else 1.0)
                    for c, s, v in zip(full["train_counts"], full["s"], full["variant"])]
    full = full.rename(columns={"never_predicted": "obs"})
    for name, d in (("main grid, five original bounded heads", main), ("all fixed-s, incl. second tier", full)):
        ex = d["obs"] == d["pred"]
        loss = d["pred"] >= 1
        rows.append(dict(pool=name, n=len(d), exact=ex.mean(), guess_zero=(d["obs"] == 0).mean(),
                         n_loss_predicted=int(loss.sum()), exact_when_loss_predicted=ex[loss].mean(),
                         observed_below_bound=int((d["obs"] < d["pred"]).sum())))
    return pd.DataFrame(rows)


def learned_scale(df):
    """Learned-s CE models with post-hoc adjustment (tau = 1): how often the learned s ends below the threshold."""
    d = df[(df["s_label"] == "learned") & (df["loss"] == "CE") & (df["variant"] == "posthoc1")].copy()
    d["below"] = d["s_final"] < 0.5 * np.log(d["rho_num"].astype(float))
    return d.groupby(["model", "rho_num"]).agg(n=("below", "size"), share_below=("below", "mean"),
                                               s_mean=("s_final", "mean"), s_min=("s_final", "min"),
                                               s_max=("s_final", "max"),
                                               never_predicted=("never_predicted", "mean")).reset_index()


def h2a_planned_mann_whitney(df):
    """The test written in the analysis plan: per-evaluation loss against C-host, below versus above the
    threshold, unpaired one-sided Mann-Whitney U. The paper's primary H2a test is the paired Wilcoxon."""
    base = df[(df["tags_str"] == "") & (df["rho_str"] != "nat") & df["dataset"].isin(["toniot8", "ciciot8"])]
    w = _with_reff(base)
    ph = w[(w["variant"] == "posthoc1") & w["model"].isin(["C-bounded", "Q"]) & w["s_label"].isin(FIXED_S)]
    host = w[(w["variant"] == "posthoc1") & (w["model"] == "C-host")][KEYS + ["macro_f1"]]
    m = ph.merge(host, on=KEYS, suffixes=("", "_host"))
    m["drop"] = 100 * (m["macro_f1"] - m["macro_f1_host"])
    below, above = m.loc[m["x_reach"] < 1, "drop"], m.loc[m["x_reach"] >= 1, "drop"]
    u = sps.mannwhitneyu(below, above, alternative="less")
    return pd.DataFrame([dict(n_below=len(below), n_above=len(above), mean_below=below.mean(),
                              mean_above=above.mean(), U=u.statistic, p_one_sided=u.pvalue)])


def h3_by_scale_and_loss(df):
    """Q minus C-bounded at matched s, by loss and s, per dataset and pooled (95% t intervals over cells)."""
    base = df[(df["tags_str"] == "") & (df["rho_str"] != "nat") & df["dataset"].isin(["toniot8", "ciciot8"])]
    d = paired(base, {"model": "Q", "variant": "raw", "loss": ["CE", "LA"], "s_label": FIXED_S},
               {"model": "C-bounded", "variant": "raw"}, match_s=True)
    rows = []
    for (loss, s), g in d.groupby(["loss", "s_label"]):
        for ds, gg in [("both", g)] + list(g.groupby("dataset")):
            per_cell = gg.groupby("cell")["diff"].mean() * 100
            per_run = gg["diff"] * 100
            n_cells = len(per_cell)
            if n_cells >= 2:
                h = sps.t.ppf(0.975, n_cells - 1) * per_cell.std(ddof=1) / np.sqrt(n_cells)
            else:
                h = sps.t.ppf(0.975, len(per_run) - 1) * per_run.std(ddof=1) / np.sqrt(len(per_run))
            rows.append(dict(loss=loss, s=float(s), dataset=ds, n_runs=len(gg), n_cells=n_cells,
                             est=per_cell.mean(), lo95=per_cell.mean() - h, hi95=per_cell.mean() + h,
                             share_positive=float((per_run > 0).mean())))
    return pd.DataFrame(rows).sort_values(["loss", "s", "dataset"])


def operating_point(df, results, fpr_target=0.05):
    """Detector view: alert when max_attack z - z_benign exceeds a threshold chosen on the balanced validation
    set so that 5% of benign validation flows raise an alert; report test benign false-alarm rate, attack
    detection rate (any attack flagged as an attack) and detection of the two rarest classes."""
    pick = df[(df["rho_str"] == "1000") & (df["tags_str"] == "") & df["dataset"].isin(["toniot8", "ciciot8"])
              & (df["variant"].isin(["raw", "posthoc1"]))
              & (((df["model"].isin(["Q", "Q-noent", "C-bounded"])) & (df["s_label"] == "4"))
                 | df["model"].isin(["C-host", "MLP"]))]
    pick = pick[(pick["loss"].isin(["CE", "LA"])) & ~((pick["variant"] == "posthoc1") & (pick["loss"] != "CE"))]
    rows = []
    for r in pick.itertuples():
        z = store.load_logits(results, r.run_id)
        if z is None:
            continue
        names = list(r.class_names)
        b = names.index(BENIGN[r.dataset])
        adj = -np.asarray(z["log_pi"]) if r.variant == "posthoc1" else 0.0

        def score(zz):
            zz = np.asarray(zz) + adj
            att = np.delete(zz, b, axis=1).max(1)
            return att - zz[:, b]

        sv, st = score(z["z_val"]), score(z["z_test"])
        yv, yt = np.asarray(z["y_val"]), np.asarray(z["y_test"])
        thr = np.quantile(sv[yv == b], 1 - fpr_target)
        alert = st > thr
        # argmax decision (no threshold) for comparison
        zz = np.asarray(z["z_test"]) + adj
        argmax_alert = zz.argmax(1) != b
        counts = np.asarray(z["counts"])
        tail = np.argsort(counts)[:2]
        rows.append(dict(dataset=r.dataset, model=r.model, mode=("post-hoc" if r.variant == "posthoc1" else r.loss),
                         seed=r.seed, benign_far_argmax=argmax_alert[yt == b].mean(),
                         benign_far=alert[yt == b].mean(), detection=alert[yt != b].mean(),
                         tail_detection=alert[np.isin(yt, tail)].mean()))
    d = pd.DataFrame(rows)
    return d.groupby(["dataset", "model", "mode"]).agg(
        seeds=("seed", "nunique"), benign_far_argmax=("benign_far_argmax", "mean"), benign_far=("benign_far", "mean"),
        detection=("detection", "mean"), tail_detection=("tail_detection", "mean")).reset_index()


def hardware_intervals():
    rows = []
    for label, k, n in (("benign assigned to backdoor", 28, 38), ("benign still correct", 8, 38),
                        ("benign misclassified", 30, 38)):
        lo = sps.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
        hi = sps.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
        rows.append(dict(quantity=label, k=k, n=n, share=k / n, lo95=lo, hi95=hi))
    rows.append(dict(quantity="contraction needed for the predicted collapse (threshold / 2 r_eff)",
                     k=np.nan, n=np.nan, share=4.604241920231168 / (2 * 2.3496558996241035), lo95=np.nan,
                     hi95=np.nan))
    return pd.DataFrame(rows)



def all_tables(df, results):
    """Every second-revision table, keyed by name."""
    return dict(reachability_pools=reachability_pools(df), learned_scale=learned_scale(df),
                h2a_mann_whitney=h2a_planned_mann_whitney(df), h3_by_scale_loss=h3_by_scale_and_loss(df),
                hardware_intervals=hardware_intervals(), operating_point=operating_point(df, results))
