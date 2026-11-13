"""Revision additions: Lemma 4 against the simulator, the classical control heads, the revision tier,
learning-rate tags, capped validation, small-sample pooling, the revision analyses and hardware helpers."""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bbd import config, data, hardware as hw, models, qsim, revision, runner, stats

INPUT = Path(os.environ["BBD_INPUT"])


# ----------------------------------------------------------------------------- theory
def test_proposition4_entanglement_free_head_is_a_sinusoid():
    """<Z_k> of the entanglement-free circuit equals u_x sin(theta) + u_z cos(theta) (Lemma 4)."""
    torch.manual_seed(0)
    n, k, L = 10, 8, 3
    phi = torch.empty(L, n, 2, dtype=torch.float64).uniform_(-math.pi, math.pi)
    th0 = torch.zeros(1, n, dtype=torch.float64)
    th1 = torch.full((1, n), math.pi / 2, dtype=torch.float64)
    uz = qsim.expectations(th0, phi, k, False, False, cdtype=torch.complex128)[0]
    ux = qsim.expectations(th1, phi, k, False, False, cdtype=torch.complex128)[0]
    theta = torch.rand(64, n, dtype=torch.float64) * math.pi
    ez = qsim.expectations(theta, phi, k, False, False, cdtype=torch.complex128)
    pred = ux * torch.sin(theta[:, :k]) + uz * torch.cos(theta[:, :k])
    assert torch.allclose(ez, pred, atol=1e-10)
    assert (ux ** 2 + uz ** 2 <= 1 + 1e-10).all()          # A_k <= 1
    # with CZ gates the identity fails (the head is no longer a product of single-qubit sinusoids)
    ez_e = qsim.expectations(theta, phi, k, True, False, cdtype=torch.complex128)
    assert not torch.allclose(ez_e, pred, atol=1e-3)


@pytest.mark.parametrize("name", ["C-sin-diag", "C-diag", "C-sin", "C-bounded-init", "C-host-nobias"])
def test_control_heads(name):
    s = None if name == "C-host-nobias" else 2.0
    spec = config.RunSpec("toniot8", 100, name, s, "CE", 0)
    m = models.build_model(spec, d=12, k=8)
    z = m(torch.randn(256, 12) * 3)
    assert z.shape == (256, 8)
    if s is not None:
        assert z.abs().max() <= s + 1e-6 and m.bound == pytest.approx(s)
    else:
        assert m.bound is None and m.head.v.bias is None


def test_c_sin_diag_matches_qnoent_at_init():
    """At initialisation both compute about -s sin(pi h / 2) per class."""
    torch.manual_seed(1)
    h = torch.rand(100, 10) * 2 - 1
    head = models.DiagSinHead(10, 8, 2.0)
    with torch.no_grad():
        head.phi.zero_()
        z = head(h)
    assert torch.allclose(z, -2.0 * torch.sigmoid(torch.tensor(4.0)) * torch.sin(math.pi * h[:, :8] / 2), atol=1e-5)


# ----------------------------------------------------------------------------- grid
def test_revision_tier():
    g = config.build_grid(("revision",))
    assert len(g) == 1690 and sum(s.is_quantum for s in g) == 634
    ids = {s.run_id for s in g}
    assert "toniot8_r100_Q_s2.35_CE_seed0" in ids and "toniot8_r100_Q-noent_s3_LA_seed0" in ids
    lr = [s for s in g if any(t.startswith("lr") for t in s.tags)]
    assert len(lr) == 288 and {config.tag_lr(s.tags) for s in lr} == set(config.LR_SWEEP)
    assert {s.seed for s in lr} == {1, 2, 3}
    assert sum("valcap" in s.tags for s in g) == 200
    # no revision run collides with an original run
    orig = {s.run_id for s in config.build_grid(("hw", "core", "sweep", "ablation", "ref"))}
    assert not ids & orig
    assert len(config.build_tuning()) == 3 * 13 * 2 * 8


def test_capped_validation_index():
    y = np.repeat(np.arange(4), 100)
    idx = runner.capped_val_index(y, [500, 50, 7, 0], seed=0)
    assert np.bincount(y[idx], minlength=4).tolist() == [100, 50, 7, 1]


# ----------------------------------------------------------------------------- statistics
def test_pooled_small_sample():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(2):
        for i in range(10):
            rows.append(dict(diff=0.02 + 0.03 * c + rng.normal(0, 0.01), cell=f"c{c}", unit=f"c{c}-{i}"))
    r = stats.pooled(pd.DataFrame(rows))
    assert r["df"] == 1 and r["method"] == "cluster-mean"
    # two cells: the t(1) interval is much wider than a normal one would be
    assert (r["ci90"][1] - r["ci90"][0]) > 4 * 1.645 * 2 * r["se"] / 2
    assert r["pi95"][0] < r["ci95"][0] and r["pi95"][1] > r["ci95"][1]
    ct = stats.cell_table(pd.DataFrame(rows), delta=0.01)
    assert len(ct) == 2 and ct["outside_margin"].all()


# ----------------------------------------------------------------------------- runs and analyses
@pytest.fixture(scope="module")
def rev_results():
    try:
        for name in ("toniot8", "ciciot8"):
            data.load_prepared(name)
    except FileNotFoundError:
        data.write_synthetic_toniot(INPUT / "ton-iot-rev" / "train_test_network.csv")
        data.write_synthetic_ciciot(INPUT / "ciciot2023-rev" / "csv")
        src = data.find_sources(INPUT)
        for p in {**data.prepare_toniot(src["toniot"]), **data.prepare_ciciot(src["ciciot"], cap=2000)}.values():
            data.save_prepared(p)
    res = Path(os.environ["BBD_WORK"]) / "results_rev"
    cache = runner.DataCache("cpu")
    hp = {"lr": 1e-2, "wd": 0.0}
    specs = []
    for ds in ("toniot8", "ciciot8"):
        for seed in (0, 1, 2):
            for loss in ("CE", "LA"):
                specs += [config.RunSpec(ds, 100, "C-host", None, loss, seed),
                          config.RunSpec(ds, 100, "C-host-nobias", None, loss, seed),
                          config.RunSpec(ds, 100, "C-bounded", 1.0, loss, seed),
                          config.RunSpec(ds, 100, "C-sin-diag", 1.0, loss, seed),
                          config.RunSpec(ds, 100, "Q", 1.0, loss, seed),
                          config.RunSpec(ds, 100, "Q-noent", 1.0, loss, seed),
                          config.RunSpec(ds, 100, "C-bounded", 1.0, loss, seed, ("valcap",)),
                          config.RunSpec(ds, 100, "C-bounded", 1.0, loss, seed, ("lr0.03",))]
    for s in specs:
        recs = runner.execute(s, hp, cache, "cpu", res)
        runner.append(res, 0, recs)
    return res


def test_tags_change_training(rev_results):
    recs = runner.read_records(rev_results)
    lr = [r for r in recs if "lr0.03" in r["tags"] and r["variant"] == "raw"]
    assert lr and all(r["hp"] == {"lr": 0.03, "wd": 0.0} for r in lr)
    cap = [r for r in recs if "valcap" in r["tags"] and r["variant"] == "raw"]
    std = [r for r in recs if r["model"] == "C-bounded" and not r["tags"] and r["variant"] == "raw"]
    assert cap and min(r["n_val_fit"] for r in cap) < max(r["n_val_fit"] for r in std)


def test_revision_analyses(rev_results, tmp_path):
    df = stats.records_frame(rev_results)
    summ, per_model = revision.reachability_pretest(df, rev_results)
    assert len(summ) == 6 and (summ["observed_lt"] == 0).all()
    dec = revision.h1_decomposition(df, rev_results)
    assert len(dec) and dec["net_loss"].between(-1, 1).all()
    g = revision.pooled_grid(df, "C-sin-diag", "C-bounded", s_values=("1",))
    assert set(g["loss"]) == {"CE", "LA"}
    out = revision.run_all(df, rev_results, tmp_path / "t", tmp_path / "f", verbose=False)
    assert out["hyperparameters"] is not None
    leak, comp = out["leakage"]
    assert len(leak) == 2 and "share_within_0.1" in leak
    assert out["natural_prior"] is not None


# ----------------------------------------------------------------------------- hardware helpers
def test_hardware_session2_helpers():
    y = np.repeat(np.arange(8), 75)
    keep = hw.every_other(y)
    assert len(keep) == 304 and (np.bincount(y[keep]) == 38).all()
    sh, est = hw.plan_session_shots([360] * 4, remaining=316)
    assert sh == 512 and est < 0.8 * 316
    sh, _ = hw.plan_session_shots([360] * 4, remaining=60)
    assert sh in (128, None)
    pm = hw.paired_mcnemar(np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]), np.array([0, 1, 1, 2]))
    assert pm["a_only"] == 1 and pm["b_only"] == 0
    inp = dict(s=2.35, log_pi=np.log(np.array([100.0, 1.0]) / 101.0), ez_test_sim=np.array([[1.0, -1.0], [-1.0, 1.0]]))
    pred = hw.h5_prediction(inp, alpha=0.955)
    assert pred["unreachable_sim"] == 0 and pred["unreachable_hw"] == 1
