"""End to end on synthetic data with the real Kaggle schemas: prepare -> tune -> grid -> resume ->
stats -> tables -> figures -> hardware analysis (on FakeKingston)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from bbd import config, data, runner, stats

INPUT = Path(os.environ["BBD_INPUT"])


@pytest.fixture(scope="module")
def prepared():
    data.write_synthetic_toniot(INPUT / "ton-iot" / "train_test_network.csv")
    data.write_synthetic_ciciot(INPUT / "ciciot2023" / "csv")
    src = data.find_sources(INPUT)
    assert len(src["toniot"]) == 1 and len(src["ciciot"]) == 4
    out = {**data.prepare_toniot(src["toniot"]), **data.prepare_ciciot(src["ciciot"], cap=2000)}
    for p in out.values():
        data.save_prepared(p)
    return out


def test_prepared_datasets(prepared):
    t9, t10, c8 = prepared["toniot8"], prepared["toniot10"], prepared["ciciot8"]
    assert (t9.k, t10.k, c8.k) == (8, 10, 8)
    assert not {"mitm", "ransomware"} & set(t9.class_names) and {"mitm", "ransomware"} <= set(t10.class_names)
    assert t9.k == 8 and t10.k == 10
    assert t9.class_names[0] == "normal" and c8.class_names[0] == "DDoS"
    assert not any(c.split("=")[0] in data.TONIOT_ID_COLS for c in t9.feature_names)
    assert t9.meta["dedup"]["duplicate_rows"] >= 200
    for p in prepared.values():
        test = np.bincount(p.y[p.split == 2], minlength=p.k)
        val = np.bincount(p.y[p.split == 1], minlength=p.k)
        assert (test == config.N_TEST).all() and (val == config.N_VAL).all()
        assert np.isfinite(p.X).all() and np.abs(p.X).max() <= 6


def test_training_subsets(prepared):
    p = prepared["toniot8"]
    idx, counts = data.sample_train(p, 100, seed=3)
    assert len(np.unique(idx)) == len(idx) and (p.split[idx] == 0).all()
    assert counts[0] / counts[-1] == pytest.approx(100, rel=0.15)
    idx2, _ = data.sample_train(p, 100, seed=3)
    assert (np.sort(idx) == np.sort(idx2)).all()


@pytest.fixture(scope="module")
def results(prepared, tmp_path_factory):
    res = Path(os.environ["BBD_WORK"]) / "results"
    # a slice of the tuning stage and of the grid, enough for every figure
    tune = [t for t in config.build_tuning() if t[0].dataset == "toniot8" and t[2] < 2]
    cache = runner.DataCache("cpu")
    for spec, hp, i in tune:
        runner.append(res, 0, runner.execute(spec, hp, cache, "cpu", None, trial=i))
    hps = runner.best_hyperparameters(res)
    assert "toniot8|Q|CE" in hps
    grid = [s for s in config.build_grid(("hw", "core", "sweep", "ref"))
            if s.seed < 2 and s.dataset == "toniot8" and (s.s in (None, 1.0, 2.0, 3.0))
            and s.model in ("Q", "Q-noent", "C-bounded", "C-host", "C-cosine", "MLP", "LightGBM")]
    for s in grid:
        hp = hps.get(config.tune_key(s.dataset, s.family, s.loss), runner.DEFAULT_HP)
        runner.append(res, 0, runner.execute(s, hp, cache, "cpu", res))
    return res


def test_ledger_resume_and_carry(results, tmp_path):
    done = runner.done_keys(results)
    assert any("_Q_s2_LA_seed0" in d for d in done)
    st = runner.status(results)
    assert st["errors"] == 0
    # carry forward into a fresh results folder from an "attached input"
    fake_input = tmp_path / "input" / "prev-version" / "results"
    import shutil
    shutil.copytree(results, fake_input)
    fresh = tmp_path / "fresh"
    n = runner.carry_forward(fresh, tmp_path / "input", verbose=False)
    assert n > 0 and runner.done_keys(fresh) == done


def test_records_and_stats(results):
    df = stats.records_frame(results)
    assert {"raw", "posthoc1", "shots1024"} <= set(df["variant"])
    q = df[(df["model"] == "Q") & (df["variant"] == "raw")]
    assert (q["r_eff"] <= q["s_final"] + 1e-6).all()           # a bounded head never exceeds its bound
    ph = df[df["variant"].str.startswith("posthoc")]
    raw = df[df["variant"] == "raw"].set_index("run_id")["r_eff"]
    from bbd.theory import unreachable_count
    for _, r in ph.iterrows():                                   # Lemma 2 holds on every run
        c = np.asarray(r["train_counts"], float)
        assert r["never_predicted"] >= unreachable_count(c / c.sum(), raw[r["run_id"]], r["tau"])
    h = stats.hypothesis_table(df)
    assert {"H1", "H3"} <= set(h["hypothesis"])


def test_tables_and_figures(results):
    from bbd import figures, tables
    df = stats.records_frame(results)
    out = Path(os.environ["BBD_WORK"]) / "figures"
    tables.write_all(df, stats.hypothesis_table(df), Path(os.environ["BBD_WORK"]) / "tables")
    shot_df = figures.shot_curve(results, df, seeds=range(2), shots=(64, 256, 1024))
    made = figures.make_all(df, out, results, None, shot_df)
    failed = [k for k, v in made.items() if not v and not k.startswith("fig_hw") and "ciciot" not in k]
    assert not failed, failed


def test_hardware_on_fake_backend(results):
    pytest.importorskip("qiskit_ibm_runtime")
    from bbd import figures, hardware as hw
    rid = "toniot8_r100_Q_s3_LA_seed0"
    inp = hw.model_inputs(results, rid, n_test=45, n_cal=18)
    assert hw.verify_against_qiskit(inp) < 1e-6
    backend = hw.get_backend(fake=True)
    path = hw.best_path(backend)["path"]
    qc = hw.build_circuit(inp["phi"], inp["k"], inp["entangle"], inp["reupload"])
    pubs, keys = [], []
    for lay, layout in (("A", path), ("B", path[::-1])):
        isa = hw.to_isa(qc, backend, layout)
        assert hw.check_isa(isa, qc, layout)["ok"]
        pubs.append((isa, np.vstack([inp["theta_test"], inp["theta_cal"]]), 256))
        keys.append(lay)
    c0, c1 = hw.calibration_circuits(10)
    pubs += [(hw.to_isa(c0, backend, path), None, 1024), (hw.to_isa(c1, backend, path), None, 1024)]
    res = hw.make_sampler(backend).run(pubs).result()
    nt = len(inp["theta_test"])
    hw_z = {}
    for i, lay in enumerate(keys):
        z = hw.counts_to_z(res[i].data.c, inp["k"])
        hw_z[("LA", lay)] = (z[:nt], z[nt:])
    cal = {"z0": hw.counts_to_z(res[2].data.c, 10)[0], "z1": hw.counts_to_z(res[3].data.c, 10)[0]}
    out = hw.analyse({"LA": inp}, hw_z, path, cal, 256, out=Path(os.environ["BBD_WORK"]) / "hardware")
    assert "spearman" in out and set(out["conditions"]["LA / layout A"]) >= {"raw", "global", "affine", "readout"}
    figdir = Path(os.environ["BBD_WORK"]) / "figures"
    assert figures.fig_hw_conditions(out, figdir) and figures.fig_hw_bias(out, figdir)
    assert figures.fig_hw_confusion(out, figdir)
