"""Paper figures are drawn from released outputs only; and the effective-parameter counts quoted in the paper."""
import numpy as np
import pandas as pd
import pytest

from bbd import paper_figures, paper_numbers


def _hw(seed):
    rng = np.random.default_rng(seed)
    cond = lambda: {"recall": rng.uniform(0.3, 1.0, 8).tolist()}
    conf = lambda: np.diag(np.full(8, 30)).tolist()
    return {"conditions": {k: {"raw": cond(), "sim_exact": cond()} for k in ("LA / layout A", "LA / layout B")},
            "confusion": {"CE235 + post-hoc / layout A": {c: conf() for c in ("sim_exact", "raw", "global", "readout")}}}


def _ledger():
    rows = []
    counts = (1000 ** -(np.arange(8) / 7) * 2000).round().astype(int).tolist()
    for i, (model, s) in enumerate([("Q", 1.0), ("C-bounded", 2.0), ("C-cosine", 4.0), ("C-host", None)]):
        for variant in ("raw", "posthoc1"):
            rows.append(dict(run_id=f"r{i}", tier="core", model=model, s=s, s_label="x" if s else "na",
                             variant=variant, tags_str="", train_counts=counts, tau=1.0, never_predicted=i % 3))
    rows.append(dict(rows[1], run_id="rev", tier="revision"))            # second tier must be excluded
    return pd.DataFrame(rows)


def test_paper_figures_from_outputs(tmp_path):
    rev = tmp_path / "rev"
    rev.mkdir()
    pd.DataFrame({"host": "C-host", "s": [1, 2, 4], "lost": [0.2, 0.1, 0.05], "gained": [0.03, 0.04, 0.05],
                  "saturated": [0.3, 0.1, 0.02], "share_of_lost_on_saturated": [0.7, 0.35, 0.1]}
                 ).to_csv(rev / "rev_h1_decomposition.csv", index=False)
    made, stats = paper_figures.make_all(_ledger(), rev, _hw(0), _hw(1), tmp_path / "fig", verbose=False)
    assert {p.name for p in made} == {f"{n}.{e}" for n in ("fig1_reachability", "fig_h1", "fig_hw") for e in ("pdf", "png")}
    assert stats["n"] == 3 and stats["never_below"] <= 1.0                # three bounded main-grid post-hoc rows


@pytest.mark.parametrize("k, n_params, n_cz", [(8, 44, 17), (10, 50, 18)])
def test_effective_parameters_and_cz(k, n_params, n_cz):
    """Sect. 4: with K read qubits on a 10-qubit, 3-layer path, only n_params of 60 angles and n_cz of 27 CZ gates
    can change <Z_0..Z_{K-1}>."""
    assert paper_numbers.effective_counts(k) == (n_params, n_cz)


def test_sinusoid_identity():
    assert paper_numbers.sinusoid_identity_error() < 1e-10


def test_agrees_rounding():
    assert paper_numbers._agrees("74.1 against 63.3", "74.1 against 63.3") == "yes"
    assert paper_numbers._agrees("74.1 against 64.0", "74.1 against 63.3") == "no"
    assert paper_numbers._agrees("-21.7 [-24.7, -18.7], p = 8.6e-12", "-21.7 [-24.7, -18.7], p = 8.6e-12") == "yes"
