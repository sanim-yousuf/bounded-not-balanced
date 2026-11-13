import numpy as np
import pytest

from bbd import theory
from bbd.metrics import ece, evaluate


def test_clipped_form_attains_box_optimum():
    rng = np.random.default_rng(0)
    for t in range(60):
        k = 6
        eta = rng.dirichlet(np.ones(k) * 0.5)
        s = rng.uniform(0.2, 3)
        delta = np.log(rng.dirichlet(np.ones(k))) if t % 2 else np.zeros(k)
        z_star = theory.clipped_minimiser(eta, delta, s)
        z_pgd = theory.projected_gradient_minimiser(eta, delta, s, iters=8000)
        assert theory.expected_ce(z_star, eta, delta) <= theory.expected_ce(z_pgd, eta, delta) + 1e-9
        assert z_star[np.argmax(np.log(eta) - delta)] >= z_star.max() - 1e-9     # Bayes class attains the max


@pytest.mark.parametrize("s,expected", [(0.5, [6, 8, 8]), (1, [2, 6, 7]), (2, [0, 2, 4]), (4, [0, 0, 0])])
def test_unreachable_table_from_plan(s, expected):
    assert [theory.unreachable_count_exponential(10, r, s) for r in (10, 100, 1000)] == expected


def test_reachability_is_exact():
    rng = np.random.default_rng(1)
    k, s = 6, 1.0
    pi = theory.exponential_counts(10_000, k, 100) / 10_000
    reach = theory.reachable(pi / pi.sum(), s)
    z = rng.uniform(-s, s, (200_000, k))
    z[np.arange(len(z)), rng.integers(0, k, len(z))] = s                  # push some corners
    pred = np.argmax(z - np.log(pi / pi.sum()), axis=1)
    seen = np.isin(np.arange(k), pred)
    assert not np.any(seen & ~reach)                                       # nothing unreachable is ever predicted


def test_metrics_match_sklearn():
    from sklearn.metrics import f1_score, balanced_accuracy_score
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, 500)
    z = rng.normal(size=(500, 5))
    z[np.arange(500), y] += 1.0
    m = evaluate(z, y, 5, bound=None)
    assert m["macro_f1"] == pytest.approx(f1_score(y, z.argmax(1), average="macro"))
    assert m["bacc"] == pytest.approx(balanced_accuracy_score(y, z.argmax(1)))


def test_ece_and_never_predicted():
    y = np.array([0, 1, 2, 0])
    z = np.array([[5., 0, 0], [5, 0, 0], [5, 0, 0], [5, 0, 0]])
    m = evaluate(z, y, 3, bound=5.0)
    assert m["never_predicted"] == 2
    assert m["saturation"] == 0.0
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(probs, np.array([0, 1])) == pytest.approx(0.0)
