"""Theory used throughout the paper.

Lemma 1 (clipped-target form). For logits in [-s, s]^K and loss -log softmax(z + delta)_y,
a pointwise minimiser of the expected loss is z*_k = clip(log eta_k - delta_k + c, -s, s) for a
scalar c, unique up to a common shift when no coordinate clips.

Lemma 2 (post-hoc reachability). With an unbiased head, post-hoc adjustment
argmax_k z_k - tau log pi_k can predict class h for some input iff 2 r > tau log(pi_h / pi_min),
where r is the half-range of the logits (the nominal bound s, or the effective range).
"""
from __future__ import annotations

import numpy as np


def exponential_counts(total: int, k: int, rho: float) -> np.ndarray:
    """Class sizes n_r = n_max * rho^(-r/(K-1)), r = 0 (most frequent) .. K-1, summing to ~total."""
    r = np.arange(k)
    shape = rho ** (-r / (k - 1))
    n = total * shape / shape.sum()
    return np.maximum(np.round(n).astype(int), 1)


def reach_threshold(rho: float, tau: float = 1.0) -> float:
    """Smallest half-range at which every class stays reachable: s > tau log(rho) / 2."""
    return tau * np.log(rho) / 2.0


def reachable(pi, r: float, tau: float = 1.0) -> np.ndarray:
    """Boolean per class: can post-hoc adjustment ever predict it?"""
    pi = np.asarray(pi, dtype=np.float64)
    return 2.0 * r > tau * np.log(pi / pi.min())


def unreachable_count(pi, r: float, tau: float = 1.0) -> int:
    return int((~reachable(pi, r, tau)).sum())


def unreachable_count_exponential(k: int, rho: float, s: float, tau: float = 1.0) -> int:
    ratios = rho ** (np.arange(k) / (k - 1))          # pi_h / pi_min, index 0 = rarest
    return int(np.sum(2 * s <= tau * np.log(ratios)))


def rarest_wins_everywhere(pi, r: float, tau: float = 1.0) -> bool:
    pi = np.sort(np.asarray(pi, dtype=np.float64))
    return 2.0 * r <= tau * np.log(pi[1] / pi[0])


def clipped_minimiser(eta, delta, s: float, iters: int = 200) -> np.ndarray:
    """Lemma 1: z* = clip(log eta - delta + c, -s, s), with c the fixed point c = log sum exp(z* + delta)."""
    eta, delta = np.asarray(eta, float), np.asarray(delta, float)
    target = np.log(eta) - delta

    def resid(c):
        z = np.clip(target + c, -s, s)
        w = z + delta
        m = w.max()
        return c - (m + np.log(np.exp(w - m).sum()))

    lo, hi = -100.0, 100.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:
            hi = mid
        else:
            lo = mid
    return np.clip(target + 0.5 * (lo + hi), -s, s)


def expected_ce(z, eta, delta) -> float:
    w = np.asarray(z) + np.asarray(delta)
    m = w.max()
    return float(-(np.asarray(eta) * (w - (m + np.log(np.exp(w - m).sum())))).sum())


def projected_gradient_minimiser(eta, delta, s: float, iters: int = 20000, lr: float = 0.5) -> np.ndarray:
    """Reference solver for tests: projected gradient descent on the box."""
    eta, delta = np.asarray(eta, float), np.asarray(delta, float)
    z = np.zeros_like(eta)
    for _ in range(iters):
        w = z + delta
        p = np.exp(w - w.max())
        p /= p.sum()
        z = np.clip(z - lr * (p - eta), -s, s)
    return z
