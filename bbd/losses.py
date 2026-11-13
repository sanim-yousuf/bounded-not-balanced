"""Training losses and post-hoc adjustment.

CE   plain cross-entropy
CB   class-balanced cross-entropy (Cui et al., CVPR 2019): weights from the effective
     number of samples, beta = 0.9999, normalised to sum to K
LA   logit-adjusted loss (Menon et al., ICLR 2021): softmax over z + tau * log(pi)
post-hoc adjustment (Menon et al.): predict argmax_k z_k - tau * log(pi_k)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def class_balanced_weights(counts, beta: float = 0.9999) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-12)
    return w * len(counts) / w.sum()


class Objective:
    def __init__(self, loss: str, counts, tau: float = 1.0, device="cpu"):
        self.loss = loss
        counts = np.asarray(counts, dtype=np.float64)
        pi = counts / counts.sum()
        self.log_pi = torch.as_tensor(np.log(pi), dtype=torch.float32, device=device)
        self.tau = tau
        self.weight = (torch.as_tensor(class_balanced_weights(counts), dtype=torch.float32, device=device)
                       if loss == "CB" else None)

    def __call__(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss == "LA":
            return F.cross_entropy(z + self.tau * self.log_pi, y)
        if self.loss == "CB":
            return F.cross_entropy(z, y, weight=self.weight)
        return F.cross_entropy(z, y)


def posthoc(z: np.ndarray, log_pi: np.ndarray, tau: float) -> np.ndarray:
    return z - tau * np.asarray(log_pi)[None, :]
