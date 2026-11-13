"""Shared encoder and the head ladder.

Every model is  x -> Encoder -> h in (-1, 1)^n -> Head -> logits z in R^K.
Heads differ only in how h becomes z, so paired comparisons isolate the head.

  Q              s * <Z_k>   (state-vector simulation, CZ on a path)
  Q-noent        Q without CZ gates: a product state, classically closed-form
  C-bounded      s * tanh(V h)                       no bias
  C-bounded-wide s * tanh(V2 tanh(V1 h))             twice the head parameters
  C-cosine       s * cos(h, w_k)                     standard long-tail bounded head
  C-host         V h + c                             unbounded
  MLP            unrestricted reference (wider encoder, unbounded head)

Revision controls (they separate what Q-noent actually does from what a dense tanh head does):
  C-sin-diag     s * A_k cos(pi/2 (h_k + 1) - phi_k)   the closed form of Q-noent (Lemma 4), classical
  C-diag         s * tanh(a_k h_k + b_k)               one encoder unit per class, gain init pi/2
  C-sin          s * cos(w_k . h + b_k)                sinusoid on a dense linear map
  C-bounded-init s * tanh(V h + c), V init (pi/2) I    dense tanh with inner bias and Q-matched initial gain
  C-host-nobias  V h                                   unbounded, no output bias (isolates the bound in H1)

`s` is a fixed float, or "learned" (s = softplus(u), initialised at 1).
The "bias" tag adds a learned per-class bias to a bounded head; it can absorb the prior.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import qsim
from .config import ENC_HIDDEN, N_LAYERS, N_QUBITS, RunSpec


class Encoder(nn.Module):
    def __init__(self, d: int, n: int = N_QUBITS, hidden: int = ENC_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, n))

    def forward(self, x):
        return torch.tanh(self.net(x))


class Scale(nn.Module):
    """Fixed or learned logit bound s."""

    def __init__(self, s):
        super().__init__()
        self.learned = s == "learned"
        if self.learned:
            self.u = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))       # softplus(u) = 1
        else:
            self.register_buffer("s", torch.tensor(float(s)))

    def forward(self):
        return F.softplus(self.u) + 1e-3 if self.learned else self.s

    def value(self) -> float:
        return float(self.forward().detach().cpu())


class QuantumHead(nn.Module):
    def __init__(self, n: int, k: int, s, layers: int = N_LAYERS, entangle: bool = True, reupload: bool = False):
        super().__init__()
        self.n, self.k, self.entangle, self.reupload = n, k, entangle, reupload
        self.phi = nn.Parameter(torch.empty(layers, n, 2).uniform_(-0.1, 0.1))
        self.scale = Scale(s)

    def angles(self, h):
        return qsim.encoding_angles(h)

    def expectations(self, h):
        return qsim.expectations(self.angles(h), self.phi, self.k, self.entangle, self.reupload)

    def forward(self, h):
        return self.scale() * self.expectations(h)


class BoundedHead(nn.Module):
    def __init__(self, n: int, k: int, s, wide: bool = False):
        super().__init__()
        self.wide = wide
        if wide:
            self.v1 = nn.Linear(n, n)
        self.v = nn.Linear(n, k, bias=False)
        self.scale = Scale(s)

    def forward(self, h):
        if self.wide:
            h = torch.tanh(self.v1(h))
        return self.scale() * torch.tanh(self.v(h))


class CosineHead(nn.Module):
    def __init__(self, n: int, k: int, s):
        super().__init__()
        self.w = nn.Parameter(torch.randn(k, n) / math.sqrt(n))
        self.scale = Scale(s)

    def forward(self, h):
        return self.scale() * F.normalize(h, dim=-1, eps=1e-6) @ F.normalize(self.w, dim=-1).t()


class HostHead(nn.Module):
    def __init__(self, n: int, k: int, bias: bool = True):
        super().__init__()
        self.v = nn.Linear(n, k, bias=bias)

    def forward(self, h):
        return self.v(h)


class DiagSinHead(nn.Module):
    """Classical closed form of Q-noent: z_k = s A_k cos(pi/2 (h_k + 1) - phi_k), A_k in (0, 1).

    Initialised like Q-noent with near-zero rotation angles (A_k ~ 1, phi_k ~ 0)."""

    def __init__(self, n: int, k: int, s):
        super().__init__()
        assert k <= n
        self.k = k
        self.u = nn.Parameter(torch.full((k,), 4.0))                 # A = sigmoid(u) ~ 0.98
        self.phi = nn.Parameter(torch.empty(k).uniform_(-0.1, 0.1))
        self.scale = Scale(s)

    def forward(self, h):
        theta = (math.pi / 2) * (h[:, : self.k] + 1.0)
        return self.scale() * torch.sigmoid(self.u) * torch.cos(theta - self.phi)


class DiagTanhHead(nn.Module):
    """One encoder unit per class through tanh: z_k = s tanh(a_k h_k + b_k), gain initialised at pi/2."""

    def __init__(self, n: int, k: int, s):
        super().__init__()
        assert k <= n
        self.k = k
        self.a = nn.Parameter(torch.full((k,), math.pi / 2))
        self.b = nn.Parameter(torch.zeros(k))
        self.scale = Scale(s)

    def forward(self, h):
        return self.scale() * torch.tanh(self.a * h[:, : self.k] + self.b)


class SinHead(nn.Module):
    """Sinusoid on a dense linear map: z_k = s cos(w_k . h + b_k), b_k initialised at pi/2 (odd at init)."""

    def __init__(self, n: int, k: int, s):
        super().__init__()
        self.v = nn.Linear(n, k)
        with torch.no_grad():
            self.v.bias.fill_(math.pi / 2)
        self.scale = Scale(s)

    def forward(self, h):
        return self.scale() * torch.cos(self.v(h))


class BoundedInitHead(nn.Module):
    """Dense tanh head with an inner bias and a Q-matched initial gain: V = (pi/2) [I_k 0] + small noise."""

    def __init__(self, n: int, k: int, s):
        super().__init__()
        self.v = nn.Linear(n, k)
        with torch.no_grad():
            self.v.weight.normal_(0.0, 0.05)
            self.v.weight[:, :k].add_(torch.eye(k) * (math.pi / 2))
            self.v.bias.zero_()
        self.scale = Scale(s)

    def forward(self, h):
        return self.scale() * torch.tanh(self.v(h))


class Net(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, k: int, bias: bool = False):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.class_bias = nn.Parameter(torch.zeros(k)) if bias else None

    def forward(self, x):
        z = self.head(self.encoder(x))
        return z + self.class_bias if self.class_bias is not None else z

    @property
    def bound(self):
        """Nominal logit bound s (None for unbounded heads)."""
        sc = getattr(self.head, "scale", None)
        return sc.value() if sc is not None else None


class MLP(nn.Module):
    """Unrestricted reference: wider encoder, unbounded linear head."""

    def __init__(self, d: int, k: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, k))
        self.class_bias = None

    def forward(self, x):
        return self.net(x)

    bound = None


def build_model(spec: RunSpec, d: int, k: int, n: int = N_QUBITS) -> nn.Module:
    if spec.model == "MLP":
        return MLP(d, k)
    enc = Encoder(d, n)
    if spec.model == "Q":
        head = QuantumHead(n, k, spec.s, entangle=True, reupload="reupload" in spec.tags)
    elif spec.model == "Q-noent":
        head = QuantumHead(n, k, spec.s, entangle=False, reupload="reupload" in spec.tags)
    elif spec.model == "C-bounded":
        head = BoundedHead(n, k, spec.s)
    elif spec.model == "C-bounded-wide":
        head = BoundedHead(n, k, spec.s, wide=True)
    elif spec.model == "C-cosine":
        head = CosineHead(n, k, spec.s)
    elif spec.model == "C-host":
        head = HostHead(n, k)
    elif spec.model == "C-host-nobias":
        head = HostHead(n, k, bias=False)
    elif spec.model == "C-sin-diag":
        head = DiagSinHead(n, k, spec.s)
    elif spec.model == "C-diag":
        head = DiagTanhHead(n, k, spec.s)
    elif spec.model == "C-sin":
        head = SinHead(n, k, spec.s)
    elif spec.model == "C-bounded-init":
        head = BoundedInitHead(n, k, spec.s)
    else:
        raise ValueError(spec.model)
    return Net(enc, head, k, bias="bias" in spec.tags)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith("encoder"))
    return {"params_total": int(total), "params_head": int(head)}
