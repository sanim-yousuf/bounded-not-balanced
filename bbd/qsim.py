"""Batched state-vector simulator for the quantum head, in PyTorch.

Conventions match Qiskit exactly, so a trained head can be exported to a QuantumCircuit
and run on IBM hardware without any re-mapping:

* qubit q is bit q of the basis-state index (little-endian),
* RY(t) = [[cos t/2, -sin t/2], [sin t/2, cos t/2]],  RZ(t) = diag(e^{-it/2}, e^{+it/2}),
* each variational layer applies RY(a) then RZ(b) on every qubit, then CZ on the path
  edges (0,1), (1,2), ..., (n-2, n-1)  (all CZs commute, so their order is irrelevant),
* the head reads <Z_k> on qubit k for class k.

Circuit
-------
  |0>^n -- RY(theta(x)) -- [ RY(a_lq) RZ(b_lq) , CZ-path ] x L   (with optional re-uploading of
  RY(theta(x)) before every layer after the first)

The state is kept as a (B, 2^n) complex tensor. A single-qubit gate on qubit q is a
broadcast 2x2 matmul on the view (B, 2^(n-q-1), 2, 2^q), so the cost per gate is one
pass over the state; the simulator is memory-bandwidth bound and cheap for n = 10.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch


# ----------------------------------------------------------------------------- static tables
@lru_cache(maxsize=None)
def _bits(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)     # (2^n, n), column q = bit q


def path_edges(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


@lru_cache(maxsize=None)
def _cz_diag_np(n: int) -> np.ndarray:
    b = _bits(n)
    d = np.ones(2 ** n)
    for a, c in path_edges(n):
        d *= np.where((b[:, a] & b[:, c]) == 1, -1.0, 1.0)
    return d


def cz_diag(n: int, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(_cz_diag_np(n), device=device, dtype=dtype)


def z_matrix(n: int, k: int, device, dtype=torch.float32) -> torch.Tensor:
    """(2^n, k) matrix with entry +1/-1 = eigenvalue of Z on qubit j for basis state i."""
    return torch.as_tensor(1 - 2 * _bits(n)[:, :k], device=device, dtype=dtype)


# ----------------------------------------------------------------------------- gates
def ry_rz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RZ(b) @ RY(a) as complex 2x2 matrices; a, b broadcast to shape S -> (S, 2, 2)."""
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    e = torch.polar(torch.ones_like(b), -b / 2)        # e^{-ib/2}
    ec = e.conj()
    row0 = torch.stack([e * c, -e * s], dim=-1)
    row1 = torch.stack([ec * s, ec * c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def ry_matrix(t: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def apply_1q(psi: torch.Tensor, U: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """Apply U on qubit q. psi (B, 2^n); U (2,2) shared or (B,2,2) per sample."""
    B = psi.shape[0]
    v = psi.reshape(B, 2 ** (n - q - 1), 2, 2 ** q)
    if U.dim() == 2:
        out = torch.matmul(U.to(psi.dtype), v)
    else:
        out = torch.matmul(U.to(psi.dtype)[:, None], v)
    return out.reshape(B, 2 ** n)


def product_state(theta: torch.Tensor, cdtype=torch.complex64) -> torch.Tensor:
    """RY(theta_q)|0> on every qubit, as a (B, 2^n) state (qubit 0 = least significant bit)."""
    B, n = theta.shape
    amp = torch.stack([torch.cos(theta / 2), torch.sin(theta / 2)], dim=-1)   # (B, n, 2)
    psi = amp[:, n - 1]
    for q in range(n - 2, -1, -1):
        psi = (psi[:, :, None] * amp[:, q][:, None, :]).reshape(B, -1)
    return psi.to(cdtype)


# ----------------------------------------------------------------------------- fused gate blocks
GROUP = 5     # qubits fused per block: a layer of n single-qubit gates becomes ceil(n/5) 32x32 matmuls


def _groups(n: int, m: int = GROUP) -> list[tuple[int, int]]:
    return [(q0, min(m, n - q0)) for q0 in range(0, n, m)]


def kron_block(U: torch.Tensor) -> torch.Tensor:
    """U (..., m, 2, 2) for qubits q0..q0+m-1  ->  (..., 2^m, 2^m) acting on the group index
    sum_j bit_{q0+j} 2^j, i.e. U_{q0+m-1} (x) ... (x) U_{q0} (highest qubit first, Qiskit order)."""
    out = U[..., -1, :, :]
    for j in range(U.shape[-3] - 2, -1, -1):
        a, b = out, U[..., j, :, :]
        out = (a[..., :, None, :, None] * b[..., None, :, None, :]).reshape(
            *a.shape[:-2], a.shape[-2] * 2, a.shape[-1] * 2)
    return out


def apply_block(psi: torch.Tensor, M: torch.Tensor, q0: int, m: int, n: int) -> torch.Tensor:
    """Apply a 2^m x 2^m block (shared, or per sample (B, 2^m, 2^m)) on qubits q0..q0+m-1."""
    B = psi.shape[0]
    v = psi.reshape(B, 2 ** (n - q0 - m), 2 ** m, 2 ** q0)
    out = torch.matmul(M.to(psi.dtype) if M.dim() == 2 else M.to(psi.dtype)[:, None], v)
    return out.reshape(B, 2 ** n)


# ----------------------------------------------------------------------------- forward
def statevector(theta: torch.Tensor, phi: torch.Tensor, entangle: bool = True, reupload: bool = False,
                cdtype=torch.complex64) -> torch.Tensor:
    """Final state for encoding angles theta (B, n) and variational angles phi (L, n, 2)."""
    B, n = theta.shape
    L = phi.shape[0]
    psi = product_state(theta, cdtype)
    rdtype = torch.float64 if cdtype == torch.complex128 else torch.float32
    cz = cz_diag(n, theta.device, rdtype) if entangle else None
    groups = _groups(n)
    gates = ry_rz(phi[..., 0], phi[..., 1]).to(cdtype)                      # (L, n, 2, 2)
    blocks = [[kron_block(gates[layer, q0:q0 + m]) for q0, m in groups] for layer in range(L)]
    if reupload:
        enc = ry_matrix(theta).to(cdtype)                                   # (B, n, 2, 2)
        enc_blocks = [kron_block(enc[:, q0:q0 + m]) for q0, m in groups]    # (B, 2^m, 2^m) each
    for layer in range(L):
        if reupload and layer > 0:
            for (q0, m), M in zip(groups, enc_blocks):
                psi = apply_block(psi, M, q0, m, n)
        for (q0, m), M in zip(groups, blocks[layer]):
            psi = apply_block(psi, M, q0, m, n)
        if entangle:
            psi = psi * cz
    return psi


def statevector_reference(theta, phi, entangle=True, reupload=False, cdtype=torch.complex64):
    """Gate-by-gate version (one pass per single-qubit gate); kept for tests and benchmarks."""
    B, n = theta.shape
    psi = product_state(theta, cdtype)
    rdtype = torch.float64 if cdtype == torch.complex128 else torch.float32
    cz = cz_diag(n, theta.device, rdtype) if entangle else None
    enc = ry_matrix(theta) if reupload else None
    gates = ry_rz(phi[..., 0], phi[..., 1])
    for layer in range(phi.shape[0]):
        if reupload and layer > 0:
            for q in range(n):
                psi = apply_1q(psi, enc[:, q], q, n)
        for q in range(n):
            psi = apply_1q(psi, gates[layer, q], q, n)
        if entangle:
            psi = psi * cz
    return psi


def expectations(theta: torch.Tensor, phi: torch.Tensor, k: int, entangle: bool = True,
                 reupload: bool = False, cdtype=torch.complex64) -> torch.Tensor:
    """<Z_j> for j < k, shape (B, k)."""
    psi = statevector(theta, phi, entangle, reupload, cdtype)
    probs = psi.real ** 2 + psi.imag ** 2
    return probs @ z_matrix(theta.shape[1], k, theta.device, probs.dtype)


def probabilities(theta, phi, entangle=True, reupload=False) -> torch.Tensor:
    psi = statevector(theta, phi, entangle, reupload)
    return psi.real ** 2 + psi.imag ** 2


@torch.no_grad()
def sampled_expectations(probs: torch.Tensor, k: int, shots: int, generator: torch.Generator | None = None,
                         chunk: int = 2048) -> torch.Tensor:
    """Shot-noise estimate of <Z_j>, j < k, from exact basis-state probabilities (B, 2^n)."""
    out = []
    weights = torch.as_tensor(1 << np.arange(k), device=probs.device)
    for i in range(0, probs.shape[0], chunk):
        p = probs[i:i + chunk].clamp_min(0)
        idx = torch.multinomial(p, shots, replacement=True, generator=generator)   # (b, shots)
        bits = (idx[..., None] & weights) > 0                                     # (b, shots, k)
        out.append(1.0 - 2.0 * bits.float().mean(dim=1))
    return torch.cat(out)


def encoding_angles(h: torch.Tensor) -> torch.Tensor:
    """Map encoder outputs h in (-1, 1) to RY angles in (0, pi); injective, so <Z> = cos(theta) is too."""
    return (math.pi / 2) * (h + 1.0)
