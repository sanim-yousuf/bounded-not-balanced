import numpy as np
import pytest
import torch

from bbd import qsim

I2 = np.eye(2)
Z = np.diag([1.0, -1.0])


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], complex)


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def on(U, q, n):
    """Dense operator for U on qubit q (qubit 0 = least significant), Qiskit ordering."""
    out = np.array([[1.0 + 0j]])
    for j in reversed(range(n)):
        out = np.kron(out, U if j == q else I2)
    return out


def reference(theta, phi, entangle, reupload):
    n = len(theta)
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1
    for q in range(n):
        psi = on(ry(theta[q]), q, n) @ psi
    cz = qsim._cz_diag_np(n)
    for layer in range(phi.shape[0]):
        if reupload and layer > 0:
            for q in range(n):
                psi = on(ry(theta[q]), q, n) @ psi
        for q in range(n):
            psi = on(rz(phi[layer, q, 1]) @ ry(phi[layer, q, 0]), q, n) @ psi
        if entangle:
            psi = cz * psi
    return np.array([np.real(psi.conj() @ on(Z, q, n) @ psi) for q in range(n)])


@pytest.mark.parametrize("entangle,reupload", [(True, False), (False, False), (True, True)])
def test_matches_dense_reference(entangle, reupload):
    rng = np.random.default_rng(0)
    n, L, B = 5, 3, 4
    th = rng.uniform(0, np.pi, (B, n))
    phi = rng.uniform(-np.pi, np.pi, (L, n, 2))
    got = qsim.expectations(torch.tensor(th), torch.tensor(phi), n, entangle, reupload, torch.complex128).numpy()
    for b in range(B):
        np.testing.assert_allclose(got[b], reference(th[b], phi, entangle, reupload), atol=1e-10)


def test_matches_qiskit():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from bbd.hardware import build_circuit
    rng = np.random.default_rng(1)
    n, L, k = 6, 3, 5
    phi = rng.uniform(-np.pi, np.pi, (L, n, 2))
    th = rng.uniform(0, np.pi, (3, n))
    ours = qsim.expectations(torch.tensor(th), torch.tensor(phi), k, True, False, torch.complex128).numpy()
    qc = build_circuit(phi, k, measure=False)
    for i in range(3):
        sv = Statevector(qc.assign_parameters(th[i]))
        ref = [sv.expectation_value(SparsePauliOp.from_sparse_list([("Z", [j], 1.0)], n)).real for j in range(k)]
        np.testing.assert_allclose(ours[i], ref, atol=1e-10)


def test_gradients_finite_difference():
    torch.manual_seed(0)
    th = torch.rand(2, 4, dtype=torch.float64) * 3
    phi = (torch.rand(2, 4, 2, dtype=torch.float64) - 0.5).requires_grad_()
    f = lambda p: qsim.expectations(th, p, 3, True, False, torch.complex128).sum()
    assert torch.autograd.gradcheck(f, (phi,), eps=1e-6, atol=1e-6)


def test_shot_sampling_unbiased():
    th = torch.rand(64, 4) * 3
    phi = torch.rand(2, 4, 2) - 0.5
    exact = qsim.expectations(th, phi, 4)
    pr = qsim.probabilities(th, phi)
    est = qsim.sampled_expectations(pr, 4, 4096, torch.Generator().manual_seed(0))
    assert float((est - exact).abs().mean()) < 0.02


def test_encoding_range():
    h = torch.linspace(-0.999, 0.999, 11)
    a = qsim.encoding_angles(h)
    assert float(a.min()) > 0 and float(a.max()) < np.pi and torch.all(a[1:] > a[:-1])
