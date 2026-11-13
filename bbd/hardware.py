"""Inference of trained quantum heads on IBM Quantum hardware (ibm_kingston), and the H4 / H5 analysis.

Design (see the research plan, "Hardware experiment")
  * The head is trained on a logical path (CZ on edges (i, i+1)). Any physical 10-qubit path of the
    heavy-hex lattice implements exactly that circuit with zero SWAPs; we pick the best-calibrated one.
  * Layout A maps logical qubit i to path[i]; layout B reverses the path. Every CZ stays in place,
    but each class is read on a different physical qubit. This swap is the causal test of H4.
  * One job carries every PUB (cost probe aside), so all conditions share one calibration window:
      LA model / layout A, LA model / layout B, CE model (post-hoc) / layout A, and two readout
      calibration circuits (all-0, all-1) on the path qubits.
  * Inputs: ~600 test inputs stratified by class + ~50 validation inputs for the affine calibration.
  * Inference conditions, all from the same raw counts:
      sim_exact  exact simulation          sim_shots  simulation with the same shots
      raw        hardware counts           global     one scalar 1/alpha fitted on calibration inputs
      affine     per-qubit (alpha_k, beta_k) fitted on calibration inputs
      readout    tensored readout inversion from the all-0 / all-1 circuits (exact for single-qubit
                 marginals, i.e. what M3 reduces to for <Z_k>)
Execution uses SamplerV2 (qiskit-ibm-runtime 0.51, pinned in notebook 03). The Open Plan allows job
mode, which is what this module uses; job ids are saved so a restarted session can fetch results.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .config import HW_DIR, N_LAYERS, N_QUBITS
from .metrics import evaluate

# ----------------------------------------------------------------------------- connection
def connect(token: str | None = None, instance: str | None = None, channel: str = "ibm_quantum_platform",
            plans_preference=("open",)):
    """Connect to IBM Quantum Platform. Without an explicit instance CRN, the Open (free) plan is preferred,
    so no paid instance is used by accident."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    kw = dict(channel=channel, token=token)
    if instance:
        kw["instance"] = instance
    elif plans_preference:
        kw["plans_preference"] = list(plans_preference)
    return QiskitRuntimeService(**kw)


def describe_account(service, backend=None) -> dict:
    info = {}
    try:
        info["usage"] = service.usage()
    except Exception as e:
        info["usage"] = f"unavailable: {e!r}"
    try:
        info["account"] = {k: v for k, v in service.active_account().items() if k not in ("token",)}
    except Exception:
        pass
    if backend is not None and not is_fake(backend):
        try:
            st = backend.status()
            info["backend"] = dict(name=backend.name, operational=st.operational, pending_jobs=st.pending_jobs)
        except Exception:
            pass
    return info


def get_backend(service=None, name: str = "ibm_kingston", fake: bool = False):
    if fake or service is None:
        from qiskit_ibm_runtime import fake_provider
        cls = {"ibm_kingston": "FakeKingston", "ibm_fez": "FakeFez", "ibm_marrakesh": "FakeMarrakesh",
               "ibm_torino": "FakeTorino"}.get(name, "FakeKingston")
        return getattr(fake_provider, cls)()
    return service.backend(name)


def is_fake(backend) -> bool:
    return type(backend).__name__.startswith("Fake")


# ----------------------------------------------------------------------------- calibration data
def qubit_errors(backend) -> dict:
    """Per-qubit readout error and asymmetry, per-edge CZ error, from the backend target / properties."""
    t = backend.target
    ro = {}
    for (q,), props in t["measure"].items():
        ro[q] = props.error if props is not None and props.error is not None else np.nan
    two = "cz" if "cz" in t.operation_names else ("ecr" if "ecr" in t.operation_names else "cx")
    cz = {}
    for qs, props in t[two].items():
        e = props.error if props is not None and props.error is not None else np.nan
        cz[tuple(sorted(qs))] = min(cz.get(tuple(sorted(qs)), np.inf), e)
    asym = {}
    try:
        p = backend.properties()
        for q in ro:
            try:
                p10 = p.qubit_property(q, "prob_meas1_prep0")[0]
                p01 = p.qubit_property(q, "prob_meas0_prep1")[0]
                asym[q] = (float(p10), float(p01))
            except Exception:
                pass
    except Exception:
        pass
    return dict(readout=ro, cz=cz, asym=asym, two_qubit_gate=two)


def best_path(backend, n: int = N_QUBITS, max_readout: float = 0.08, max_cz: float = 0.05):
    """Best simple path of n qubits: maximise sum log(1-readout) + sum log(1-cz) over the path."""
    err = qubit_errors(backend)
    ro, cz = err["readout"], err["cz"]
    good = {q for q, e in ro.items() if np.isfinite(e) and e < max_readout}
    adj = {q: [] for q in good}
    for (a, b), e in cz.items():
        if a in good and b in good and np.isfinite(e) and e < max_cz:
            adj[a].append((b, math.log1p(-e)))
            adj[b].append((a, math.log1p(-e)))
    node = {q: math.log1p(-ro[q]) for q in good}
    best = (-np.inf, None)

    def dfs(path, score, seen):
        nonlocal best
        if len(path) == n:
            if score > best[0]:
                best = (score, list(path))
            return
        for nb, w in adj[path[-1]]:
            if nb not in seen:
                seen.add(nb)
                path.append(nb)
                dfs(path, score + w + node[nb], seen)
                path.pop()
                seen.discard(nb)

    for q in sorted(good):
        dfs([q], node[q], {q})
    if best[1] is None:
        raise RuntimeError("no qualifying path found; relax max_readout / max_cz")
    path = best[1]
    if path[0] > path[-1]:
        path = path[::-1]
    return path_info(backend, path, err)


def path_info(backend, path, err: dict | None = None) -> dict:
    """Today's error rates along a given path (also used for a resumed job's stored path)."""
    err = err or qubit_errors(backend)
    ro, cz = err["readout"], err["cz"]
    path = [int(q) for q in path]
    r = [float(ro.get(q, np.nan)) for q in path]
    c = [float(cz.get(tuple(sorted(e)), np.nan)) for e in zip(path, path[1:])]
    return dict(path=path, log_fidelity=float(np.sum(np.log1p(-np.array(r + c)))), readout=r, cz=c,
                asym=[err["asym"].get(q) for q in path], two_qubit_gate=err["two_qubit_gate"])


# ----------------------------------------------------------------------------- circuits
def build_circuit(phi: np.ndarray, k: int, entangle: bool = True, reupload: bool = False, measure: bool = True):
    """Qiskit circuit identical to qsim.statevector; encoding angles are a ParameterVector 'theta'."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    L, n, _ = phi.shape
    theta = ParameterVector("theta", n)
    qc = QuantumCircuit(n, k if measure else 0)
    for q in range(n):
        qc.ry(theta[q], q)
    for layer in range(L):
        if reupload and layer > 0:
            for q in range(n):
                qc.ry(theta[q], q)
        for q in range(n):
            qc.ry(float(phi[layer, q, 0]), q)
            qc.rz(float(phi[layer, q, 1]), q)
        if entangle:
            for start in (0, 1):
                for i in range(start, n - 1, 2):
                    qc.cz(i, i + 1)
    if measure:
        qc.measure(range(k), range(k))
    return qc


def calibration_circuits(n: int):
    from qiskit import QuantumCircuit
    zero = QuantumCircuit(n, n, name="cal0")
    zero.measure(range(n), range(n))
    one = QuantumCircuit(n, n, name="cal1")
    one.x(range(n))
    one.measure(range(n), range(n))
    return zero, one


def to_isa(circuit, backend, layout: list[int], optimization_level: int = 1):
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    return pm.run(circuit)


def two_qubit_count(circuit) -> int:
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def check_isa(isa, logical, layout):
    """No SWAPs (same 2-qubit gate count) and measurements land on the intended physical qubits."""
    n2_log, n2_isa = two_qubit_count(logical), two_qubit_count(isa)
    measured = {}
    for inst in isa.data:
        if inst.operation.name == "measure":
            q = isa.find_bit(inst.qubits[0]).index
            c = isa.find_bit(inst.clbits[0]).index
            measured[c] = q
    expected = {c: layout[c] for c in range(len(measured))}
    ok = n2_log == n2_isa and measured == expected
    return dict(ok=bool(ok), two_qubit_logical=n2_log, two_qubit_isa=n2_isa, measured=measured, expected=expected,
                depth=isa.depth())


# ----------------------------------------------------------------------------- model inputs
def stratified(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.flatnonzero(y == c), size=min(per_class, int((y == c).sum())), replace=False)
           for c in np.unique(y)]
    return np.sort(np.concatenate(idx))


def model_inputs(results, run_id: str, n_test: int = 600, n_cal: int = 50, seed: int = 0) -> dict:
    """Encoding angles and exact simulated <Z_k> for hardware test and calibration inputs."""
    import torch
    from . import qsim
    from .data import load_prepared
    from .runner import load_model
    model, spec, ck = load_model(results, run_id, "cpu")
    p = load_prepared(spec.dataset)
    k = p.k
    Xt, yt = p.part("test")
    Xv, yv = p.part("val")
    it = stratified(yt, max(1, n_test // k), seed)
    iv = stratified(yv, max(1, math.ceil(n_cal / k)), seed + 1)
    with torch.no_grad():
        def run(X):
            h = model.encoder(torch.as_tensor(X))
            th = model.head.angles(h)
            ez = qsim.expectations(th.double(), model.head.phi.double(), k, model.head.entangle,
                                   model.head.reupload, cdtype=torch.complex128)
            pr = qsim.probabilities(th, model.head.phi, model.head.entangle, model.head.reupload)
            return th.numpy().astype(np.float64), ez.numpy(), pr
        th_t, ez_t, pr_t = run(Xt[it])
        th_c, ez_c, _ = run(Xv[iv])
    counts = np.asarray(ck["counts"], float)
    return dict(run_id=run_id, spec=spec.to_dict(), k=k, s=float(model.bound), class_names=p.class_names,
                log_pi=np.log(counts / counts.sum()), phi=model.head.phi.detach().double().numpy(),
                entangle=model.head.entangle, reupload=model.head.reupload,
                theta_test=th_t, y_test=yt[it], ez_test_sim=ez_t, probs_test=pr_t,
                theta_cal=th_c, y_cal=yv[iv], ez_cal_sim=ez_c)


def verify_against_qiskit(inp: dict, n_check: int = 8, tol: float = 1e-6) -> float:
    """Max |<Z_k>| difference between our simulator and qiskit.quantum_info.Statevector."""
    from qiskit.quantum_info import SparsePauliOp, Statevector
    qc = build_circuit(inp["phi"], inp["k"], inp["entangle"], inp["reupload"], measure=False)
    n = qc.num_qubits
    ops = [SparsePauliOp.from_sparse_list([("Z", [j], 1.0)], n) for j in range(inp["k"])]
    worst = 0.0
    for i in range(min(n_check, len(inp["theta_test"]))):
        sv = Statevector(qc.assign_parameters(inp["theta_test"][i]))
        ez = np.array([sv.expectation_value(o).real for o in ops])
        worst = max(worst, float(np.abs(ez - inp["ez_test_sim"][i]).max()))
    if worst > tol:
        raise AssertionError(f"simulator and Qiskit disagree: max diff {worst:.2e}")
    return worst


# ----------------------------------------------------------------------------- execution
def _jobs_file() -> Path:
    HW_DIR.mkdir(parents=True, exist_ok=True)
    return HW_DIR / "jobs.json"


def record_job(tag: str, job, meta: dict):
    f = _jobs_file()
    jobs = json.loads(f.read_text()) if f.exists() else {}
    jobs[tag] = dict(job_id=job.job_id(), submitted=time.strftime("%Y-%m-%d %H:%M:%S"), **meta)
    f.write_text(json.dumps(jobs, indent=1, default=str))


def find_job(job_id: str) -> dict:
    """Saved settings of a submitted job (path, order, shots, inputs), from this session's hardware/ folder or
    from an attached output of an earlier session, so a resumed job is analysed exactly as it was submitted."""
    from .config import INPUT_ROOT
    files = [_jobs_file()] + sorted(Path(INPUT_ROOT).glob("**/hardware/jobs.json"))
    for f in files:
        if not f.exists():
            continue
        for tag, meta in json.loads(f.read_text()).items():
            if meta.get("job_id") == job_id:
                cur = _jobs_file()
                jobs = json.loads(cur.read_text()) if cur.exists() else {}
                jobs.setdefault(tag, meta)                  # keep the record in this session's output too
                cur.write_text(json.dumps(jobs, indent=1, default=str))
                return dict(meta, tag=tag)
    raise FileNotFoundError(f"job {job_id} not found in any hardware/jobs.json; attach the output of the notebook "
                            f"version that submitted it")


def make_sampler(backend, dd: bool = True):
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    if dd and not is_fake(backend):
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
    return sampler


def submit(backend, pubs: list, tag: str, meta: dict, dd: bool = True):
    """Run PUBs [(isa_circuit, param_values or None, shots), ...] as one job and record its id."""
    sampler = make_sampler(backend, dd)
    job = sampler.run(pubs)
    record_job(tag, job, meta)
    return job


def job_usage_seconds(job) -> float | None:
    for getter in (lambda: job.usage(), lambda: job.metrics()["usage"]["quantum_seconds"],
                   lambda: job.metrics()["usage"]["seconds"]):
        try:
            v = getter()
            if v is not None:
                return float(v)
        except Exception:
            continue
    return None


def counts_to_z(bitarray, k: int) -> np.ndarray:
    """<Z_j>, j < k, per parameter set, from a SamplerV2 BitArray of shape (N,) or ()."""
    bools = bitarray.to_bool_array(order="little")          # (..., shots, nbits), bit j at index j
    z = 1.0 - 2.0 * bools[..., :k].mean(axis=-2)
    return np.atleast_2d(z)


# ----------------------------------------------------------------------------- planning
def plan_shots(per_shot_seconds: float, n_inputs: int, n_pubs: int, minutes_available: float = 9.0,
               choices=(1024, 512, 256)) -> int:
    for sh in choices:
        if per_shot_seconds * sh * n_inputs * n_pubs <= 60 * minutes_available:
            return sh
    return choices[-1]


# ----------------------------------------------------------------------------- analysis
def _correct(ez_hw_test, ez_hw_cal, ez_sim_cal, ro_a=None, ro_b=None):
    out = {"raw": ez_hw_test}
    alpha_g = float((ez_hw_cal * ez_sim_cal).sum() / max((ez_sim_cal ** 2).sum(), 1e-12))
    out["global"] = ez_hw_test / alpha_g
    k = ez_hw_test.shape[1]
    alpha, beta = np.ones(k), np.zeros(k)
    for j in range(k):
        A = np.vstack([ez_sim_cal[:, j], np.ones(len(ez_sim_cal))]).T
        (a, b), *_ = np.linalg.lstsq(A, ez_hw_cal[:, j], rcond=None)
        alpha[j], beta[j] = a, b
    out["affine"] = (ez_hw_test - beta) / np.where(np.abs(alpha) < 1e-3, 1e-3, alpha)
    if ro_a is not None:
        out["readout"] = np.clip((ez_hw_test - ro_b) / ro_a, -1, 1)
    return out, dict(alpha_global=alpha_g, alpha=alpha.tolist(), beta=beta.tolist())


def _boot_ci(z, y, k, metric, bound, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == c) for c in range(k)]
    vals = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in classes if len(ix)])
        vals.append(evaluate(z[idx], y[idx], k, bound)[metric])
    return tuple(np.quantile(vals, [0.025, 0.975]).tolist())


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    from scipy.stats import binomtest
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def readout_from_cal(z0: np.ndarray, z1: np.ndarray, measured_logical_to_phys: list[int], path: list[int]):
    """Per-logical-qubit readout scale a_j and offset b_j from the all-0 / all-1 circuits on the path.

    z0, z1: <Z> on each path position (n,), from preparing |0..0> and |1..1>.
    <Z>_meas = (1 - e01 - e10) <Z> + (e10 - e01), with e01 = P(1|0) and e10 = P(0|1).
    """
    e01 = (1 - z0) / 2
    e10 = (1 + z1) / 2
    pos = {q: i for i, q in enumerate(path)}
    a = np.array([1 - e01[pos[q]] - e10[pos[q]] for q in measured_logical_to_phys])
    b = np.array([e10[pos[q]] - e01[pos[q]] for q in measured_logical_to_phys])
    return a, b, dict(e01=e01.tolist(), e10=e10.tolist())


def analyse(inputs: dict, hw_z: dict, path: list[int], cal: dict | None, shots: int, posthoc_keys=(),
            out: Path = HW_DIR) -> dict:
    """Build the hardware results dict used by the figures and the H4 / H5 tests.

    inputs   {model_key: model_inputs(...)}
    hw_z     {(model_key, layout): (ez_test (N,k), ez_cal (M,k))}
    cal      {"z0": (n,), "z1": (n,)} readout calibration on the path, or None
    """
    import torch
    from scipy.stats import spearmanr
    from . import qsim
    from .losses import posthoc
    res = dict(path=path, shots=shots, conditions={}, confusion={}, bias=[], qubits=[], fits={}, mcnemar={})
    for (mk, lay), (ez_t, ez_c) in hw_z.items():
        inp = inputs[mk]
        k, s, y = inp["k"], inp["s"], inp["y_test"]
        layout = path if lay == "A" else path[::-1]
        phys = layout[:k]
        ro_a = ro_b = None
        if cal is not None:
            ro_a, ro_b, ro_meta = readout_from_cal(np.asarray(cal["z0"]), np.asarray(cal["z1"]), phys, path)
        conds, fit = _correct(ez_t, ez_c, inp["ez_cal_sim"], ro_a, ro_b)
        g = torch.Generator().manual_seed(7)
        conds = dict(sim_exact=inp["ez_test_sim"],
                     sim_shots=qsim.sampled_expectations(inp["probs_test"], k, shots, g).numpy(), **conds)
        is_ph = mk in posthoc_keys
        name = f"{mk}" + (" + post-hoc" if is_ph else "") + f" / layout {lay}"
        res["conditions"][name] = {}
        correct_sim = None
        for c, ez in conds.items():
            z = s * ez
            zz = posthoc(z, inp["log_pi"], 1.0) if is_ph else z
            m = evaluate(zz, y, k, None if is_ph else s)
            m["macro_f1_ci"] = _boot_ci(zz, y, k, "macro_f1", None)
            m["tail_fpr_ci"] = _boot_ci(zz, y, k, "tail_fpr", None)
            res["conditions"][name][c] = {kk: m[kk] for kk in ("macro_f1", "bacc", "tail_recall", "tail_fpr",
                                                               "never_predicted", "pred_share", "recall",
                                                               "macro_f1_ci", "tail_fpr_ci")}
            corr = zz.argmax(1) == y
            if c == "sim_exact":
                correct_sim = corr
            else:
                res["mcnemar"].setdefault(name, {})[c] = {
                    str(cls): mcnemar(correct_sim[y == cls], corr[y == cls]) for cls in range(k)}
            conf = np.zeros((k, k), int)
            np.add.at(conf, (y, zz.argmax(1)), 1)
            res["confusion"].setdefault(name, {})[c] = conf.tolist()
        res["fits"][name] = fit
        if not is_ph:
            sim_share = np.array(res["conditions"][name]["sim_exact"]["pred_share"])
            raw_share = np.array(res["conditions"][name]["raw"]["pred_share"])
            beta_used = ro_b if ro_b is not None else np.array(fit["beta"])
            for j in range(k):
                res["bias"].append(dict(model=mk, layout=lay, class_index=j, class_name=inp["class_names"][j],
                                        physical=int(phys[j]), beta=float(beta_used[j]),
                                        beta_fit=float(fit["beta"][j]), alpha_fit=float(fit["alpha"][j]),
                                        over_prediction=float(raw_share[j] - sim_share[j])))
            if lay == "A":
                res["qubits"] = [dict(physical=int(q), alpha=float(a), beta=float(b))
                                 for q, a, b in zip(phys, fit["alpha"], beta_used)]
    if res["bias"]:
        b = res["bias"]
        r = spearmanr([x["beta"] for x in b], [x["over_prediction"] for x in b])
        res["spearman"] = dict(r=float(r.statistic), p=float(r.pvalue), n=len(b))
        follow = []
        by = {}
        for x in b:
            by.setdefault((x["model"], x["class_index"]), {})[x["layout"]] = x
        for v in by.values():
            if "A" in v and "B" in v:
                db = v["B"]["beta"] - v["A"]["beta"]
                do = v["B"]["over_prediction"] - v["A"]["over_prediction"]
                follow.append(np.sign(db) == np.sign(do))
        if follow:
            res["follows_qubit"] = dict(fraction=float(np.mean(follow)), n=len(follow))
    if cal is not None:
        res["readout_calibration"] = {"z0": list(map(float, cal["z0"])), "z1": list(map(float, cal["z1"]))}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "hw_results.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def h5_tail_inflation(res: dict) -> dict:
    """H5: post-hoc model predicts the tail more often on raw hardware than in simulation."""
    out = {}
    for name, c in res["conditions"].items():
        if "post-hoc" in name:
            out[name] = dict(tail_pred_sim=c["sim_exact"]["tail_fpr"], tail_pred_raw=c["raw"]["tail_fpr"],
                             tail_pred_global=c.get("global", {}).get("tail_fpr"))
    return out


# ----------------------------------------------------------------------------- revision: hardware session 2
SESSION1_SECONDS_PER_SHOT = 278.0 / 1_024_000      # measured on ibm_kingston, 2026-10-30


def every_other(y: np.ndarray) -> np.ndarray:
    """A stratified half of the session-1 inputs (every other input of each class), so sessions pair."""
    return np.sort(np.concatenate([np.flatnonzero(y == c)[::2] for c in np.unique(y)]))


def subset_inputs(inp: dict, keep: np.ndarray) -> dict:
    out = dict(inp)
    for key in ("theta_test", "y_test", "ez_test_sim"):
        out[key] = inp[key][keep]
    out["probs_test"] = inp["probs_test"][keep]
    out["keep"] = keep
    return out


def remaining_seconds(service) -> float | None:
    try:
        u = service.usage()
        return float(u.get("usage_remaining_seconds"))
    except Exception:
        return None


def plan_session_shots(n_inputs_per_pub: list[int], remaining: float | None, cal_shots: int = 8192,
                       options=(512, 384, 256, 128), safety: float = 0.8, per_shot: float = SESSION1_SECONDS_PER_SHOT):
    """Largest shot count whose estimated QPU time fits safety x remaining quota."""
    for sh in options:
        est = per_shot * (sh * sum(n_inputs_per_pub) + 2 * cal_shots)
        if remaining is None or est <= safety * remaining:
            return sh, est
    return None, per_shot * (options[-1] * sum(n_inputs_per_pub) + 2 * cal_shots)


def _decisions(inp: dict, ez: np.ndarray, is_posthoc: bool) -> np.ndarray:
    from .losses import posthoc
    z = inp["s"] * ez
    return (posthoc(z, inp["log_pi"], 1.0) if is_posthoc else z).argmax(1)


def paired_mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> dict:
    """Exact McNemar test on per-input correctness of two conditions evaluated on the same inputs."""
    ca, cb = pred_a == y, pred_b == y
    return dict(acc_a=float(ca.mean()), acc_b=float(cb.mean()), a_only=int((ca & ~cb).sum()),
                b_only=int((~ca & cb).sum()), p=mcnemar(ca, cb))


def shot_band(inp: dict, shots: int, is_posthoc: bool = False, n_draws: int = 200, seed: int = 0) -> dict:
    """Distribution of macro-F1 under finite-shot sampling alone (no hardware noise): mean and 95% band."""
    import torch
    from . import qsim
    from .losses import posthoc
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_draws):
        ez = qsim.sampled_expectations(inp["probs_test"], inp["k"], shots, g).numpy()
        z = inp["s"] * ez
        zz = posthoc(z, inp["log_pi"], 1.0) if is_posthoc else z
        vals.append(evaluate(zz, inp["y_test"], inp["k"], None if is_posthoc else inp["s"])["macro_f1"])
    v = np.asarray(vals)
    return dict(mean=float(v.mean()), lo=float(np.quantile(v, 0.025)), hi=float(np.quantile(v, 0.975)))


def h5_prediction(inp: dict, alpha: float, tau: float = 1.0) -> dict:
    """Lemma 3 + 2: classes post-hoc adjustment cannot reach once hardware contracts the logits by alpha."""
    from .theory import unreachable_count
    pi = np.exp(inp["log_pi"])
    z = inp["s"] * inp["ez_test_sim"]
    r = float(0.5 * (z.max(1) - z.min(1)).max())
    return dict(r_eff_sim=r, alpha=float(alpha), unreachable_sim=unreachable_count(pi, r, tau),
                unreachable_hw=unreachable_count(pi, alpha * r, tau), threshold=float(tau * np.log(pi.max() / pi.min())),
                two_alpha_r=float(2 * alpha * r))


def draw_circuit(phi_shape=(1, N_QUBITS, 2), k: int = 9, out: Path | None = None):
    """Schematic of the head for the paper: encoding, one variational layer (repeated L times), readout of <Z_k>."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate, ParameterVector
    L, n, _ = phi_shape
    th = ParameterVector("θ", n)
    a = ParameterVector("a", n)
    b = ParameterVector("b", n)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(th[q], q)
    qc.barrier(label="× L")
    for q in range(n):
        qc.ry(a[q], q)
        qc.rz(b[q], q)
    for start in (0, 1):
        for i in range(start, n - 1, 2):
            qc.cz(i, i + 1)
    qc.barrier(label="read")
    readout = Gate("⟨Z⟩", 1, [])
    for j in range(k):
        qc.append(readout, [j])
    fig = qc.draw("mpl", style="bw", fold=-1, scale=0.55)
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(Path(out) / f"fig0_circuit.{ext}", bbox_inches="tight", dpi=300)
    return fig
