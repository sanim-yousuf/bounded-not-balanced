"""Generate the three revision notebooks (05, 06, 07). Run after tools/build_notebooks.py.

  05_revision_runs        GPU: the `revision` tier (controls, Q-noent at small s, lr sweep, capped validation)
  06_hardware_session2    IBM: replication, a real test of Lemma 3 (H5), entanglement-free circuit
  07_revision_analysis    corrected statistics and every revision analysis; writes revision_assets.zip
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BOOT, N02, OUT, PIP_QISKIT, nb  # noqa: E402

LAUNCH = [c for c in N02 if c[0] == "code" and c[1].startswith("def launch")]
assert len(LAUNCH) == 1

# ============================================================================ 05
N05 = [
("md", """# 05 · Revision runs (peer-review controls)

**Settings:** Accelerator **GPU T4 x2**, Internet off, *Save Version → Save & Run All (Commit)*.
**Inputs:** `bbd-code` (the revised version), the output of notebook **01**, and the output of notebook **02**
(its ledgers mark the original runs and tuning as done, so nothing is repeated).

Runs only the `revision` tier, about 3 GPU-hours on two T4s:

| Block | What | Runs | Answers |
| --- | --- | --- | --- |
| R0 | Q at s = 2.35 (CE) and Q-noent at s = 3 (LA), TON_IoT, rho = 100, seed 0 | 2 | models for hardware session 2 (notebook 06) |
| R1 | Q-noent at s = 1, 1.5 and learned; both datasets, all rho, CE and LA, 10 seeds | 360 | is Q equivalent to Q-noent where the Q - C-bounded gap is largest? |
| R2 | classical readout controls C-sin-diag, C-diag, C-sin, C-bounded-init at s = 1, 2, 4; 5 seeds | 720 | what in Q-noent beats s tanh(Vh): the sinusoid, the diagonal readout or the initial gain? |
| R3 | learning-rate sweep (4 values, wd 0) for Q, Q-noent, C-bounded at s = 1, 4; rho = 100; seeds 1-3 | 288 | is the Q - C-bounded gap a tuning artefact? |
| R4 | early stopping on a validation set capped at each class's training count; rho = 100, 1000; 5 seeds | 200 | does the balanced validation set inflate rare-class results? |
| R5 | unbounded head without output bias (C-host-nobias), 10 seeds | 120 | does H1 confound the bound with the output bias? |

Tuning first runs 8 trials for each new head family (240 short classical trials)."""),
("code", BOOT),
("code", '''import json, subprocess, time, torch
from bbd import config, runner
BUDGET_HOURS = float(os.environ.get("BBD_BUDGET_HOURS", 11.3))
TIERS = "revision"
N_WORKERS = max(1, torch.cuda.device_count())
START = time.time(); DEADLINE = START + BUDGET_HOURS * 3600
RESULTS = config.RESULTS_DIR; RESULTS.mkdir(parents=True, exist_ok=True)
n = runner.carry_forward(RESULTS, config.INPUT_ROOT, kinds=("ledger",))   # done-markers only; packs stay in 02's output
print(f"carried forward {n} ledger files; workers: {N_WORKERS}")
from bbd.data import find_prepared_dir
for ds in config.DATASETS:
    print(ds, "prepared at", find_prepared_dir(ds))
print(json.dumps(runner.status(RESULTS, tiers=("revision",)), indent=1))'''),
] + LAUNCH + [
("md", "## Tuning of the new head families (skipped once complete)"),
("code", '''st = runner.status(RESULTS, tiers=("revision",))
if st["tuning"][0] < st["tuning"][1]:
    print("tuning exit codes:", launch("tune"))
hps = runner.best_hyperparameters(RESULTS)
import pandas as pd
new = {k: v for k, v in hps.items() if any(m in k for m in ("C-sin", "C-diag", "C-bounded-init", "C-host-nobias"))}
display(pd.DataFrame([dict(key=k, **v) for k, v in sorted(new.items())]))'''),
("md", "## Revision grid"),
("code", '''if time.time() < DEADLINE - 600:
    print("grid exit codes:", launch("run"))
print(json.dumps(runner.status(RESULTS, tiers=("revision",)), indent=1))
errs = [r for r in runner.read_records(RESULTS) if r.get("stage") == "error"]
print(len(errs), "errors")
for r in errs[-5:]:
    print(r["run_id"], r["error"]); print(r["traceback"][-600:])'''),
("md", "## Quick look (the full analysis is notebook 07)"),
("code", '''from bbd import stats, revision
df = stats.records_frame(RESULTS)
display(revision.pooled_grid(df, "Q", "Q-noent").round(2))
ctl = revision.controls_table(df)
if len(ctl):
    display(ctl[(ctl.b == "C-bounded") & (ctl.s == "all")].round(2))'''),
]

# ============================================================================ 06
N06 = [
("md", """# 06 · Hardware session 2: replication and a real test of Lemma 3

**Settings:** Accelerator *None*, Internet **on**, Secrets `IBM_QUANTUM_TOKEN` (and optionally `IBM_QUANTUM_INSTANCE`).
**Inputs:** `bbd-code` (revised), the outputs of notebooks **01**, **02**, **03** (session 1: its `jobs.json`
holds the qubit path and the job id) and **05** (the two new hardware models).

One job, sized to the QPU time you have left: the notebook reads your remaining Open Plan seconds and picks
512, 384, 256 or 128 shots (512 shots need about 3.4 QPU-minutes). It uses the session-1 qubit path and 304 of
the 600 session-1 test inputs (every other input of each class, 38 per class) plus the 56 calibration inputs:

* the LA model on layout A and on layout B again: replication, a paired test of A against B, and a paired test of
  session 1 against session 2 on the same inputs;
* a post-hoc model at s = 2.35, just above the rho = 100 threshold. Lemma 3 predicts that a measured
  contraction alpha < 0.98 makes the most frequent class unreachable on raw hardware and that the global rescale
  restores it, so H5 becomes a real test;
* the entanglement-free circuit (Q-noent) on layout A, to separate CZ-gate noise from readout noise.

Run once with `DRY_RUN = True` (simulated kingston, no token, no cost), then with `DRY_RUN = False`.
If the queue outlasts the session, commit, attach this notebook's output and set `RESUME_JOB`."""),
("code", PIP_QISKIT),
("code", BOOT),
("code", '''DRY_RUN = True
BACKEND = "ibm_kingston"
SESSION1_JOB = "dbi9i95ep8hs73andgs0"   # main job of notebook 03 (session 1)
RESUME_JOB = None                      # a session-2 job id, to fetch a job submitted earlier (real runs only)
MODELS = {"LA": "toniot8_r100_Q_s3_LA_seed0",          # replication, layouts A and B
          "CE235": "toniot8_r100_Q_s2.35_CE_seed0",    # H5 test (post-hoc), layout A
          "QNE": "toniot8_r100_Q-noent_s3_LA_seed0"}   # entanglement-free, layout A
POSTHOC = ("CE235",)
SAFETY = 0.8                           # use at most 80% of the remaining QPU seconds
if DRY_RUN:
    RESUME_JOB = None'''),
("code", '''import json, time, numpy as np, pandas as pd
from bbd import config, hardware as hw, runner, store
RESULTS = config.RESULTS_DIR
runner.carry_forward(RESULTS, config.INPUT_ROOT, kinds=("ledger",))
missing = [r for r in MODELS.values() if not store.has(RESULTS, "ckpt", r)]
assert not missing, f"checkpoints missing: {missing}. Attach the outputs of notebooks 02 and 05."
S1 = None
try:
    S1 = hw.find_job(SESSION1_JOB)          # path and settings of session 1
    print("session 1:", S1["submitted"], "path", S1["path"], "shots", S1["shots"])
except FileNotFoundError as e:
    print("session 1 record not found (attach the output of notebook 03):", e)
SAVED = hw.find_job(RESUME_JOB) if RESUME_JOB else None
if DRY_RUN:
    service, backend = None, hw.get_backend(name=BACKEND, fake=True)
    remaining = None
else:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    try:
        instance = sec.get_secret("IBM_QUANTUM_INSTANCE")
    except Exception:
        instance = None
    service = hw.connect(sec.get_secret("IBM_QUANTUM_TOKEN"), instance)
    backend = service.backend(BACKEND)
    remaining = hw.remaining_seconds(service)
print(backend.name, "| remaining QPU seconds:", remaining)'''),
("code", '''full = {k: hw.model_inputs(RESULTS, rid, 600, 50) for k, rid in MODELS.items()}
keep = hw.every_other(full["LA"]["y_test"])
if DRY_RUN:
    keep = keep[::4]                        # the noisy simulator is slow; a quarter checks the whole pipeline
inputs = {k: hw.subset_inputs(v, keep) for k, v in full.items()}
for k, inp in inputs.items():
    print(k, inp["run_id"], "| s =", inp["s"], "| test", len(inp["y_test"]), "| cal", len(inp["y_cal"]),
          "| max |sim - Qiskit| = %.1e" % hw.verify_against_qiskit(inp))
if SAVED:
    path = list(SAVED["path"])
elif S1 is not None and not DRY_RUN:
    path = list(S1["path"])                 # same physical qubits as session 1
else:
    path = hw.best_path(backend)["path"]
bp = hw.path_info(backend, path)
print("path:", path); print("readout errors today:", np.round(bp["readout"], 4)); print("CZ errors today:", np.round(bp["cz"], 4))
pubs_spec = [("LA", "A"), ("LA", "B"), ("CE235", "A"), ("QNE", "A")]
isa = {}
for k, lay in pubs_spec:
    layout = path if lay == "A" else path[::-1]
    circ = hw.build_circuit(inputs[k]["phi"], inputs[k]["k"], inputs[k]["entangle"], inputs[k]["reupload"])
    isa[(k, lay)] = hw.to_isa(circ, backend, layout)
    chk = hw.check_isa(isa[(k, lay)], circ, layout)
    print(k, lay, {x: chk[x] for x in ("ok", "two_qubit_isa", "depth")}); assert chk["ok"], chk
c0, c1 = hw.calibration_circuits(len(path))
cal_isa = [hw.to_isa(c0, backend, path), hw.to_isa(c1, backend, path)]
config.HW_DIR.mkdir(parents=True, exist_ok=True)
(config.HW_DIR / "path_session2.json").write_text(json.dumps(dict(bp, backend=backend.name, time=time.ctime()), default=str, indent=1))'''),
("code", '''n_per_pub = [len(inputs[k]["theta_test"]) + len(inputs[k]["theta_cal"]) for k, _ in pubs_spec]
if SAVED:
    SHOTS = SAVED["shots"]
elif DRY_RUN:
    SHOTS = 128
else:
    SHOTS, est = hw.plan_session_shots(n_per_pub, remaining, cal_shots=8192, safety=SAFETY)
    assert SHOTS is not None, f"not enough QPU time left: need about {est:.0f} s, have {remaining} s"
    print(f"{SHOTS} shots per input, estimated {est:.0f} QPU seconds of {remaining:.0f} remaining")'''),
("code", '''if RESUME_JOB is None:
    pubs = [(isa[(k, lay)], np.vstack([inputs[k]["theta_test"], inputs[k]["theta_cal"]]), SHOTS) for k, lay in pubs_spec]
    cal_shots = 2048 if DRY_RUN else 8192
    pubs += [(cal_isa[0], None, cal_shots), (cal_isa[1], None, cal_shots)]
    job = hw.submit(backend, pubs, "session2", dict(order=pubs_spec, shots=SHOTS, path=path, backend=backend.name,
                                                    n_test=600, n_cal=50, keep=keep.tolist(), models=MODELS,
                                                    dry_run=DRY_RUN))
    RESUME_JOB = job.job_id(); print("SESSION-2 JOB ID:", RESUME_JOB, " (saved in hardware/jobs.json)")
else:
    job = service.job(RESUME_JOB)
while not DRY_RUN and str(job.status()) not in ("DONE", "JobStatus.DONE", "ERROR", "CANCELLED"):
    print(time.strftime("%H:%M"), job.status()); time.sleep(120)
res = job.result(); print("usage", hw.job_usage_seconds(job), "s")'''),
("code", '''hw_z = {}
for i, (k, lay) in enumerate(pubs_spec):
    nt = len(inputs[k]["theta_test"])
    z = hw.counts_to_z(res[i].data.c, inputs[k]["k"])
    hw_z[(k, lay)] = (z[:nt], z[nt:])
n = len(pubs_spec)
cal = {"z0": hw.counts_to_z(res[n].data.c, len(path))[0], "z1": hw.counts_to_z(res[n + 1].data.c, len(path))[0]}
OUT2 = config.HW_DIR / "session2"
out = hw.analyse(inputs, hw_z, path, cal, SHOTS, posthoc_keys=POSTHOC, out=OUT2)
display(pd.DataFrame({nm: {c: round(v["macro_f1"], 3) for c, v in cs.items()} for nm, cs in out["conditions"].items()}).T)'''),
("md", "## Tests"),
("code", '''rep = {"shots": SHOTS, "job": RESUME_JOB}
y = inputs["LA"]["y_test"]
# 1. shot-noise band: is the hardware loss larger than finite sampling alone explains?
rep["shot_band"] = {f"{k}/{lay}": hw.shot_band(inputs[k], SHOTS, k in POSTHOC) for k, lay in pubs_spec}
# 2. paired layout test: same session, same inputs
pa = hw._decisions(inputs["LA"], hw_z[("LA", "A")][0], False)
pb = hw._decisions(inputs["LA"], hw_z[("LA", "B")][0], False)
rep["layout_A_vs_B"] = hw.paired_mcnemar(pa, pb, y)
# 3. Lemma 3 / H5: classes predicted unreachable after the measured contraction, against what happened
alpha = out["fits"]["CE235 + post-hoc / layout A"]["alpha_global"]
rep["h5_prediction"] = hw.h5_prediction(inputs["CE235"], alpha)
c = out["conditions"]["CE235 + post-hoc / layout A"]
rep["h5_observed"] = {cond: dict(never_predicted=c[cond]["never_predicted"], tail_fpr=c[cond]["tail_fpr"],
                                 recall_most_frequent=c[cond]["recall"][0], macro_f1=c[cond]["macro_f1"])
                      for cond in ("sim_exact", "raw", "global", "readout")}
# 4. entanglement-free circuit: hardware loss with and without CZ gates, same layout
cq, cn = out["conditions"]["LA / layout A"], out["conditions"]["QNE / layout A"]
rep["cz_vs_readout"] = dict(Q_loss=cq["sim_exact"]["macro_f1"] - cq["raw"]["macro_f1"],
                            Qnoent_loss=cn["sim_exact"]["macro_f1"] - cn["raw"]["macro_f1"])
print(json.dumps(rep, indent=1, default=float))'''),
("code", '''# 5. session 1 against session 2 on the same 304 inputs (fetching a finished job costs no QPU time)
if not DRY_RUN and S1 is not None:
    r1 = service.job(SESSION1_JOB).result()
    order1 = [tuple(o) for o in S1["order"]]
    for lay in ("A", "B"):
        i = order1.index(("LA", lay))
        z1 = hw.counts_to_z(r1[i].data.c, full["LA"]["k"])[:600][keep]
        p1 = hw._decisions(inputs["LA"], z1, False)
        p2 = hw._decisions(inputs["LA"], hw_z[("LA", lay)][0], False)
        rep[f"session1_vs_2_layout_{lay}"] = hw.paired_mcnemar(p1, p2, y)
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("session")}, indent=1))
OUT2.mkdir(parents=True, exist_ok=True)
(OUT2 / "hw2_tests.json").write_text(json.dumps(rep, indent=1, default=float))
print("saved", OUT2 / "hw2_tests.json")'''),
]

# ============================================================================ 07
N07 = [
("md", """# 07 · Revision analysis: corrected statistics, controls, sensitivity checks

**Settings:** Accelerator *None*, Internet on.
**Inputs:** `bbd-code` (revised) and the outputs of notebooks **01**, **02**, **03**, **05** and **06**.

* Hypothesis table with small-sample t inference, between-cell SD and 95% prediction intervals; H2a paired within
  baseline runs; every row labelled primary, pre-specified follow-up or exploratory.
* Reachability predicted *before* the test set is seen (nominal s; validation-set range).
* H1 decomposition (saturated against unsaturated inputs) and the unbounded head without output bias.
* Q against Q-noent at every scale; classical readout controls; learning-rate sweep; capped validation;
  near-duplicate (leakage) audit; benign false-alarm rates at the natural prior; selected hyperparameters.
* All main figures redrawn with the corrected code, plus the revision figures. Writes `revision_assets.zip`."""),
("code", PIP_QISKIT),
("code", BOOT),
("code", '''import json
from pathlib import Path
import pandas as pd
from bbd import config, figures, revision, runner, stats, tables
RESULTS = config.RESULTS_DIR
runner.carry_forward(RESULTS, config.INPUT_ROOT, kinds=("ledger",))
hw_out = None
for f in sorted(Path(config.INPUT_ROOT).glob("**/hardware/hw_results.json")):
    hw_out = json.loads(f.read_text()); print("hardware session 1:", f)
df = stats.records_frame(RESULTS)
print(len(df), "records,", df["run_id"].nunique(), "runs")'''),
("md", "## Hypotheses (corrected inference)"),
("code", '''hyp = stats.hypothesis_table(df)
pd.set_option("display.width", 220)
display(hyp.reindex(columns=["hypothesis", "role", "claim", "test", "method", "n", "n_cells", "est", "ci90", "ci95",
                             "pi95", "tau", "p", "p_holm"]))
tables.write_all(df, hyp, config.TABLE_DIR)
hyp.to_csv(config.TABLE_DIR / "table_hypotheses_full.csv", index=False)
base = df[(df.tags_str == "") & (df.rho_str != "nat") & df.dataset.isin(["toniot8", "ciciot8"])]
cells = stats.cell_table(stats.paired(base, {"model": "Q", "variant": "raw", "loss": ["CE", "LA"], "s_label": stats.FIXED_S},
                                      {"model": "Q-noent", "variant": "raw"}))
cells.to_csv(config.TABLE_DIR / "rev_q_vs_qnoent_cells.csv", index=False)
print("Q - Q-noent cells inside / outside the 1-point margin:", int(cells.inside_margin.sum()), "/",
      int(cells.outside_margin.sum()), "of", len(cells))'''),
("md", "## Revision analyses"),
("code", '''rev = revision.run_all(df, RESULTS, config.TABLE_DIR, config.FIG_DIR)
for name in ("reachability_pretest", "h1_decomposition", "h1_nobias", "q_vs_qnoent", "lr_sweep", "valcap", "leakage"):
    v = rev.get(name)
    print("=" * 20, name)
    for frame in (v if isinstance(v, tuple) else (v,)):
        if isinstance(frame, pd.DataFrame) and len(frame):
            display(frame.round(3))'''),
("code", '''ctl = rev.get("controls")
if isinstance(ctl, pd.DataFrame) and len(ctl):
    display(ctl[ctl.s == "all"].round(2))
npt = rev.get("natural_prior")
if isinstance(npt, pd.DataFrame) and len(npt):
    display(npt.round(3))'''),
("md", "## Hardware session 2"),
("code", '''for f in sorted(Path(config.INPUT_ROOT).glob("**/hardware/session2/hw2_tests.json")):
    print(f); print(f.read_text()[:4000])
for f in sorted(Path(config.INPUT_ROOT).glob("**/hardware/session2/hw_results.json")):
    h2 = json.loads(f.read_text())
    display(pd.DataFrame({nm: {c: round(v["macro_f1"], 3) for c, v in cs.items()} for nm, cs in h2["conditions"].items()}).T)'''),
("md", "## Figures (main figures redrawn with the corrected code, plus revision figures)"),
("code", '''shot_df = figures.shot_curve(RESULTS, df)
made = figures.make_all(df, config.FIG_DIR, RESULTS, hw_out, shot_df)
from IPython.display import Image, display
for p in sorted(config.FIG_DIR.glob("*.png")):
    print(p.name); display(Image(str(p), width=860))'''),
("code", '''import zipfile
ZIP = config.WORK_ROOT / "revision_assets.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in (config.FIG_DIR, config.TABLE_DIR):
        for p in folder.iterdir():
            z.write(p, f"{folder.name}/{p.name}")
    for f in Path(config.INPUT_ROOT).glob("**/hardware/session2/*.json"):
        z.write(f, f"hardware_session2/{f.name}")
print("wrote", ZIP)'''),
]

if __name__ == "__main__":
    for name, cells in (("05_revision_runs", N05), ("06_hardware_session2", N06), ("07_revision_analysis", N07)):
        nbf.write(nb(cells), OUT / f"{name}.ipynb")
        print("wrote", OUT / f"{name}.ipynb")
