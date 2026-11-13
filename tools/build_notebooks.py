"""Generate the five Kaggle notebooks from one place (python tools/build_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"

BOOT = '''# --- bootstrap: find the bbd-code dataset, copy it to a writable folder, import it
import glob, os, shutil, sys
# only the bbd-code dataset counts: attached notebook outputs never contain the code (it is copied to /tmp)
hits = [h for h in sorted(glob.glob("/kaggle/input/**/bbd/__init__.py", recursive=True)) if "/bbd_code/" not in h]
hits = hits or sorted(glob.glob("./bbd/__init__.py"))
if not hits:
    raise FileNotFoundError("Attach the 'bbd-code' dataset to this notebook (KAGGLE_INSTRUCTIONS.md, step 2).")
if len(hits) > 1:
    print("warning: several copies of the code are attached; using", hits[0])
SRC = os.path.dirname(os.path.dirname(os.path.abspath(hits[0])))
CODE = "/tmp/bbd_code" if os.path.isdir("/kaggle/working") else SRC
if CODE != SRC:
    shutil.rmtree(CODE, ignore_errors=True)
    shutil.copytree(SRC, CODE, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results", "work"))
sys.path.insert(0, CODE)
os.environ["PYTHONPATH"] = CODE + os.pathsep + os.environ.get("PYTHONPATH", "")
import bbd
print("bbd", bbd.__version__, "loaded from", SRC)'''

PIP_QISKIT = '''# Pinned: these are the versions the hardware code was tested with (SamplerV2 API).
%pip install -q "qiskit==2.6.1" "qiskit-ibm-runtime==0.51.0" "qiskit-aer==0.18.0" pylatexenc'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


# ============================================================================ 00
N00 = [
("md", """# 00 · Setup and tests

**Settings:** Accelerator *GPU T4 x2* (or None), Internet **on** (installs Qiskit for the hardware tests).
**Inputs:** the `bbd-code` dataset.

This notebook checks the environment, runs the full test suite and benchmarks the simulator on the GPU,
then estimates how many Kaggle GPU-hours the grid will need. It writes nothing that later notebooks need."""),
("code", BOOT),
("code", '''import platform, subprocess, torch, psutil
print("python", platform.python_version(), "| torch", torch.__version__, "| CUDA", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}:", torch.cuda.get_device_name(i))
print("CPUs", os.cpu_count(), "| RAM %.1f GB" % (psutil.virtual_memory().total / 2**30),
      "| disk free %.1f GB" % (shutil.disk_usage("/kaggle/working" if os.path.isdir("/kaggle/working") else ".").free / 2**30))'''),
("code", PIP_QISKIT),
("md", "## Test suite\nSimulator against a dense reference and Qiskit, gradients, Lemma 1 and Lemma 2, metrics, the data pipeline on synthetic files with the real schemas, tuning, grid resume and carry-forward, statistics, every figure, and the hardware pipeline on a simulated `ibm_kingston` (FakeKingston). Takes about 3–6 minutes."),
("code", '''r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore", f"{CODE}/tests"],
                   capture_output=True, text=True, cwd=CODE)
print(r.stdout[-6000:]); print(r.stderr[-3000:])
assert r.returncode == 0, "tests failed - do not start the grid"'''),
("md", "## GPU benchmark and grid estimate"),
("code", '''from bbd import bench
n_workers = max(1, torch.cuda.device_count())
ms, est = bench.report(n_workers=n_workers)
print({k: f"{v:.2f} ms/step" for k, v in ms.items()})
print(f"grid: {est['n_grid']} runs ({est['n_quantum']} quantum) -> about {est['grid_hours']:.1f} h wall-clock on {n_workers} GPU(s)")
print(f"tuning: about {est['tuning_hours']:.1f} h.  Kaggle counts GPU quota per session-hour, so quota used ~ wall-clock.")'''),
]

# ============================================================================ 01
N01 = [
("md", """# 01 · Prepare the datasets

**Settings:** Accelerator *None*, Internet off.
**Inputs:** `bbd-code`, a TON_IoT dataset (the `train_test_network` CSV), and a CICIoT2023 dataset (the CSV part files).

The notebook finds both datasets by their **columns**, not their folder names, so any Kaggle copy with the
original schema works. It drops identifiers that leak the label (timestamps, IPs, ports), removes rows whose
features appear with conflicting labels and exact duplicates, fixes balanced test (300/class) and validation
(100/class) sets, and saves three prepared datasets: `toniot8` (main, 8 classes: without MITM, whose
labels are unreliable, and ransomware, of which only ~1% of rows survive de-duplication), `toniot10` (all classes,
sensitivity check) and `ciciot8` (8 categories). CICIoT2023 is streamed file by file and capped at 60,000
rows per class, sampled uniformly over all files so the rare Web and BruteForce rows are all kept.
Expect about 20–40 minutes for CICIoT2023."""),
("code", BOOT),
("code", '''from bbd import data, config
src = data.find_sources(config.INPUT_ROOT)
for k, v in src.items():
    print(f"{k}: {len(v)} file(s)", *[f"  {p}" for p in v[:3]], "  ..." if len(v) > 3 else "", sep="\\n")
assert src["toniot"], "No TON_IoT file found (needs columns type, src_bytes, conn_state). See KAGGLE_INSTRUCTIONS.md."
assert src["ciciot"], "No CICIoT2023 files found (needs columns label, flow_duration, Protocol Type). See KAGGLE_INSTRUCTIONS.md."'''),
("code", '''import time
t = time.time(); ton = data.prepare_toniot(src["toniot"]); print(f"TON_IoT done in {time.time()-t:.0f}s")
t = time.time(); cic = data.prepare_ciciot(src["ciciot"], cap=60_000); print(f"CICIoT2023 done in {time.time()-t:.0f}s")
prepared = {**ton, **cic}
for p in prepared.values():
    data.save_prepared(p)
print("saved to", config.PREPARED_DIR)'''),
("code", '''import pandas as pd
rows = []
for name, p in prepared.items():
    for i, c in enumerate(p.class_names):
        rows.append(dict(dataset=name, rank=i, cls=c, raw_rows=p.meta["raw_counts"].get(c, 0),
                         survival=round(p.meta["survival"][c], 3), natural_est=int(p.natural_counts[i]),
                         pool=int(p.pool_counts()[i]), val=int(((p.y == i) & (p.split == 1)).sum()),
                         test=int(((p.y == i) & (p.split == 2)).sum())))
summary = pd.DataFrame(rows)
summary.to_csv(config.PREPARED_DIR / "dataset_summary.csv", index=False)
display(summary)
for name, p in prepared.items():
    print(name, "features:", p.d, "| cleaning:", p.meta["dedup"], "| natural rho ~ %.0f" % (p.natural_counts.max() / p.natural_counts.min()))'''),
("code", '''from bbd import figures
from IPython.display import Image, display
figures.fig_datasets([prepared["toniot8"], prepared["ciciot8"]], config.FIG_DIR)
display(Image(str(config.FIG_DIR / "fig_datasets.png"), width=900))'''),
]

# ============================================================================ 02
N02 = [
("md", """# 02 · Run the grid (resumable, two GPUs)

**Settings:** Accelerator **GPU T4 x2**, Internet off, *Save Version → Save & Run All (Commit)*.
**Inputs:** `bbd-code`, the output of notebook 01, and — from the second version on — the output of the
**previous version of this notebook** (Add Input → Your Work → this notebook).

Each committed version runs until its time budget (default 11.3 h, inside Kaggle's 12 h limit), then stops
cleanly. Everything finished is saved under `/kaggle/working/results`. The next version carries those results
forward and continues where the last one stopped; finished runs are never repeated. Order of work:
tuning (8 trials per head and loss) → the two models sent to hardware → seeds 0–4 of the core grid →
references → seeds 5–9 → scale sweep → ablations. You can start the hardware notebook as soon as the first
version has finished (the hardware models run first)."""),
("code", BOOT),
("code", '''import json, subprocess, time, torch
from bbd import config, runner
BUDGET_HOURS = float(os.environ.get("BBD_BUDGET_HOURS", 11.3))   # stop launching runs; Kaggle stops a session at 12 h
TIERS = os.environ.get("BBD_TIERS", "hw,core,sweep,ablation,ref")
N_WORKERS = max(1, torch.cuda.device_count())
START = time.time(); DEADLINE = START + BUDGET_HOURS * 3600
RESULTS = config.RESULTS_DIR; RESULTS.mkdir(parents=True, exist_ok=True)
n = runner.carry_forward(RESULTS, config.INPUT_ROOT)
print(f"carried forward {n} files; workers: {N_WORKERS}")
from bbd.data import find_prepared_dir
for ds in config.DATASETS:
    print(ds, "prepared at", find_prepared_dir(ds))
print(json.dumps(runner.status(RESULTS), indent=1))'''),
("code", '''def launch(stage, deadline=DEADLINE):
    """Start one worker per GPU, stream progress, return when all workers exit."""
    launch_id = f"{stage}-{int(time.time())}"
    (RESULTS / "logs").mkdir(exist_ok=True)
    procs = []
    for i in range(N_WORKERS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i) if torch.cuda.is_available() else "",
                   BBD_LAUNCH=launch_id, BBD_THREADS="2")
        log = open(RESULTS / "logs" / f"{launch_id}_w{i}.log", "w")
        cmd = [sys.executable, "-m", "bbd.runner", stage, "--worker", str(i), "--n-workers", str(N_WORKERS),
               "--deadline", str(deadline), "--tiers", TIERS, "--results", str(RESULTS)]
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=CODE), log))
    last = 0
    while any(p.poll() is None for p, _ in procs):
        time.sleep(30)
        if time.time() - last > 600:
            last = time.time()
            print(time.strftime("%H:%M"), json.dumps(runner.status(RESULTS)), flush=True)
            for i in range(N_WORKERS):
                tail = (RESULTS / "logs" / f"{launch_id}_w{i}.log").read_text().splitlines()[-2:]
                print(f"   w{i}:", *tail, sep="\\n      ")
    for p, log in procs:
        log.close()
    return [p.returncode for p, _ in procs]'''),
("md", "## Tuning (skipped once complete)"),
("code", '''st = runner.status(RESULTS)
if st["tuning"][0] < st["tuning"][1]:
    print("tuning exit codes:", launch("tune"))
hps = runner.best_hyperparameters(RESULTS)
print(len(hps), "tuned (dataset | family | loss) keys")
import pandas as pd
display(pd.DataFrame([dict(key=k, **v) for k, v in sorted(hps.items())]))'''),
("md", "## Grid"),
("code", '''if time.time() < DEADLINE - 600:
    print("grid exit codes:", launch("run"))
print(json.dumps(runner.status(RESULTS), indent=1))'''),
("code", '''errs = [r for r in runner.read_records(RESULTS) if r.get("stage") == "error"]
print(len(errs), "errors")
for r in errs[-5:]:
    print(r["run_id"], r["error"]); print(r["traceback"][-600:])'''),
("md", "## Quick look (sanity check, not the analysis)"),
("code", '''from bbd import stats
df = stats.records_frame(RESULTS)
if len(df):
    q = df[(df.variant.isin(["raw", "posthoc1"])) & (df.tags_str == "")]
    piv = q.pivot_table(index=["dataset", "model", "s_label"], columns=["rho_str", "cond"], values="macro_f1", aggfunc="mean")
    display((100 * piv).round(1))'''),
]

# ============================================================================ 03
N03 = [
("md", """# 03 · Hardware: inference on ibm_kingston

**Settings:** Accelerator *None*, Internet **on**, Add-ons → **Secrets**: `IBM_QUANTUM_TOKEN` (your API key) and,
optionally, `IBM_QUANTUM_INSTANCE` (the CRN of your Open Plan instance).
**Inputs:** `bbd-code`, the output of notebook 01 and the output of notebook 02 (it needs the checkpoints of the
two hardware models, which the grid trains first).

Run it once with `DRY_RUN = True`: it does everything against a simulated `ibm_kingston` (FakeKingston) with no
token and no cost. Then set `DRY_RUN = False`. The real run submits a small **cost probe** first, measures the
QPU seconds per shot from it, and sizes the main job to fit `MINUTES_AVAILABLE` of your Open Plan allowance
(10 minutes per 28 days). The main job carries every circuit (LA model in layouts A and B, the CE model with
post-hoc adjustment in layout A, and two readout-calibration circuits) so all conditions share one calibration
window. Job ids are saved to `/kaggle/working/hardware/jobs.json`; if the queue outlasts the session, set
`RESUME_JOB` to the id and run again."""),
("code", PIP_QISKIT),
("code", BOOT),
("code", '''DRY_RUN = True                 # True: FakeKingston, no token, no cost.  False: the real ibm_kingston
BACKEND = "ibm_kingston"
N_TEST, N_CAL = 600, 50        # stratified test inputs, validation inputs for the affine calibration
MINUTES_AVAILABLE = 8.0        # QPU minutes you are willing to spend on the main job
RESUME_JOB = None              # paste a main-job id to fetch a job submitted in an earlier session (real runs only;
                               # attach that session's output so its hardware/jobs.json is found)
MODELS = {"LA": "toniot8_r100_Q_s3_LA_seed0", "CE": "toniot8_r100_Q_s3_CE_seed0"}
POSTHOC = ("CE",)              # the CE model is evaluated with post-hoc adjustment (H5)
if DRY_RUN:                    # the noisy simulator is slow: a smaller run checks the whole pipeline
    N_TEST, N_CAL = 90, 27
    RESUME_JOB = None'''),
("code", '''import json, time, numpy as np
from bbd import config, hardware as hw, runner, store
RESULTS = config.RESULTS_DIR
runner.carry_forward(RESULTS, config.INPUT_ROOT, kinds=("ledger",))   # artifacts are read from the inputs in place
SAVED = None
if RESUME_JOB:                 # analyse the job exactly as it was submitted
    SAVED = hw.find_job(RESUME_JOB)
    BACKEND, N_TEST, N_CAL, MODELS = SAVED["backend"], SAVED["n_test"], SAVED["n_cal"], SAVED["models"]
    print("resuming", RESUME_JOB, "submitted", SAVED["submitted"], "on", BACKEND, "path", SAVED["path"])
missing = [r for r in MODELS.values() if not store.has(RESULTS, "ckpt", r)]
assert not missing, f"checkpoints missing: {missing}. Attach the output of notebook 02 (its first version trains these)."
if DRY_RUN:
    service, backend = None, hw.get_backend(name=BACKEND, fake=True)
else:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    token = sec.get_secret("IBM_QUANTUM_TOKEN")
    try:
        instance = sec.get_secret("IBM_QUANTUM_INSTANCE")
    except Exception:
        instance = None
    service = hw.connect(token, instance)
    backend = service.backend(BACKEND)
print(backend.name, json.dumps(hw.describe_account(service, backend) if service else {"mode": "dry run"}, default=str, indent=1))'''),
("code", '''inputs = {k: hw.model_inputs(RESULTS, rid, N_TEST, N_CAL) for k, rid in MODELS.items()}
for k, inp in inputs.items():
    print(k, inp["run_id"], "| s =", inp["s"], "| test", len(inp["y_test"]), "| cal", len(inp["y_cal"]),
          "| max |sim - Qiskit| = %.1e" % hw.verify_against_qiskit(inp))'''),
("code", '''bp = hw.best_path(backend) if SAVED is None else hw.path_info(backend, SAVED["path"])   # resumed: the submitted path
path = bp["path"]
print("path:", path); print("readout errors:", np.round(bp["readout"], 4)); print("CZ errors:", np.round(bp["cz"], 4))
circ, isa = {}, {}
for k, inp in inputs.items():
    circ[k] = hw.build_circuit(inp["phi"], inp["k"], inp["entangle"], inp["reupload"])
    for lay, layout in (("A", path), ("B", path[::-1])):
        isa[(k, lay)] = hw.to_isa(circ[k], backend, layout)
        chk = hw.check_isa(isa[(k, lay)], circ[k], layout)
        print(k, lay, {x: chk[x] for x in ("ok", "two_qubit_isa", "depth")})
        assert chk["ok"], chk
c0, c1 = hw.calibration_circuits(len(path))
cal_isa = [hw.to_isa(c0, backend, path), hw.to_isa(c1, backend, path)]
config.HW_DIR.mkdir(parents=True, exist_ok=True)
(config.HW_DIR / "path.json").write_text(json.dumps(dict(bp, backend=backend.name, time=time.ctime()), default=str, indent=1))'''),
("md", "## Cost probe (20 inputs × 1,024 shots)"),
("code", '''if RESUME_JOB is None:
    th = inputs["LA"]["theta_test"][:20]
    probe = hw.submit(backend, [(isa[("LA", "A")], th, 256 if DRY_RUN else 1024)], "probe", dict(n=20, shots=1024))
    print("probe job:", probe.job_id()); pr = probe.result()
    used = hw.job_usage_seconds(probe)
    per_shot = (used / (20 * 1024)) if used else 3e-4
    print(f"usage {used} s  ->  {per_shot * 1e6:.0f} us per shot (fallback 300 us if unreported)")
    n_inputs = len(inputs["LA"]["theta_test"]) + len(inputs["LA"]["theta_cal"])
    SHOTS = 256 if DRY_RUN else hw.plan_shots(per_shot, n_inputs, n_pubs=3, minutes_available=MINUTES_AVAILABLE)
    est_min = per_shot * SHOTS * n_inputs * 3 / 60
    print(f"main job: {SHOTS} shots per input, about {est_min:.1f} QPU minutes")'''),
("md", "## Main job"),
("code", '''if RESUME_JOB is None:
    pubs, order = [], []
    for k, lay in (("LA", "A"), ("LA", "B"), ("CE", "A")):
        inp = inputs[k]
        pubs.append((isa[(k, lay)], np.vstack([inp["theta_test"], inp["theta_cal"]]), SHOTS)); order.append((k, lay))
    pubs += [(cal_isa[0], None, 2048 if DRY_RUN else 8192), (cal_isa[1], None, 2048 if DRY_RUN else 8192)]
    job = hw.submit(backend, pubs, "main", dict(order=order, shots=SHOTS, path=path, backend=backend.name,
                                                n_test=N_TEST, n_cal=N_CAL, models=MODELS, dry_run=DRY_RUN))
    RESUME_JOB = job.job_id()
    print("MAIN JOB ID:", RESUME_JOB, " (saved in hardware/jobs.json)")
else:
    job = service.job(RESUME_JOB)
    order, SHOTS = [tuple(o) for o in SAVED["order"]], SAVED["shots"]
t0 = time.time()
while not DRY_RUN and str(job.status()) not in ("DONE", "JobStatus.DONE", "ERROR", "CANCELLED"):
    print(time.strftime("%H:%M"), job.status()); time.sleep(120)
res = job.result()
print("done after %.0f min; usage %s s" % ((time.time() - t0) / 60, hw.job_usage_seconds(job)))'''),
("code", '''hw_z = {}
for i, (k, lay) in enumerate(order):
    nt = len(inputs[k]["theta_test"])
    z = hw.counts_to_z(res[i].data.c, inputs[k]["k"])
    hw_z[(k, lay)] = (z[:nt], z[nt:])
cal = {"z0": hw.counts_to_z(res[len(order)].data.c, len(path))[0], "z1": hw.counts_to_z(res[len(order) + 1].data.c, len(path))[0]}
out = hw.analyse(inputs, hw_z, path, cal, SHOTS, posthoc_keys=POSTHOC)
import pandas as pd
tab = pd.DataFrame({name: {c: round(v["macro_f1"], 3) for c, v in conds.items()} for name, conds in out["conditions"].items()}).T
display(tab)
print("H4  Spearman(beta_k, over-prediction):", out.get("spearman"))
print("H4  over-prediction follows the qubit across layouts:", out.get("follows_qubit"))
print("H5  tail predictions, post-hoc model:", json.dumps(hw.h5_tail_inflation(out), indent=1))'''),
("code", '''from bbd import figures
from IPython.display import Image, display
for f in (figures.fig_hw_conditions, figures.fig_hw_bias, figures.fig_hw_confusion):
    for p in f(out, config.FIG_DIR):
        if str(p).endswith(".png"): display(Image(str(p), width=900))
hw.draw_circuit((1, 10, 2), inputs["LA"]["k"], config.FIG_DIR)'''),
]

# ============================================================================ 04
N04 = [
("md", """# 04 · Analysis, tables and figures

**Settings:** Accelerator *None*, Internet on (only for the circuit drawing; everything else works offline).
**Inputs:** `bbd-code`, the output of notebook 01, the output of the **latest** version of notebook 02 (it already
contains all earlier versions' results), and the output of notebook 03 if the hardware run is done.

Produces every figure (PDF + 300 dpi PNG, drawn at Springer LNCS size), the LaTeX tables and the
pre-registered hypothesis tests, and zips them as `paper_assets.zip`. It can be run at any point of the grid:
figures whose runs are missing are skipped."""),
("code", PIP_QISKIT),
("code", BOOT),
("code", '''import json, shutil
from pathlib import Path
from bbd import config, figures, runner, stats, tables
RESULTS = config.RESULTS_DIR
runner.carry_forward(RESULTS, config.INPUT_ROOT, kinds=("ledger",))   # artifacts are read from the inputs in place
hw_out = None
for f in sorted(Path(config.INPUT_ROOT).glob("**/hardware/hw_results.json")):
    hw_out = json.loads(f.read_text()); print("hardware results:", f)
print(json.dumps(runner.status(RESULTS), indent=1))
df = stats.records_frame(RESULTS)
print(len(df), "records,", df["run_id"].nunique(), "runs")'''),
("md", "## Pre-registered hypotheses (H1–H3) and the saturation regression"),
("code", '''hyp = stats.hypothesis_table(df)
display(hyp.reindex(columns=["hypothesis", "role", "claim", "test", "n", "est", "ci90", "pi95", "p", "p_holm"]))
print("saturation regression (tail-recall gain vs saturation):", stats.saturation_regression(df))'''),
("code", '''tables.write_all(df, hyp, config.TABLE_DIR)
print(sorted(p.name for p in config.TABLE_DIR.iterdir()))
print((config.TABLE_DIR / "table_main_toniot8.tex").read_text()[:1500])'''),
("md", "## Figures"),
("code", '''shot_df = figures.shot_curve(RESULTS, df)        # CPU simulation from saved checkpoints
made = figures.make_all(df, config.FIG_DIR, RESULTS, hw_out, shot_df)
try:
    from bbd.hardware import draw_circuit
    draw_circuit((1, 10, 2), 9, config.FIG_DIR); print("ok   fig0_circuit")
except Exception as e:
    print("skip fig0_circuit:", e)
from bbd import data
try:
    figures.fig_datasets([data.load_prepared("toniot8"), data.load_prepared("ciciot8")], config.FIG_DIR)
except Exception as e:
    print("skip fig_datasets:", e)'''),
("code", '''from IPython.display import Image, display
for p in sorted(config.FIG_DIR.glob("*.png")):
    print(p.name); display(Image(str(p), width=860))'''),
("code", '''import zipfile
ZIP = config.WORK_ROOT / "paper_assets.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in (config.FIG_DIR, config.TABLE_DIR):
        for p in folder.iterdir():
            z.write(p, f"{folder.name}/{p.name}")
print("wrote", ZIP)'''),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, cells in (("00_setup_and_tests", N00), ("01_prepare_data", N01), ("02_run_grid", N02),
                        ("03_hardware_ibm", N03), ("04_analysis_figures", N04)):
        nbf.write(nb(cells), OUT / f"{name}.ipynb")
        print("wrote", OUT / f"{name}.ipynb")
