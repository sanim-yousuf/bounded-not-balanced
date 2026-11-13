# Running *Bounded, Not Balanced* on Kaggle

Everything runs on Kaggle: four dataset uploads/attachments, eight notebooks (00–04 for the paper, 05–07 for the peer-review revision), and one IBM Quantum API key.
The grid is 2,550 runs (1,090 quantum) plus 384 tuning trials. Notebook 00 measures the T4 speed and prints the
GPU-hours the grid needs before you commit to it. For scale: on a 2-core CPU the same grid needs about 47 hours,
and a T4 is several times faster for the batched simulator. The hardware part costs about 4 QPU-minutes of IBM's
free Open Plan.

| Notebook | Accelerator | Internet | Inputs to attach | Time | What it writes |
| --- | --- | --- | --- | --- | --- |
| `00_setup_and_tests` | GPU T4 x2 | on | `bbd-code` | ~10 min | nothing needed later (test log, GPU estimate) |
| `01_prepare_data` | None | off | `bbd-code`, TON_IoT, CICIoT2023 | 20–40 min | `prepared/` (3 datasets) |
| `02_run_grid` | **GPU T4 x2** | off | `bbd-code`, output of 01, previous output of 02 | ≤11.5 h per version, as many versions as notebook 00 estimates | `results/` (ledgers, artifact packs) |
| `03_hardware_ibm` | None | on | `bbd-code`, outputs of 01 and 02, Secrets | 15 min + IBM queue | `hardware/` |
| `04_analysis_figures` | None | on | `bbd-code`, outputs of 01, 02 (latest), 03 | 10–20 min | `figures/`, `tables/`, `paper_assets.zip` |

---

## Step 1 — IBM Quantum API key (needed only for notebook 03)

1. Sign in at <https://quantum.cloud.ibm.com>.
2. On the dashboard, create an **API key** (it is shown once; copy it).
3. Open **Instances** and copy the **CRN** of your *Open Plan* instance. It is optional: without it the code asks
   IBM to prefer the Open (free) plan (`plans_preference=["open"]`), so a paid instance is never picked by accident.
4. On the Open Plan you have 10 QPU-minutes per rolling 28 days. `ibm_kingston` is available to Open Plan users.

## Step 2 — Upload the code as a Kaggle dataset

1. Kaggle → **Datasets → New Dataset** → upload `bbd_kaggle.zip` → title **`bbd-code`** → *Private* → **Create**.
   Kaggle unpacks the zip; the notebooks find the code anywhere under `/kaggle/input`.
2. To change the code later: open the dataset → **New Version** → upload the new zip. Notebooks pick up the
   new version after you re-attach it (Input panel → ⋮ → *Update version*).

## Step 3 — Find the two datasets on Kaggle

Attach them with **Add Input** in notebook 01. Notebook 01 identifies files by their **columns**, so any copy
with the original schema works, and it prints exactly which files it found.

**TON_IoT (network, train/test file).** Used: `arnobbhowmik/ton-iot-network-dataset`, file
`train_test_network.csv`, 211,043 rows (50,000 normal; 20,000 per attack class; 1,043 MITM). The widely cited
version of the file has 461,043 rows (300,000 normal) with the same attack rows; either works, because the
controlled-imbalance grid needs at most ~12,000 normal rows. State the 211,043-row version in the paper.

**CICIoT2023.** Used: `himadri07/ciciot2023`, a pre-split copy (`CICIOT23/{train,validation,test}/*.csv`,
about 7.8 M rows, the class proportions of the full 46.7 M-row release). Notebook 01 merges the three files and
makes its own splits. Copies with the original `part-*.csv` files work the same way.
Any copy with the original 34 labels or the 8 category labels works. If a copy renames columns, notebook 01
stops with a message listing what it expected.

## Step 4 — Import the notebooks

For each `.ipynb` in `notebooks/`: Kaggle → **Create → New Notebook → File → Import Notebook** → upload.
Then, in the right-hand panel, set **Accelerator** and **Internet** as in the table above, and attach the inputs
(**Add Input** → *Your Work* for your own datasets and notebook outputs).

## Step 5 — Run, in this order

### 00 · Setup and tests (interactive)
Attach `bbd-code`, GPU T4 x2, Internet on, **Run All**. All tests must pass. The last cell prints the measured
ms/step on the T4 and the estimated grid hours. If the estimate is far above 30 h, reduce the tiers in
notebook 02 (see *Tuning the budget* below).

### 01 · Prepare data (commit)
Attach `bbd-code`, the TON_IoT dataset and the CICIoT2023 dataset. Accelerator None, Internet off.
Click **Save Version → Save & Run All (Commit)**. Its output (`/kaggle/working/prepared`) is what notebook 02
reads. Check the printed class table: `toniot8` has 8 classes with *normal* first (no MITM, no ransomware), `toniot10` has 10, `ciciot8` has 8 with *DDoS* first.

### 02 · Run the grid (commit, repeat until done)
1. Attach `bbd-code` and **the output of notebook 01** (Add Input → Your Work → Notebooks → 01_prepare_data).
2. Accelerator **GPU T4 x2**, Internet off. **Save Version → Save & Run All (Commit)**.
3. The version stops cleanly at 11.3 h (edit `BUDGET_HOURS`). It runs tuning first (384 short trials), then the
   two hardware models, then the grid by priority.
4. When it finishes: open the notebook again, **Add Input → Your Work → Notebooks → 02_run_grid** (its latest
   version's output), and commit again. The new version copies the earlier results and continues; nothing
   finished is repeated. Repeat until the status cell shows every tier complete.
5. Kaggle's quota meter (top of the editor) shows hours left this week. Each committed version uses up to its
   session time.
6. The output stays small in file count: ledgers are one JSONL file per worker per version, and logits and
   checkpoints are packed into zip files of 250 runs each (`results/packs/`). The whole grid is a few dozen
   files and roughly 1–2 GB.

### 03 · Hardware (after the first version of 02)
1. Attach `bbd-code`, the outputs of 01 and 02. Accelerator None, Internet **on**.
2. **Add-ons → Secrets → Add a new secret**: label `IBM_QUANTUM_TOKEN`, value = your API key. Optionally add
   `IBM_QUANTUM_INSTANCE` = your Open Plan CRN. Tick *Attach* for this notebook.
3. Run with `DRY_RUN = True` first: the whole pipeline runs on a simulated `ibm_kingston` with no token and no cost.
4. Set `DRY_RUN = False` and **Run All** interactively (not as a commit, so you can watch the queue).
   - It verifies the simulator against Qiskit, picks the best 10-qubit path on kingston from today's calibration,
     checks the transpiled circuits have zero SWAPs, then runs a 20-input **cost probe** and sizes the main job to
     `MINUTES_AVAILABLE`.
   - The main job id is printed and saved to `hardware/jobs.json`, together with the qubit path, shot count and
     inputs it was submitted with. Open Plan jobs can queue for hours. If the session ends first, commit the
     notebook (so `jobs.json` is saved), attach that version's output to the notebook, set `RESUME_JOB = "<id>"`
     and run again. Nothing is re-submitted, and the result is analysed with the stored path and settings even if
     today's calibration would pick another path.
5. Commit the notebook afterwards (Save Version) so `hardware/hw_results.json` and the figures are kept as output.

### 04 · Analysis and figures (any time; final run at the end)
Attach `bbd-code`, the output of 01, the output of the **latest** version of 02 (it already contains all
earlier versions) and the output of 03. Accelerator None. Run All. Download `paper_assets.zip` from the Output tab.

---

## Step 6 — Revision runs for the peer review (notebooks 05 to 07)

These add the controls and checks the review asked for. Nothing from notebooks 01–04 is re-run.

**Before you start:** open `bbd-code` → **New Version**, delete every old file and folder, upload only the new
`bbd_kaggle.zip`, and save. Then, in each notebook below, update the `bbd-code` input to that version
(Input panel → ⋮ → *Update version*). The dataset must hold one copy of the code.

### 05 · Revision runs (commit, about 3 h on GPU T4 x2)
Attach `bbd-code`, the output of **01** and the output of **02**. Accelerator **GPU T4 x2**, Internet off.
**Save Version → Save & Run All (Commit)**. It tunes the five new head families (240 short trials), then runs
1,690 runs (634 quantum) in this order: the two hardware models → Q-noent at s = 1, 1.5, learned → classical
readout controls and the unbiased host → learning-rate sweep and capped-validation runs. If it stops at the
time budget, attach this notebook's own output as well and commit again; finished runs are not repeated.

### 06 · Hardware session 2 (interactive, then commit)
Attach `bbd-code` and the outputs of **01**, **02**, **03** and **05**. Accelerator None, Internet **on**, the
`IBM_QUANTUM_TOKEN` secret attached (as for notebook 03).
1. Run with `DRY_RUN = True` first (simulated kingston, no cost).
2. Set `DRY_RUN = False` and Run All. The notebook reads your remaining Open Plan seconds and sizes the job to 80%
   of them (512 shots need about 3.4 QPU-minutes; it falls back to 384, 256 or 128 shots). It stops with a
   message if there is not enough time left; then wait for the 28-day window to roll and run it later.
3. It uses session 1's qubit path (from notebook 03's `jobs.json`) and 304 of the same test inputs, so the two
   sessions can be compared input by input. Fetching session 1's results again costs no QPU time.
4. Commit afterwards so `hardware/session2/` is saved. If the queue outlasts the session, commit, attach this
   notebook's output, set `RESUME_JOB` to the printed id and run again.

### 07 · Revision analysis (any time after 05; final run after 06)
Attach `bbd-code` and the outputs of **01**, **02**, **03**, **05** and **06**. Accelerator None. Run All.
Download `revision_assets.zip` (tables `rev_*.csv`, `table_hypotheses_full.csv`, all figures redrawn, and the
session-2 hardware results).

### What each review point maps to

| Review point | Where it is answered |
| --- | --- |
| "Observed ≥ predicted" is true by construction | `rev_reachability_pretest_*.csv`: prediction from nominal s and from validation logits |
| Q-noent not run where the gap is largest | 05 block R1; `rev_q_vs_qnoent.csv`, `rev_q_vs_qnoent_cells.csv`, H3-ent-all row |
| Gap confounded with parameterisation / initialisation | 05 block R2; `rev_controls.csv`, `rev_fig_scale_gap` |
| Tuning at one seed and one scale | 05 block R3; `rev_lr_sweep_*.csv`, `rev_fig_lr_sweep` |
| H1 narrated as agreement; output-bias confound | `rev_h1_decomposition.csv`, `rev_h1_nobias.csv` (block R5) |
| Statistics (normal quantiles, 2-cell H2b, unpaired H2a, hidden heterogeneity) | `stats.pooled` (t, df = cells − 1, tau, prediction interval), paired H2a, `cell_table`, role column |
| Balanced validation set | 05 block R4; `rev_valcap_*.csv` |
| CICIoT2023 / TON_IoT near-duplicates | `rev_leakage_*.csv` (audit, and comparisons recomputed without near-duplicates) |
| Deployment metric | `rev_natural_prior.csv` (benign false-alarm rate, natural-prior precision/F1) |
| Hyperparameters not reported | `rev_hyperparameters.csv` |
| Single hardware session; H5 could not trigger; CZ vs readout | notebook 06: replication, paired A/B and session tests, s = 2.35 post-hoc model, Q-noent on hardware, shot-noise band |

## What the figures are

| File | Content |
| --- | --- |
| `fig0_circuit` | one layer of the quantum head (symbolic angles) |
| `fig_datasets` | natural class sizes after cleaning, and the controlled training profiles |
| `fig1_reachability` | (a) Lemma 2 phase diagram; (b) predicted vs observed never-predicted classes on every run |
| `fig2_main` | macro-F1 vs ρ for every head under CE, CE + post-hoc and LA-loss, both datasets |
| `fig3_collapse` | gap to C-host against the normalised range 2r_eff/(τ log ρ): post-hoc collapses below 1, LA does not |
| `fig4_forest` | paired differences with the ±1-point TOST band: Q − C-bounded, Q − Q-noent, C-bounded − C-host |
| `fig5_tail_tradeoff` | tail recall against tail false-positive rate |
| `fig6_saturation` | tail-recall gain of bounded heads against their saturation rate (H1) |
| `fig7_perclass_*` | per-class recall heat maps at ρ = 100 |
| `fig8_calibration` | ECE against the logit bound; reliability diagrams |
| `fig9_range_shots` | effective vs nominal range; decisions changed by shot noise vs shots |
| `fig10_optimisation` | best epoch per head and validation curves (optimisation-budget check) |
| `fig_hw_conditions` | macro-F1 and tail FPR: simulation, raw hardware, global, affine, readout inversion |
| `fig_hw_bias` | H4: over-prediction vs readout offset for both layouts; per-qubit α and β |
| `fig_hw_confusion` | confusion matrices: simulation, raw hardware, affine-corrected |

All figures are drawn at the Springer LNCS text width (122 mm) with 8 pt type, saved as vector PDF (embedded
TrueType fonts) and 300 dpi PNG, with one colour per head (Okabe–Ito, colour-blind safe) in every figure.

## Tuning the budget

- Fewer tiers: in notebook 02 set `TIERS = "hw,core,ref"` (core grid + references only, about 60% of the cost),
  then add `sweep` and `ablation` in a later version.
- Smoke test of the whole notebook in minutes: before the imports in notebook 02, set
  `os.environ.update(BBD_TRAIN_SIZE="2000", BBD_MAX_EPOCHS="5")` and `TIERS = "hw"`. Remove it for the real run,
  and delete `/kaggle/working/results` (or start a fresh notebook) so smoke results are not carried forward.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Attach the 'bbd-code' dataset` | Add Input → Your Work → Datasets → bbd-code |
| `No TON_IoT file found` / `No CICIoT2023 files found` | the attached copy has different column names; try another copy from Step 3 |
| `prepared dataset ... not found` in 02/03/04 | attach the **output of notebook 01** (it must have been committed) |
| `checkpoints missing` in 03 | attach the output of notebook 02; its first version trains the two hardware models first |
| `job ... not found in any hardware/jobs.json` | attach the output of the notebook 03 version that submitted the job |
| `pool too small for ...` warning | a rare class has fewer pool rows than the target count; the run continues with the realised ratio (`actual_rho`) |
| `unreadable pack ... skipped` | a session was killed mid-write; the runs in that pack lose their logits (metrics in the ledger are intact) |
| tests fail in 00 | do not start the grid; the failing test name says which part |
| `no tuning result for ...` in a worker log | a tuning trial failed; the worker uses defaults (lr 3e-3); see the errors cell |
| IBM job stays `QUEUED` | normal on the Open Plan; note the job id, set `RESUME_JOB`, come back later |
| `plans_preference` / instance errors | add the `IBM_QUANTUM_INSTANCE` secret with your Open Plan CRN |
| SamplerV2 deprecation warning | expected with qiskit-ibm-runtime 0.51; versions are pinned in the notebooks |
