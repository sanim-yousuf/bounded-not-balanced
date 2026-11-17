# Bounded, Not Balanced

Code, run records and hardware results for the paper

> **Bounded, Not Balanced: What Quantum Classifier Heads Do to Rare-Attack Detection**
> Sanim Yousuf Fahim and Mohammad Arif Hossain

A variational quantum classifier head returns expectation values, so its logits are confined to
[−s, s]. This repository tests what that bound does to rare-class detection on two IoT intrusion
datasets (TON_IoT and CICIoT2023), compares the quantum head with classical heads that share the
same output range, and checks the predictions on an IBM Heron r2 processor (`ibm_kingston`).

## Main results

| Question | Result |
| --- | --- |
| Does the bound rebalance classes? | No. Bounded heads score lower than an unbounded head under cross-entropy; the cause is saturation. |
| When can post-hoc logit adjustment work? | Only if 2r > τ log(π_h/π_min) (Lemma 2). The condition predicts the outcome in 85.3% of 3,002 fixed-scale evaluations. |
| Is the quantum head special? | Classical heads with the same range recover most of its gain. Entanglement adds about one macro-F1 point under logit adjustment. |
| How does it compare with trees? | No neural head reaches LightGBM at ρ = 1000 (91.8 on TON_IoT, 71.6 on CICIoT2023). |
| What happens on hardware? | The per-qubit noise model (Lemma 3) predicts which class is lost and to which class. In the second session it predicted total loss of benign traffic; 30 of 38 benign inputs were lost. |

The paper has the full tables, the statistical tests and the limits of each claim.

## Repository layout

```
bbd/                   the package (module map in bbd/__init__.py)
notebooks/             Kaggle notebooks 00 to 07, without outputs
tests/                 pytest suite (39 tests)
tools/                 notebook builders and check_paper_numbers.py
results/
  ledger/              one JSON line per run and evaluation variant
                       20261027-*: main grid, 2,550 runs (notebook 02)
                       20261030-*: revision runs, 1,690 runs (notebook 05)
  logs/                training and tuning logs of both grids
  tables/              CSV and LaTeX tables (notebook 07)
  hardware/session1/   IBM job dbi9i95ep8hs73andgs0 (notebook 03)
  hardware/session2/   IBM job dbiucdghoj4c73b6urqg (notebooks 06 and 07)
  prepared/            class counts and split metadata (no feature data)
  paper_figures/       the data figures as printed, and paper_numbers.csv
docs/
  KAGGLE_INSTRUCTIONS.md   how to rerun everything on Kaggle
  DESIGN.md                grid sizes and methodology notes
```

## Reproduce the paper from the released records

This needs no GPU, no dataset download and no IBM account.

```bash
git clone https://github.com/<USER>/bounded-not-balanced.git
cd bounded-not-balanced
pip install -r requirements.txt
python -m pytest tests                 # 39 tests, about 5 minutes on a CPU
```

**Figures 2, 4 and 5** (repository only):

```bash
python -m bbd.paper_figures results results/tables \
    results/hardware/session1/hw_results.json \
    results/hardware/session2/hw_results.json \
    out/
```

**Every number the tables do not print** (with the release assets unpacked):

```bash
python tools/check_paper_numbers.py
```

It lists each value beside the one in the paper. All 28 checkable values agree. The 512-shot
standard deviation (Section 6) also needs the prepared TON_IoT arrays, which notebook 01 builds
from the public data; with them, all 29 agree.

**Second-revision tables** (learned scales, reachability baselines, the planned H2a test, H3 by scale and
loss, the operating point, hardware intervals) come from the cell "Second-revision analyses" of notebook 07;
`python tools/revision_round2.py` recomputes them locally with the release assets unpacked. Released copies:
`results/tables/round2/`.

**Correction.** An earlier version of `results/tables/rev_natural_prior.csv` pooled post-hoc adjustment of CE-trained and class-balanced C-host runs. The analysis code now labels the latter `CB+post-hoc`; only the C-host post-hoc rows changed .

## Rerun the experiments

The notebooks are written for Kaggle (GPU T4 x2). `docs/KAGGLE_INSTRUCTIONS.md` gives the order,
the inputs each notebook needs and the expected run time.

| Notebook | Purpose |
| --- | --- |
| 00 | setup, tests, GPU time estimate |
| 01 | data cleaning and splits |
| 02 | main grid (resumable) |
| 03 | hardware session 1 |
| 04 | first analysis and figures |
| 05 | revision runs |
| 06 | hardware session 2 |
| 07 | final statistics, paper figures, paper numbers |

Notebooks 03 and 06 submit jobs to IBM Quantum. They read the API key from Kaggle Secrets
(`IBM_QUANTUM_TOKEN`); never paste a key into a notebook. Rerunning them uses QPU time and gives
new measurements, so the hardware numbers in the paper come from the two recorded jobs above.

## Data

The datasets are not redistributed here. Notebook 01 builds the prepared arrays from the public
Kaggle copies:

| Dataset | Kaggle copy | Rows used |
| --- | --- | --- |
| TON_IoT (network) | `arnobbhowmik/ton-iot-network-dataset` | 211,043 |
| CICIoT2023 | `himadri07/ciciot2023` | 7,845,673 |

Both copies are subsets of the official releases (the TON_IoT copy has 50,000 benign rows; the
CICIoT2023 copy is part of about 46.7 million rows). Please cite the original dataset papers
when you use them.

## Requirements

Python 3 with PyTorch, NumPy, pandas, SciPy, scikit-learn, statsmodels, LightGBM, Matplotlib and
pyarrow. The quantum packages are pinned to the versions used for the paper: `qiskit==2.6.1`,
`qiskit-ibm-runtime==0.51.0`, `qiskit-aer==0.18.0`.

## Citation

```bibtex
@inproceedings{fahim2027bounded,
  title     = {Bounded, Not Balanced: What Quantum Classifier Heads Do to Rare-Attack Detection},
  author    = {Fahim, Sanim Yousuf and Hossain, Mohammad Arif},
  booktitle = {<VENUE, to be filled in after acceptance>},
  year      = {2027}
}
```

## License

Code: MIT (see `LICENSE`). Result files under `results/` and the release assets: CC BY 4.0.

## Acknowledgments

We acknowledge the use of IBM Quantum services for this work. The views expressed are those of
the authors and do not reflect the official policy or position of IBM or the IBM Quantum team.

Questions and bug reports: please open an issue.
