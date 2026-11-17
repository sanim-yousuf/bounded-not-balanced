# Released records

Every file here is an unmodified output of one of the Kaggle notebooks in `notebooks/`.

| Folder | Written by | Contents |
| --- | --- | --- |
| `ledger/` | notebooks 02 and 05 | one JSON line per run and evaluation variant. Files `20261027-*`: main grid (2,550 runs). Files `20261030-*`: revision runs (1,690 runs). |
| `logs/` | notebooks 02 and 05 | tuning and training logs, one per worker. IDs `1793118053`, `1793120604`: main grid. IDs `1793379631`, `1793380173`: revision runs. |
| `tables/` | notebook 07 | main tables, hypothesis tests and every revision analysis (CSV and LaTeX) |
| `hardware/session1/` | notebook 03 | `hw_results.json`, `jobs.json`, `path.json` for IBM job `dbi9i95ep8hs73andgs0` |
| `hardware/session2/` | notebooks 06 and 07 | `hw_results.json`, `hw2_tests.json` for IBM job `dbiucdghoj4c73b6urqg` |
| `prepared/` | notebook 01 | class counts and split metadata for each dataset (no feature data) |
| `paper_figures/` | notebook 07 | the data figures as printed, and `paper_numbers.csv` |
| `tables/round2/` | notebook 07 ("Second-revision analyses") | numbers added in the second revision: reachability baselines, learned scales, planned H2a test, H3 by scale and loss, operating point, hardware intervals |

`paper_numbers.csv` lists each manuscript number that no table prints, beside the value
recomputed from these records. `tools/check_paper_numbers.py` recomputes it.

Use the tables in `tables/`. Notebook 04 also wrote a `table_hypotheses` file during the first
analysis; notebook 07 replaced it with the statistics reported in the paper.

Model checkpoints and per-run logits (`packs/`) are release assets; see the top-level README.
