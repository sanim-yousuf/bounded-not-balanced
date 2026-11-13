"""Recompute the manuscript numbers that no table prints and compare each with the paper.

Run from the repository root after unpacking the release assets (see README):

    python tools/check_paper_numbers.py            # writes out/paper_numbers.csv

Without the release assets, rows that need a model checkpoint are reported as missing.
The 512-shot standard deviations also need the prepared TON_IoT arrays from notebook 01.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BBD_WORK", str(ROOT / "work"))
os.environ.setdefault("BBD_INPUT", str(ROOT / "input"))

from bbd.paper_numbers import paper_numbers  # noqa: E402
from bbd.stats import records_frame  # noqa: E402

RES = ROOT / "results"


def main() -> int:
    table = paper_numbers(
        records_frame(RES),
        results=RES,
        rev_dir=RES / "tables",
        hw1=RES / "hardware/session1/hw_results.json",
        hw2=RES / "hardware/session2/hw_results.json",
        hw2_tests=RES / "hardware/session2/hw2_tests.json",
        dataset_summary=RES / "prepared/dataset_summary.csv",
    )
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "paper_numbers.csv", index=False)
    checked = table[table.agrees != "info"]
    agree = int((checked.agrees == "yes").sum())
    print(f"{agree} of {len(checked)} checkable values agree with the paper; table in out/paper_numbers.csv")
    rest = checked[checked.agrees != "yes"]
    if len(rest):
        print(rest[["claim", "value"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
