"""Recompute the second-revision tables outside Kaggle (same code as notebook 07).

    python tools/revision_round2.py      # needs results/packs from the release assets
"""
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BBD_WORK", str(ROOT / "work"))
os.environ.setdefault("BBD_INPUT", str(ROOT / "input"))

from bbd import round2  # noqa: E402
from bbd.stats import records_frame  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "out" / "revision_round2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = round2.all_tables(records_frame(RES), RES)
    pd.set_option("display.width", 200)
    for name, t in tables.items():
        t.to_csv(OUT / f"{name}.csv", index=False)
        print("=" * 12, name)
        print(t.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
