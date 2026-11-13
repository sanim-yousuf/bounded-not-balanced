"""Quality-assurance run on synthetic data with the real schemas (not used on Kaggle).

Runs a reduced grid (both datasets, 3 seeds, all heads, CE and LA, three rho) on CPU so the
figures, tables and statistics can be inspected end to end. Resumable: call repeatedly with a time
limit until it reports nothing pending.

  python tools/qa_synthetic.py prepare
  python tools/qa_synthetic.py run --minutes 9
  python tools/qa_synthetic.py hardware
  python tools/qa_synthetic.py figures
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("BBD_QA", "/tmp/bbd_qa"))
os.environ.update(BBD_WORK=str(ROOT / "work"), BBD_INPUT=str(ROOT / "input"), BBD_TRAIN_SIZE="3000",
                  BBD_MAX_EPOCHS="40", BBD_PATIENCE="8", BBD_N_TEST="120", BBD_N_VAL="50", BBD_QGROUP="4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bbd import config, data, runner  # noqa: E402


def grid():
    keep = []
    for s in config.build_grid(("hw", "core", "sweep", "ref")):
        if s.seed > 2 or s.dataset not in ("toniot8", "ciciot8"):
            continue
        if s.tier == "sweep" and s.s != 3.0:
            continue
        keep.append(s)
    return keep


def main():
    cmd = sys.argv[1]
    if cmd == "prepare":
        data.write_synthetic_toniot(ROOT / "input" / "ton" / "train_test_network.csv", n_normal=9000, n_attack=1500,
                                    n_mitm=500)
        data.write_synthetic_ciciot(ROOT / "input" / "cic" / "csv", n_files=4, rows_per_file=12000)
        src = data.find_sources(ROOT / "input")
        for p in {**data.prepare_toniot(src["toniot"]), **data.prepare_ciciot(src["ciciot"], cap=4000)}.values():
            data.save_prepared(p)
        print("prepared")
    elif cmd == "run":
        minutes = float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else 9
        stop = time.time() + 60 * minutes
        res = config.RESULTS_DIR
        done = runner.done_keys(res)
        cache = runner.DataCache("cpu")
        import torch
        torch.set_num_threads(2)
        hp = {"lr": 1e-2, "wd": 0.0}
        todo = [s for s in grid() if s.run_id not in done]
        print(len(todo), "pending")
        n = 0
        for s in todo:
            if time.time() > stop:
                break
            runner.append(res, 0, runner.execute(s, hp, cache, "cpu", res))
            n += 1
        print(n, "done this call;", len(todo) - n, "left")
    elif cmd == "hardware":
        import numpy as np
        from bbd import figures, hardware as hw
        res = config.RESULTS_DIR
        inputs = {"LA": hw.model_inputs(res, "toniot8_r100_Q_s3_LA_seed0", 90, 27),
                  "CE": hw.model_inputs(res, "toniot8_r100_Q_s3_CE_seed0", 90, 27)}
        backend = hw.get_backend(fake=True)
        path = hw.best_path(backend)["path"]
        pubs, order = [], []
        for k, lay in (("LA", "A"), ("LA", "B"), ("CE", "A")):
            inp = inputs[k]
            layout = path if lay == "A" else path[::-1]
            isa = hw.to_isa(hw.build_circuit(inp["phi"], inp["k"]), backend, layout)
            pubs.append((isa, np.vstack([inp["theta_test"], inp["theta_cal"]]), 256))
            order.append((k, lay))
        c0, c1 = hw.calibration_circuits(10)
        pubs += [(hw.to_isa(c0, backend, path), None, 4096), (hw.to_isa(c1, backend, path), None, 4096)]
        r = hw.make_sampler(backend).run(pubs).result()
        hw_z = {}
        for i, (k, lay) in enumerate(order):
            nt = len(inputs[k]["theta_test"])
            z = hw.counts_to_z(r[i].data.c, inputs[k]["k"])
            hw_z[(k, lay)] = (z[:nt], z[nt:])
        cal = {"z0": hw.counts_to_z(r[3].data.c, 10)[0], "z1": hw.counts_to_z(r[4].data.c, 10)[0]}
        out = hw.analyse(inputs, hw_z, path, cal, 256, posthoc_keys=("CE",))
        print({n: {c: round(v["macro_f1"], 3) for c, v in d.items()} for n, d in out["conditions"].items()})
        print(out.get("spearman"), out.get("follows_qubit"))
        hw.draw_circuit((1, 10, 2), 9, config.FIG_DIR)
    elif cmd == "figures":
        import json
        from bbd import figures, stats, tables
        res = config.RESULTS_DIR
        df = stats.records_frame(res)
        hwf = config.HW_DIR / "hw_results.json"
        hw_out = json.loads(hwf.read_text()) if hwf.exists() else None
        hyp = stats.hypothesis_table(df)
        print(hyp[["hypothesis", "n", "est", "p", "p_holm"]])
        tables.write_all(df, hyp, config.TABLE_DIR)
        shot_df = figures.shot_curve(res, df, seeds=range(3), shots=(64, 128, 256, 512, 1024, 2048, 4096))
        figures.make_all(df, config.FIG_DIR, res, hw_out, shot_df)
        figures.fig_datasets([data.load_prepared("toniot8"), data.load_prepared("ciciot8")], config.FIG_DIR)


if __name__ == "__main__":
    main()
