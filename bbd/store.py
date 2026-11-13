"""Run artifacts (logits, checkpoints) packed into a few zip files instead of thousands of loose files.

Kaggle notebook outputs are kept small in file count: each worker appends its artifacts to
results/packs/<session>_w<i>_<n>.zip, rolling over every PACK_SIZE entries. Readers look in the
current results folder and in every attached `results/` folder under /kaggle/input, so notebooks
03 and 04 can read artifacts straight from attached outputs without copying them.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from .config import INPUT_ROOT, RESULTS_DIR

PACK_SIZE = 250


class PackWriter:
    def __init__(self, results: Path, session: str, worker: int):
        self.dir = Path(results) / "packs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session, self.worker = session, worker
        self.index, self.count = 0, 0
        while self._path().exists():
            self.index += 1

    def _path(self) -> Path:
        return self.dir / f"{self.session}_w{self.worker}_{self.index:03d}.zip"

    def put(self, name: str, payload: bytes):
        if self.count >= PACK_SIZE:
            self.index, self.count = self.index + 1, 0
        with zipfile.ZipFile(self._path(), "a", zipfile.ZIP_STORED) as z:
            z.writestr(name, payload)
        self.count += 1
        _INDEX.clear()                                  # new entries: rebuild the lookup on next read


def npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def torch_bytes(obj) -> bytes:
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


# ----------------------------------------------------------------------------- reading
_INDEX: dict = {}


def roots(results: Path | None = None) -> list[Path]:
    out = [Path(results or RESULTS_DIR)]
    out += [p for p in sorted(Path(INPUT_ROOT).glob("**/results")) if p.is_dir()]
    seen, uniq = set(), []
    for p in out:
        r = p.resolve() if p.exists() else p
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return uniq


def _index(results: Path | None) -> dict:
    key = str(results)
    if key not in _INDEX:
        idx = {}
        for root in roots(results):
            for sub in ("logits", "ckpt"):                       # loose files (older layout, tests)
                d = root / sub
                if d.exists():
                    for f in d.iterdir():
                        idx.setdefault(f"{sub}/{f.name}", ("file", f))
            pk = root / "packs"
            if pk.exists():
                for z in sorted(pk.glob("*.zip")):
                    try:
                        with zipfile.ZipFile(z) as zz:
                            for n in zz.namelist():
                                idx.setdefault(n, ("zip", z))
                    except zipfile.BadZipFile:
                        print(f"warning: unreadable pack {z} skipped")
        _INDEX[key] = idx
    return _INDEX[key]


def read(results: Path | None, name: str) -> bytes | None:
    hit = _index(results).get(name)
    if hit is None:
        return None
    kind, path = hit
    if kind == "file":
        return Path(path).read_bytes()
    with zipfile.ZipFile(path) as z:
        return z.read(name)


def has(results, kind: str, run_id: str) -> bool:
    ext = "npz" if kind == "logits" else "pt"
    return f"{kind}/{run_id}.{ext}" in _index(results)


def load_logits(results, run_id: str):
    b = read(results, f"logits/{run_id}.npz")
    return None if b is None else np.load(io.BytesIO(b))


def load_ckpt(results, run_id: str, device="cpu"):
    import torch
    b = read(results, f"ckpt/{run_id}.pt")
    if b is None:
        raise FileNotFoundError(f"checkpoint {run_id} not found in {roots(results)}")
    return torch.load(io.BytesIO(b), map_location=device, weights_only=False)


def refresh():
    _INDEX.clear()
