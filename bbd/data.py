"""TON_IoT and CICIoT2023: discovery, cleaning, fixed splits and controlled-imbalance sampling.

Pipeline (run once, in notebook 01; CPU only)
  1. find the attached datasets under /kaggle/input by their columns, not their folder names
  2. stream the CSV / Parquet files, keeping at most `cap` rows per class with uniform random
     priority keys (so rare classes spread over hundreds of CICIoT2023 part files are all kept)
  3. drop identifiers that leak the label (TON_IoT: timestamps, IPs, ports) and free-text fields
  4. remove conflicting duplicates (same features, different labels), then exact duplicates
  5. order classes by natural frequency (index 0 = most frequent); fixed balanced test and
     validation sets per class (seed 2027); everything else is the training pool
  6. features: sign(x) log1p(|x|), standardised on the pool, clipped to [-6, 6]; one-hot for
     low-cardinality categoricals (top 8 levels + "other")
Each run then draws its own training subset from the pool with `sample_train`.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_SEED, DATASETS, N_TEST, N_VAL, PREPARED_DIR, TRAIN_SIZE, INPUT_ROOT
from .theory import exponential_counts

# ----------------------------------------------------------------------------- schemas
TONIOT_ID_COLS = ["ts", "src_ip", "src_port", "dst_ip", "dst_port"]
TONIOT_TEXT_COLS = ["dns_query", "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent", "http_referrer",
                    "weird_addl", "http_orig_mime_types", "http_resp_mime_types"]
TONIOT_TARGET = "type"
TONIOT_DROP = ["label"]                        # binary label, redundant with `type`

CICIOT_TARGET = "label"
CICIOT_CATEGORIES = ["DDoS", "DoS", "Mirai", "Benign", "Spoofing", "Recon", "Web", "BruteForce"]
_CIC_WEB = {"sqlinjection", "xss", "uploading_attack", "backdoor_malware", "browserhijacking", "commandinjection"}
_CIC_SPOOF = {"mitm-arpspoofing", "dns_spoofing"}


def ciciot_category(label: str) -> str | None:
    """Map one of the 34 CICIoT2023 labels (or an 8-class label) to its category."""
    l = str(label).strip().lower()
    direct = {c.lower(): c for c in CICIOT_CATEGORIES}
    if l in direct:
        return direct[l]
    if l in ("benigntraffic", "benign_traffic"):
        return "Benign"
    if l.startswith("ddos"):
        return "DDoS"
    if l.startswith("dos"):
        return "DoS"
    if l.startswith("mirai"):
        return "Mirai"
    if l.startswith("recon") or l == "vulnerabilityscan":
        return "Recon"
    if l in _CIC_SPOOF:
        return "Spoofing"
    if l in _CIC_WEB:
        return "Web"
    if l == "dictionarybruteforce":
        return "BruteForce"
    return None


def _norm(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip())


# ----------------------------------------------------------------------------- discovery
def _header(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        return [_norm(c) for c in pq.read_schema(path).names]
    with open(path, "r", errors="ignore") as f:
        return [_norm(c) for c in f.readline().strip().split(",")]


def identify(cols: list[str]) -> str | None:
    low = {c.lower() for c in cols}
    if {"type", "src_bytes", "conn_state"} <= low:
        return "toniot"
    if "label" in low and ({"flow_duration", "protocol type"} <= low or {"flow_duration", "header_length"} <= low):
        return "ciciot"
    return None


def find_sources(root: Path | str = INPUT_ROOT) -> dict[str, list[Path]]:
    """Locate TON_IoT and CICIoT2023 files under `root` by their columns."""
    found: dict[str, list[Path]] = {"toniot": [], "ciciot": []}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith((".csv", ".parquet")):
                continue
            p = Path(dirpath) / fn
            try:
                kind = identify(_header(p))
            except Exception:
                continue
            if kind:
                found[kind].append(p)
    # TON_IoT: prefer the curated train_test_network file when present
    tt = [p for p in found["toniot"] if "train_test" in p.name.lower()]
    if tt:
        found["toniot"] = tt[:1]
    for k in found:
        found[k] = sorted(found[k])
    return found


# ----------------------------------------------------------------------------- streaming reader
def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        try:
            import pyarrow.csv as pacsv
            df = pacsv.read_csv(path, read_options=pacsv.ReadOptions(block_size=1 << 26)).to_pandas()
        except Exception:
            df = pd.read_csv(path, low_memory=False)
    df.columns = [_norm(c) for c in df.columns]
    return df


def stream_capped(paths: list[Path], label_fn, cap: int | None, seed: int = DATA_SEED, verbose: bool = True):
    """Read files one by one, keep <= cap rows per class chosen uniformly at random over ALL files.

    Returns (kept DataFrame with column '_y' = class name, raw per-class counts over all files).
    """
    rng = np.random.default_rng(seed)
    kept = None
    raw_counts: dict[str, int] = {}
    for i, p in enumerate(paths):
        df = _read_table(p)
        y = label_fn(df)
        ok = y.notna()
        if (~ok).any():
            warnings.warn(f"{p.name}: {int((~ok).sum())} rows with unmapped labels dropped")
        df = df.loc[ok].copy()
        df["_y"] = y[ok].values
        for c, n in df["_y"].value_counts().items():
            raw_counts[c] = raw_counts.get(c, 0) + int(n)
        if cap is not None:
            df["_u"] = rng.random(len(df))
            kept = df if kept is None else pd.concat([kept, df], ignore_index=True)
            kept = (kept.sort_values("_u").groupby("_y", sort=False, group_keys=False).head(cap))
        else:
            kept = df if kept is None else pd.concat([kept, df], ignore_index=True)
        if verbose and (i % 10 == 0 or i == len(paths) - 1):
            print(f"  [{i + 1}/{len(paths)}] {p.name}: kept {len(kept):,} rows")
    kept = kept.drop(columns=["_u"], errors="ignore").reset_index(drop=True)
    return kept, raw_counts


# ----------------------------------------------------------------------------- cleaning
def dedup(df: pd.DataFrame, feature_cols: list[str], label_col: str = "_y", round_decimals: int | None = None):
    """Drop rows whose features appear with more than one label, then exact duplicates."""
    feats = df[feature_cols]
    if round_decimals is not None:
        num = feats.select_dtypes(include=[np.number]).columns
        feats = feats.copy()
        feats[num] = feats[num].round(round_decimals)
    h = pd.util.hash_pandas_object(feats, index=False).values
    tmp = pd.DataFrame({"h": h, "y": df[label_col].values})
    n_labels = tmp.groupby("h")["y"].nunique()
    conflict = tmp["h"].map(n_labels).values > 1
    dup = pd.Series(h).duplicated(keep="first").values
    keep = ~conflict & ~dup
    stats = {"rows_in": int(len(df)), "conflicting_rows": int(conflict.sum()),
             "duplicate_rows": int((dup & ~conflict).sum()), "rows_out": int(keep.sum())}
    return df.loc[keep].reset_index(drop=True), stats


def encode_features(df: pd.DataFrame, feature_cols: list[str], pool_mask: np.ndarray, top_levels: int = 8):
    """Numeric: signed log1p, standardised on the pool; categorical: one-hot of the top levels."""
    blocks, names = [], []
    for c in feature_cols:
        col = df[c]
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().mean() > 0.98:                         # numeric column
            v = num.fillna(0.0).to_numpy(np.float64)
            v = np.sign(v) * np.log1p(np.abs(v))
            mu, sd = v[pool_mask].mean(), v[pool_mask].std()
            if sd < 1e-12:
                continue                                       # constant on the pool: no information
            blocks.append(np.clip((v - mu) / sd, -6, 6)[:, None])
            names.append(c)
        else:                                                   # categorical column
            s = col.astype(str).fillna("nan")
            levels = s[pool_mask].value_counts().index[:top_levels].tolist()
            if len(levels) <= 1:
                continue
            for lv in levels:
                blocks.append((s.values == lv).astype(np.float64)[:, None])
                names.append(f"{c}={lv}")
            other = ~s.isin(levels).values
            if other.any():
                blocks.append(other.astype(np.float64)[:, None])
                names.append(f"{c}=other")
    return np.hstack(blocks).astype(np.float32), names


def fixed_splits(y: np.ndarray, k: int, n_test: int = N_TEST, n_val: int = N_VAL, seed: int = DATA_SEED):
    """0 = training pool, 1 = validation, 2 = test. Balanced and identical for every run."""
    rng = np.random.default_rng(seed)
    split = np.zeros(len(y), dtype=np.uint8)
    notes = {}
    for c in range(k):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        t = min(n_test, int(0.3 * len(idx)))
        v = min(n_val, int(0.1 * len(idx)))
        if t < n_test or v < n_val:
            notes[c] = f"class {c} has {len(idx)} rows: test {t}, val {v}"
            warnings.warn(notes[c])
        split[idx[:t]] = 2
        split[idx[t:t + v]] = 1
    return split, notes


# ----------------------------------------------------------------------------- prepared data
@dataclass
class Prepared:
    name: str
    X: np.ndarray
    y: np.ndarray
    split: np.ndarray
    class_names: list
    natural_counts: np.ndarray           # estimated per-class counts in the full data after cleaning
    feature_names: list
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.class_names)

    @property
    def d(self) -> int:
        return self.X.shape[1]

    def part(self, which: str):
        code = {"pool": 0, "val": 1, "test": 2}[which]
        m = self.split == code
        return self.X[m], self.y[m]

    def pool_counts(self) -> np.ndarray:
        return np.bincount(self.y[self.split == 0], minlength=self.k)


def save_prepared(p: Prepared, out_dir: Path = PREPARED_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{p.name}.npz", X=p.X, y=p.y, split=p.split)
    meta = dict(p.meta, name=p.name, class_names=p.class_names, natural_counts=p.natural_counts.tolist(),
                feature_names=p.feature_names, n=int(len(p.y)), d=int(p.d))
    (out_dir / f"{p.name}.json").write_text(json.dumps(meta, indent=1))


def find_prepared_dir(name: str) -> Path:
    """Look in /kaggle/working first, then in any attached notebook output."""
    for base in [PREPARED_DIR, *sorted(Path(INPUT_ROOT).glob("**/prepared"))]:
        if (Path(base) / f"{name}.npz").exists():
            return Path(base)
    raise FileNotFoundError(f"prepared dataset {name!r} not found; run notebook 01 and attach its output")


def load_prepared(name: str, base: Path | None = None) -> Prepared:
    base = Path(base) if base else find_prepared_dir(name)
    arr = np.load(base / f"{name}.npz")
    meta = json.loads((base / f"{name}.json").read_text())
    return Prepared(name, arr["X"], arr["y"].astype(np.int64), arr["split"], meta["class_names"],
                    np.asarray(meta["natural_counts"]), meta["feature_names"], meta)


def _finish(name, df, feature_cols, raw_counts, dedup_stats, source_files, drop_classes, extra_meta) -> Prepared:
    df = df[~df["_y"].isin(drop_classes)].reset_index(drop=True)
    kept_counts = df["_y"].value_counts()
    # natural count of the full data after cleaning = raw count x per-class survival rate of cleaning
    surv = extra_meta.get("survival", {})
    nat = {c: raw_counts.get(c, 0) * surv.get(c, 1.0) for c in kept_counts.index}
    order = sorted(nat, key=lambda c: (-nat[c], c))            # index 0 = most frequent
    natural = np.array([nat[c] for c in order])
    cls_index = {c: i for i, c in enumerate(order)}
    y = df["_y"].map(cls_index).to_numpy(np.int64)
    split, notes = fixed_splits(y, len(order))
    X, names = encode_features(df, feature_cols, split == 0)
    meta = dict(source_files=[str(p) for p in source_files], raw_counts=raw_counts, dedup=dedup_stats,
                kept_counts={c: int(kept_counts[c]) for c in order}, split_notes=notes,
                dropped_classes=list(drop_classes), **{k: v for k, v in extra_meta.items() if k != "survival"},
                survival={c: float(surv.get(c, 1.0)) for c in order})
    return Prepared(name, X, y, split, order, natural, names, meta)


def prepare_toniot(paths: list[Path], cap: int | None = None) -> dict[str, Prepared]:
    def label_fn(df):
        col = [c for c in df.columns if c.lower() == TONIOT_TARGET][0]
        return df[col].astype(str).str.strip().str.lower()

    df, raw = stream_capped(paths, label_fn, cap)
    lower = {c.lower(): c for c in df.columns}
    drop = [lower[c] for c in TONIOT_ID_COLS + TONIOT_TEXT_COLS + TONIOT_DROP + [TONIOT_TARGET] if c in lower]
    features = [c for c in df.columns if c not in drop and c != "_y"]
    n_before = df["_y"].value_counts()
    df, st = dedup(df, features)
    survival = (df["_y"].value_counts() / n_before).fillna(0).to_dict()
    out = {}
    for name, (src, drop_classes) in DATASETS.items():
        if src == "toniot":
            out[name] = _finish(name, df, features, raw, st, paths, drop_classes,
                                {"dataset": "TON_IoT", "dropped_columns": drop, "survival": survival})
    return out


def prepare_ciciot(paths: list[Path], cap: int = 60_000) -> dict[str, Prepared]:
    def label_fn(df):
        col = [c for c in df.columns if c.lower() == CICIOT_TARGET][0]
        return df[col].map(ciciot_category)

    df, raw = stream_capped(paths, label_fn, cap)
    features = [c for c in df.columns if c.lower() != CICIOT_TARGET and c != "_y"]
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64).fillna(0.0)
    n_before = df["_y"].value_counts()
    df, st = dedup(df, features, round_decimals=6)
    survival = (df["_y"].value_counts() / n_before).fillna(0).to_dict()
    return {name: _finish(name, df, features, raw, st, paths, drop,
                          {"dataset": "CICIoT2023", "cap_per_class": cap, "survival": survival,
                           "split_note": "merged part files carry no capture id; split is random after deduplication"})
            for name, (src, drop) in DATASETS.items() if src == "ciciot"}


# ----------------------------------------------------------------------------- per-run sampling
def train_counts(p: Prepared, rho, total: int = TRAIN_SIZE) -> np.ndarray:
    if rho == "nat":
        share = p.natural_counts / p.natural_counts.sum()
        target = np.maximum(np.round(share * total).astype(int), 5)
    else:
        target = exponential_counts(total, p.k, float(rho))
    pool = p.pool_counts()
    capped = target > pool
    if capped.any() and (p.name, str(rho), total) not in _CAP_WARNED:
        _CAP_WARNED.add((p.name, str(rho), total))
        names = ", ".join(f"{p.class_names[c]} ({target[c]} -> {pool[c]})" for c in np.flatnonzero(capped))
        warnings.warn(f"{p.name} rho={rho}: pool too small for {names}; the realised imbalance (actual_rho in the "
                      f"ledger) differs from the target", stacklevel=2)
    return np.minimum(target, pool)


_CAP_WARNED: set = set()


def sample_train(p: Prepared, rho, seed: int, total: int = TRAIN_SIZE):
    """Indices (into p.X) of one training subset, and its per-class counts."""
    counts = train_counts(p, rho, total)
    rng = np.random.default_rng(10_007 * seed + (0 if rho == "nat" else int(rho)))
    pool = p.split == 0
    idx = []
    for c in range(p.k):
        cand = np.flatnonzero(pool & (p.y == c))
        idx.append(rng.choice(cand, size=counts[c], replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx, counts


# ----------------------------------------------------------------------------- synthetic data (tests, dry runs)
def write_synthetic_toniot(path: Path, n_normal: int = 6000, n_attack: int = 800, n_mitm: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    types = ["normal"] + ["backdoor", "ddos", "dos", "injection", "password", "ransomware", "scanning", "xss"] + ["mitm"]
    sizes = [n_normal] + [n_attack] * 8 + [n_mitm]
    rows = []
    for ti, (t, n) in enumerate(zip(types, sizes)):
        loc = rng.normal(size=6) * 1.5 + ti * 0.3
        num = np.abs(rng.normal(loc, 1.0, size=(n, 6))) * 100
        df = pd.DataFrame({
            "ts": rng.integers(1.5e9, 1.6e9, n), "src_ip": [f"192.168.1.{i % 250}" for i in range(n)],
            "src_port": rng.integers(1, 65535, n), "dst_ip": "10.0.0.1", "dst_port": rng.integers(1, 1024, n),
            "proto": rng.choice(["tcp", "udp", "icmp"], n, p=[0.6, 0.3, 0.1]),
            "service": rng.choice(["-", "http", "dns", "ssl"], n), "duration": num[:, 0] / 100,
            "src_bytes": num[:, 1].astype(int), "dst_bytes": num[:, 2].astype(int),
            "conn_state": rng.choice(["S0", "SF", "REJ", "OTH"], n), "missed_bytes": 0,
            "src_pkts": num[:, 3].astype(int), "src_ip_bytes": num[:, 4].astype(int),
            "dst_pkts": num[:, 5].astype(int), "dst_ip_bytes": (num[:, 4] * 0.5).astype(int),
            "dns_query": "-", "dns_qclass": 0, "dns_qtype": rng.integers(0, 3, n), "dns_rcode": 0,
            "dns_AA": "-", "dns_RD": "-", "dns_RA": "-", "dns_rejected": "-", "ssl_version": "-",
            "ssl_cipher": "-", "ssl_resumed": "-", "ssl_established": "-", "ssl_subject": "-", "ssl_issuer": "-",
            "http_trans_depth": "-", "http_method": rng.choice(["-", "GET", "POST"], n), "http_uri": "-",
            "http_version": "-", "http_request_body_len": 0, "http_response_body_len": 0, "http_status_code": 0,
            "http_user_agent": "-", "http_orig_mime_types": "-", "http_resp_mime_types": "-",
            "weird_name": "-", "weird_addl": "-", "weird_notice": "-",
            "label": int(t != "normal"), "type": t})
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df = pd.concat([df, df.sample(200, random_state=seed)], ignore_index=True)     # exact duplicates
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sample(frac=1, random_state=seed).to_csv(path, index=False)
    return path


def write_synthetic_ciciot(folder: Path, n_files: int = 4, rows_per_file: int = 8000, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = ["flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
            "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
            "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
            "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv",
            "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius",
            "Covariance", "Variance", "Weight"]
    labels = (["DDoS-ICMP_Flood"] * 40 + ["DoS-UDP_Flood"] * 20 + ["Mirai-greeth_flood"] * 12 + ["BenignTraffic"] * 8
              + ["DNS_Spoofing"] * 6 + ["Recon-PortScan"] * 5 + ["SqlInjection", "XSS"] * 2 + ["DictionaryBruteForce"] * 3)
    cats = sorted({ciciot_category(l) for l in labels})
    centers = {c: rng.normal(size=len(cols)) * 1.2 for c in cats}
    folder.mkdir(parents=True, exist_ok=True)
    for f in range(n_files):
        lab = rng.choice(labels, rows_per_file)
        X = np.vstack([np.abs(rng.normal(centers[ciciot_category(l)], 1.0)) * 10 for l in lab])
        df = pd.DataFrame(X, columns=cols)
        df["label"] = lab
        df.to_csv(folder / f"part-{f:05d}-synthetic.csv", index=False)
    return folder
