"""Per-run evaluation.

Classes are indexed by natural-frequency rank (0 = most frequent), so head / medium / tail
are the first, middle and last thirds of the class indices.
"""
from __future__ import annotations

import numpy as np


def splits_by_rank(k: int) -> dict:
    a, b = int(np.ceil(k / 3)), int(np.ceil(2 * k / 3))
    return {"head": list(range(0, a)), "medium": list(range(a, b)), "tail": list(range(b, k))}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(err)


def reliability(probs: np.ndarray, y: np.ndarray, bins: int = 15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((0.5 * (lo + hi), conf[m].mean(), correct[m].mean(), int(m.sum())))
    return np.array(rows)


def evaluate(z: np.ndarray, y: np.ndarray, k: int, bound: float | None = None) -> dict:
    """All per-run metrics from logits z (N, K) and labels y (N,)."""
    z = np.asarray(z, dtype=np.float64)
    pred = z.argmax(1)
    conf = np.zeros((k, k), dtype=np.int64)
    np.add.at(conf, (y, pred), 1)
    support = conf.sum(1)
    predicted = conf.sum(0)
    tp = np.diag(conf).astype(float)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    groups = splits_by_rank(k)
    tail = np.array(groups["tail"])
    non_tail = ~np.isin(y, tail)
    probs = softmax(z)
    spread = (z.max(1) - z.min(1)) / 2.0
    out = {
        "macro_f1": float(f1.mean()),
        "bacc": float(recall.mean()),
        "acc": float((pred == y).mean()),
        "head_recall": float(recall[groups["head"]].mean()),
        "medium_recall": float(recall[groups["medium"]].mean()) if groups["medium"] else float("nan"),
        "tail_recall": float(recall[tail].mean()),
        "tail_fpr": float(np.isin(pred[non_tail], tail).mean()) if non_tail.any() else float("nan"),
        "never_predicted": int((predicted == 0).sum()),
        "r_eff": float(spread.max()),          # effective half-range reached on this set
        "r_eff95": float(np.quantile(spread, 0.95)),
        "ece": ece(probs, y),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()),
        "recall": recall.round(5).tolist(),
        "precision": precision.round(5).tolist(),
        "pred_share": (predicted / max(len(y), 1)).round(5).tolist(),
    }
    if bound is not None:
        top2 = np.sort(z, axis=1)[:, -2:]
        out["saturation"] = float(np.mean(np.all(top2 >= 0.95 * bound, axis=1)))
    return out
