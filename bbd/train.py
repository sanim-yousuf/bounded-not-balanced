"""Early-stopped minibatch training (AdamW), selecting the epoch with the best balanced-validation macro-F1."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .config import BATCH_SIZE, MAX_EPOCHS, PATIENCE


def macro_f1_torch(pred: torch.Tensor, y: torch.Tensor, k: int) -> float:
    conf = torch.bincount(y * k + pred, minlength=k * k).view(k, k).double()
    tp = conf.diag()
    prec = tp / conf.sum(0).clamp_min(1)
    rec = tp / conf.sum(1).clamp_min(1)
    f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec), torch.zeros_like(tp))
    return float(f1.mean())


@torch.no_grad()
def predict_logits(model, X: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    model.eval()
    out = [model(X[i:i + batch]) for i in range(0, X.shape[0], batch)]
    return torch.cat(out).float()


def fit(model, Xtr, ytr, Xval, yval, objective, k: int, lr: float = 3e-3, wd: float = 0.0,
        max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, batch: int = BATCH_SIZE, seed: int = 0):
    """Train `model` in place; returns (best_state_dict, history dict)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = Xtr.shape[0]
    best, best_state, best_epoch, bad = -1.0, None, -1, 0
    hist = {"train_loss": [], "val_macro_f1": []}
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(Xtr.device)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            loss = objective(model(Xtr[b]), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        zval = predict_logits(model, Xval)
        f1 = macro_f1_torch(zval.argmax(1), yval, k)
        hist["train_loss"].append(tot / n)
        hist["val_macro_f1"].append(f1)
        if not np.isfinite(hist["train_loss"][-1]):
            break
        if f1 > best + 1e-6:
            best, best_epoch, bad = f1, epoch, 0
            best_state = copy.deepcopy({kk: v.detach().cpu() for kk, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist.update(best_epoch=best_epoch, epochs_run=epoch + 1, best_val_macro_f1=best, train_seconds=time.time() - t0)
    return best_state, hist
