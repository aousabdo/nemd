"""Parametric-POU baseline end-to-end classification sweep.

Isolates whether NAFB's advantage comes from the POU constraint
(shared by this baseline) or from the flexibility of the learned
neural analyzer (only in NAFB). 3 seeds x 3 SNRs.

Output: paper/figures/phase3_exp3_passB/parametric_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.applications.classification.dataset import make_splits
from experiments.applications.classification.features import imf_features
from experiments.applications.classification.models import FeatureMLP
from experiments.applications.classification.parametric_pou import ParametricPOUFilterBank
from experiments.applications.classification.sweep_passB import pick_device, seed_everything


def train_parametric(
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    sample_rate, n_samples, n_classes, n_epochs, seed, device,
    num_imfs: int = 3, mlp_hidden: int = 320, lr: float = 1e-4,
    batch_size: int = 64,
) -> dict:
    seed_everything(seed)
    frontend = ParametricPOUFilterBank(
        num_imfs=num_imfs, sample_rate=sample_rate, n_samples=n_samples,
    ).to(device)
    head = FeatureMLP(
        in_dim=3 * num_imfs, hidden_dim=mlp_hidden, n_classes=n_classes,
    ).to(device)
    params = list(frontend.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
    )
    best = 0.0
    best_state = None
    t0 = time.perf_counter()
    for ep in range(n_epochs):
        frontend.train(); head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            imfs, _, _ = frontend(xb)
            feats = imf_features(imfs, sample_rate=sample_rate)
            loss = F.cross_entropy(head(feats), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        frontend.eval(); head.eval()
        with torch.no_grad():
            imfs_va, _, _ = frontend(X_va.to(device))
            va = (head(imf_features(imfs_va, sample_rate=sample_rate))
                  .argmax(-1).cpu() == y_va).float().mean().item()
        if va > best:
            best = va
            best_state = {
                "f": copy.deepcopy(frontend.state_dict()),
                "h": copy.deepcopy(head.state_dict()),
            }
    if best_state is not None:
        frontend.load_state_dict(best_state["f"])
        head.load_state_dict(best_state["h"])
    frontend.eval(); head.eval()
    with torch.no_grad():
        imfs_te, _, _ = frontend(X_te.to(device))
        preds = head(imf_features(imfs_te, sample_rate=sample_rate)).argmax(-1).cpu()
    acc = (preds == y_te).float().mean().item()
    n_params = sum(p.numel() for p in params)
    return {
        "test_acc": acc, "n_params": n_params,
        "wallclock_sec": time.perf_counter() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--out", type=str,
                   default="paper/figures/phase3_exp3_passB/parametric_results.json")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = pick_device()
    print(f"Device: {device}  seeds: {seeds}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fs = 1000.0
    n_samples = 1024
    results: dict = {"config": {"seeds": seeds, "n_epochs": args.n_epochs},
                     "parametric_pou": {}}

    for snr in (3.0, 10.0, 20.0):
        for seed in seeds:
            splits = make_splits(
                n_train_per_class=1000, n_val_per_class=200,
                n_test_per_class=200, n_samples=n_samples, fs=fs,
                snr_db=snr, seed=seed,
            )
            X_tr, y_tr = splits["train"]
            X_va, y_va = splits["val"]
            X_te, y_te = splits["test"]
            snr_key = f"snr_{int(snr)}_db"
            r = train_parametric(
                X_tr, y_tr, X_va, y_va, X_te, y_te,
                sample_rate=fs, n_samples=n_samples, n_classes=3,
                n_epochs=args.n_epochs, seed=seed, device=device,
            )
            results["parametric_pou"].setdefault(snr_key, {})[f"seed_{seed}"] = r
            print(f"  SNR {snr:.0f} seed={seed}: acc={r['test_acc']:.4f}  "
                  f"params={r['n_params']}")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
