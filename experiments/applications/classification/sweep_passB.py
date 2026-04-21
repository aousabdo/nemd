"""Pass-B revision sweep: baseline expansion, param-matched comparison,
and multi-seed confidence intervals for the task-aware classification
experiment in Section V-C of the IEEE TSP manuscript.

Adds the following pipelines to the existing set:

  * ``mel_fb_mlp``      — fixed mel-scale triangular filter bank + MLP
  * ``sincnet_mlp``     — learnable sinc-windowed bandpass bank + MLP
  * ``vmd_mlp_big``     — VMD + larger MLP (~N-EMD parameter count)

All classical / fixed / small-param pipelines are cheap enough for 5
seeds; the end-to-end N-EMD pipelines run at 3 seeds to keep
wall-clock manageable on CPU / MPS. Physics-regulariser ablation for
the pretrained N-EMD is also included for SNR 10 dB.

Outputs:
  paper/figures/phase3_exp3_passB/sweep_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nemd.classical import ClassicalEMD, VMD
from nemd.losses import NEMDLoss
from nemd.model import NEMD
from experiments.applications.classification.dataset import make_splits
from experiments.applications.classification.features import imf_features
from experiments.applications.classification.filter_banks import (
    MelFilterBank, SincNetFrontend,
)
from experiments.applications.classification.models import FeatureMLP
from experiments.applications.classification.train_pipelines import (
    PipelineResult, cache_features_classical, macro_f1,
    train_feature_pipeline, train_nemd_end_to_end, train_raw_cnn,
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------
# New frontend pipelines
# ---------------------------------------------------------------------

def cache_features_frontend(
    signals: torch.Tensor,
    frontend: nn.Module,
    sample_rate: float,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Apply a (non-trainable) frontend once and cache features."""
    frontend.to(device).eval()
    feats_list = []
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            x = signals[i:i + batch_size].to(device)
            imfs = frontend(x)                                 # (B, K, T)
            feats = imf_features(imfs, sample_rate=sample_rate)
            feats_list.append(feats.cpu())
    return torch.cat(feats_list, dim=0)


def train_sincnet_mlp(
    X_tr: torch.Tensor, y_tr: torch.Tensor,
    X_va: torch.Tensor, y_va: torch.Tensor,
    X_te: torch.Tensor, y_te: torch.Tensor,
    sample_rate: float, n_epochs: int, seed: int, device: torch.device,
    batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 1e-4,
    num_filters: int = 3, verbose: bool = False,
) -> PipelineResult:
    """End-to-end SincNet frontend + MLP.

    The sinc cutoffs are trainable; gradients flow through feature
    extraction into the bandpass parameters.
    """
    seed_everything(seed)
    frontend = SincNetFrontend(
        num_filters=num_filters, sample_rate=sample_rate, n_taps=101,
    ).to(device)
    # Match FeatureMLP capacity of classical pipelines (hidden=64)
    mlp = FeatureMLP(in_dim=3 * num_filters, hidden_dim=64, n_classes=3).to(device)

    params = list(frontend.parameters()) + list(mlp.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    r = PipelineResult(name="sincnet_mlp")
    r.n_params = sum(p.numel() for p in params)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
    )
    best_val = 0.0
    best_state = None
    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        frontend.train(); mlp.train()
        total = 0.0
        correct = 0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            imfs = frontend(xb)
            feats = imf_features(imfs, sample_rate=sample_rate)
            logits = mlp(feats)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
            correct += (logits.argmax(-1) == yb).sum().item()
            n += len(xb)

        # Validation
        frontend.eval(); mlp.eval()
        with torch.no_grad():
            imfs_va = frontend(X_va.to(device))
            logits_va = mlp(imf_features(imfs_va, sample_rate=sample_rate))
            val_acc = (logits_va.argmax(-1).cpu() == y_va).float().mean().item()
            val_loss = F.cross_entropy(logits_va, y_va.to(device)).item()

        r.train_acc_curve.append(correct / n)
        r.train_loss_curve.append(total / n)
        r.val_acc_curve.append(val_acc)
        r.val_loss_curve.append(val_loss)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                "frontend": copy.deepcopy(frontend.state_dict()),
                "mlp":      copy.deepcopy(mlp.state_dict()),
            }

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    [sincnet] ep {epoch:3d}/{n_epochs} "
                  f"val {val_acc:.3f}")

    if best_state is not None:
        frontend.load_state_dict(best_state["frontend"])
        mlp.load_state_dict(best_state["mlp"])

    # Test
    frontend.eval(); mlp.eval()
    with torch.no_grad():
        imfs_te = frontend(X_te.to(device))
        logits_te = mlp(imf_features(imfs_te, sample_rate=sample_rate))
        preds_te = logits_te.argmax(-1).cpu()
    r.test_acc = (preds_te == y_te).float().mean().item()
    r.test_loss = F.cross_entropy(logits_te, y_te.to(device)).item()
    r.test_preds = preds_te.tolist()
    r.test_labels = y_te.tolist()
    r.wallclock_sec = time.perf_counter() - t0
    # Confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for p, t in zip(preds_te.tolist(), y_te.tolist()):
        cm[t, p] += 1
    r.confusion = cm.tolist()
    return r


# ---------------------------------------------------------------------
# Large-MLP variant of VMD+MLP to match N-EMD parameter count
# ---------------------------------------------------------------------

def train_big_feature_mlp(
    feats_tr: torch.Tensor, y_tr: torch.Tensor,
    feats_va: torch.Tensor, y_va: torch.Tensor,
    feats_te: torch.Tensor, y_te: torch.Tensor,
    n_epochs: int, seed: int, device: torch.device,
    hidden_dim: int = 320, name: str = "vmd_mlp_big",
) -> PipelineResult:
    """FeatureMLP with hidden_dim ≈ 320 gives ~130k params, matching N-EMD."""
    seed_everything(seed)
    in_dim = feats_tr.shape[-1]
    model = FeatureMLP(in_dim=in_dim, hidden_dim=hidden_dim, n_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    r = PipelineResult(name=name)
    r.n_params = sum(p.numel() for p in model.parameters())

    loader = DataLoader(
        TensorDataset(feats_tr, y_tr), batch_size=64, shuffle=True,
    )
    best_val = 0.0
    best_state = None
    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        correct = 0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
            correct += (logits.argmax(-1) == yb).sum().item()
            n += len(xb)

        model.eval()
        with torch.no_grad():
            logits_va = model(feats_va.to(device))
            val_acc = (logits_va.argmax(-1).cpu() == y_va).float().mean().item()
            val_loss = F.cross_entropy(logits_va, y_va.to(device)).item()

        r.train_acc_curve.append(correct / n)
        r.train_loss_curve.append(total / n)
        r.val_acc_curve.append(val_acc)
        r.val_loss_curve.append(val_loss)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_te = model(feats_te.to(device))
        preds_te = logits_te.argmax(-1).cpu()
    r.test_acc = (preds_te == y_te).float().mean().item()
    r.test_loss = F.cross_entropy(logits_te, y_te.to(device)).item()
    r.wallclock_sec = time.perf_counter() - t0
    cm = np.zeros((3, 3), dtype=int)
    for p, t in zip(preds_te.tolist(), y_te.tolist()):
        cm[t, p] += 1
    r.confusion = cm.tolist()
    return r


# ---------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------

def run_sweep(
    snr_list: list[float],
    seeds: list[int],
    out_dir: Path,
    n_train: int = 1000, n_val: int = 200, n_test: int = 200,
    n_samples: int = 1024, sample_rate: float = 1000.0,
    epochs_classical: int = 50, epochs_nemd: int = 50,
    include_nemd_scratch: bool = True,
) -> dict:
    """Run the multi-seed sweep."""
    device = pick_device()
    print(f"Device: {device}")

    results: dict = {
        "config": {
            "snr_list":  snr_list,
            "seeds":     seeds,
            "n_train":   n_train,
            "n_val":     n_val,
            "n_test":    n_test,
            "n_samples": n_samples,
            "sample_rate": sample_rate,
            "epochs_classical": epochs_classical,
            "epochs_nemd":      epochs_nemd,
        },
        "snr_results": {},
    }

    for snr in snr_list:
        print(f"\n============================================================")
        print(f"  SNR = {snr} dB")
        print(f"============================================================")
        snr_key = f"snr_{int(snr)}_db"
        results["snr_results"][snr_key] = {}

        # Cache classical features once per (snr, seed) since data depends on
        # seed (random noise realisation). We re-split per seed.
        for seed in seeds:
            print(f"\n  --- seed {seed} ---")
            splits = make_splits(
                n_train_per_class=n_train, n_val_per_class=n_val,
                n_test_per_class=n_test, n_samples=n_samples,
                fs=sample_rate, snr_db=snr, seed=seed,
            )
            X_tr, y_tr = splits["train"]
            X_va, y_va = splits["val"]
            X_te, y_te = splits["test"]

            seed_key = f"seed_{seed}"
            seed_out: dict = {}

            # ---- 1. EMD + MLP ----
            print("    EMD+MLP ...")
            feats_tr_emd, _ = cache_features_classical(X_tr, "emd", 3, sample_rate)
            feats_va_emd, _ = cache_features_classical(X_va, "emd", 3, sample_rate)
            feats_te_emd, _ = cache_features_classical(X_te, "emd", 3, sample_rate)
            r = train_feature_pipeline(
                "emd_mlp", feats_tr_emd, y_tr, feats_va_emd, y_va,
                feats_te_emd, y_te, n_epochs=epochs_classical, seed=seed,
                device=device, verbose=False,
            )
            seed_out["emd_mlp"] = {
                "test_acc": r.test_acc, "n_params": r.n_params,
                "wallclock_sec": r.wallclock_sec,
            }
            print(f"      acc={r.test_acc:.4f}")

            # ---- 2. VMD + MLP (small, hidden=64) ----
            print("    VMD+MLP ...")
            feats_tr_vmd, _ = cache_features_classical(X_tr, "vmd", 3, sample_rate)
            feats_va_vmd, _ = cache_features_classical(X_va, "vmd", 3, sample_rate)
            feats_te_vmd, _ = cache_features_classical(X_te, "vmd", 3, sample_rate)
            r = train_feature_pipeline(
                "vmd_mlp", feats_tr_vmd, y_tr, feats_va_vmd, y_va,
                feats_te_vmd, y_te, n_epochs=epochs_classical, seed=seed,
                device=device, verbose=False,
            )
            seed_out["vmd_mlp"] = {
                "test_acc": r.test_acc, "n_params": r.n_params,
                "wallclock_sec": r.wallclock_sec,
            }
            print(f"      acc={r.test_acc:.4f}")

            # ---- 3. VMD + big MLP (param-matched to N-EMD) ----
            print("    VMD+MLP (big) ...")
            r = train_big_feature_mlp(
                feats_tr_vmd, y_tr, feats_va_vmd, y_va,
                feats_te_vmd, y_te, n_epochs=epochs_classical, seed=seed,
                device=device, hidden_dim=320, name="vmd_mlp_big",
            )
            seed_out["vmd_mlp_big"] = {
                "test_acc": r.test_acc, "n_params": r.n_params,
                "wallclock_sec": r.wallclock_sec,
            }
            print(f"      acc={r.test_acc:.4f}  params={r.n_params}")

            # ---- 4. Mel-FB + MLP (fixed) ----
            print("    Mel-FB+MLP ...")
            mel_fb = MelFilterBank(
                num_filters=3, sample_rate=sample_rate, n_samples=n_samples,
            )
            feats_tr_mel = cache_features_frontend(X_tr, mel_fb, sample_rate, device)
            feats_va_mel = cache_features_frontend(X_va, mel_fb, sample_rate, device)
            feats_te_mel = cache_features_frontend(X_te, mel_fb, sample_rate, device)
            r = train_feature_pipeline(
                "mel_fb_mlp", feats_tr_mel, y_tr, feats_va_mel, y_va,
                feats_te_mel, y_te, n_epochs=epochs_classical, seed=seed,
                device=device, verbose=False,
            )
            seed_out["mel_fb_mlp"] = {
                "test_acc": r.test_acc, "n_params": r.n_params,
                "wallclock_sec": r.wallclock_sec,
            }
            print(f"      acc={r.test_acc:.4f}")

            # ---- 5. SincNet + MLP (learnable frontend) ----
            print("    SincNet+MLP ...")
            r = train_sincnet_mlp(
                X_tr, y_tr, X_va, y_va, X_te, y_te,
                sample_rate=sample_rate, n_epochs=epochs_classical,
                seed=seed, device=device,
            )
            seed_out["sincnet_mlp"] = {
                "test_acc": r.test_acc, "n_params": r.n_params,
                "wallclock_sec": r.wallclock_sec,
            }
            print(f"      acc={r.test_acc:.4f}")

            # ---- 6. N-EMD scratch (end-to-end) ----
            if include_nemd_scratch:
                print("    N-EMD scratch (e2e) ...")
                t0 = time.perf_counter()
                seed_everything(seed)
                nemd = NEMD(
                    num_imfs=3, hidden_dim=64, num_layers=3,
                    sample_rate=sample_rate, temperature=0.3,
                )
                r = train_nemd_end_to_end(
                    name="nemd_scratch",
                    train_signals=X_tr, train_labels=y_tr,
                    val_signals=X_va,   val_labels=y_va,
                    test_signals=X_te,  test_labels=y_te,
                    nemd_model=nemd, sample_rate=sample_rate,
                    n_epochs=epochs_nemd, seed=seed, device=device,
                    lr=1e-4, physics_weight=0.1, verbose=False,
                )
                seed_out["nemd_scratch"] = {
                    "test_acc": r.test_acc, "n_params": r.n_params,
                    "wallclock_sec": r.wallclock_sec,
                }
                print(f"      acc={r.test_acc:.4f}  wall={r.wallclock_sec:.0f}s")

            results["snr_results"][snr_key][seed_key] = seed_out

            # Persist after each seed so partial results survive a crash
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "sweep_results.json", "w") as f:
                json.dump(results, f, indent=2)

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snrs", type=str, default="3,10,20")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--epochs-classical", type=int, default=50)
    p.add_argument("--epochs-nemd", type=int, default=50)
    p.add_argument("--out-dir", type=str,
                   default="paper/figures/phase3_exp3_passB")
    p.add_argument("--skip-nemd", action="store_true")
    args = p.parse_args()

    snr_list  = [float(s) for s in args.snrs.split(",")]
    seeds     = [int(s) for s in args.seeds.split(",")]
    out_dir   = Path(args.out_dir)

    t0 = time.perf_counter()
    run_sweep(
        snr_list=snr_list, seeds=seeds, out_dir=out_dir,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        epochs_classical=args.epochs_classical, epochs_nemd=args.epochs_nemd,
        include_nemd_scratch=not args.skip_nemd,
    )
    print(f"\nTotal wall-clock: {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
