"""Step 6: Cognitive load classification on real pupil data.

Three task formulations:
  - 2-class: control vs memory
  - 3-class: load level (5-digit / 9-digit / 13-digit, task type collapsed)
  - 6-class: full (control_5 / control_9 / control_13 / memory_5 / memory_9 / memory_13)

SUBJECT-LEVEL split: 70/15/15 on subjects (not epochs).
Multiple random seeds for stable estimates.

Pipelines: Raw CNN, EMD+MLP, VMD+MLP, N-EMD end-to-end (pretrained + scratch).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nemd.classical import ClassicalEMD, VMD
from nemd.model import NEMD
from nemd.train import TrainConfig
from nemd.utils import to_numpy
from nemd.data.pupil_loader import stream_pupil_data, load_events
from nemd.data.pupil_preprocessing import PupilPreprocessor, PupilPreprocessConfig
from experiments.applications.classification.features import imf_features
from experiments.applications.classification.models import (
    FeatureMLP, RawSignalCNN, NEMDClassifier, ModeCNN, NEMDClassifierCNN,
)
from nemd.losses import NEMDLoss


# -----------------------------------------------------------------------
# Label mapping for different task formulations
# -----------------------------------------------------------------------

LABEL_MAP_2CLASS = {
    "control_5": 0, "control_9": 0, "control_13": 0,
    "memory_5": 1, "memory_9": 1, "memory_13": 1,
}
LABEL_MAP_3CLASS = {
    "control_5": 0, "memory_5": 0,
    "control_9": 1, "memory_9": 1,
    "control_13": 2, "memory_13": 2,
}
LABEL_MAP_6CLASS = {
    "control_5": 0, "control_9": 1, "control_13": 2,
    "memory_5": 3, "memory_9": 4, "memory_13": 5,
}

TASK_CONFIGS = {
    "2class": {"label_map": LABEL_MAP_2CLASS, "n_classes": 2, "name": "control vs memory"},
    "3class": {"label_map": LABEL_MAP_3CLASS, "n_classes": 3, "name": "load level (5/9/13)"},
    "6class": {"label_map": LABEL_MAP_6CLASS, "n_classes": 6, "name": "full 6-class"},
}


# -----------------------------------------------------------------------
# Data loading and subject-level splitting
# -----------------------------------------------------------------------

def load_all_epochs(
    cache_dir: str = "data/ds003838_cache",
    repo_dir: str = "data/ds003838",
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Load all preprocessed epochs, conditions, and subject IDs."""
    cache = Path(cache_dir)
    repo = Path(repo_dir)
    parquets = sorted(cache.glob("*_eye0.parquet"))

    proc = PupilPreprocessor(PupilPreprocessConfig())
    all_epochs, all_conds, all_subs = [], [], []

    for pq in parquets:
        sub = pq.stem.split("_")[0]
        try:
            df = pd.read_parquet(pq)
            events = load_events(str(repo), sub)
            result = proc.preprocess_subject(df, events)
            for ep, cond in zip(result["epochs"], result["conditions"]):
                all_epochs.append(ep)
                all_conds.append(cond)
                all_subs.append(sub)
        except Exception:
            pass

    return all_epochs, all_conds, all_subs


def subject_level_split(
    all_subs: list[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """Split subjects into train/val/test sets."""
    rng = np.random.default_rng(seed)
    unique_subs = sorted(set(all_subs))
    rng.shuffle(unique_subs)
    n = len(unique_subs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_subs = set(unique_subs[:n_train])
    val_subs = set(unique_subs[n_train:n_train + n_val])
    test_subs = set(unique_subs[n_train + n_val:])
    return train_subs, val_subs, test_subs


def make_tensors(
    epochs: list[np.ndarray],
    conds: list[str],
    subs: list[str],
    sub_set: set[str],
    label_map: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter epochs by subject set and convert to tensors."""
    X, y = [], []
    for ep, cond, sub in zip(epochs, conds, subs):
        if sub in sub_set and cond in label_map:
            X.append(ep)
            y.append(label_map[cond])
    if not X:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return torch.from_numpy(np.stack(X)).float(), torch.tensor(y, dtype=torch.long)


# -----------------------------------------------------------------------
# Pipeline runners (simplified from Phase 3 Exp 3)
# -----------------------------------------------------------------------

def _train_mlp_on_features(
    train_feats, train_labels, val_feats, val_labels,
    n_classes, n_epochs=50, lr=1e-3, device="cpu",
):
    model = FeatureMLP(in_dim=train_feats.shape[-1], hidden_dim=64, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=64, shuffle=True)
    best_acc, best_state = 0, None
    for ep in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(val_feats.to(device)).argmax(-1).cpu()
        acc = (preds == val_labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)
    return model


def _eval_model(model, X, y, device, is_nemd=False, batch_size=64):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            if is_nemd:
                logits, _, _ = model(xb)
            else:
                logits = model(xb)
            all_preds.append(logits.argmax(-1).cpu())
    preds = torch.cat(all_preds)
    acc = (preds == y).float().mean().item()
    # Macro F1
    n_classes = max(y.max().item() + 1, preds.max().item() + 1)
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (y == c)).sum().item()
        fp = ((preds == c) & (y != c)).sum().item()
        fn = ((preds != c) & (y == c)).sum().item()
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1s.append(2 * p * r / (p + r + 1e-12))
    return acc, float(np.mean(f1s))


def _decompose_classical(signals, method, K, fs):
    N, T = signals.shape
    out = np.zeros((N, K, T))
    if method == "emd":
        decomp = ClassicalEMD(max_imfs=K + 1)
        t = np.arange(T) / fs
        for i in range(N):
            imfs = decomp.decompose(signals[i].numpy(), t)
            k = min(K, imfs.shape[0])
            out[i, :k] = imfs[:k]
    elif method == "vmd":
        decomp = VMD(n_modes=K)
        for i in range(N):
            out[i] = decomp.decompose(signals[i].numpy())
    return torch.from_numpy(out).float()


def run_pipeline(
    name, X_tr, y_tr, X_va, y_va, X_te, y_te,
    n_classes, nemd_ckpt=None, device="cpu", K=4, fs=100.0,
    n_epochs=50,
):
    """Run one pipeline and return (test_acc, macro_f1)."""
    if name == "raw_cnn":
        model = RawSignalCNN(n_classes=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
        best_acc, best_state = 0, None
        for ep in range(n_epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            acc, _ = _eval_model(model, X_va, y_va, device)
            if acc > best_acc:
                best_acc = acc; best_state = copy.deepcopy(model.state_dict())
        if best_state: model.load_state_dict(best_state)
        return _eval_model(model, X_te, y_te, device)

    elif name in ("emd_mlp", "vmd_mlp"):
        method = "emd" if "emd" in name else "vmd"
        imfs_tr = _decompose_classical(X_tr, method, K, fs)
        imfs_va = _decompose_classical(X_va, method, K, fs)
        imfs_te = _decompose_classical(X_te, method, K, fs)
        feats_tr = imf_features(imfs_tr, sample_rate=fs)
        feats_va = imf_features(imfs_va, sample_rate=fs)
        feats_te = imf_features(imfs_te, sample_rate=fs)
        model = _train_mlp_on_features(feats_tr, y_tr, feats_va, y_va, n_classes, device=device)
        model.eval()
        with torch.no_grad():
            preds = model(feats_te.to(device)).argmax(-1).cpu()
        acc = (preds == y_te).float().mean().item()
        f1s = []
        for c in range(n_classes):
            tp = ((preds == c) & (y_te == c)).sum().item()
            fp = ((preds == c) & (y_te != c)).sum().item()
            fn = ((preds != c) & (y_te == c)).sum().item()
            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            f1s.append(2 * p * r / (p + r + 1e-12))
        return acc, float(np.mean(f1s))

    elif name in ("emd_cnn", "vmd_cnn"):
        method = "emd" if "emd" in name else "vmd"
        imfs_tr = _decompose_classical(X_tr, method, K, fs)
        imfs_va = _decompose_classical(X_va, method, K, fs)
        imfs_te = _decompose_classical(X_te, method, K, fs)
        model = ModeCNN(n_channels=K, n_classes=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(imfs_tr, y_tr), batch_size=64, shuffle=True)
        best_acc, best_state = 0, None
        for ep in range(n_epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            acc, _ = _eval_model(model, imfs_va, y_va, device)
            if acc > best_acc:
                best_acc = acc; best_state = copy.deepcopy(model.state_dict())
        if best_state: model.load_state_dict(best_state)
        return _eval_model(model, imfs_te, y_te, device)

    elif name == "nemd_frozen_cnn":
        # Pretrained N-EMD frozen; only ModeCNN trains.
        blob = torch.load(nemd_ckpt, map_location="cpu", weights_only=False)
        config: TrainConfig = blob["config"]
        nemd = NEMD(num_imfs=K, hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers, kernel_size=config.kernel_size,
                    sample_rate=fs, temperature=0.5)
        nemd.load_state_dict(blob["state_dict"])
        nemd.to(device).eval()
        for p in nemd.parameters():
            p.requires_grad = False

        def _nemd_decompose(X):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 32):
                    xb = X[i:i + 32].to(device)
                    imfs, _, _ = nemd(xb, temperature=0.5, sort_by_centroid=True)
                    out.append(imfs.cpu())
            return torch.cat(out, dim=0)

        imfs_tr = _nemd_decompose(X_tr)
        imfs_va = _nemd_decompose(X_va)
        imfs_te = _nemd_decompose(X_te)
        model = ModeCNN(n_channels=K, n_classes=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(imfs_tr, y_tr), batch_size=64, shuffle=True)
        best_acc, best_state = 0, None
        for ep in range(n_epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            acc, _ = _eval_model(model, imfs_va, y_va, device)
            if acc > best_acc:
                best_acc = acc; best_state = copy.deepcopy(model.state_dict())
        if best_state: model.load_state_dict(best_state)
        return _eval_model(model, imfs_te, y_te, device)

    elif name in ("nemd_finetuned_cnn", "nemd_scratch_cnn"):
        # End-to-end: N-EMD + ModeCNN. Finetuned starts from pretrained ckpt
        # with small LR on N-EMD; scratch starts random.
        #
        # Hardening vs the original MLP-head end-to-end code:
        #   (1) Head warmup: train ModeCNN for ``warmup_epochs`` with N-EMD
        #       frozen so the classifier converges before gradients flow back
        #       into the decomposition (prevents chaotic N-EMD updates from
        #       an untrained head).
        #   (2) Tighter grad clip (0.5 vs 1.0) and smaller physics weight
        #       (0.01 vs 0.1) — the CNN head backward has ~T x larger grad
        #       norm than the MLP-on-stats path; loss scales retuned.
        #   (3) NaN guard: if a batch produces non-finite loss, skip the
        #       optimizer step (prevents pathological Adam-state corruption
        #       that we observed hanging the prior run for 18h).
        blob = torch.load(nemd_ckpt, map_location="cpu", weights_only=False)
        config: TrainConfig = blob["config"]
        nemd = NEMD(num_imfs=K, hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers, kernel_size=config.kernel_size,
                    sample_rate=fs, temperature=0.5)
        if name == "nemd_finetuned_cnn":
            nemd.load_state_dict(blob["state_dict"])
        clf = NEMDClassifierCNN(nemd, n_classes=n_classes, sample_rate=fs).to(device)
        physics = NEMDLoss(lambda_sharp=1.0, lambda_order=1.0, lambda_ortho=0.1,
                           lambda_balance=5.0, sample_rate=fs)
        warmup_epochs = min(3, n_epochs // 4)
        if name == "nemd_finetuned_cnn":
            nemd_lr = 1e-5
        else:
            nemd_lr = 1e-4
        # Optimizer with separate param groups so we can toggle N-EMD lr later.
        opt = torch.optim.Adam([
            {"params": clf.nemd.parameters(), "lr": 0.0},            # frozen during warmup
            {"params": clf.classifier.parameters(), "lr": 1e-3},
        ], weight_decay=1e-4)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        phys_weight = 0.01
        best_acc, best_state = 0, None
        nan_skips = 0
        for ep in range(n_epochs):
            # Unfreeze N-EMD after warmup.
            if ep == warmup_epochs:
                opt.param_groups[0]["lr"] = nemd_lr
            in_warmup = ep < warmup_epochs
            clf.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _, meta = clf(xb)
                ce = F.cross_entropy(logits, yb)
                if in_warmup:
                    loss = ce
                else:
                    imfs, _, _ = clf.nemd(xb, temperature=0.5, sort_by_centroid=True)
                    phys, _ = physics(imfs, meta)
                    loss = ce + phys_weight * phys
                if not torch.isfinite(loss):
                    nan_skips += 1
                    opt.zero_grad()
                    continue
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(clf.parameters(), 0.5)
                opt.step()
            acc, _ = _eval_model(clf, X_va, y_va, device, is_nemd=True)
            if acc > best_acc:
                best_acc = acc; best_state = copy.deepcopy(clf.state_dict())
        if nan_skips:
            print(f"    [{name}] skipped {nan_skips} non-finite batches")
        if best_state: clf.load_state_dict(best_state)
        return _eval_model(clf, X_te, y_te, device, is_nemd=True)

    elif name.startswith("nemd"):
        blob = torch.load(nemd_ckpt, map_location="cpu", weights_only=False)
        config: TrainConfig = blob["config"]
        if name == "nemd_pretrained":
            nemd = NEMD(num_imfs=K, hidden_dim=config.hidden_dim,
                        num_layers=config.num_layers, kernel_size=config.kernel_size,
                        sample_rate=fs, temperature=0.5)
            nemd.load_state_dict(blob["state_dict"])
        else:  # scratch
            nemd = NEMD(num_imfs=K, hidden_dim=config.hidden_dim,
                        num_layers=config.num_layers, kernel_size=config.kernel_size,
                        sample_rate=fs, temperature=0.5)
        clf = NEMDClassifier(nemd, n_classes=n_classes, sample_rate=fs).to(device)
        physics = NEMDLoss(lambda_sharp=1.0, lambda_order=1.0, lambda_ortho=0.1,
                           lambda_balance=5.0, sample_rate=fs)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-4, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        best_acc, best_state = 0, None
        for ep in range(n_epochs):
            clf.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _, meta = clf(xb)
                ce = F.cross_entropy(logits, yb)
                imfs, _, _ = clf.nemd(xb, temperature=0.5, sort_by_centroid=True)
                phys, _ = physics(imfs, meta)
                loss = ce + 0.1 * phys
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
                opt.step()
            acc, _ = _eval_model(clf, X_va, y_va, device, is_nemd=True)
            if acc > best_acc:
                best_acc = acc; best_state = copy.deepcopy(clf.state_dict())
        if best_state: clf.load_state_dict(best_state)
        return _eval_model(clf, X_te, y_te, device, is_nemd=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemd-ckpt", type=str, default="checkpoints_pupil/final.pt")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="paper/figures/phase4")
    parser.add_argument("--pipelines", type=str,
                        default="raw_cnn,emd_mlp,vmd_mlp,nemd_pretrained,nemd_scratch")
    parser.add_argument("--tasks", type=str, default="2class,3class,6class",
                        help="Comma-separated task names to run")
    parser.add_argument("--results-filename", type=str,
                        default="step6_classification_results.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all epochs...")
    all_epochs, all_conds, all_subs = load_all_epochs()
    print(f"Total: {len(all_epochs)} epochs, {len(set(all_subs))} subjects")
    print(f"Conditions: {dict(Counter(all_conds))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipelines = args.pipelines.split(",")
    results = {}

    task_filter = set(t.strip() for t in args.tasks.split(","))
    for task_name, task_cfg in TASK_CONFIGS.items():
        if task_name not in task_filter:
            continue
        n_classes = task_cfg["n_classes"]
        label_map = task_cfg["label_map"]
        print(f"\n{'='*60}")
        print(f"Task: {task_cfg['name']} ({task_name}, {n_classes} classes)")
        print(f"{'='*60}")

        task_results = {p: {"accs": [], "f1s": []} for p in pipelines}

        for seed in range(args.n_seeds):
            print(f"\n--- Seed {seed} ---")
            train_subs, val_subs, test_subs = subject_level_split(all_subs, seed=42 + seed)
            print(f"  Train: {len(train_subs)} subs, Val: {len(val_subs)}, Test: {len(test_subs)}")

            X_tr, y_tr = make_tensors(all_epochs, all_conds, all_subs, train_subs, label_map)
            X_va, y_va = make_tensors(all_epochs, all_conds, all_subs, val_subs, label_map)
            X_te, y_te = make_tensors(all_epochs, all_conds, all_subs, test_subs, label_map)
            print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")

            if len(X_te) == 0:
                print("  WARNING: empty test set, skipping seed")
                continue

            for pipe in pipelines:
                t0 = time.time()
                torch.manual_seed(seed)
                np.random.seed(seed)
                try:
                    acc, f1 = run_pipeline(
                        pipe, X_tr, y_tr, X_va, y_va, X_te, y_te,
                        n_classes=n_classes, nemd_ckpt=args.nemd_ckpt,
                        device=device, K=4, fs=100.0, n_epochs=args.n_epochs,
                    )
                    elapsed = time.time() - t0
                    task_results[pipe]["accs"].append(acc)
                    task_results[pipe]["f1s"].append(f1)
                    print(f"  [{pipe}] acc={acc:.4f} f1={f1:.4f} ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  [{pipe}] ERROR: {e}")

        results[task_name] = {}
        print(f"\n--- {task_cfg['name']} Summary ---")
        print(f"  {'Pipeline':<20} | {'Acc (mean±std)':>18} | {'F1 (mean±std)':>18}")
        print("  " + "-" * 62)
        for pipe in pipelines:
            accs = task_results[pipe]["accs"]
            f1s = task_results[pipe]["f1s"]
            if accs:
                print(f"  {pipe:<20} | {np.mean(accs):>6.4f} ± {np.std(accs):>6.4f} | "
                      f"{np.mean(f1s):>6.4f} ± {np.std(f1s):>6.4f}")
                results[task_name][pipe] = {
                    "acc_mean": float(np.mean(accs)),
                    "acc_std": float(np.std(accs)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                    "accs": accs,
                    "f1s": f1s,
                }

    # Save
    out_path = out_dir / args.results_filename
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
