"""Pass-B real-world benchmark: 2-class (control vs memory) pupillometry.

Prior Phase 4 work showed that 3-class and 6-class classifications on
ds003838 are near chance across every pipeline. The 2-class (memory
vs control) formulation is strictly easier and is the natural
real-world benchmark to report alongside the synthetic experiment.

Pipelines (all sharing the same MLP head capacity where applicable):
    emd_mlp, vmd_mlp, vmd_mlp_big, mel_fb_mlp, sincnet_mlp,
    nemd_scratch, raw_cnn.

Output:  paper/figures/phase4/step6_2class_passB.json
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nemd.classical import ClassicalEMD, VMD
from nemd.model import NEMD
from nemd.train import TrainConfig
from nemd.losses import NEMDLoss
from nemd.data.pupil_loader import load_events
from nemd.data.pupil_preprocessing import (
    PupilPreprocessor, PupilPreprocessConfig,
)
from experiments.applications.classification.features import imf_features
from experiments.applications.classification.filter_banks import (
    MelFilterBank, SincNetFrontend,
)
from experiments.applications.classification.models import (
    FeatureMLP, RawSignalCNN, NEMDClassifier,
)
from experiments.applications.classification.sweep_passB import pick_device


# ---------------------------------------------------------------------
# Label map and data loading
# ---------------------------------------------------------------------

LABEL_MAP = {
    "control_5": 0, "control_9": 0, "control_13": 0,
    "memory_5":  1, "memory_9":  1, "memory_13":  1,
}


def load_epochs(
    cache_dir: str = "data/ds003838_cache",
    repo_dir:  str = "data/ds003838",
) -> tuple[list[np.ndarray], list[int], list[str]]:
    cache = Path(cache_dir); repo = Path(repo_dir)
    parquets = sorted(cache.glob("*_eye0.parquet"))
    proc = PupilPreprocessor(PupilPreprocessConfig())

    epochs_all: list[np.ndarray] = []
    labels_all: list[int] = []
    subs_all:   list[str] = []
    for pq in parquets:
        sub = pq.stem.split("_")[0]
        try:
            df = pd.read_parquet(pq)
            events = load_events(str(repo), sub)
            result = proc.preprocess_subject(df, events)
        except Exception:
            continue
        for ep, cond in zip(result["epochs"], result["conditions"]):
            lbl = LABEL_MAP.get(cond)
            if lbl is None:
                continue
            epochs_all.append(ep)
            labels_all.append(lbl)
            subs_all.append(sub)
    return epochs_all, labels_all, subs_all


def subject_split(
    subs: list[str], labels: list[int], seed: int,
    test_frac: float = 0.15, val_frac: float = 0.15,
) -> tuple[list[bool], list[bool], list[bool]]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(subs))
    rng.shuffle(uniq)
    n = len(uniq)
    n_te = max(2, int(round(n * test_frac)))
    n_va = max(2, int(round(n * val_frac)))
    subs_te = set(uniq[:n_te])
    subs_va = set(uniq[n_te:n_te + n_va])
    subs_tr = set(uniq[n_te + n_va:])
    te = [s in subs_te for s in subs]
    va = [s in subs_va for s in subs]
    tr = [s in subs_tr for s in subs]
    return tr, va, te


# ---------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------

def _decompose_classical(X: np.ndarray, method: str, K: int, fs: float) -> torch.Tensor:
    N, T = X.shape
    out = np.zeros((N, K, T), dtype=np.float32)
    if method == "emd":
        dec = ClassicalEMD(max_imfs=K + 1)
        for i in range(N):
            t = np.arange(T) / fs
            imfs = dec.decompose(X[i], t)
            k = min(K, imfs.shape[0])
            out[i, :k] = imfs[:k]
    elif method == "vmd":
        dec = VMD(n_modes=K)
        for i in range(N):
            out[i] = dec.decompose(X[i])
    return torch.from_numpy(out)


def _metrics(preds: torch.Tensor, y: torch.Tensor, n_classes: int) -> tuple[float, float, float]:
    """Return (accuracy, balanced_accuracy, macro_f1) for binary/multiclass."""
    acc = (preds == y).float().mean().item()
    # Balanced accuracy = mean of per-class recall
    recalls = []
    f1s = []
    for c in range(n_classes):
        mask_c = (y == c)
        if mask_c.sum().item() == 0:
            continue
        tp = ((preds == c) & mask_c).sum().item()
        fp = ((preds == c) & (y != c)).sum().item()
        fn = ((preds != c) & mask_c).sum().item()
        recall = tp / (tp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        recalls.append(recall)
        f1s.append(f1)
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    f1 = float(np.mean(f1s)) if f1s else 0.0
    return acc, bal_acc, f1


def _train_feature_mlp(
    feats_tr, y_tr, feats_va, y_va, feats_te, y_te,
    n_classes: int, hidden: int, n_epochs: int, device,
) -> tuple[float, float]:
    model = FeatureMLP(
        in_dim=feats_tr.shape[-1], hidden_dim=hidden, n_classes=n_classes,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(feats_tr, y_tr), batch_size=64, shuffle=True,
    )
    best_acc = 0.0; best_state = None
    for ep in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(feats_va.to(device))
            val_acc = (logits.argmax(-1).cpu() == y_va).float().mean().item()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(feats_te.to(device))
        preds  = logits.argmax(-1).cpu()
    return _metrics(preds, y_te, n_classes)


def _train_frontend_mlp(
    X_tr, y_tr, X_va, y_va, X_te, y_te, frontend, sample_rate,
    n_classes, n_epochs, device, trainable: bool,
) -> tuple[float, float]:
    frontend.to(device)
    if not trainable:
        frontend.eval()
    K = frontend.num_filters
    in_dim = 3 * K
    head = FeatureMLP(in_dim=in_dim, hidden_dim=64, n_classes=n_classes).to(device)
    params = list(head.parameters())
    if trainable:
        params += list(frontend.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    best_acc = 0.0; best_state = None
    for ep in range(n_epochs):
        head.train()
        if trainable: frontend.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            imfs = frontend(xb)
            feats = imf_features(imfs, sample_rate=sample_rate)
            loss = F.cross_entropy(head(feats), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        if trainable: frontend.eval()
        with torch.no_grad():
            imfs_va = frontend(X_va.to(device))
            logits  = head(imf_features(imfs_va, sample_rate=sample_rate))
            val_acc = (logits.argmax(-1).cpu() == y_va).float().mean().item()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "head": copy.deepcopy(head.state_dict()),
                "front": copy.deepcopy(frontend.state_dict()),
            }
    if best_state is not None:
        head.load_state_dict(best_state["head"])
        frontend.load_state_dict(best_state["front"])
    head.eval(); frontend.eval()
    with torch.no_grad():
        imfs_te = frontend(X_te.to(device))
        logits  = head(imf_features(imfs_te, sample_rate=sample_rate))
        preds = logits.argmax(-1).cpu()
    return _metrics(preds, y_te, n_classes)


def _train_raw_cnn(
    X_tr, y_tr, X_va, y_va, X_te, y_te, n_classes, n_epochs, device,
) -> tuple[float, float]:
    model = RawSignalCNN(n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    best_acc = 0.0; best_state = None
    for ep in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(X_va.to(device))
            va = (logits.argmax(-1).cpu() == y_va).float().mean().item()
        if va > best_acc:
            best_acc = va; best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_te.to(device))
        preds = logits.argmax(-1).cpu()
    return _metrics(preds, y_te, n_classes)


def _train_nemd_e2e(
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    n_classes, sample_rate, n_epochs, device, ckpt_path: str | None,
) -> tuple[float, float]:
    K = 3
    if ckpt_path:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        nemd = NEMD(
            num_imfs=cfg.num_imfs, hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers, kernel_size=cfg.kernel_size,
            sample_rate=sample_rate, temperature=0.3,
        )
        nemd.load_state_dict(blob["state_dict"])
    else:
        nemd = NEMD(
            num_imfs=K, hidden_dim=64, num_layers=3,
            sample_rate=sample_rate, temperature=0.3,
        )
    clf = NEMDClassifier(
        nemd_model=nemd, n_classes=n_classes, sample_rate=sample_rate,
    ).to(device)
    phys = NEMDLoss(
        lambda_sharp=1.0, lambda_order=1.0, lambda_ortho=0.1,
        lambda_balance=5.0, sample_rate=sample_rate,
    )
    opt = torch.optim.Adam(clf.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    best_acc = 0.0; best_state = None
    for ep in range(n_epochs):
        clf.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, imfs, meta = clf(xb)
            ce = F.cross_entropy(logits, yb)
            phys_loss, _ = phys(imfs, meta)
            (ce + 0.1 * phys_loss).backward()
            opt.step(); opt.zero_grad()
        clf.eval()
        with torch.no_grad():
            logits, _, _ = clf(X_va.to(device))
            va = (logits.argmax(-1).cpu() == y_va).float().mean().item()
        if va > best_acc:
            best_acc = va; best_state = copy.deepcopy(clf.state_dict())
    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        logits, _, _ = clf(X_te.to(device))
        preds = logits.argmax(-1).cpu()
    return _metrics(preds, y_te, n_classes)


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--cache-dir", type=str, default="data/ds003838_cache")
    p.add_argument("--repo-dir",  type=str, default="data/ds003838")
    p.add_argument("--nemd-ckpt", type=str,
                   default="checkpoints_p25b_v3/final.pt")
    p.add_argument("--out", type=str,
                   default="paper/figures/phase4/step6_2class_passB.json")
    p.add_argument(
        "--pipelines", type=str,
        default="emd_mlp,vmd_mlp,vmd_mlp_big,mel_fb_mlp,sincnet_mlp,"
                "raw_cnn,nemd_scratch",
    )
    args = p.parse_args()

    seeds     = [int(s) for s in args.seeds.split(",")]
    pipelines = [p.strip() for p in args.pipelines.split(",")]

    print("Loading pupillometry data ...")
    t0 = time.perf_counter()
    epochs_all, labels_all, subs_all = load_epochs(
        cache_dir=args.cache_dir, repo_dir=args.repo_dir,
    )
    print(f"  {len(epochs_all)} epochs from {len(set(subs_all))} subjects "
          f"({time.perf_counter() - t0:.1f}s)")
    print(f"  Class counts: {Counter(labels_all)}")

    T = len(epochs_all[0])
    fs = 100.0
    print(f"  epoch length T={T}, fs={fs} Hz")

    device = pick_device()
    print(f"Device: {device}")

    X_all = torch.from_numpy(np.stack(epochs_all).astype(np.float32))
    y_all = torch.tensor(labels_all, dtype=torch.long)

    out: dict = {
        "config": {"seeds": seeds, "n_epochs": args.n_epochs,
                   "T": T, "fs": fs, "n_epochs_total": len(epochs_all),
                   "n_subjects": len(set(subs_all)),
                   "class_counts": dict(Counter(labels_all)),
                   "pipelines": ["majority"] + pipelines},
        "results": {p: {"accs": [], "bal_accs": [], "f1s": []}
                    for p in ["majority"] + pipelines},
    }

    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        tr, va, te = subject_split(subs_all, labels_all, seed)
        X_tr, y_tr = X_all[tr], y_all[tr]
        X_va, y_va = X_all[va], y_all[va]
        X_te, y_te = X_all[te], y_all[te]
        print(f"  split: tr={len(y_tr)} va={len(y_va)} te={len(y_te)}")

        # Majority-class baseline (per-split)
        maj_class = int(Counter(y_tr.tolist()).most_common(1)[0][0])
        preds_maj = torch.full_like(y_te, maj_class)
        acc_m, bal_m, f1_m = _metrics(preds_maj, y_te, n_classes=2)
        out["results"]["majority"]["accs"].append(acc_m)
        out["results"]["majority"]["bal_accs"].append(bal_m)
        out["results"]["majority"]["f1s"].append(f1_m)
        print(f"  [majority            ] acc={acc_m:.4f}  bal={bal_m:.4f}  "
              f"f1={f1_m:.4f}")

        for name in pipelines:
            t0 = time.perf_counter()
            try:
                if name in ("emd_mlp", "vmd_mlp", "vmd_mlp_big"):
                    method = "emd" if "emd" in name else "vmd"
                    imfs_tr = _decompose_classical(X_tr.numpy(), method, 3, fs)
                    imfs_va = _decompose_classical(X_va.numpy(), method, 3, fs)
                    imfs_te = _decompose_classical(X_te.numpy(), method, 3, fs)
                    f_tr = imf_features(imfs_tr, sample_rate=fs)
                    f_va = imf_features(imfs_va, sample_rate=fs)
                    f_te = imf_features(imfs_te, sample_rate=fs)
                    hidden = 320 if name == "vmd_mlp_big" else 64
                    acc, bal, f1 = _train_feature_mlp(
                        f_tr, y_tr, f_va, y_va, f_te, y_te,
                        n_classes=2, hidden=hidden, n_epochs=args.n_epochs,
                        device=device,
                    )
                elif name == "mel_fb_mlp":
                    fb = MelFilterBank(num_filters=3, sample_rate=fs, n_samples=T)
                    acc, bal, f1 = _train_frontend_mlp(
                        X_tr, y_tr, X_va, y_va, X_te, y_te,
                        frontend=fb, sample_rate=fs, n_classes=2,
                        n_epochs=args.n_epochs, device=device, trainable=False,
                    )
                elif name == "sincnet_mlp":
                    fb = SincNetFrontend(
                        num_filters=3, sample_rate=fs, n_taps=51,
                    )
                    acc, bal, f1 = _train_frontend_mlp(
                        X_tr, y_tr, X_va, y_va, X_te, y_te,
                        frontend=fb, sample_rate=fs, n_classes=2,
                        n_epochs=args.n_epochs, device=device, trainable=True,
                    )
                elif name == "raw_cnn":
                    acc, bal, f1 = _train_raw_cnn(
                        X_tr, y_tr, X_va, y_va, X_te, y_te,
                        n_classes=2, n_epochs=args.n_epochs, device=device,
                    )
                elif name == "nemd_scratch":
                    acc, bal, f1 = _train_nemd_e2e(
                        X_tr, y_tr, X_va, y_va, X_te, y_te,
                        n_classes=2, sample_rate=fs,
                        n_epochs=args.n_epochs, device=device, ckpt_path=None,
                    )
                elif name == "nemd_pretrained":
                    acc, bal, f1 = _train_nemd_e2e(
                        X_tr, y_tr, X_va, y_va, X_te, y_te,
                        n_classes=2, sample_rate=fs,
                        n_epochs=args.n_epochs, device=device,
                        ckpt_path=args.nemd_ckpt,
                    )
                else:
                    print(f"  [{name}] unknown pipeline"); continue
                dt = time.perf_counter() - t0
                out["results"][name]["accs"].append(acc)
                out["results"][name]["bal_accs"].append(bal)
                out["results"][name]["f1s"].append(f1)
                print(f"  [{name:20s}] acc={acc:.4f}  bal={bal:.4f}  "
                      f"f1={f1:.4f}  ({dt:.0f}s)")
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")

            # Persist after each pipeline
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

    # Summary
    print("\n==== Summary (2-class; balanced-acc chance = 0.5) ====")
    for name in ["majority"] + pipelines:
        r = out["results"].get(name, {})
        accs = r.get("accs", [])
        bal  = r.get("bal_accs", [])
        f1s  = r.get("f1s", [])
        if accs:
            print(
                f"  {name:20s}  acc {np.mean(accs):.4f} ± {np.std(accs):.4f}"
                f"   bal {np.mean(bal):.4f} ± {np.std(bal):.4f}"
                f"   f1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  (n={len(accs)})"
            )


if __name__ == "__main__":
    main()
