"""CWRU 4-class bearing-fault classification with the NAFB pipeline set.

Classes: normal baseline, inner race, ball, outer race (0.007" fault,
drive-end, 12 kHz). Each MAT file carries ~480k-sample vibration
traces; we segment into non-overlapping 1024-sample windows. 4-class
classification with subject/file-level split so windows from the same
MAT file stay together.

Pipelines (subset of Sec V-D's list, selected to keep the experiment
tractable):
    emd_mlp_small, vmd_mlp_small, vmd_mlp_big, raw_cnn,
    nafb_default (big head)

Output: paper/figures/phase3_exp3_passB/cwru_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nemd.classical import ClassicalEMD, VMD
from nemd.model import NEMD
from nemd.losses import NEMDLoss
from experiments.applications.classification.features import imf_features
from experiments.applications.classification.models import (
    FeatureMLP, RawSignalCNN, NEMDClassifier,
)
from experiments.applications.classification.sweep_passB import pick_device, seed_everything


CLASS_ORDER = ["normal", "inner_race", "ball", "outer_race"]
DATA = Path("data/cwru_4class")
WIN = 1024
STRIDE = 1024   # non-overlapping
FS = 12000.0


def _load_mat_series(path: Path) -> np.ndarray:
    """Return the drive-end vibration signal from a CWRU MAT file."""
    blob = scipy.io.loadmat(path, squeeze_me=True)
    # CWRU files contain fields like 'X097_DE_time' for drive-end time series.
    for k, v in blob.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and "DE" in k:
            return v.astype(np.float32)
    # Fallback: take the largest 1D array
    arrs = [(k, v) for k, v in blob.items()
            if isinstance(v, np.ndarray) and v.ndim == 1]
    if not arrs:
        raise ValueError(f"No 1D signal in {path}")
    k, v = max(arrs, key=lambda kv: kv[1].size)
    return v.astype(np.float32)


def load_windows(win: int = WIN, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return X (N, win), y (N,), file_id per window (N,)."""
    X_list, y_list, fid_list = [], [], []
    for class_idx, cls in enumerate(CLASS_ORDER):
        cls_dir = DATA / cls
        files = sorted(cls_dir.glob("*.mat"))
        if not files:
            raise FileNotFoundError(f"No files for class {cls}")
        for fp in files:
            x = _load_mat_series(fp)
            # Standardise per file
            x = (x - x.mean()) / (x.std() + 1e-8)
            for start in range(0, len(x) - win + 1, stride):
                X_list.append(x[start:start + win])
                y_list.append(class_idx)
                fid_list.append(fp.stem)
    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, fid_list


def file_level_split(fids: list[str], seed: int,
                     train_frac: float = 0.6, val_frac: float = 0.2,
                     ) -> tuple[list[bool], list[bool], list[bool]]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(fids))
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(n * train_frac))
    n_va = int(round(n * val_frac))
    tr = set(uniq[:n_tr]); va = set(uniq[n_tr:n_tr + n_va])
    te = set(uniq[n_tr + n_va:])
    return ([f in tr for f in fids],
            [f in va for f in fids],
            [f in te for f in fids])


# ---------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------

def _train_feature_mlp(f_tr, y_tr, f_va, y_va, f_te, y_te,
                       hidden, n_epochs, device, seed):
    seed_everything(seed)
    m = FeatureMLP(
        in_dim=f_tr.shape[-1], hidden_dim=hidden, n_classes=4,
    ).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(f_tr, y_tr), batch_size=64, shuffle=True,
    )
    best = 0.0; best_state = None
    for _ in range(n_epochs):
        m.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va = (m(f_va.to(device)).argmax(-1).cpu() == y_va).float().mean().item()
        if va > best:
            best = va; best_state = copy.deepcopy(m.state_dict())
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        acc = (m(f_te.to(device)).argmax(-1).cpu() == y_te).float().mean().item()
    return acc, sum(p.numel() for p in m.parameters())


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


def _train_raw_cnn(X_tr, y_tr, X_va, y_va, X_te, y_te,
                   n_epochs, device, seed):
    seed_everything(seed)
    m = RawSignalCNN(n_classes=4).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True,
    )
    best = 0.0; best_state = None
    for _ in range(n_epochs):
        m.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va = (m(X_va.to(device)).argmax(-1).cpu() == y_va).float().mean().item()
        if va > best:
            best = va; best_state = copy.deepcopy(m.state_dict())
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        acc = (m(X_te.to(device)).argmax(-1).cpu() == y_te).float().mean().item()
    return acc, sum(p.numel() for p in m.parameters())


def _train_nafb(X_tr, y_tr, X_va, y_va, X_te, y_te,
                sample_rate, n_epochs, device, seed):
    seed_everything(seed)
    nemd = NEMD(num_imfs=3, hidden_dim=64, num_layers=3,
                sample_rate=sample_rate, temperature=0.3).to(device)
    clf = NEMDClassifier(
        nemd_model=nemd, n_classes=4, mlp_hidden=320,
        sample_rate=sample_rate,
    ).to(device)
    phys = NEMDLoss(
        lambda_sharp=1.0, lambda_order=1.0, lambda_ortho=0.1,
        lambda_balance=5.0, sample_rate=sample_rate,
    )
    opt = torch.optim.Adam(clf.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True,
    )
    best = 0.0; best_state = None
    for _ in range(n_epochs):
        clf.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _, meta = clf(xb)
            # Physics loss needs the raw (B, K, T) imfs, not the 3*K features.
            imfs, _, _ = clf.nemd(
                xb, temperature=clf.temperature,
                sort_by_centroid=clf.sort_by_centroid,
            )
            ce = F.cross_entropy(logits, yb)
            ph, _ = phys(imfs, meta)
            (ce + 0.1 * ph).backward()
            opt.step(); opt.zero_grad()
        clf.eval()
        with torch.no_grad():
            va_logits, _, _ = clf(X_va.to(device))
            va = (va_logits.argmax(-1).cpu() == y_va).float().mean().item()
        if va > best:
            best = va; best_state = copy.deepcopy(clf.state_dict())
    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        te_logits, _, _ = clf(X_te.to(device))
        acc = (te_logits.argmax(-1).cpu() == y_te).float().mean().item()
    return acc, sum(p.numel() for p in clf.parameters())


# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--out", type=str,
                   default="paper/figures/phase3_exp3_passB/cwru_results.json")
    p.add_argument("--pipelines", type=str,
                   default="emd_mlp,vmd_mlp,vmd_mlp_big,raw_cnn,nafb_default")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pipelines = [p.strip() for p in args.pipelines.split(",")]

    device = pick_device()
    print(f"Loading CWRU windows (win={WIN}, stride={STRIDE}, fs={FS}) ...")
    X, y, fids = load_windows()
    print(f"  loaded: {X.shape}  {len(set(fids))} files  "
          f"{np.bincount(y)} per-class counts")

    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "config": {"fs": FS, "win": WIN, "stride": STRIDE,
                   "classes": CLASS_ORDER, "seeds": seeds,
                   "n_epochs": args.n_epochs,
                   "n_windows": int(X.shape[0]),
                   "class_counts": np.bincount(y).tolist()},
        "results": {p: {"accs": [], "n_params_reported": None}
                    for p in pipelines},
    }

    for seed in seeds:
        tr, va, te = file_level_split(fids, seed)
        X_tr, y_tr = Xt[tr], yt[tr]
        X_va, y_va = Xt[va], yt[va]
        X_te, y_te = Xt[te], yt[te]
        print(f"\n--- seed {seed}: tr={len(y_tr)} va={len(y_va)} te={len(y_te)} ---")

        for name in pipelines:
            t0 = time.perf_counter()
            try:
                if name == "emd_mlp":
                    imfs_tr = _decompose_classical(X_tr.numpy(), "emd", 3, FS)
                    imfs_va = _decompose_classical(X_va.numpy(), "emd", 3, FS)
                    imfs_te = _decompose_classical(X_te.numpy(), "emd", 3, FS)
                    f_tr = imf_features(imfs_tr, sample_rate=FS)
                    f_va = imf_features(imfs_va, sample_rate=FS)
                    f_te = imf_features(imfs_te, sample_rate=FS)
                    acc, nparams = _train_feature_mlp(
                        f_tr, y_tr, f_va, y_va, f_te, y_te,
                        hidden=64, n_epochs=args.n_epochs,
                        device=device, seed=seed,
                    )
                elif name == "vmd_mlp":
                    imfs_tr = _decompose_classical(X_tr.numpy(), "vmd", 3, FS)
                    imfs_va = _decompose_classical(X_va.numpy(), "vmd", 3, FS)
                    imfs_te = _decompose_classical(X_te.numpy(), "vmd", 3, FS)
                    f_tr = imf_features(imfs_tr, sample_rate=FS)
                    f_va = imf_features(imfs_va, sample_rate=FS)
                    f_te = imf_features(imfs_te, sample_rate=FS)
                    acc, nparams = _train_feature_mlp(
                        f_tr, y_tr, f_va, y_va, f_te, y_te,
                        hidden=64, n_epochs=args.n_epochs,
                        device=device, seed=seed,
                    )
                elif name == "vmd_mlp_big":
                    imfs_tr = _decompose_classical(X_tr.numpy(), "vmd", 3, FS)
                    imfs_va = _decompose_classical(X_va.numpy(), "vmd", 3, FS)
                    imfs_te = _decompose_classical(X_te.numpy(), "vmd", 3, FS)
                    f_tr = imf_features(imfs_tr, sample_rate=FS)
                    f_va = imf_features(imfs_va, sample_rate=FS)
                    f_te = imf_features(imfs_te, sample_rate=FS)
                    acc, nparams = _train_feature_mlp(
                        f_tr, y_tr, f_va, y_va, f_te, y_te,
                        hidden=320, n_epochs=args.n_epochs,
                        device=device, seed=seed,
                    )
                elif name == "raw_cnn":
                    acc, nparams = _train_raw_cnn(
                        X_tr, y_tr, X_va, y_va, X_te, y_te,
                        n_epochs=args.n_epochs, device=device, seed=seed,
                    )
                elif name == "nafb_default":
                    acc, nparams = _train_nafb(
                        X_tr, y_tr, X_va, y_va, X_te, y_te,
                        sample_rate=FS,
                        n_epochs=args.n_epochs, device=device, seed=seed,
                    )
                else:
                    continue
                dt = time.perf_counter() - t0
                results["results"][name]["accs"].append(acc)
                results["results"][name]["n_params_reported"] = nparams
                print(f"  [{name:16s}] acc={acc:.4f}  "
                      f"params={nparams}  wall={dt:.0f}s")
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    print("\n==== Summary (CWRU 4-class; chance 0.25) ====")
    for name in pipelines:
        accs = results["results"][name]["accs"]
        if accs:
            a = np.array(accs)
            print(f"  {name:16s}  {a.mean()*100:5.2f} \u00b1 {a.std()*100:.2f}%  "
                  f"(n={len(accs)})")


if __name__ == "__main__":
    main()
