"""Phase 3 Exp 3 runner: task-aware end-to-end classification.

Compares five pipelines:
  1. raw_cnn
  2. emd_mlp (fixed)
  3. vmd_mlp (fixed)
  4. nemd_pretrained (end-to-end fine-tune)
  5. nemd_scratch (end-to-end from random init)

Saves accuracy, F1, confusion matrices, training curves, and filter
response figures.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from nemd.model import NEMD
from nemd.train import TrainConfig
from experiments.applications.classification.dataset import make_splits
from experiments.applications.classification.train_pipelines import (
    PipelineResult,
    cache_features_classical,
    macro_f1,
    train_feature_pipeline,
    train_nemd_end_to_end,
    train_raw_cnn,
)


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 120,
})


def make_nemd_from_ckpt(ckpt_path: str, sample_rate: float) -> NEMD:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: TrainConfig = blob["config"]
    model = NEMD(
        num_imfs=config.num_imfs,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        sample_rate=sample_rate,
        temperature=0.3,
    )
    model.load_state_dict(blob["state_dict"])
    return model


def make_nemd_fresh(sample_rate: float, num_imfs: int = 3) -> NEMD:
    return NEMD(
        num_imfs=num_imfs, hidden_dim=64, num_layers=3,
        sample_rate=sample_rate, temperature=0.3,
    )


def plot_curves(results: list[PipelineResult], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        epochs = np.arange(1, len(r.val_acc_curve) + 1)
        axes[0].plot(epochs, r.train_acc_curve, "--", alpha=0.5,
                     label=f"{r.name} (train)")
        axes[0].plot(epochs, r.val_acc_curve, label=f"{r.name} (val)")
        axes[1].plot(epochs, r.val_loss_curve, label=r.name)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training / Validation accuracy", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation CE loss")
    axes[1].set_title("Validation loss", fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results: list[PipelineResult], out_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r.confusion)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{r.name}\nAcc {r.test_acc:.3f}", fontweight="bold",
                     fontsize=9)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["A", "B", "C"]); ax.set_yticklabels(["A", "B", "C"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle("Confusion matrices (test set)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_filter_responses(
    nemd_physics_only: NEMD | None,
    nemd_task_pretrained: NEMD | None,
    nemd_task_scratch: NEMD | None,
    sample_signal: torch.Tensor,
    sample_rate: float,
    out_path: Path,
) -> None:
    """Compare learned filter responses: physics-only vs task-aware."""
    variants = [
        ("Physics-only\n(Phase 2.5b+ ckpt)", nemd_physics_only),
        ("Task-aware (pretrained)", nemd_task_pretrained),
        ("Task-aware (scratch)", nemd_task_scratch),
    ]
    variants = [(n, m) for n, m in variants if m is not None]

    fig, axes = plt.subplots(len(variants), 1, figsize=(10, 3 * len(variants)))
    if len(variants) == 1:
        axes = [axes]
    x = sample_signal.unsqueeze(0)
    for ax, (name, model) in zip(axes, variants):
        model.eval()
        with torch.no_grad():
            _, _, meta = model(x, temperature=0.3, sort_by_centroid=True)
        filters = meta["filters"].squeeze(0).cpu().numpy()
        n_freqs = filters.shape[-1]
        freqs = np.linspace(0, sample_rate / 2, n_freqs)
        colors = ["C0", "C1", "C2"]
        for k in range(filters.shape[0]):
            ax.plot(freqs, filters[k], color=colors[k],
                    linewidth=1.2, label=f"Filter {k+1}")
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("H_k(f)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Learned filter responses: physics-only vs task-aware",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemd-ckpt", type=str,
                        default="checkpoints_p25b_v3/final.pt",
                        help="Pretrained N-EMD checkpoint")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--sample-rate", type=float, default=1000.0)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--epochs-classical", type=int, default=50)
    parser.add_argument("--epochs-nemd", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="paper/figures/phase3_exp3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipelines", type=str,
        default="raw_cnn,emd_mlp,vmd_mlp,nemd_pretrained,nemd_scratch",
        help="Comma-separated list of pipelines to run",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    print(f"\n=== Generating dataset "
          f"({args.n_train} / {args.n_val} / {args.n_test} per class) ===")
    t0 = time.perf_counter()
    splits = make_splits(
        n_train_per_class=args.n_train,
        n_val_per_class=args.n_val,
        n_test_per_class=args.n_test,
        n_samples=args.n_samples,
        fs=args.sample_rate,
        snr_db=args.snr_db,
        seed=args.seed,
    )
    print(f"    Data gen: {time.perf_counter() - t0:.1f}s")

    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    X_te, y_te = splits["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipelines_to_run = set(args.pipelines.split(","))
    all_results: list[PipelineResult] = []
    preprocessing_times: dict[str, float] = {}

    # ---- 1. Raw CNN ----
    if "raw_cnn" in pipelines_to_run:
        print(f"\n=== 1. Raw CNN ===")
        r = train_raw_cnn(
            "raw_cnn", X_tr, y_tr, X_va, y_va, X_te, y_te,
            n_epochs=args.epochs_classical, seed=args.seed, device=device,
        )
        print(f"    Test accuracy: {r.test_acc:.4f}  params: {r.n_params:,}  "
              f"wall {r.wallclock_sec:.1f}s")
        all_results.append(r)

    # ---- 2. EMD + MLP ----
    emd_preprocessing_time = None
    if "emd_mlp" in pipelines_to_run:
        print(f"\n=== 2. EMD + MLP ===")
        feats_tr, emd_train_t = cache_features_classical(
            X_tr, "emd", num_imfs=3, fs=args.sample_rate, verbose=True,
        )
        feats_va, _ = cache_features_classical(
            X_va, "emd", num_imfs=3, fs=args.sample_rate,
        )
        feats_te, emd_test_t = cache_features_classical(
            X_te, "emd", num_imfs=3, fs=args.sample_rate, verbose=True,
        )
        emd_preprocessing_time = emd_test_t
        preprocessing_times["emd_test_total_sec"] = emd_test_t
        preprocessing_times["emd_test_per_signal_ms"] = (
            emd_test_t * 1000 / len(X_te)
        )
        r = train_feature_pipeline(
            "emd_mlp", feats_tr, y_tr, feats_va, y_va, feats_te, y_te,
            n_epochs=args.epochs_classical, seed=args.seed, device=device,
        )
        print(f"    Test accuracy: {r.test_acc:.4f}  params: {r.n_params:,}  "
              f"wall {r.wallclock_sec:.1f}s")
        all_results.append(r)

    # ---- 3. VMD + MLP ----
    if "vmd_mlp" in pipelines_to_run:
        print(f"\n=== 3. VMD + MLP ===")
        feats_tr, _ = cache_features_classical(
            X_tr, "vmd", num_imfs=3, fs=args.sample_rate, verbose=True,
        )
        feats_va, _ = cache_features_classical(
            X_va, "vmd", num_imfs=3, fs=args.sample_rate,
        )
        feats_te, vmd_test_t = cache_features_classical(
            X_te, "vmd", num_imfs=3, fs=args.sample_rate, verbose=True,
        )
        preprocessing_times["vmd_test_total_sec"] = vmd_test_t
        preprocessing_times["vmd_test_per_signal_ms"] = (
            vmd_test_t * 1000 / len(X_te)
        )
        r = train_feature_pipeline(
            "vmd_mlp", feats_tr, y_tr, feats_va, y_va, feats_te, y_te,
            n_epochs=args.epochs_classical, seed=args.seed, device=device,
        )
        print(f"    Test accuracy: {r.test_acc:.4f}  params: {r.n_params:,}  "
              f"wall {r.wallclock_sec:.1f}s")
        all_results.append(r)

    # ---- 4. N-EMD pretrained, end-to-end ----
    nemd_pretrained_for_viz = None
    if "nemd_pretrained" in pipelines_to_run:
        print(f"\n=== 4. N-EMD (pretrained) + MLP end-to-end ===")
        nemd_model = make_nemd_from_ckpt(args.nemd_ckpt, args.sample_rate)
        r = train_nemd_end_to_end(
            "nemd_pretrained", X_tr, y_tr, X_va, y_va, X_te, y_te,
            nemd_model=nemd_model, sample_rate=args.sample_rate,
            n_epochs=args.epochs_nemd, seed=args.seed, device=device,
        )
        print(f"    Test accuracy: {r.test_acc:.4f}  params: {r.n_params:,}  "
              f"wall {r.wallclock_sec:.1f}s")
        all_results.append(r)
        nemd_pretrained_for_viz = r.model  # type: ignore[attr-defined]

    # ---- 5. N-EMD from scratch, end-to-end ----
    nemd_scratch_for_viz = None
    if "nemd_scratch" in pipelines_to_run:
        print(f"\n=== 5. N-EMD (scratch) + MLP end-to-end ===")
        nemd_model = make_nemd_fresh(args.sample_rate, num_imfs=3)
        r = train_nemd_end_to_end(
            "nemd_scratch", X_tr, y_tr, X_va, y_va, X_te, y_te,
            nemd_model=nemd_model, sample_rate=args.sample_rate,
            n_epochs=args.epochs_nemd, seed=args.seed, device=device,
        )
        print(f"    Test accuracy: {r.test_acc:.4f}  params: {r.n_params:,}  "
              f"wall {r.wallclock_sec:.1f}s")
        all_results.append(r)
        nemd_scratch_for_viz = r.model  # type: ignore[attr-defined]

    # ---- Summary table ----
    print(f"\n\n=== Summary (test set) ===")
    print(f"  {'Pipeline':<20} | {'Test Acc':>8} | {'Macro F1':>8} | "
          f"{'# params':>10} | {'Wall (s)':>9}")
    print("  " + "-" * 70)
    summary_rows = []
    for r in all_results:
        f1 = macro_f1(r.confusion)
        print(f"  {r.name:<20} | {r.test_acc:>8.4f} | {f1:>8.4f} | "
              f"{r.n_params:>10,} | {r.wallclock_sec:>9.1f}")
        summary_rows.append({
            "pipeline": r.name,
            "test_acc": r.test_acc,
            "macro_f1": f1,
            "n_params": r.n_params,
            "wallclock_sec": r.wallclock_sec,
            "confusion": r.confusion,
            "test_loss": r.test_loss,
        })

    print(f"\nPreprocessing times on test set ({len(X_te)} signals):")
    for k, v in preprocessing_times.items():
        print(f"  {k}: {v:.3f}")

    # ---- Save summary + figures ----
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "config": vars(args),
            "pipelines": summary_rows,
            "preprocessing_times": preprocessing_times,
        }, f, indent=2)
    print(f"\nSaved summary: {out_dir / 'summary.json'}")

    if len(all_results) > 0:
        plot_curves(all_results, out_dir / "training_curves.png")
        print(f"Saved: {out_dir / 'training_curves.png'}")
        plot_confusion_matrices(all_results, out_dir / "confusion_matrices.png")
        print(f"Saved: {out_dir / 'confusion_matrices.png'}")

    # Filter response comparison
    if any([nemd_pretrained_for_viz, nemd_scratch_for_viz]):
        sample_signal = X_te[0]
        physics_only = None
        try:
            physics_only = make_nemd_from_ckpt(args.nemd_ckpt, args.sample_rate)
        except Exception:
            pass
        plot_filter_responses(
            nemd_physics_only=physics_only,
            nemd_task_pretrained=(
                nemd_pretrained_for_viz.nemd if nemd_pretrained_for_viz else None
            ),
            nemd_task_scratch=(
                nemd_scratch_for_viz.nemd if nemd_scratch_for_viz else None
            ),
            sample_signal=sample_signal,
            sample_rate=args.sample_rate,
            out_path=out_dir / "filter_responses.png",
        )
        print(f"Saved: {out_dir / 'filter_responses.png'}")


if __name__ == "__main__":
    main()
