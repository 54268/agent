"""Screen Geometry Agent observations on one ORACLE leave-class-out fold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage5.protocol import build_lco_folds, class_subset  # noqa: E402
from maros_stage5.training import train_experts  # noqa: E402
from maros_staged.datasets import load_oracle_npz, load_wisig_subset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["oracle", "wisig"], default="oracle")
    parser.add_argument("--oracle-root", default="data/oracle/processed_k10u6")
    parser.add_argument("--wisig-pkl",
                        default="data/WiSig/WiSig_OpenSet_1Day_1Rx/wisig_openset_subset.pkl")
    parser.add_argument("--output", default="results/stage5_oracle/geometry_view_screen.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--views", nargs="+", default=[
        "spectral", "fft_iq", "envelope_phase", "difference_iq", "raw"])
    args = parser.parse_args()
    oracle_root = Path(args.oracle_root)
    if not oracle_root.is_absolute():
        oracle_root = ROOT / oracle_root
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    if args.dataset == "oracle":
        splits = load_oracle_npz(oracle_root, augment_train=False)
    else:
        wisig_path = Path(args.wisig_pkl)
        if not wisig_path.is_absolute():
            wisig_path = ROOT / wisig_path
        splits = load_wisig_subset(wisig_path, augment_train=False)
    fold = build_lco_folds(splits.num_known, 5, 2026)[0]
    train = class_subset(splits.train, fold.support_classes, remap=True)
    val = class_subset(splits.val, fold.support_classes, remap=True)
    base = {
        "state_dim": 128, "stem_channels": 128, "message_dim": 64,
        "prototype_temperature": 0.15, "batch_size": 512,
        "expert_epochs": args.epochs, "expert_lr": 0.001,
        "expert_weight_decay": 0.0001, "supcon_temperature": 0.1,
        "supcon_weight": 0.1, "compact_weight": 0.1, "margin_weight": 0.1,
        "prototype_margin": 0.5, "diversity_weight": 0.01,
        "reconstruction_weight": 0.05,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for view in args.views:
        print(f"[oracle:view] {view}")
        cfg = {**base, "geometry_view": view}
        _, history, best_epoch = train_experts(
            train, val, len(fold.support_classes), cfg, device, 4200)
        best = history[best_epoch - 1]
        rows.append({"view": view, "best_epoch": best_epoch,
                     "identity_val_accuracy": best["identity_val_accuracy"],
                     "geometry_val_accuracy": best["geometry_val_accuracy"],
                     "joint_mean_accuracy": 0.5 * (best["identity_val_accuracy"]
                                                    + best["geometry_val_accuracy"])})
    report = {"protocol": f"{args.dataset} fold0: "
                          f"{len(fold.support_classes)} support / {len(fold.heldout_classes)} heldout",
              "epochs": args.epochs, "rows": rows,
              "selected": max(rows, key=lambda row: row["geometry_val_accuracy"])["view"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
