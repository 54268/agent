"""Audit local prototype geometry after a completed Stage-5 run.

This script is diagnostic only: open-test metrics must never be fed back into
LCO selection.  It reports whether useful separation exists inside a local
sensor but is destroyed by a downstream gate or fusion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage5.agents import RoleStructuredExperts, UNIVERSAL_VIEW_NAMES  # noqa: E402
from maros_staged.datasets import load_oracle_npz, load_wisig_subset  # noqa: E402


@torch.no_grad()
def collect(model, dataset, device):
    rows = {"identity": [], "geometry": [], "views": [], "labels": []}
    model.eval()
    for x, y in DataLoader(dataset, batch_size=512, shuffle=False):
        out = model(x.to(device))
        rows["identity"].append(out["identity"].state.cpu().numpy())
        rows["geometry"].append(out["geometry"].state.cpu().numpy())
        rows["views"].append(out["geometry"].evidence["view_states"].cpu().numpy())
        rows["labels"].append(y.numpy())
    return {key: np.concatenate(value) for key, value in rows.items()}


def evaluate(train_z, train_y, open_z, open_y):
    train_z = train_z / np.maximum(np.linalg.norm(train_z, axis=1, keepdims=True), 1e-8)
    open_z = open_z / np.maximum(np.linalg.norm(open_z, axis=1, keepdims=True), 1e-8)
    classes = int(train_y.max()) + 1
    prototypes = np.stack([train_z[train_y == cls].mean(0) for cls in range(classes)])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    train_dist = ((train_z[:, None] - prototypes[None]) ** 2).sum(2)
    open_dist = ((open_z[:, None] - prototypes[None]) ** 2).sum(2)
    pred = open_dist.argmin(1)
    dmin = open_dist[np.arange(len(open_dist)), pred]
    means = np.asarray([train_dist[train_y == cls, cls].mean() for cls in range(classes)])
    stds = np.asarray([train_dist[train_y == cls, cls].std() + 1e-6 for cls in range(classes)])
    zscore = (dmin - means[pred]) / stds[pred]
    unknown = open_y == -1
    known = ~unknown
    return {
        "known_nearest_prototype_accuracy": float((pred[known] == open_y[known]).mean()),
        "dmin_auroc": float(roc_auc_score(unknown, dmin)),
        "class_z_auroc": float(roc_auc_score(unknown, zscore)),
        "known_dmin_mean": float(dmin[known].mean()),
        "unknown_dmin_mean": float(dmin[unknown].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config); cfg_path = cfg_path if cfg_path.is_absolute() else ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg["dataset"] == "oracle":
        source = Path(cfg["oracle_root"]); source = source if source.is_absolute() else ROOT / source
        splits = load_oracle_npz(source, augment_train=False)
    else:
        source = Path(cfg["wisig_pkl"]); source = source if source.is_absolute() else ROOT / source
        splits = load_wisig_subset(source, augment_train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoleStructuredExperts(
        splits.num_known, int(cfg["state_dim"]), int(cfg["stem_channels"]),
        int(cfg["message_dim"]), float(cfg["prototype_temperature"]),
        str(cfg["geometry_view"])).to(device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute(): checkpoint = ROOT / checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["experts"])
    train = collect(model, splits.train, device)
    opened = collect(model, splits.open_test, device)
    report = {
        "warning": "diagnostic open-test audit; never use for LCO selection",
        "identity": evaluate(train["identity"], train["labels"], opened["identity"], opened["labels"]),
        "geometry_fused": evaluate(train["geometry"], train["labels"], opened["geometry"], opened["labels"]),
        "geometry_views": {
            name: evaluate(train["views"][:, index], train["labels"],
                           opened["views"][:, index], opened["labels"])
            for index, name in enumerate(UNIVERSAL_VIEW_NAMES)
        },
    }
    output = Path(args.output); output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
