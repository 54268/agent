"""Stage-1 training: fit the shared stem + three evidence agents on known data.

No unknown sample is ever used here.  After training we collect evidence on the
known validation set (to fit empirical-CDF calibrators) and on the open test set
(known + real unknown, offline diagnosis only).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import load_oracle_npz, load_wisig_subset
from .evidence import collect_evidence
from .model import Stage1ModelConfig, ThreeAgentModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_splits(cfg: Dict):
    dataset = cfg["dataset"]
    augment = bool(cfg.get("augment_train", False))
    if dataset == "wisig":
        return load_wisig_subset(cfg["wisig_pkl"], augment_train=augment)
    if dataset == "oracle":
        return load_oracle_npz(cfg["oracle_root"], augment_train=augment)
    raise ValueError(f"unknown dataset {dataset}")


@torch.no_grad()
def _val_stats(model, loader, device) -> Dict[str, float]:
    model.eval()
    n_id = n_pr = total = 0
    rec_sum = 0.0
    gap_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, recon_classes=y)
        n_id += (out["identity"]["local_pred"] == y).sum().item()
        n_pr += (out["prototype"]["local_pred"] == y).sum().item()
        e_true = out["reconstruction"]["evidence"]["error"]
        dist = out["prototype"]["evidence"]["distances"]
        hard = dist.masked_fill(
            torch.nn.functional.one_hot(y, model.config.num_classes).bool(), float("inf")
        ).argmin(dim=1)
        e_wrong = model.reconstruction.reconstruction_error(x, out["h_s"], hard)
        rec_sum += e_true.sum().item()
        gap_sum += (e_wrong - e_true).sum().item()
        total += len(y)
    return {"val_identity_acc": n_id / total, "val_prototype_acc": n_pr / total,
            "val_recon_mse": rec_sum / total, "val_recon_wrong_gap": gap_sum / total}


def train_stage1(config: Dict) -> Dict:
    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = _load_splits(config)
    bs = int(config.get("batch_size", 256))
    train_loader = DataLoader(splits.train, batch_size=bs, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(splits.val, batch_size=512, shuffle=False, num_workers=0)
    open_loader = DataLoader(splits.open_test, batch_size=512, shuffle=False, num_workers=0)

    model_cfg = Stage1ModelConfig(
        num_classes=splits.num_known,
        embedding_dim=int(config.get("embedding_dim", 128)),
        stem_channels=int(config.get("stem_channels", 128)),
        proto_temperature=float(config.get("proto_temperature", 1.0)),
        proto_momentum=float(config.get("proto_momentum", 0.9)),
        compact_weight=float(config.get("compact_weight", 0.1)),
        margin_weight=float(config.get("margin_weight", 0.1)),
        margin=float(config.get("margin", 1.0)),
        recon_weight=float(config.get("recon_weight", 1.0)),
        proto_weight=float(config.get("proto_weight", 1.0)),
        recon_top_k=int(config.get("recon_top_k", 3)),
        latent_channels=int(config.get("latent_channels", 16)),
        signal_length=int(splits.signal_length),
        rec_wrong_weight=float(config.get("rec_wrong_weight", 1.0)),
        rec_margin=float(config.get("rec_margin", 0.1)),
    )
    model = ThreeAgentModel(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-3)),
                            weight_decay=float(config.get("weight_decay", 1e-4)))
    epochs = int(config.get("epochs", 40))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"[stage1] dataset={config['dataset']} device={device} "
          f"K={splits.num_known} train={len(splits.train)} val={len(splits.val)} "
          f"open={len(splits.open_test)} epochs={epochs}")

    model.refresh_prototypes(train_loader, device)
    history = []
    best_score, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = {}
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x, recon_classes=y)
            losses = model.losses(out, y)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            model.prototype.update_ema(
                out["prototype"]["embedding"].detach(), y
            )
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach()) * len(y)
        scheduler.step()
        model.refresh_prototypes(train_loader, device)

        n_train = len(splits.train)
        train_loss = {k: v / n_train for k, v in running.items()}
        stats = _val_stats(model, val_loader, device)
        row = {"epoch": epoch, **train_loss, **stats}
        history.append(row)
        score = 0.5 * (stats["val_identity_acc"] + stats["val_prototype_acc"])
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  ep{epoch:03d} loss={train_loss['total']:.4f} "
                  f"id_acc={stats['val_identity_acc']:.4f} pr_acc={stats['val_prototype_acc']:.4f} "
                  f"rec={stats['val_recon_mse']:.4f} gap={stats['val_recon_wrong_gap']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[stage1] best epoch={best_epoch} mean val acc={best_score:.4f} "
          f"elapsed={time.time()-t0:.1f}s")

    # evidence: calibrators are fit on known validation only
    val_data, calibrators = collect_evidence(model, val_loader, device,
                                             top_k=model_cfg.recon_top_k)
    open_data, _ = collect_evidence(model, open_loader, device,
                                    top_k=model_cfg.recon_top_k, calibrators=calibrators)

    import dataclasses
    torch.save({"state_dict": model.state_dict(), "config": dataclasses.asdict(model_cfg),
                "num_classes": splits.num_known, "best_epoch": best_epoch},
               out_dir / "model.pt")
    np.savez_compressed(out_dir / "evidence_open.npz", **open_data)
    np.savez_compressed(out_dir / "evidence_val.npz", **val_data)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return {"out_dir": str(out_dir), "history": history, "best_epoch": best_epoch,
            "val_data": val_data, "open_data": open_data, "splits": splits, "model": model,
            "device": device}
