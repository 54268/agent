"""Evidence calibration, cross-agent disagreement and evidence collection.

Stage 1 deliberately uses *no real unknown* for calibration.  Each raw
unknown-oriented score (1-confidence, nearest-prototype distance, best-case
reconstruction error) is mapped to ``u in [0, 1]`` with the empirical CDF of
that score on the known validation set, so ``u = 0.95`` literally means "more
anomalous than 95% of known validation samples".  Disagreement follows Section
10 of the design doc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


class EmpiricalCdfCalibrator:
    def __init__(self, values: np.ndarray):
        self.ref = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        rank = np.searchsorted(self.ref, values, side="right")
        return rank / max(len(self.ref), 1)


class ClassConditionalCdf:
    """Unknown score = empirical CDF on known validation, with shrinkage.

    A global CDF is blended with a per-class CDF keyed on the predicted known
    class (design doc Sections 8.4 / 9.3 class-conditional thresholds).  Only
    known validation samples are used; small classes shrink toward the global
    CDF via ``n / (n + n_prior)``.
    """

    def __init__(self, values: np.ndarray, classes: np.ndarray, n_prior: float = 50.0):
        values = np.asarray(values, dtype=np.float64)
        classes = np.asarray(classes).reshape(-1)
        self.global_cdf = EmpiricalCdfCalibrator(values)
        self.n_prior = float(n_prior)
        self.class_cdfs: Dict[int, EmpiricalCdfCalibrator] = {}
        for cls in np.unique(classes):
            self.class_cdfs[int(cls)] = EmpiricalCdfCalibrator(values[classes == cls])

    def score(self, values: np.ndarray, classes: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        classes = np.asarray(classes).reshape(-1)
        u_global = self.global_cdf.score(values)
        u = np.empty_like(u_global)
        for cls, cdf in self.class_cdfs.items():
            mask = classes == cls
            if not np.any(mask):
                continue
            u_local = cdf.score(values[mask])
            n = len(cdf.ref)
            alpha = n / (n + self.n_prior)
            u[mask] = alpha * u_local + (1.0 - alpha) * u_global[mask]
        missing = np.ones(len(values), dtype=bool)
        for cls in self.class_cdfs:
            missing[classes == cls] = False
        if np.any(missing):
            u[missing] = u_global[missing]
        return u


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence normalised to [0, 1]."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * (np.log(a) - np.log(b)), axis=1)

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / np.log(2.0)


@torch.no_grad()
def collect_evidence(model, loader: DataLoader, device: torch.device,
                     top_k: int = 3, calibrators: Optional[Dict[str, EmpiricalCdfCalibrator]] = None):
    """Run the three agents over a loader and gather per-sample evidence."""
    model.eval()
    records = {k: [] for k in (
        "y", "pred_id", "pred_proto", "rec_cand", "conf", "entropy", "margin_id",
        "d1", "d2", "margin_proto", "rec_err", "rec_best", "rec_err_true",
        "p_id", "p_proto", "z_id", "z_proto", "z_rec",
    )}

    for x, y in loader:
        x = x.to(device)
        out = model(x)
        id_ev = out["identity"]["evidence"]
        pr_ev = out["prototype"]["evidence"]
        h_s = out["h_s"]

        fused_p = 0.5 * id_ev["prob"] + 0.5 * pr_ev["prob"]
        topk = fused_p.topk(top_k, dim=-1).indices
        cand = topk[:, 0]
        rec_err = model.reconstruction.reconstruction_error(x, h_s, cand).cpu().numpy()
        rec_best = model.reconstruction.best_case_error(x, h_s, topk).cpu().numpy()

        records["y"].append(y.numpy())
        records["pred_id"].append(id_ev["pred"].cpu().numpy())
        records["pred_proto"].append(pr_ev["pred"].cpu().numpy())
        records["rec_cand"].append(cand.cpu().numpy())
        records["conf"].append(id_ev["confidence"].cpu().numpy())
        records["entropy"].append(id_ev["entropy"].cpu().numpy())
        records["margin_id"].append(id_ev["margin"].cpu().numpy())
        records["d1"].append(pr_ev["d1"].cpu().numpy())
        records["d2"].append(pr_ev["d2"].cpu().numpy())
        records["margin_proto"].append(pr_ev["margin"].cpu().numpy())
        records["rec_err"].append(rec_err)
        records["rec_best"].append(rec_best)
        records["p_id"].append(id_ev["prob"].cpu().numpy())
        records["p_proto"].append(pr_ev["prob"].cpu().numpy())
        records["z_id"].append(out["identity"]["embedding"].cpu().numpy())
        records["z_proto"].append(out["prototype"]["embedding"].cpu().numpy())
        records["z_rec"].append(
            model.reconstruction(x, h_s, id_ev["pred"])["embedding"].cpu().numpy()
        )

        # reconstruction under the ground-truth class (known splits only; -1 for unknown)
        y_dev = y.to(device)
        known_mask = (y >= 0)
        rec_true = np.full(len(y), np.nan, dtype=np.float32)
        if bool(known_mask.any()):
            err = model.reconstruction.reconstruction_error(
                x[known_mask], h_s[known_mask], y_dev[known_mask]
            )
            rec_true[known_mask.numpy()] = err.cpu().numpy()
        records["rec_err_true"].append(rec_true)

    data = {k: (np.concatenate(v, axis=0) if k.startswith(("p_", "z_")) else np.concatenate(v))
            for k, v in records.items()}

    num_classes = data["p_id"].shape[1]
    # raw unknown-oriented score and the known class used for class-conditional CDF
    raw = {
        "identity": (1.0 - data["conf"], data["pred_id"]),
        "prototype": (data["d1"], data["pred_proto"]),
        "reconstruction": (data["rec_err"], data["rec_cand"]),
    }
    if calibrators is None:
        calibrators = {name: ClassConditionalCdf(vals, cls) for name, (vals, cls) in raw.items()}
    u = {f"u_{name}": calibrators[name].score(vals, cls) for name, (vals, cls) in raw.items()}
    data.update(u)

    # disagreement (Section 10)
    d_ip = js_divergence(data["p_id"], data["p_proto"])
    d_label = (data["pred_id"] != data["pred_proto"]).astype(np.float64)
    d_confgeo = data["conf"] * u["u_prototype"]
    d_confrec = data["conf"] * u["u_reconstruction"]
    data["d_ip"] = d_ip
    data["d_label"] = d_label
    data["d_confgeo"] = d_confgeo
    data["d_confrec"] = d_confrec
    data["disagreement"] = 0.25 * (d_ip + d_label + d_confgeo + d_confrec)

    # fixed label-free fusions of calibrated unknown evidence:
    #  - mean : consensus (works when channels are similarly strong)
    #  - max  : OR rule (works when channels catch different unknowns)
    u_stack = np.stack([u["u_identity"], u["u_prototype"], u["u_reconstruction"]], axis=1)
    data["u_fused"] = u_stack.mean(axis=1)
    data["u_fused_mean"] = data["u_fused"]
    data["u_fused_max"] = u_stack.max(axis=1)
    data["closed_pred"] = np.where(
        data["p_id"][np.arange(len(data["y"])), :].max(axis=1)
        >= data["p_proto"][np.arange(len(data["y"])), :].max(axis=1),
        data["pred_id"], data["pred_proto"],
    )
    return data, calibrators
