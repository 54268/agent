"""Generate PUG in I/Q space and certify it with fresh A/B/C evidence."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from maros_staged.evidence import ClassConditionalCdf
from maros_stage11.reviewer import signal_descriptor


def iq_candidates(train: dict, max_seeds_per_class: int, etas: tuple[float, ...],
                  seed: int) -> dict:
    """Move boundary Known I/Q toward a nearest rival class; never copy reports."""
    y = np.asarray(train["labels"], dtype=int)
    if np.any(y < 0):
        raise ValueError("PUG generation may use Known train I/Q only")
    iq = np.asarray(train["iq"], dtype=np.float32)
    desc = signal_descriptor(iq).astype(np.float64)
    scale = np.maximum(desc.std(axis=0), 0.1)
    z = (desc - desc.mean(axis=0)) / scale
    classes = np.unique(y)
    proto = np.stack([z[y == k].mean(axis=0) for k in classes])
    p = train["report"]
    boundary_score = (1 - np.minimum(p["a_margin"], p["b_margin"]) +
                      np.asarray(p["disagree"], dtype=float) +
                      np.asarray(train["u_proto"], dtype=float))
    rng = np.random.default_rng(seed)
    generated, sources, rivals, eta_values = [], [], [], []
    for cls in classes:
        local = np.flatnonzero(y == cls)
        ranked = local[np.argsort(boundary_score[local])[-max_seeds_per_class:]]
        for index in ranked:
            distances = ((proto - z[index]) ** 2).mean(axis=1)
            distances[int(np.flatnonzero(classes == cls)[0])] = np.inf
            rival_class = int(classes[np.argmin(distances)])
            candidates = np.flatnonzero(y == rival_class)
            rival = int(candidates[np.argmin(
                ((z[candidates] - z[index]) ** 2).mean(axis=1))])
            for eta in etas:
                # A bounded cross-class interpolation stays in consumable I/Q.
                # The certificate, not this generator, decides if it is Unknown-like.
                candidate = (1 - eta) * iq[index] + eta * iq[rival]
                generated.append(candidate.astype(np.float32))
                sources.append(int(index))
                rivals.append(rival)
                eta_values.append(float(eta))
    if not generated:
        raise RuntimeError("no PUG boundary seeds")
    return {"iq": np.stack(generated),
            "source_index": np.asarray(sources, dtype=int),
            "rival_index": np.asarray(rivals, dtype=int),
            "eta": np.asarray(eta_values, dtype=float)}


class SupportEvidence:
    """C sees only A/B reports plus candidate-conditioned prototype/EVT scalars."""

    def fit(self, calibration: dict):
        labels = np.asarray(calibration["b_pred"], dtype=int)
        self.open_cdf = ClassConditionalCdf(calibration["openmax_raw"], labels)
        self.geo_cdf = ClassConditionalCdf(calibration["b_d1"], labels)
        self.id_cdf = ClassConditionalCdf(calibration["a_d1"],
                                          np.asarray(calibration["a_pred"], dtype=int))
        return self

    def transform(self, data: dict) -> dict:
        b = np.asarray(data["b_pred"], dtype=int)
        a = np.asarray(data["a_pred"], dtype=int)
        u_open = self.open_cdf.score(data["openmax_raw"], b)
        u_proto = self.geo_cdf.score(data["b_d1"], b)
        u_id = self.id_cdf.score(data["a_d1"], a)
        data["u_open"] = u_open
        data["u_proto"] = u_proto
        data["u_idproto"] = u_id
        return data


def certificate(generated: dict, pseudo: dict, train: dict, calibration: dict,
                *, known_energy_min: float = 0.5,
                known_energy_max: float = 1.5) -> dict:
    """Fresh A/B reports and C support jointly certify plausible pseudo Unknown."""
    if len(generated["iq"]) != len(pseudo["labels"]):
        raise ValueError("every generated I/Q must have a fresh A/B result")
    if np.any(pseudo["labels"] != -1):
        raise ValueError("pseudo labels must be Unknown; never source Known labels")
    if "fresh_forward" not in pseudo or not pseudo["fresh_forward"]:
        raise ValueError("certificate refuses copied A/B reports")
    p = pseudo["report"]
    source = generated["source_index"]
    src_energy = np.mean(train["iq"][source] ** 2, axis=(1, 2))
    new_energy = np.mean(generated["iq"] ** 2, axis=(1, 2))
    ratio = new_energy / np.maximum(src_energy, 1e-10)
    plausible_energy = (ratio >= known_energy_min) & (ratio <= known_energy_max)
    # Distance-to-Known sanity check uses held-out Known calibration distribution,
    # never a formal Unknown or LCO proxy-Unknown label.
    train_desc = signal_descriptor(train["iq"])
    val_desc = signal_descriptor(calibration["iq"])
    pseudo_desc = signal_descriptor(generated["iq"])
    scaler = StandardScaler().fit(train_desc)
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1).fit(
        scaler.transform(train_desc))
    val_near = nn.kneighbors(scaler.transform(val_desc), return_distance=True)[0][:, 0]
    pseudo_near = nn.kneighbors(scaler.transform(pseudo_desc), return_distance=True)[0][:, 0]
    distance_cap = float(np.quantile(val_near, 0.99) * 1.5)
    not_too_far = pseudo_near <= distance_cap
    c_out = (np.asarray(pseudo["u_open"]) >= 0.9) & (
        (np.asarray(pseudo["u_proto"]) >= 0.9) |
        (np.asarray(pseudo["u_idproto"]) >= 0.9))
    conflict = (p["disagree"] | (p["a_margin"] < 0.15) |
                (p["b_margin"] < 0.15))
    hard = (~p["disagree"] & (p["a_conf"] >= 0.6) &
            (p["b_conf"] >= 0.6) & (p["a_margin"] >= 0.15) &
            (p["b_margin"] >= 0.15))
    # C's open-support test is mandatory for both boundary and hard type.
    accepted = plausible_energy & not_too_far & c_out & (conflict | hard)
    return {"accepted": accepted, "boundary": accepted & conflict,
            "hard": accepted & hard, "plausible_energy": plausible_energy,
            "not_too_far": not_too_far, "c_outside_support": c_out,
            "distance_cap": distance_cap,
            "nearest_known_distance": pseudo_near,
            "energy_ratio": ratio}


def pug_features(data: dict) -> np.ndarray:
    p = data["report"]
    return np.column_stack([
        data["u_open"], data["u_proto"], data["u_idproto"],
        1 - np.asarray(p["a_conf"]), 1 - np.asarray(p["b_conf"]),
        np.asarray(p["disagree"], dtype=float),
        1 - np.asarray(p["a_margin"]), 1 - np.asarray(p["b_margin"]),
        p["b_view_anomaly"],
    ]).astype(np.float64)


class PUGTool:
    """Candidate-sensitive binary open-space tool, not a K-class identity head."""

    def __init__(self, seed=42):
        self.seed = int(seed)
        self.enabled = False

    def fit(self, known: dict, pseudo: dict, mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(pseudo["labels"]):
            raise ValueError("PUG mask must identify freshly inferred pseudo rows")
        if mask.sum() < 12:
            return self
        rng = np.random.default_rng(self.seed)
        k = min(len(known["labels"]), 2 * int(mask.sum()))
        index = rng.choice(len(known["labels"]), k, replace=False)
        x_known = pug_features(known)[index]
        x_pseudo = pug_features(pseudo)[mask]
        x = np.vstack([x_known, x_pseudo])
        y = np.r_[np.zeros(len(x_known)), np.ones(len(x_pseudo))]
        self.scaler = StandardScaler().fit(x)
        self.model = LogisticRegression(max_iter=500, class_weight="balanced",
                                        random_state=self.seed).fit(
            self.scaler.transform(x), y)
        self.enabled = True
        return self

    def raw_score(self, data: dict) -> np.ndarray:
        if not self.enabled:
            return np.zeros(len(data["labels"]), dtype=np.float64)
        return self.model.predict_proba(
            self.scaler.transform(pug_features(data)))[:, 1]


def apply_increment(base: np.ndarray, tool: np.ndarray, weight: float) -> np.ndarray:
    """C can add evidence; it cannot replace a stronger Stage-5 risk signal."""
    base = np.asarray(base, dtype=np.float64)
    tool = np.asarray(tool, dtype=np.float64)
    return np.clip(base + float(weight) * np.maximum(tool - base, 0), 0, 1)
