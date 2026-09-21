"""Independent C support-domain tools; no A/B hidden states or K-way head."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def probabilities(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def reports(a_logits: np.ndarray, b_logits: np.ndarray,
            view_weights: np.ndarray, view_distances: np.ndarray) -> dict:
    """Public report only: identities, confidence, competition, view source/anomaly."""
    a, b = probabilities(a_logits), probabilities(b_logits)
    def summary(p):
        order = np.argsort(p, axis=1)
        top = order[:, -1]
        second = order[:, -2]
        return top, second, p[np.arange(len(p)), top], (
            p[np.arange(len(p)), top] - p[np.arange(len(p)), second]), (
            -(p * np.log(p.clip(1e-12))).sum(axis=1))
    at, a2, ac, am, ae = summary(a)
    bt, b2, bc, bm, be = summary(b)
    vw = np.asarray(view_weights, dtype=np.float64)
    vd = np.asarray(view_distances, dtype=np.float64)
    if vw.ndim != 2 or vd.ndim != 3 or vw.shape != vd.shape[:2]:
        raise ValueError("B public view evidence dimensions differ")
    b_distance = np.sum(vw * vd[np.arange(len(vd)), :, bt], axis=1)
    return {"a_top1": at, "a_top2": a2, "a_conf": ac, "a_margin": am,
            "a_entropy": ae, "b_top1": bt, "b_top2": b2,
            "b_conf": bc, "b_margin": bm, "b_entropy": be,
            "b_view_source": vw.argmax(axis=1),
            "b_view_anomaly": b_distance,
            "b_view_concentration": vw.max(axis=1),
            "disagree": at != bt,
            "review_recommended": (at != bt) | (am < 0.15) | (bm < 0.15)}


def signal_descriptor(iq: np.ndarray) -> np.ndarray:
    """C-owned lightweight signal representation, independent of A/B encoders."""
    x = np.asarray(iq, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("C expects [N,2,T] I/Q")
    n, _, t = x.shape
    if t % 16:
        raise ValueError("I/Q length must be divisible by 16")
    z = x[:, 0] + 1j * x[:, 1]
    amp = np.abs(z)
    phase_step = np.angle(z[:, 1:] * np.conj(z[:, :-1]))
    phase_step = np.pad(phase_step, ((0, 0), (1, 0)))
    spectra = np.abs(np.fft.fft(z, axis=1)) / np.sqrt(t)
    parts = [x[:, 0], x[:, 1], amp, phase_step, spectra]
    return np.concatenate([v.reshape(n, 16, t // 16).mean(axis=2)
                           for v in parts], axis=1).astype(np.float32)


def _ranks(values: np.ndarray, candidates: np.ndarray,
           memories: dict[int, np.ndarray], fallback: np.ndarray) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float64)
    for i, (value, candidate) in enumerate(zip(values, candidates)):
        ref = memories.get(int(candidate), fallback)
        out[i] = np.searchsorted(ref, value, side="right") / len(ref)
    return out


class CReviewer:
    """Candidate validator with C-private prototypes, PCA and boundary PUG."""

    def __init__(self, seed: int = 42, pug_eta: float = 1.5):
        self.seed = int(seed)
        self.pug_eta = float(pug_eta)

    def fit(self, descriptors: np.ndarray, labels: np.ndarray,
            public: dict) -> "CReviewer":
        x = np.asarray(descriptors, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if len(x) != len(y) or np.any(y < 0):
            raise ValueError("C enrollment must contain Known samples only")
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), 0.05)
        z = np.clip((x - self.mean) / self.scale, -12, 12)
        self.classes = np.unique(y)
        self.prototypes = np.vstack([z[y == k].mean(0) for k in self.classes])
        self.variances = np.vstack([np.maximum(z[y == k].var(0), 0.25)
                                    for k in self.classes])
        self.pca = PCA(n_components=min(24, z.shape[1], len(z) - 1),
                       svd_solver="full").fit(z)
        train_candidate = np.asarray(public["a_top1"], dtype=np.int64)
        train_distance = self._distance(z, y)
        train_recon = self._reconstruction(z)
        self.distance_memory = {int(k): np.sort(train_distance[y == k])
                                for k in self.classes}
        self.distance_global = np.sort(train_distance)
        self.recon_memory = np.sort(train_recon)
        # Boundary seeds: disagreement, low margin or class-edge. Never proxy/formal Unknown.
        edge = _ranks(train_distance, y, self.distance_memory,
                      self.distance_global) >= 0.8
        boundary = (np.asarray(public["disagree"]) |
                    (np.asarray(public["a_margin"]) < 0.15) |
                    (np.asarray(public["b_margin"]) < 0.15) | edge)
        seed_indices = np.flatnonzero(boundary)
        if len(seed_indices) < 16:
            seed_indices = np.argsort(train_distance)[-max(16, len(y) // 5):]
        own = self.prototypes[y[seed_indices]]
        pair_distance = ((own[:, None, :] - self.prototypes[None, :, :]) ** 2).mean(2)
        pair_distance[np.arange(len(seed_indices)), y[seed_indices]] = np.inf
        rival = self.prototypes[pair_distance.argmin(1)]
        pseudo = z[seed_indices] + self.pug_eta * (z[seed_indices] - own) + 0.35 * (rival - own)
        known_features = self._raw_features(z, train_candidate, public)
        pseudo_public = {key: np.asarray(value)[seed_indices]
                         for key, value in public.items()}
        pseudo_features = self._raw_features(pseudo, y[seed_indices], pseudo_public)
        rng = np.random.default_rng(self.seed)
        known_indices = rng.choice(len(z), size=min(len(z), len(seed_indices) * 2), replace=False)
        fit_x = np.vstack([known_features[known_indices], pseudo_features])
        fit_y = np.r_[np.zeros(len(known_indices)), np.ones(len(pseudo_features))]
        self.pug_standard_mean = fit_x.mean(0)
        self.pug_standard_scale = np.maximum(fit_x.std(0), 1e-3)
        self.pug = LogisticRegression(max_iter=500, class_weight="balanced",
                                      random_state=self.seed).fit(
            (fit_x - self.pug_standard_mean) / self.pug_standard_scale, fit_y)
        self.pug_training = {"boundary_seed_count": int(len(seed_indices)),
                             "pseudo_count": int(len(pseudo_features)),
                             "train_known_count": int(len(known_indices))}
        return self

    def _z(self, x):
        return np.clip((np.asarray(x, dtype=np.float64) - self.mean) / self.scale, -12, 12)

    def _distance(self, z, candidate):
        candidate = np.asarray(candidate, dtype=np.int64)
        if np.any(candidate < 0) or np.any(candidate >= len(self.prototypes)):
            raise ValueError("candidate class outside C support")
        return np.mean((z - self.prototypes[candidate]) ** 2 /
                       self.variances[candidate], axis=1)

    def _reconstruction(self, z):
        return np.mean((z - self.pca.inverse_transform(self.pca.transform(z))) ** 2,
                       axis=1)

    def _raw_features(self, z, candidate, public):
        d = self._distance(z, candidate)
        r = self._reconstruction(z)
        return np.column_stack([d, r, 1 - np.asarray(public["a_conf"]),
                                1 - np.asarray(public["b_conf"]),
                                np.asarray(public["b_view_anomaly"]),
                                np.asarray(public["disagree"], dtype=float)])

    def scores(self, descriptors: np.ndarray, candidate: np.ndarray,
               public: dict) -> dict[str, np.ndarray]:
        z = self._z(descriptors)
        d = self._distance(z, candidate)
        r = self._reconstruction(z)
        proto = _ranks(d, candidate, self.distance_memory, self.distance_global)
        recon = np.searchsorted(self.recon_memory, r, side="right") / len(self.recon_memory)
        f = self._raw_features(z, candidate, public)
        pug = self.pug.predict_proba((f - self.pug_standard_mean) /
                                     self.pug_standard_scale)[:, 1]
        lite = 0.8 * proto + 0.2 * (1 - np.minimum(np.asarray(public["a_conf"]),
                                                  np.asarray(public["b_conf"])))
        full = 0.55 * proto + 0.20 * recon + 0.25 * pug
        return {"lite": lite, "full": full, "no_pug": 0.70 * proto + 0.30 * recon,
                "no_proto": 0.45 * recon + 0.55 * pug,
                "no_reconstruction": 0.65 * proto + 0.35 * pug,
                "prototype": proto, "reconstruction": recon, "pug": pug}


def route(public: dict, scores: dict, mode: str,
          lite_gate: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    n = len(public["a_top1"])
    disagree = np.asarray(public["disagree"], dtype=bool)
    low = (np.asarray(public["a_margin"]) < 0.15) | (
        np.asarray(public["b_margin"]) < 0.15) | (
        np.asarray(public["a_conf"]) < 0.55) | (
        np.asarray(public["b_conf"]) < 0.55)
    if mode == "all_full":
        full = np.ones(n, dtype=bool)
    elif mode == "disagreement_only":
        full = disagree
    elif mode == "disagreement_low":
        full = disagree | low
    elif mode == "disagreement_low_lite":
        full = disagree | low | (np.asarray(scores["lite"]) >= lite_gate)
    elif mode == "lite_only":
        full = np.zeros(n, dtype=bool)
        return np.asarray(scores["lite"]), full
    else:
        raise ValueError(f"unknown route: {mode}")
    return np.where(full, scores["full"], 0.0), full
